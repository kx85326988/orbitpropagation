using LinearAlgebra
using StaticArrays
using Plots
using Printf
using Base64
using Dates
using Statistics
using SatelliteToolbox 

# --- 物理定数 ---
const mu_earth = 3.986004418e14  # 地心重力定数 (m^3/s^2)
const J2_coeff = 1.08263e-3     # J2係数
const R_E = 6378137.0            # 地球半径 (m)

# --- 主衛星の初期軌道要素 ---
a_c_stm_init = 6903137.0        # m
e_c_stm_init = 0.0022           #
i_c_stm_init = deg2rad(97.65)   #
Omega_c_stm_init = deg2rad(0.0) #
omega_c_stm_init = deg2rad(0.0) #
M_c_stm_init = deg2rad(0.0)     #

# --- 編隊飛行関連パラメータ ---
delta_v_magnitude = 0.1 # m/s
delta_a_dot_drag_initial_normalized = -1.0e-7 # [1/s]
delta_B_initial_param = 0.01 # 仮の差動弾道係数

# --- 構造体定義 ---
# プログラム内で一貫して使用するためのカスタム構造体
struct OrbitalElementsClassical
    a::Float64; e::Float64; i::Float64; RAAN::Float64
    omega::Float64; f_true::Float64; n::Float64; M::Float64
end

# ECI状態を渡すためのカスタム構造体
struct CartesianStateECI
    r_vec::SVector{3, Float64}
    v_vec::SVector{3, Float64}
end

struct QuasiNonsingularROEsKoenig
    delta_a_norm::Float64; delta_lambda::Float64; delta_ex::Float64
    delta_ey::Float64; delta_ix::Float64; delta_iy::Float64
end

@enum PerturbationType begin KEPLER_ONLY; J2_ONLY; DRAG_ONLY; J2_AND_DRAG end
@enum DragModelTypeForSTM begin NO_DRAG; DENSITY_MODEL_FREE; DENSITY_MODEL_SPECIFIC end
@enum SeparationPlane begin RT_PLANE; RN_PLANE; NT_PLANE end

"""
ECI座標系の位置・速度ベクトルから古典軌道要素を計算する (SatelliteToolbox.jlを使用)。
"""
function sv_to_orbital_elements(state::CartesianStateECI, epoch::Float64 = 0.0)::OrbitalElementsClassical
    # ★★★ 修正: OrbitStateVector構造体を作成してsv_to_keplerに渡す ★★★
    # OrbitStateVectorは 時刻, 位置ベクトル, 速度ベクトル を引数に取る
    sv = OrbitStateVector(epoch, state.r_vec, state.v_vec)
    
    # sv_to_keplerはOrbitStateVectorを引数に取り、KeplerianElementsを返す
    # muは内部でデフォルト値が使われる
    kep = SatelliteToolbox.sv_to_kepler(sv)
    
    # fからMを計算
    M_val = SatelliteToolbox.true_to_mean_anomaly(kep.e, kep.f)
    
    # nをaから手動で計算 (KeplerianElements構造体はnを含まないため)
    n_val = sqrt(mu_earth / kep.a^3)
    
    return OrbitalElementsClassical(
        kep.a, kep.e, kep.i, kep.Ω, kep.ω, kep.f,
        n_val, # 計算したnを使用
        M_val
    )
end

"""
カスタムのOrbitalElementsClassical構造体からECI座標系の位置・速度ベクトルを計算する。
"""
function orbital_elements_to_sv(oe::OrbitalElementsClassical, epoch::Float64 = 0.0)::SVector{6,Float64}
    # Mからfへの変換
    f_true_val = SatelliteToolbox.mean_to_true_anomaly(oe.e, oe.M)
    
    # SatelliteToolbox.jlのKeplerianElements構造体を作成
    keps = KeplerianElements(epoch, oe.a, oe.e, oe.i, oe.RAAN, oe.omega, f_true_val)
    
    # kepler_to_svはKeplerianElementsを引数に取る
    sv_out = SatelliteToolbox.kepler_to_sv(keps)
    
    return vcat(sv_out.r, sv_out.v)
end

function cw_to_eci_deputy_state(r_chief_eci::SVector{3,Float64}, v_chief_eci::SVector{3,Float64}, dr_lvlh::SVector{3,Float64}, dv_lvlh::SVector{3,Float64})::CartesianStateECI
    r_c_hat=normalize(r_chief_eci); h_c_vec=cross(r_chief_eci,v_chief_eci); h_c_hat_val=normalize(h_c_vec)
    if norm(h_c_vec)<1e-9; t_c_hat_temp=normalize(v_chief_eci); if abs(dot(r_c_hat,t_c_hat_temp))>1.0-1e-6; temp_axis=abs(t_c_hat_temp[1])<0.9 ? SVector(1.0,0,0) : SVector(0,1.0,0); h_c_hat_val=normalize(cross(t_c_hat_temp,temp_axis)); else; h_c_hat_val=normalize(cross(r_c_hat,t_c_hat_temp)); end; end
    t_c_hat_final=normalize(cross(h_c_hat_val,r_c_hat)); dcm_lvlh_to_eci=hcat(r_c_hat,t_c_hat_final,h_c_hat_val)
    dr_eci=dcm_lvlh_to_eci*dr_lvlh; r_deputy_eci=r_chief_eci+dr_eci
    omega_lvlh_scalar=dot(h_c_vec,h_c_hat_val)/(norm(r_chief_eci)^2)
    omega_vector_lvlh_frame=SVector(0.0,0.0,omega_lvlh_scalar)
    dv_eci_relative=dcm_lvlh_to_eci*(dv_lvlh+cross(omega_vector_lvlh_frame,dr_lvlh))
    v_deputy_eci=v_chief_eci+dv_eci_relative
    return CartesianStateECI(r_deputy_eci,v_deputy_eci)
end

function orbital_elements_to_qns_roe_koenig(oe_c::OrbitalElementsClassical, oe_d::OrbitalElementsClassical)::QuasiNonsingularROEsKoenig
    delta_a_norm_val=(oe_d.a-oe_c.a)/oe_c.a; term_lambda_deputy=oe_d.M+oe_d.omega+oe_d.RAAN*cos(oe_d.i); term_lambda_chief=oe_c.M+oe_c.omega+oe_c.RAAN*cos(oe_c.i)
    delta_lambda_val=mod(term_lambda_deputy-term_lambda_chief+pi,2*pi)-pi; delta_ex_val=oe_d.e*cos(oe_d.omega)-oe_c.e*cos(oe_c.omega); delta_ey_val=oe_d.e*sin(oe_d.omega)-oe_c.e*sin(oe_c.omega)
    delta_ix_val=oe_d.i-oe_c.i; delta_Omega_val=mod(oe_d.RAAN-oe_c.RAAN+pi,2*pi)-pi; delta_iy_val=delta_Omega_val*sin(oe_c.i)
    return QuasiNonsingularROEsKoenig(delta_a_norm_val,delta_lambda_val,delta_ex_val,delta_ey_val,delta_ix_val,delta_iy_val)
end

function eci_to_hill(r_eci::SVector{3,Float64}, v_eci::SVector{3,Float64}, r_chief_eci::SVector{3,Float64}, v_chief_eci::SVector{3,Float64})::SVector{3,Float64}
    r_c_hat=normalize(r_chief_eci); h_c_vec=cross(r_chief_eci,v_chief_eci); h_c_hat_val=normalize(h_c_vec)
    if norm(h_c_vec)<1e-9; t_c_hat_temp=normalize(v_chief_eci); if abs(dot(r_c_hat,t_c_hat_temp))>1.0-1e-6; temp_axis=abs(t_c_hat_temp[1])<0.9 ? SVector(1.0,0,0) : SVector(0,1.0,0); h_c_hat_val=normalize(cross(t_c_hat_temp,temp_axis)); else; h_c_hat_val=normalize(cross(r_c_hat,t_c_hat_temp)); end; end
    t_c_hat_final=normalize(cross(h_c_hat_val,r_c_hat)); dcm_eci_to_hill=transpose(hcat(r_c_hat,t_c_hat_final,h_c_hat_val))
    dr_eci = r_eci - r_chief_eci
    return dcm_eci_to_hill * dr_eci
end

"""
ECI座標系での相対位置ベクトルを、主衛星基準のHill座標系に変換する。
"""
function eci_to_hill_rel_pos(r_deputy_eci::SVector{3,Float64}, r_chief_eci::SVector{3,Float64}, v_chief_eci::SVector{3,Float64})::SVector{3,Float64}
    # 主衛星基準のHill座標系の軸を計算
    r_c_hat = normalize(r_chief_eci)
    h_c_vec = cross(r_chief_eci, v_chief_eci)
    h_c_hat_val = normalize(h_c_vec)
    if norm(h_c_vec) < 1e-9
        t_c_hat_temp = normalize(v_chief_eci)
        if abs(dot(r_c_hat, t_c_hat_temp)) > 1.0 - 1e-6
            temp_axis = abs(t_c_hat_temp[1]) < 0.9 ? SVector(1.0,0,0) : SVector(0,1.0,0)
            h_c_hat_val = normalize(cross(t_c_hat_temp, temp_axis))
        else
            h_c_hat_val = normalize(cross(r_c_hat, t_c_hat_temp))
        end
    end
    t_c_hat_final = normalize(cross(h_c_hat_val, r_c_hat))
    
    # ECIからHillへの回転行列を作成
    dcm_eci_to_hill = transpose(hcat(r_c_hat, t_c_hat_final, h_c_hat_val))
    
    # ECI座標系での相対位置ベクトルを計算
    dr_eci = r_deputy_eci - r_chief_eci
    
    # Hill座標系に変換
    return dcm_eci_to_hill * dr_eci
end

function calculate_j2_perturbation_eci(r_eci_vec::SVector{3,Float64}, mu::Float64, j2_val::Float64, r_earth_eq::Float64)::SVector{3,Float64}
    x,y,z=r_eci_vec; r_sq=dot(r_eci_vec,r_eci_vec); r=sqrt(r_sq); if r<1e-3; return SVector(0.0,0.0,0.0); end
    term_common=-1.5*mu*j2_val*r_earth_eq^2/(r^5); z_sq_r_sq=(z^2)/r_sq
    ax=term_common*x*(1.0-5.0*z_sq_r_sq); ay=term_common*y*(1.0-5.0*z_sq_r_sq); az=term_common*z*(3.0-5.0*z_sq_r_sq)
    return SVector(ax,ay,az)
end

function eci_to_hill_relative_acceleration(a_relative_eci::SVector{3,Float64}, r_chief_eci::SVector{3,Float64}, v_chief_eci::SVector{3,Float64})::SVector{3,Float64}
    r_c_hat=normalize(r_chief_eci); h_c_vec=cross(r_chief_eci,v_chief_eci); h_c_hat_val=normalize(h_c_vec)
    if norm(h_c_vec)<1e-9; t_c_hat_temp=normalize(v_chief_eci); if abs(dot(r_c_hat,t_c_hat_temp))>1.0-1e-6; temp_axis=abs(t_c_hat_temp[1])<0.9 ? SVector(1.0,0,0) : SVector(0,1.0,0); h_c_hat_val=normalize(cross(t_c_hat_temp,temp_axis)); else; h_c_hat_val=normalize(cross(r_c_hat,t_c_hat_temp)); end; end
    t_c_hat_final=normalize(cross(h_c_hat_val,r_c_hat)); dcm_eci_to_hill=transpose(hcat(r_c_hat,t_c_hat_final,h_c_hat_val))
    a_relative_hill=dcm_eci_to_hill*a_relative_eci
    return a_relative_hill
end

function final_roe_to_deputy_oe(oe_chief_final::OrbitalElementsClassical, final_roes::SVector{7,Float64})::OrbitalElementsClassical
    ac,ec,ic,Omegac,omegac,Mc=oe_chief_final.a,oe_chief_final.e,oe_chief_final.i,oe_chief_final.RAAN,oe_chief_final.omega,oe_chief_final.M
    delta_a_norm_val=final_roes[1]; delta_lambda_val=final_roes[2]; delta_ex_val=final_roes[3]; delta_ey_val=final_roes[4]; delta_ix_val=final_roes[5]; delta_iy_val=final_roes[6]
    ad=ac*(1.0+delta_a_norm_val); id=ic+delta_ix_val; Omegad=Omegac
    if abs(sin(ic))>1e-7; Omegad=Omegac+delta_iy_val/sin(ic); end; Omegad=mod(Omegad,2*pi)
    X=delta_ex_val+ec*cos(omegac); Y=delta_ey_val+ec*sin(omegac); ed=sqrt(X^2+Y^2); if ed<1e-10; ed=1e-10; end
    omegad=0.0; if ed>1e-9; omegad=atan(Y,X); if omegad<0.0; omegad+=2*pi; end; end
    Md=delta_lambda_val+(Mc+omegac+Omegac*cos(ic))-(omegad+Omegad*cos(id)); Md=mod(Md,2*pi); if Md<0.0; Md+=2*pi; end
    nd=sqrt(mu_earth/abs(ad)^3)
    f_true_d_val = SatelliteToolbox.mean_to_true_anomaly(ed, Md)
    return OrbitalElementsClassical(ad,ed,id,Omegad,omegad,f_true_d_val,nd,Md)
end

function get_secular_j2_rates_koenig(ac::Float64, ec::Float64, ic::Float64)::Tuple{Float64,Float64}
    n_c=sqrt(mu_earth/ac^3); eta_c=sqrt(1.0-ec^2); if eta_c<1e-9; eta_c=1e-9; end
    common_factor=(3.0/4.0)*J2_coeff*(R_E/ac)^2*n_c/(eta_c^4)
    omega_dot=common_factor*(5.0*cos(ic)^2-1.0); Omega_dot=common_factor*(-2.0*cos(ic))
    return omega_dot,Omega_dot
end

function get_J_qns_augmented_koenig(omega_c_val::Float64)::SMatrix{7,7,Float64}
    J_aug=@MMatrix fill(0.0,7,7); J_aug[1,1]=1.0; J_aug[2,2]=1.0; cos_wc=cos(omega_c_val); sin_wc=sin(omega_c_val)
    J_aug[3,3]=cos_wc; J_aug[3,4]=sin_wc; J_aug[4,3]=-sin_wc; J_aug[4,4]=cos_wc
    J_aug[5,5]=1.0; J_aug[6,6]=1.0; J_aug[7,7]=1.0
    return SMatrix(J_aug)
end

function get_J_qns_inv_augmented_koenig(omega_c_val::Float64)::SMatrix{7,7,Float64}
    J_inv_aug=@MMatrix fill(0.0,7,7); J_inv_aug[1,1]=1.0; J_inv_aug[2,2]=1.0; cos_wc=cos(omega_c_val); sin_wc=sin(omega_c_val)
    J_inv_aug[3,3]=cos_wc; J_inv_aug[3,4]=-sin_wc; J_inv_aug[4,3]=sin_wc; J_inv_aug[4,4]=cos_wc
    J_inv_aug[5,5]=1.0; J_inv_aug[6,6]=1.0; J_inv_aug[7,7]=1.0
    return SMatrix(J_inv_aug)
end

function get_A_prime_qns_augmented_koenig_selectable(ac_val::Float64, ec_val::Float64, ic_val::Float64, omegac_val::Float64, include_j2::Bool, include_drag_effects::Bool, drag_model_type::DragModelTypeForSTM)::Tuple{SMatrix{7,7,Float64}, SMatrix{7,7,Float64}, SMatrix{7,7,Float64}}
    A_kep_p=@MMatrix zeros(Float64,7,7); A_j2_p=@MMatrix zeros(Float64,7,7); A_drag_p=@MMatrix zeros(Float64,7,7)
    n_c=sqrt(mu_earth/ac_val^3); A_kep_p[2,1]=-1.5*n_c
    if include_j2
        eta_c=sqrt(1.0-ec_val^2); if eta_c<1e-9; eta_c=1e-9; end
        kappa_J2=(3.0/4.0)*J2_coeff*(R_E^2*sqrt(mu_earth))/(ac_val^(3.5)*eta_c^4)
        E_f=1.0+eta_c; F_f=4.0+3.0*eta_c; G_f=1.0/eta_c^2
        cos_i=cos(ic_val); sin_i=sin(ic_val)
        P_g=3.0*cos_i^2-1.0; Q_g=5.0*cos_i^2-1.0; S_g=sin(2.0*ic_val); T_g=sin_i^2
        ex_c=ec_val*cos(omegac_val); ey_c=ec_val*sin(omegac_val)
        A_j2_p[2,1]=-3.5*kappa_J2*E_f*P_g; A_j2_p[2,3]=kappa_J2*F_f*G_f*ec_val*P_g; A_j2_p[2,5]=-kappa_J2*F_f*S_g
        A_j2_p[4,1]=-3.5*kappa_J2*ec_val*Q_g; A_j2_p[4,3]=4*kappa_J2*Q_g*G_f*ec_val^2; A_j2_p[4,5]=-5*kappa_J2*ec_val*S_g
        A_j2_p[6,1]=3.5*kappa_J2*S_g; A_j2_p[6,3]=-4*kappa_J2*G_f*S_g*ec_val; A_j2_p[6,5]=2*kappa_J2*T_g
    end
    if include_drag_effects
        if drag_model_type==DENSITY_MODEL_FREE; A_drag_p[1,7]=1.0; A_drag_p[3,7]=(1.0-ec_val);
        elseif drag_model_type==DENSITY_MODEL_SPECIFIC; println("警告: DENSITY_MODEL_SPECIFIC のプラント行列は未実装です。") end
    end
    return SMatrix(A_kep_p), SMatrix(A_j2_p), SMatrix(A_drag_p)
end

function get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_prime::SMatrix{7,7,Float64}, A_j2_prime::SMatrix{7,7,Float64}, A_drag_prime::SMatrix{7,7,Float64}, t_prop::Float64, ec_val_for_drag_effect::Float64, include_drag_effects::Bool, drag_model_type::DragModelTypeForSTM)::SMatrix{7,7,Float64}
    A_kep_J2_prime=A_kep_prime+A_j2_prime
    if drag_model_type == DENSITY_MODEL_FREE && include_drag_effects
    Phi_drag_prime = SMatrix{7,7,Float64}(I) + A_drag_prime * t_prop

    # TODO: 結合項を有効にするとDRAG_ONLYケースで発散する問題がある。
    #       A_kep_J2_prime と Integral_Phi_drag_prime の積の計算、
    #       または A_drag_p の定義を再検証する必要がある。
    # Integral_Phi_drag_prime = SMatrix{7,7,Float64}(I) * t_prop + A_drag_prime * (t_prop^2 / 2.0)
    # CouplingTerm_B = A_kep_J2_prime * Integral_Phi_drag_prime
    # return Phi_drag_prime + CouplingTerm_B

    return Phi_drag_prime # 現在は結合項を無効化して暫定対応
    elseif drag_model_type==DENSITY_MODEL_SPECIFIC&&include_drag_effects; println("警告: DENSITY_MODEL_SPECIFIC のSTM計算は未実装です。"); return SMatrix{7,7,Float64}(I)+(A_kep_J2_prime+A_drag_prime)*t_prop
    else; return SMatrix{7,7,Float64}(I)+A_kep_J2_prime*t_prop; end
end

# --- 外れ値を除去するヘルパー関数 ---
function filter_outliers_iqr(data_vector::Vector{Float64})
    finite_data = filter(isfinite, data_vector)
    if isempty(finite_data) || length(unique(finite_data)) == 1
        return data_vector 
    end
    
    Q1 = quantile(finite_data, 0.25)
    Q3 = quantile(finite_data, 0.75)
    IQR_val = Q3 - Q1
    
    # IQRが0の場合は外れ値なしとする
    if IQR_val == 0
        return data_vector
    end
    
    lower_bound = Q1 - 1.5 * IQR_val
    upper_bound = Q3 + 1.5 * IQR_val
    
    return map(x -> (isfinite(x) && (x < lower_bound || x > upper_bound)) ? NaN : x, data_vector)
end

# --- 状態再構成プロセスを検証するためのテスト関数
function run_state_reconstruction_test()
    println("\n\n--- 最終ROEからの状態再構成プロセスの検証を開始します ---")

    # --- 1. 既知の初期状態を準備 ---
    
    # 主衛星の初期状態 (テスト用にシンプルな値を使用)
    oe_chief_initial = OrbitalElementsClassical(
        7000e3,      # a
        0.01,        # e
        deg2rad(50.0), # i
        deg2rad(10.0), # RAAN
        deg2rad(20.0), # omega
        0.0, # f_true (Mから計算するのでダミー)
        0.0, # n (aから計算するのでダミー)
        deg2rad(30.0)  # M
    )
    
    # 主衛星のECI状態を計算
    posvel_chief_initial_eci = orbital_elements_to_sv(oe_chief_initial)
    r_chief_initial_eci = SVector{3}(posvel_chief_initial_eci[1:3])
    v_chief_initial_eci = SVector{3}(posvel_chief_initial_eci[4:6])
    # ECIから再度OEを計算し、一貫性を保つ
    oe_chief_initial = sv_to_orbital_elements(CartesianStateECI(r_chief_initial_eci, v_chief_initial_eci))
    
    # 副衛星の初期状態 (主衛星に対して意図的にずれを持たせる)
    oe_deputy_initial_true = OrbitalElementsClassical(
        oe_chief_initial.a + 10.0, # a_d = a_c + 10m
        oe_chief_initial.e + 0.0001, # e_d
        oe_chief_initial.i + deg2rad(0.01), # i_d
        oe_chief_initial.RAAN + deg2rad(0.01), # RAAN_d
        oe_chief_initial.omega + deg2rad(0.02), # omega_d
        0.0, 0.0, # f, n (ダミー)
        oe_chief_initial.M + deg2rad(0.03)  # M_d
    )
    
    posvel_deputy_initial_eci_true = orbital_elements_to_sv(oe_deputy_initial_true)
    r_deputy_initial_eci_true = SVector{3}(posvel_deputy_initial_eci_true[1:3])
    v_deputy_initial_eci_true = SVector{3}(posvel_deputy_initial_eci_true[4:6])
    # こちらもECIから再計算
    oe_deputy_initial_true = sv_to_orbital_elements(CartesianStateECI(r_deputy_initial_eci_true, v_deputy_initial_eci_true))


    println("--- 1. 検証用の「真の」状態を設定 ---")
    println("主衛星の真のECI位置: ", r_chief_initial_eci)
    println("副衛星の真のECI位置: ", r_deputy_initial_eci_true)


    # --- 2. 順変換 (ECI -> ROE) ---
    println("\n--- 2. 順変換 (ECI -> ROE) を実行 ---")
    true_roes = orbital_elements_to_qns_roe_koenig(oe_chief_initial, oe_deputy_initial_true)
    
    # 7次元ベクトルに拡張 (aug_paramはダミー)
    true_roes_augmented = SVector(
        true_roes.delta_a_norm, true_roes.delta_lambda,
        true_roes.delta_ex, true_roes.delta_ey,
        true_roes.delta_ix, true_roes.delta_iy,
        0.0 
    )
    println("計算された「真の」ROE: ", true_roes)


    # --- 3. 逆変換 (ROE -> ECI) ---
    println("\n--- 3. 逆変換 (ROE -> ECI) を実行 ---")
    # 逆変換関数に「真の」ROEと主衛星のOEを入力
    oe_deputy_reconstructed = final_roe_to_deputy_oe(oe_chief_initial, true_roes_augmented)
    posvel_deputy_reconstructed_eci = orbital_elements_to_sv(oe_deputy_reconstructed)
    r_deputy_reconstructed_eci = SVector{3}(posvel_deputy_reconstructed_eci[1:3])
    v_deputy_reconstructed_eci = SVector{3}(posvel_deputy_reconstructed_eci[4:6])
    
    println("再構成された副衛星のECI位置: ", r_deputy_reconstructed_eci)


    # --- 4. 比較・検証 ---
    println("\n--- 4. 比較・検証 ---")
    position_error_vec = r_deputy_initial_eci_true - r_deputy_reconstructed_eci
    velocity_error_vec = v_deputy_initial_eci_true - v_deputy_reconstructed_eci
    position_error_norm = norm(position_error_vec)
    velocity_error_norm = norm(velocity_error_vec)

    @printf "位置ベクトルの誤差 (ノルム): %.4e m\n" position_error_norm
    @printf "速度ベクトルの誤差 (ノルム): %.4e m/s\n" velocity_error_norm

    if position_error_norm < 1e-6 && velocity_error_norm < 1e-9 # 許容誤差
        println("検証結果: 順変換と逆変換は整合しています。状態再構成プロセスは正常です。")
    else
        println("警告: 状態再構成プロセスに大きな誤差または不安定性が存在します。")
        println("位置誤差ベクトル: ", position_error_vec)
        println("速度誤差ベクトル: ", velocity_error_vec)
    end
end

# --- STMの伝播プロセスを検証するためのデバッグ関数 (10軌道周期版)
function debug_stm_propagation_long_term()
    println("\n\n--- STMの長期伝播(10軌道)デバッグを開始します ---")

    # --- 1. 単純な初期条件を設定 ---
    println("\n--- 1. 単純な初期条件を設定 ---")
    
    # 主衛星: 赤道上の真円軌道 (a=7000km, e=0, i=0)
    # ※ 数値計算上、eとiは完全な0を避ける
    oe_c_debug_initial_struct = OrbitalElementsClassical(7000e3, 0.0022, deg2rad(97.65), 0.0, 0.0, 0.0, 0.0, 0.0)
    
    posvel_c_eci_vec = orbital_elements_to_sv(oe_c_debug_initial_struct)
    r_c_eci = SVector{3}(posvel_c_eci_vec[1:3])
    v_c_eci = SVector{3}(posvel_c_eci_vec[4:6])
    oe_c_debug = sv_to_orbital_elements(CartesianStateECI(r_c_eci, v_c_eci))

    # 副衛星: T方向に0.1m/sで分離 (初期δaが最大になるケース)
    dv_lvlh_debug = SVector(0.0, 0.1, 0.0)
    dr_lvlh_debug = SVector(0.0, 0.0, 0.0)
    state_d_eci = cw_to_eci_deputy_state(r_c_eci, v_c_eci, dr_lvlh_debug, dv_lvlh_debug)
    oe_d_debug = sv_to_orbital_elements(state_d_eci)
    
    # 初期ROEを計算
    roe_initial_debug = orbital_elements_to_qns_roe_koenig(oe_c_debug, oe_d_debug)
    roe_aug_init_debug = SVector(
        roe_initial_debug.delta_a_norm, roe_initial_debug.delta_lambda, roe_initial_debug.delta_ex,
        roe_initial_debug.delta_ey, roe_initial_debug.delta_ix, roe_initial_debug.delta_iy, 0.0
    )

    println("主衛星初期OE: a=$(oe_c_debug.a), e=$(oe_c_debug.e), i=$(rad2deg(oe_c_debug.i))")
    println("副衛星初期OE: a=$(oe_d_debug.a), e=$(oe_d_debug.e), i=$(rad2deg(oe_d_debug.i))")
    @printf "初期ROE: δa_norm: %.3e\n" roe_aug_init_debug[1]

    # --- 2. 10軌道周期後の状態を2つの方法で計算 ---
    println("\n--- 2. 10軌道周期後の状態を計算 (STM vs 数値プロパゲータ) ---")
    
    # 伝播時間 = 10軌道周期
    t_prop_debug = 10.0 * 2.0 * pi / oe_c_debug.n
    println("伝播時間: $t_prop_debug 秒 (10軌道周期)")

    # (A) STMによる伝播 (J2摂動のみを考慮)
    include_j2 = true; include_drag = false; drag_model = NO_DRAG
    A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(
        oe_c_debug.a, oe_c_debug.e, oe_c_debug.i, oe_c_debug.omega,
        include_j2, include_drag, drag_model
    )
    STM_prime_debug = get_STM_prime_qns_augmented_koenig_model_selectable(
        A_kep_p, A_j2_p, A_drag_p, t_prop_debug, oe_c_debug.e, include_drag, drag_model
    )
    roe_final_stm = STM_prime_debug * roe_aug_init_debug

    # (B) SatelliteToolboxのプロパゲータによる伝播 (比較用の「真値」)
    keps_c = KeplerianElements(0.0, oe_c_debug.a, oe_c_debug.e, oe_c_debug.i, oe_c_debug.RAAN, oe_c_debug.omega, oe_c_debug.f_true)
    keps_d = KeplerianElements(0.0, oe_d_debug.a, oe_d_debug.e, oe_d_debug.i, oe_d_debug.RAAN, oe_d_debug.omega, oe_d_debug.f_true)
    
    j2_prop_c = Propagators.init(Val(:J2), keps_c)
    j2_prop_d = Propagators.init(Val(:J2), keps_d)

    rf_c, vf_c = Propagators.propagate!(j2_prop_c, t_prop_debug)
    rf_d, vf_d = Propagators.propagate!(j2_prop_d, t_prop_debug)

    oe_c_final_propagator = sv_to_orbital_elements(CartesianStateECI(rf_c, vf_c))
    oe_d_final_propagator = sv_to_orbital_elements(CartesianStateECI(rf_d, vf_d))

    # --- 3. 最終分離距離の比較 ---
    final_separation_stm = norm(orbital_elements_to_sv(final_roe_to_deputy_oe(oe_c_final_propagator, roe_final_stm))[1:3] - rf_c)
    final_separation_propagator = norm(rf_d - rf_c)

    @printf "\n最終分離距離 (STM予測): %.2f km\n" (final_separation_stm / 1000.0)
    @printf "最終分離距離 (プロパゲータ): %.2f km\n" (final_separation_propagator / 1000.0)

    if final_separation_propagator > 0 && abs(final_separation_stm - final_separation_propagator) / final_separation_propagator < 0.1 # 10%以内ならOK
        println("\n検証結果: STMの予測は、10軌道周期後でも数値プロパゲータの結果と概ね一致しています。")
    else
        println("\n警告: STMの予測が数値プロパゲータの結果と大きく乖離しています。STMの線形モデルが長時間伝播で発散している可能性があります。")
    end
end

# 軌道傾斜角の影響を検証するためのデバッグ関数
function debug_with_inclination()
    println("\n\n--- 軌道傾斜角の影響検証デバッグを開始します ---")

    # --- 1. 初期条件を設定 (i のみ main_simulation の値を使用) ---
    println("\n--- 1. 初期条件を設定 (i = 97.65 deg) ---")
    
    # ★★★ 修正: 引数を8つに合わせる (f_trueとnにダミー値0.0を追加) ★★★
    oe_c_debug_initial_struct = OrbitalElementsClassical(
        7000e3,           # a
        0.0022,             # e (円軌道のまま)
        deg2rad(97.65),   # ★ i を変更
        0.0,              # RAAN
        0.0,              # omega
        0.0,              # f_true (Mから計算するのでダミー)
        0.0,              # n (aから計算するのでダミー)
        0.0               # M
    )
    
    posvel_c_eci_vec = orbital_elements_to_sv(oe_c_debug_initial_struct)
    r_c_eci = SVector{3}(posvel_c_eci_vec[1:3])
    v_c_eci = SVector{3}(posvel_c_eci_vec[4:6])
    oe_c_debug = sv_to_orbital_elements(CartesianStateECI(r_c_eci, v_c_eci))

    # 副衛星: T方向に0.1m/sで分離
    dv_lvlh_debug = SVector(0.0, 0.1, 0.0)
    dr_lvlh_debug = SVector(0.0, 0.0, 0.0)
    state_d_eci = cw_to_eci_deputy_state(r_c_eci, v_c_eci, dr_lvlh_debug, dv_lvlh_debug)
    oe_d_debug = sv_to_orbital_elements(state_d_eci)
    
    # 初期ROEを計算
    roe_initial_debug = orbital_elements_to_qns_roe_koenig(oe_c_debug, oe_d_debug)
    roe_aug_init_debug = SVector(
        roe_initial_debug.delta_a_norm, roe_initial_debug.delta_lambda, roe_initial_debug.delta_ex,
        roe_initial_debug.delta_ey, roe_initial_debug.delta_ix, roe_initial_debug.delta_iy, 0.0
    )

    println("主衛星初期OE: a=$(oe_c_debug.a), e=$(oe_c_debug.e), i=$(rad2deg(oe_c_debug.i))")
    @printf "初期ROE: δa_norm: %.3e\n" roe_aug_init_debug[1]

    # --- 2. 10軌道周期後の状態を計算 ---
    println("\n--- 2. 10軌道周期後の状態を計算 (J2_ONLY STM) ---")
    
    t_prop_debug = 10.0 * 2.0 * pi / oe_c_debug.n
    println("伝播時間: $t_prop_debug 秒 (10軌道周期)")

    # (A) STMによる伝播
    include_j2 = true; include_drag = false; drag_model = NO_DRAG
    A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(
        oe_c_debug.a, oe_c_debug.e, oe_c_debug.i, oe_c_debug.omega,
        include_j2, include_drag, drag_model
    )
    
    # ★★★ J変換を追加（i, ωが0でなくなったため） ★★★
    omega_c_ti = oe_c_debug.omega
    omega_dot_j2, Omega_dot_j2 = get_secular_j2_rates_koenig(oe_c_debug.a, oe_c_debug.e, oe_c_debug.i)
    omega_c_tf = oe_c_debug.omega + omega_dot_j2 * t_prop_debug
    
    J_ti = get_J_qns_augmented_koenig(omega_c_ti)
    J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)
    roe_prime_init = J_ti * roe_aug_init_debug

    STM_prime_debug = get_STM_prime_qns_augmented_koenig_model_selectable(
        A_kep_p, A_j2_p, A_drag_p, t_prop_debug, oe_c_debug.e, include_drag, drag_model
    )
    roe_prime_final = STM_prime_debug * roe_prime_init
    roe_final_stm = J_tf_inv * roe_prime_final


    # --- 3. 最終分離距離の計算 ---
    # 比較のため、数値プロパゲータの結果も計算
    keps_c = KeplerianElements(0.0, oe_c_debug.a, oe_c_debug.e, oe_c_debug.i, oe_c_debug.RAAN, oe_c_debug.omega, oe_c_debug.f_true)
    j2_prop_c = Propagators.init(Val(:J2), keps_c)
    rf_c, vf_c = Propagators.propagate!(j2_prop_c, t_prop_debug)
    oe_c_final_propagator = sv_to_orbital_elements(CartesianStateECI(rf_c, vf_c))

    oe_deputy_final_stm = final_roe_to_deputy_oe(oe_c_final_propagator, roe_final_stm)
    posvel_deputy_final_stm = orbital_elements_to_sv(oe_deputy_final_stm)
    r_deputy_final_stm = SVector{3}(posvel_deputy_final_stm[1:3])
    
    final_separation_stm = norm(r_deputy_final_stm - rf_c)
    
    @printf "\n軌道傾斜角 i=%.2f deg の場合:\n" rad2deg(oe_c_debug.i)
    @printf "  最終分離距離 (STM予測): %.2f km\n" (final_separation_stm / 1000.0)
end

# STMの内部計算を検証するためのデバッグ関数
function debug_stm_components()
    println("\n\n--- STM結合ロジックのデバッグを開始します ---")

    # --- 1. main_simulation と同じ現実的な初期条件を設定 ---
    println("\n--- 1. main_simulation と同じ初期条件を設定 ---")
    oe_chief_initial_for_sv = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, 0.0, M_c_stm_init)
    posvel_chief_initial_eci_vec = orbital_elements_to_sv(oe_chief_initial_for_sv)
    r_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[1:3])
    v_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[4:6])
    oe_chief_eval = sv_to_orbital_elements(CartesianStateECI(r_chief_init_eci, v_chief_init_eci))

    # 90°分離に固定
    angle_rad_val = deg2rad(90.0)
    dv_R_val_comp=delta_v_magnitude*cos(angle_rad_val); dv_T_val_comp=delta_v_magnitude*sin(angle_rad_val);
    dv_lvlh_vec=SVector(dv_R_val_comp,dv_T_val_comp,0.0); dr_lvlh_vec=SVector(10.0,0.0,0.0)
    state_deputy_init_eci=cw_to_eci_deputy_state(r_chief_init_eci,v_chief_init_eci,dr_lvlh_vec,dv_lvlh_vec)
    r_dep_init_eci=state_deputy_init_eci.r_vec; v_dep_init_eci=state_deputy_init_eci.v_vec
    oe_dep_init=sv_to_orbital_elements(CartesianStateECI(r_dep_init_eci,v_dep_init_eci))
    qns_roes_init=orbital_elements_to_qns_roe_koenig(oe_chief_eval,oe_dep_init)
    
    aug_param_val=delta_a_dot_drag_initial_normalized
    roe_aug_init_vec_temp=MVector{7,Float64}(qns_roes_init.delta_a_norm,qns_roes_init.delta_lambda,qns_roes_init.delta_ex,qns_roes_init.delta_ey,qns_roes_init.delta_ix,qns_roes_init.delta_iy,aug_param_val)
    roe_aug_init_vec=SVector(roe_aug_init_vec_temp)
    
    # 伝播時間 = 10軌道周期
    t_prop = 10.0 * 2.0 * pi * sqrt(oe_chief_eval.a^3 / mu_earth)
    println("伝播時間: $t_prop 秒 (10軌道周期)")

    # --- 2. プラント行列の各成分を計算・表示 ---
    println("\n--- 2. プラント行列の各成分を計算・表示 ---")
    A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(
        oe_chief_eval.a, oe_chief_eval.e, oe_chief_eval.i, oe_chief_eval.omega, 
        true, true, DENSITY_MODEL_FREE
    )
    A_kep_J2_prime = A_kep_p + A_j2_p

    println("A_kep_p (ケプラー項) のノルム: ", norm(A_kep_p))
    println("A_j2_p (J2項) のノルム: ", norm(A_j2_p))
    println("A_drag_p (抗力項) のノルム: ", norm(A_drag_p))
    
    # --- 3. STMの各構成要素を計算・表示 ---
    println("\n--- 3. STMの各構成要素を計算・表示 ---")
    
    # Part A: 抗力のみの効果
    Phi_drag_prime = SMatrix{7,7,Float64}(I) + A_drag_p * t_prop
    
    # Part B: 結合効果
    Integral_Phi_drag_prime = SMatrix{7,7,Float64}(I) * t_prop + A_drag_p * (t_prop^2 / 2.0)
    CouplingTerm_B = A_kep_J2_prime * Integral_Phi_drag_prime
    
    # 最終STM
    STM_prime = Phi_drag_prime + CouplingTerm_B

    println("Φ_drag' (Part A) のノルム: ", norm(Phi_drag_prime))
    println("∫Φ_drag' dt のノルム: ", norm(Integral_Phi_drag_prime))
    println("A_kep_j2' * ∫Φ_drag' dt (Part B) のノルム: ", norm(CouplingTerm_B))
    println("最終的なSTM' のノルム: ", norm(STM_prime))

    # --- 4. 最終的な分離距離を計算 ---
    println("\n--- 4. 最終的な分離距離を計算 ---")
    omega_c_ti=oe_chief_eval.omega; omega_dot_j2,Omega_dot_j2=get_secular_j2_rates_koenig(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i)
    oe_chief_at_tf=OrbitalElementsClassical(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,mod(oe_chief_eval.RAAN+Omega_dot_j2*t_prop,2*pi),mod(oe_chief_eval.omega+omega_dot_j2*t_prop,2*pi),0.0,oe_chief_eval.n,mod(oe_chief_eval.M+oe_chief_eval.n*t_prop,2*pi))
    oe_chief_at_tf=OrbitalElementsClassical(oe_chief_at_tf.a,oe_chief_at_tf.e,oe_chief_at_tf.i,oe_chief_at_tf.RAAN,oe_chief_at_tf.omega,SatelliteToolbox.mean_to_true_anomaly(oe_chief_at_tf.e,oe_chief_at_tf.M),oe_chief_at_tf.n,oe_chief_at_tf.M)
    omega_c_tf_val=oe_chief_at_tf.omega; J_ti=get_J_qns_augmented_koenig(omega_c_ti); J_tf_inv=get_J_qns_inv_augmented_koenig(omega_c_tf_val); roe_prime_init=J_ti*roe_aug_init_vec
    
    roe_prime_final=STM_prime*roe_prime_init; roe_aug_final_vec=J_tf_inv*roe_prime_final

    posvel_chief_final_eci=orbital_elements_to_sv(oe_chief_at_tf); r_chief_eci_final_val=SVector{3}(posvel_chief_final_eci[1:3]); 
    oe_deputy_at_tf=final_roe_to_deputy_oe(oe_chief_at_tf,roe_aug_final_vec)
    posvel_deputy_final_eci=orbital_elements_to_sv(oe_deputy_at_tf); r_deputy_eci_final_val=SVector{3}(posvel_deputy_final_eci[1:3])
    
    final_separation_distance = norm(r_deputy_eci_final_val - r_chief_eci_final_val)
    @printf "最終分離距離: %.2f km\n" (final_separation_distance / 1000.0)
end

# --- STMとプロパゲータの比較デバッグ関数 ---
function debug_reconstruction_with_propagator()
    println("\n\n--- J2_ONLYケースの状態再構成デバッグを開始します ---")

    # --- 1. main_simulationと同じ現実的な初期条件を設定 ---
    println("\n--- 1. 初期条件を設定 (i=97.65deg, e=0.0022) ---")
    
    oe_c_initial = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, 0.0, M_c_stm_init)
    posvel_c_initial = orbital_elements_to_sv(oe_c_initial)
    r_c_initial = SVector{3}(posvel_c_initial[1:3])
    v_c_initial = SVector{3}(posvel_c_initial[4:6])
    oe_c_eval = sv_to_orbital_elements(CartesianStateECI(r_c_initial, v_c_initial))

    # 90°分離に固定
    angle_rad = deg2rad(90.0)
    dv_R_val_comp=delta_v_magnitude*cos(angle_rad); dv_T_val_comp=delta_v_magnitude*sin(angle_rad)
    dv_lvlh_vec=SVector(dv_R_val_comp,dv_T_val_comp,0.0); dr_lvlh_vec=SVector(10.0,0.0,0.0)
    state_d_initial = cw_to_eci_deputy_state(r_c_initial, v_c_initial, dr_lvlh_vec, dv_lvlh_vec)
    oe_d_initial = sv_to_orbital_elements(state_d_initial)
    
    roe_initial = orbital_elements_to_qns_roe_koenig(oe_c_eval, oe_d_initial)
    roe_aug_init_debug = SVector(
        roe_initial.delta_a_norm, roe_initial.delta_lambda, roe_initial.delta_ex,
        roe_initial.delta_ey, roe_initial.delta_ix, roe_initial.delta_iy, 0.0
    )

    # --- 2. 10軌道周期後の状態を2つの方法で計算 ---
    println("\n--- 2. 10軌道周期後の状態を計算 (STM vs 数値プロパゲータ) ---")
    t_prop = 10.0 * 2.0 * pi * sqrt(oe_c_eval.a^3 / mu_earth)
    println("伝播時間: $t_prop 秒 (10軌道周期)")

    # (A) STMによる伝播
    omega_c_ti=oe_c_eval.omega; omega_dot_j2,Omega_dot_j2=get_secular_j2_rates_koenig(oe_c_eval.a,oe_c_eval.e,oe_c_eval.i)
    omega_c_tf=mod(oe_c_eval.omega+omega_dot_j2*t_prop,2*pi)
    J_ti=get_J_qns_augmented_koenig(omega_c_ti); J_tf_inv=get_J_qns_inv_augmented_koenig(omega_c_tf)
    roe_prime_init = J_ti * roe_aug_init_debug
    A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(oe_c_eval.a,oe_c_eval.e,oe_c_eval.i,omega_c_ti,true,false,NO_DRAG)
    STM_prime = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_p,A_j2_p,A_drag_p,t_prop,oe_c_eval.e,false,NO_DRAG)
    roe_prime_final = STM_prime * roe_prime_init
    roe_final_stm = J_tf_inv * roe_prime_final

    # (B) SatelliteToolboxのプロパゲータによる「真値」の計算
    keps_c_init = KeplerianElements(0.0, oe_c_eval.a, oe_c_eval.e, oe_c_eval.i, oe_c_eval.RAAN, oe_c_eval.omega, oe_c_eval.f_true)
    keps_d_init = KeplerianElements(0.0, oe_d_initial.a, oe_d_initial.e, oe_d_initial.i, oe_d_initial.RAAN, oe_d_initial.omega, oe_d_initial.f_true)
    j2_prop_c = Propagators.init(Val(:J2), keps_c_init)
    j2_prop_d = Propagators.init(Val(:J2), keps_d_init)
    rf_c, vf_c = Propagators.propagate!(j2_prop_c, t_prop)
    rf_d, vf_d = Propagators.propagate!(j2_prop_d, t_prop)
    oe_c_final_prop = sv_to_orbital_elements(CartesianStateECI(rf_c, vf_c))
    oe_d_final_prop = sv_to_orbital_elements(CartesianStateECI(rf_d, vf_d))
    roe_final_prop = orbital_elements_to_qns_roe_koenig(oe_c_final_prop, oe_d_final_prop)
    
    # --- 3. 最終ROEの比較 ---
    println("\n--- 3. 最終ROEの比較 ---")
    println("          \t| STMによる予測 \t| 数値プロパゲータによる真値")
    println("--------------------------------------------------------------------")
    @printf "δa_norm \t| %.3e \t| %.3e\n" roe_final_stm[1] roe_final_prop.delta_a_norm
    @printf "δλ (rad)\t| %.3e \t| %.3e\n" roe_final_stm[2] roe_final_prop.delta_lambda
    @printf "δex \t\t| %.3e \t| %.3e\n" roe_final_stm[3] roe_final_prop.delta_ex
    @printf "δey \t\t| %.3e \t| %.3e\n" roe_final_stm[4] roe_final_prop.delta_ey
    @printf "δix (rad)\t| %.3e \t| %.3e\n" roe_final_stm[5] roe_final_prop.delta_ix
    @printf "δiy (rad)\t| %.3e \t| %.3e\n" roe_final_stm[6] roe_final_prop.delta_iy

    # --- 4. 最終相対位置の比較 ---
    # STMから再構成
    oe_deputy_reconstructed = final_roe_to_deputy_oe(oe_c_final_prop, roe_final_stm)
    posvel_deputy_reconstructed = orbital_elements_to_sv(oe_deputy_reconstructed)
    
    # ★★★ 修正: SVectorとして明示的に取り出す ★★★
    r_deputy_reconstructed = SVector{3}(posvel_deputy_reconstructed[1:3])
    v_deputy_reconstructed = SVector{3}(posvel_deputy_reconstructed[4:6])

    # ★★★ 修正: 正しい引数で eci_to_hill を呼び出す ★★★
    rel_pos_stm_hill = eci_to_hill(r_deputy_reconstructed, v_deputy_reconstructed, rf_c, vf_c)

    # プロパゲータから計算
    rel_pos_prop_hill = eci_to_hill(rf_d, vf_d, rf_c, vf_c)
    
    println("\n--- 4. 最終相対位置(Hill)の比較 [m] ---")
    println("      \t| STMから再構成 \t| 数値プロパゲータによる真値")
    println("----------------------------------------------------------")
    @printf "R (x) \t| %12.3f \t| %12.3f\n" rel_pos_stm_hill[1] rel_pos_prop_hill[1]
    @printf "T (y) \t| %12.3f \t| %12.3f\n" rel_pos_stm_hill[2] rel_pos_prop_hill[2]
    @printf "N (z) \t| %12.3f \t| %12.3f\n" rel_pos_stm_hill[3] rel_pos_prop_hill[3]
end

# --- プロットとHTMLレポート作成関数 ---
function plot_results(angles_plot_list, roe_data_log, perturbation_setting, drag_model_setting, separation_plane_setting)
    if isempty(angles_plot_list)
        println("プロットするデータがありません。")
        return
    end

    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    html_filename = "simulation_report_$(timestamp)_$(lowercase(string(perturbation_setting)))_$(lowercase(string(drag_model_setting)))_$(lowercase(string(separation_plane_setting))).html"
    
    open(html_filename, "w") do f
        write(f, "<html><head><title>Simulation Report</title>")
        write(f, "<style> body { font-family: sans-serif; } h1, h2 { color: #333; } div.plot-container { page-break-inside: avoid; margin-bottom: 30px; padding-top: 20px; } img { border: 1px solid #ccc; max-width: 100%; height: auto; } table { border-collapse: collapse; width: 50%; margin-bottom: 20px; } th, td { border: 1px solid #ddd; padding: 8px; text-align: left; } th { background-color: #f2f2f2; } </style>")
        write(f, "</head><body>")
        write(f, "<h1>Simulation Report</h1>")
        
        write(f, "<h2>Initial Conditions</h2>")
        write(f, "<table>")
        write(f, "<tr><th>Parameter</th><th>Value</th></tr>")
        write(f, "<tr><td>Perturbation Setting</td><td>$perturbation_setting</td></tr>")
        write(f, "<tr><td>Drag Model for STM</td><td>$drag_model_setting</td></tr>")
        write(f, "<tr><td>Separation Plane</td><td>$separation_plane_setting</td></tr>")
        write(f, "<tr><td>Chief Semi-major Axis</td><td>$a_c_stm_init m</td></tr>")
        write(f, "<tr><td>Chief Eccentricity</td><td>$e_c_stm_init</td></tr>")
        write(f, "<tr><td>Chief Inclination</td><td>$(rad2deg(i_c_stm_init)) deg</td></tr>")
        write(f, "<tr><td>Chief RAAN</td><td>$(rad2deg(Omega_c_stm_init)) deg</td></tr>")
        write(f, "<tr><td>Chief Arg. of Perigee</td><td>$(rad2deg(omega_c_stm_init)) deg</td></tr>")
        write(f, "<tr><td>Chief Mean Anomaly</td><td>$(rad2deg(M_c_stm_init)) deg</td></tr>")
        write(f, "<tr><td>Delta-V Magnitude</td><td>$delta_v_magnitude m/s</td></tr>")
        if drag_model_setting == DENSITY_MODEL_FREE
            write(f, "<tr><td>Assumed delta_a_dot_drag</td><td>$delta_a_dot_drag_initial_normalized [1/s]</td></tr>")
        end
        write(f, "</table>")

        plot_title_main_suffix = " (Pert: $perturbation_setting, DragM: $drag_model_setting, Plane: $separation_plane_setting)"
        xlims_plot = (0.0, angles_plot_list[end])

        function plot_to_base64_string(p)
            io = IOBuffer()
            show(io, MIME"image/png"(), p)
            return base64encode(take!(io))
        end

        write(f, "<div class='plot-container'><h2>Initial Relative Orbital Elements (ROE)</h2>")
        initial_roe_plots_list = []
        push!(initial_roe_plots_list, plot(angles_plot_list, filter_outliers_iqr(roe_data_log[:initial_delta_a]), title="Initial δa_norm"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="δa_norm"))
        push!(initial_roe_plots_list, plot(angles_plot_list, filter_outliers_iqr(roe_data_log[:initial_delta_lambda]), title="Initial δλ"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="δλ (deg)"))
        push!(initial_roe_plots_list, plot(angles_plot_list, filter_outliers_iqr(roe_data_log[:initial_delta_ex]), title="Initial δex"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="δex"))
        push!(initial_roe_plots_list, plot(angles_plot_list, filter_outliers_iqr(roe_data_log[:initial_delta_ey]), title="Initial δey"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="δey"))
        push!(initial_roe_plots_list, plot(angles_plot_list, filter_outliers_iqr(roe_data_log[:initial_delta_ix]), title="Initial δix"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="δix (deg)"))
        push!(initial_roe_plots_list, plot(angles_plot_list, filter_outliers_iqr(roe_data_log[:initial_delta_iy]), title="Initial δiy"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="δiy (deg)", xlabel="Sep. Angle (deg)"))
        plot_obj_initial_roes = plot(initial_roe_plots_list..., layout=(3,2), size=(1000,900), titlefont=font(6), tickfont=font(5), guidefont=font(6))
        plot_base64_initial_roes = plot_to_base64_string(plot_obj_initial_roes)
        write(f, "<img src=\"data:image/png;base64,$(plot_base64_initial_roes)\"/></div>")
        println("初期ROEプロットをHTMLに埋め込みました。")

        valid_final_indices = .!isnan.(roe_data_log[:cost]) .& .!isinf.(roe_data_log[:cost])
        if sum(valid_final_indices) > 0
            write(f, "<div class='plot-container'><h2>Final Relative Orbital Elements (ROE) and Performance</h2>")
            angles_for_final_plot = angles_plot_list[valid_final_indices]
            final_roe_plots_list = []
            push!(final_roe_plots_list, plot(angles_for_final_plot, filter_outliers_iqr(roe_data_log[:final_delta_a][valid_final_indices]), title="Final δa_norm"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="δa_norm"))
            push!(final_roe_plots_list, plot(angles_for_final_plot, filter_outliers_iqr(roe_data_log[:final_delta_lambda][valid_final_indices]), title="Final δλ"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="δλ (deg)"))
            push!(final_roe_plots_list, plot(angles_for_final_plot, filter_outliers_iqr(roe_data_log[:final_delta_ex][valid_final_indices]), title="Final δex"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="δex"))
            push!(final_roe_plots_list, plot(angles_for_final_plot, filter_outliers_iqr(roe_data_log[:final_delta_ey][valid_final_indices]), title="Final δey"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="δey"))
            push!(final_roe_plots_list, plot(angles_for_final_plot, filter_outliers_iqr(roe_data_log[:final_delta_ix][valid_final_indices]), title="Final δix"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="δix (deg)"))
            push!(final_roe_plots_list, plot(angles_for_final_plot, filter_outliers_iqr(roe_data_log[:final_delta_iy][valid_final_indices]), title="Final δiy"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="δiy (deg)"))
            push!(final_roe_plots_list, plot(angles_for_final_plot, filter_outliers_iqr(roe_data_log[:cost][valid_final_indices]), title="Cost"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="Cost", yscale=:identity))
            push!(final_roe_plots_list, plot(angles_for_final_plot, filter_outliers_iqr(roe_data_log[:t_final][valid_final_indices])./3600.0, title="Time to Target"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="Time (hours)", xlabel="Sep. Angle (deg)"))
            plot_obj_final_roes = plot(final_roe_plots_list..., layout=(4,2), size=(1000,1200), titlefont=font(6), tickfont=font(5), guidefont=font(6))
            plot_base64_final_roes = plot_to_base64_string(plot_obj_final_roes)
            write(f, "<img src=\"data:image/png;base64,$(plot_base64_final_roes)\"/></div>")
            println("最終ROEプロットをHTMLに埋め込みました。")
        else
            write(f, "<h2>Final Relative Orbital Elements (ROE)</h2><p>有効なデータポイントがありませんでした。</p>")
        end
        
        write(f, "<div class='plot-container'><h2>Initial Relative J2 Perturbation Force</h2>")
        if !isempty(roe_data_log[:initial_rel_j2_norm])
            plot_j2_initial_list = []
            plot_title_suffix_j2_init = " (Initial Rel. J2)"
            push!(plot_j2_initial_list, plot(angles_plot_list, filter_outliers_iqr(roe_data_log[:initial_rel_j2_r]), title="J2 (Radial Hill)"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="a_R (m/s^2)"))
            push!(plot_j2_initial_list, plot(angles_plot_list, filter_outliers_iqr(roe_data_log[:initial_rel_j2_t]), title="J2 (Along-track Hill)"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="a_T (m/s^2)"))
            push!(plot_j2_initial_list, plot(angles_plot_list, filter_outliers_iqr(roe_data_log[:initial_rel_j2_n]), title="J2 (Cross-track Hill)"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="a_N (m/s^2)"))
            push!(plot_j2_initial_list, plot(angles_plot_list, filter_outliers_iqr(roe_data_log[:initial_rel_j2_norm]), title="J2 (Norm ECI)"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="||a_J2_rel|| (m/s^2)", xlabel="Sep. Angle (deg)"))
            plot_obj_j2_initial = plot(plot_j2_initial_list..., layout=(2,2), size=(1000,700), titlefont=font(6), tickfont=font(5), guidefont=font(6))
            plot_base64_j2_initial = plot_to_base64_string(plot_obj_j2_initial)
            write(f, "<img src=\"data:image/png;base64,$(plot_base64_j2_initial)\"/></div>")
            println("初期相対J2摂動力プロットをHTMLに埋め込みました。")
        end

        write(f, "<div class='plot-container'><h2>Final Relative J2 Perturbation Force</h2>")
        valid_final_j2_indices = .!isnan.(roe_data_log[:final_rel_j2_norm])
        if sum(valid_final_j2_indices) > 0
            angles_for_final_j2_plot = angles_plot_list[valid_final_j2_indices]
            plot_j2_final_list = []
            plot_title_suffix_j2_final = " (Final Rel. J2)"
            push!(plot_j2_final_list, plot(angles_for_final_j2_plot, filter_outliers_iqr(roe_data_log[:final_rel_j2_r][valid_final_j2_indices]), title="J2 (Radial Hill)"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="a_R (m/s^2)"))
            # T方向のプロット
            push!(plot_j2_final_list, plot(angles_for_final_j2_plot, filter_outliers_iqr(roe_data_log[:final_rel_j2_t][valid_final_j2_indices]), title="J2 (Along-track Hill)"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="a_T (m/s^2)"))
            # N方向のプロット
            push!(plot_j2_final_list, plot(angles_for_final_j2_plot, filter_outliers_iqr(roe_data_log[:final_rel_j2_n][valid_final_j2_indices]), title="J2 (Cross-track Hill)"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="a_N (m/s^2)"))
        
            push!(plot_j2_final_list, plot(angles_for_final_j2_plot, filter_outliers_iqr(roe_data_log[:final_rel_j2_norm][valid_final_j2_indices]), title="J2 (Norm ECI)"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="||a_J2_rel|| (m/s^2)", xlabel="Sep. Angle (deg)"))
            plot_obj_j2_final = plot(plot_j2_final_list..., layout=(2,2), size=(1000,700), titlefont=font(6), tickfont=font(5), guidefont=font(6))
            plot_base64_j2_final = plot_to_base64_string(plot_obj_j2_final)
            write(f, "<img src=\"data:image/png;base64,$(plot_base64_j2_final)\"/></div>")
            println("最終相対J2摂動力プロットをHTMLに埋め込みました。")
        else
            write(f, "<p>有効なデータポイントがありませんでした（または全てNaNでした）。</p>")
        end
        
        write(f, "</body></html>")
    end
    println("HTMLレポートを保存しました: $html_filename")
end

# --- メイン処理 (固定時間伝播に修正) ---
function main_simulation(
    perturbation_setting::PerturbationType,
    drag_model_setting::DragModelTypeForSTM,
    separation_plane_setting::SeparationPlane
)
    println("\n\n最適分離方向探索 (Pert: $perturbation_setting, DragModel: $drag_model_setting, Plane: $separation_plane_setting) を開始します...")
    include_j2_active=(perturbation_setting == J2_ONLY || perturbation_setting == J2_AND_DRAG)
    include_drag_active_stm=(drag_model_setting != NO_DRAG && (perturbation_setting == DRAG_ONLY || perturbation_setting == J2_AND_DRAG) )

    oe_chief_initial_for_sv = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, 0.0, M_c_stm_init)
    posvel_chief_initial_eci_vec = orbital_elements_to_sv(oe_chief_initial_for_sv)
    r_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[1:3]); v_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[4:6])
    oe_chief_eval = sv_to_orbital_elements(CartesianStateECI(r_chief_init_eci, v_chief_init_eci))

    optimal_angle_deg=-1.0; min_cost=Inf; angles_plot_list=Float64[]
    roe_data_log=Dict(
        :initial_delta_a=>Float64[], :initial_delta_lambda=>Float64[], :initial_delta_ex=>Float64[], :initial_delta_ey=>Float64[], :initial_delta_ix=>Float64[], :initial_delta_iy=>Float64[],
        :final_delta_a=>Float64[], :final_delta_lambda=>Float64[], :final_delta_ex=>Float64[], :final_delta_ey=>Float64[], :final_delta_ix=>Float64[], :final_delta_iy=>Float64[],
        :cost=>Float64[], :t_final=>Float64[],
        :initial_rel_j2_r=>Float64[], :initial_rel_j2_t=>Float64[], :initial_rel_j2_n=>Float64[], :initial_rel_j2_norm=>Float64[],
        :final_rel_j2_r=>Float64[], :final_rel_j2_t=>Float64[], :final_rel_j2_n=>Float64[], :final_rel_j2_norm=>Float64[]
    )
    println("Pert: $perturbation_setting, J2: $include_j2_active, Drag STM active: $include_drag_active_stm (Model: $drag_model_setting), Separation Plane: $separation_plane_setting")

    fixed_propagation_time = 10.0 * 2.0 * pi * sqrt(oe_chief_eval.a^3 / mu_earth)
    println("Fixed propagation time set to 10 orbits: $(fixed_propagation_time) seconds")

    # # ★★★ デバッグのための修正 ★★★
    # local fixed_propagation_time
    # if perturbation_setting == DRAG_ONLY
    #     # DRAG_ONLYのケースだけ、1軌道周期でテスト
    #     fixed_propagation_time = 1.0 * 2.0 * pi * sqrt(oe_chief_eval.a^3 / mu_earth)
    #     println("DEBUG: Propagation time for DRAG_ONLY set to 1 orbit: $(fixed_propagation_time) seconds")
    # else
    #     # 他のケースでは10軌道周期
    #     fixed_propagation_time = 10.0 * 2.0 * pi * sqrt(oe_chief_eval.a^3 / mu_earth)
    # end
    # # ★★★ ここまで ★★★

    angle_step_val=10.0
    for angle_val in 0.0:angle_step_val:(360.0-angle_step_val)
        angle_rad_val=deg2rad(angle_val)
        dv_R_val_comp=0.0; dv_T_val_comp=0.0; dv_N_val_comp=0.0
        if separation_plane_setting==RT_PLANE; dv_R_val_comp=delta_v_magnitude*cos(angle_rad_val); dv_T_val_comp=delta_v_magnitude*sin(angle_rad_val);
        elseif separation_plane_setting==RN_PLANE; dv_R_val_comp=delta_v_magnitude*cos(angle_rad_val); dv_N_val_comp=delta_v_magnitude*sin(angle_rad_val);
        elseif separation_plane_setting==NT_PLANE; dv_T_val_comp=delta_v_magnitude*cos(angle_rad_val); dv_N_val_comp=delta_v_magnitude*sin(angle_rad_val); end
        dv_lvlh_vec=SVector(dv_R_val_comp,dv_T_val_comp,dv_N_val_comp); dr_lvlh_vec=SVector(10.0,0.0,0.0)
        state_deputy_init_eci=cw_to_eci_deputy_state(r_chief_init_eci,v_chief_init_eci,dr_lvlh_vec,dv_lvlh_vec)
        r_dep_init_eci=state_deputy_init_eci.r_vec; v_dep_init_eci=state_deputy_init_eci.v_vec
        aj2_chief_init=calculate_j2_perturbation_eci(r_chief_init_eci,mu_earth,J2_coeff,R_E)
        aj2_dep_init=calculate_j2_perturbation_eci(r_dep_init_eci,mu_earth,J2_coeff,R_E)
        aj2_rel_init_eci=aj2_dep_init-aj2_chief_init
        aj2_rel_init_hill=eci_to_hill_relative_acceleration(aj2_rel_init_eci,r_chief_init_eci,v_chief_init_eci)
        push!(roe_data_log[:initial_rel_j2_r],aj2_rel_init_hill[1]); push!(roe_data_log[:initial_rel_j2_t],aj2_rel_init_hill[2]); push!(roe_data_log[:initial_rel_j2_n],aj2_rel_init_hill[3]); push!(roe_data_log[:initial_rel_j2_norm],norm(aj2_rel_init_eci))
        oe_dep_init=sv_to_orbital_elements(CartesianStateECI(r_dep_init_eci,v_dep_init_eci))
        qns_roes_init=orbital_elements_to_qns_roe_koenig(oe_chief_eval,oe_dep_init)
        push!(roe_data_log[:initial_delta_a],qns_roes_init.delta_a_norm); push!(roe_data_log[:initial_delta_lambda],rad2deg(qns_roes_init.delta_lambda)); push!(roe_data_log[:initial_delta_ex],qns_roes_init.delta_ex); push!(roe_data_log[:initial_delta_ey],qns_roes_init.delta_ey); push!(roe_data_log[:initial_delta_ix],rad2deg(qns_roes_init.delta_ix)); push!(roe_data_log[:initial_delta_iy],rad2deg(qns_roes_init.delta_iy))
        aug_param_val=0.0
        if drag_model_setting==DENSITY_MODEL_FREE; aug_param_val=delta_a_dot_drag_initial_normalized; elseif drag_model_setting==DENSITY_MODEL_SPECIFIC; aug_param_val=delta_B_initial_param; end
        roe_aug_init_vec_temp=MVector{7,Float64}(qns_roes_init.delta_a_norm,qns_roes_init.delta_lambda,qns_roes_init.delta_ex,qns_roes_init.delta_ey,qns_roes_init.delta_ix,qns_roes_init.delta_iy,aug_param_val)
        if !include_drag_active_stm; roe_aug_init_vec_temp[7]=0.0; end
        roe_aug_init_vec=SVector(roe_aug_init_vec_temp)
        
        tf_val=fixed_propagation_time
        push!(angles_plot_list,angle_val)
        push!(roe_data_log[:t_final],tf_val)
        
        omega_c_ti=oe_chief_eval.omega; omega_dot_j2,Omega_dot_j2=include_j2_active ? get_secular_j2_rates_koenig(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i) : (0.0,0.0)
        oe_chief_at_tf=OrbitalElementsClassical(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,mod(oe_chief_eval.RAAN+Omega_dot_j2*tf_val,2*pi),mod(oe_chief_eval.omega+omega_dot_j2*tf_val,2*pi),0.0,oe_chief_eval.n,mod(oe_chief_eval.M+oe_chief_eval.n*tf_val,2*pi))
        oe_chief_at_tf=OrbitalElementsClassical(oe_chief_at_tf.a,oe_chief_at_tf.e,oe_chief_at_tf.i,oe_chief_at_tf.RAAN,oe_chief_at_tf.omega,SatelliteToolbox.mean_to_true_anomaly(oe_chief_at_tf.e,oe_chief_at_tf.M),oe_chief_at_tf.n,oe_chief_at_tf.M)
        
        omega_c_tf_val=oe_chief_at_tf.omega; J_ti=get_J_qns_augmented_koenig(omega_c_ti); J_tf_inv=get_J_qns_inv_augmented_koenig(omega_c_tf_val); roe_prime_init=J_ti*roe_aug_init_vec
        
        # --- ★★★ ここからが修正箇所 ★★★ ---
        
        # 1. プラント行列の各成分を個別に取得する
        A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(
            oe_chief_eval.a, oe_chief_eval.e, oe_chief_eval.i, omega_c_ti, 
            include_j2_active, include_drag_active_stm, drag_model_setting
        )
        
        # 2. 分離されたプラント行列をSTM計算関数に渡す
        STM_prime = get_STM_prime_qns_augmented_koenig_model_selectable(
            A_kep_p, A_j2_p, A_drag_p, 
            tf_val, 
            oe_chief_eval.e, # この引数も渡す必要がある
            include_drag_active_stm, 
            drag_model_setting
        )
        # --- ★★★ ここまでが修正箇所 ★★★ ---
        
        roe_prime_final=STM_prime*roe_prime_init; roe_aug_final_vec=J_tf_inv*roe_prime_final
        
        push!(roe_data_log[:final_delta_a],roe_aug_final_vec[1]); push!(roe_data_log[:final_delta_lambda],rad2deg(roe_aug_final_vec[2])); push!(roe_data_log[:final_delta_ex],roe_aug_final_vec[3]); push!(roe_data_log[:final_delta_ey],roe_aug_final_vec[4]); push!(roe_data_log[:final_delta_ix],rad2deg(roe_aug_final_vec[5])); push!(roe_data_log[:final_delta_iy],rad2deg(roe_aug_final_vec[6]))
        
        cost_da_penalty=1000.0*roe_aug_final_vec[1]^2; cost_others=norm(view(roe_aug_final_vec,2:6)); 
        total_cost=cost_da_penalty+cost_others; push!(roe_data_log[:cost],total_cost)
        
        posvel_chief_final_eci=orbital_elements_to_sv(oe_chief_at_tf)
        r_chief_eci_final_val=SVector{3}(posvel_chief_final_eci[1:3]); v_chief_eci_final_val=SVector{3}(posvel_chief_final_eci[4:6])
        oe_deputy_at_tf=final_roe_to_deputy_oe(oe_chief_at_tf,roe_aug_final_vec)
        posvel_deputy_final_eci=orbital_elements_to_sv(oe_deputy_at_tf)
        r_deputy_eci_final_val=SVector{3}(posvel_deputy_final_eci[1:3])
        
        aj2_chief_final_eci=calculate_j2_perturbation_eci(r_chief_eci_final_val,mu_earth,J2_coeff,R_E); aj2_deputy_final_eci=calculate_j2_perturbation_eci(r_deputy_eci_final_val,mu_earth,J2_coeff,R_E)
        aj2_relative_final_eci=aj2_deputy_final_eci-aj2_chief_final_eci; aj2_relative_final_hill=eci_to_hill_relative_acceleration(aj2_relative_final_eci,r_chief_eci_final_val,v_chief_eci_final_val)
        push!(roe_data_log[:final_rel_j2_r],aj2_relative_final_hill[1]); push!(roe_data_log[:final_rel_j2_t],aj2_relative_final_hill[2]); push!(roe_data_log[:final_rel_j2_n],aj2_relative_final_hill[3]); push!(roe_data_log[:final_rel_j2_norm],norm(aj2_relative_final_eci))
        
        final_separation_distance = norm(r_deputy_eci_final_val - r_chief_eci_final_val)
        if mod(angle_val, 90) == 0
            @printf "  [Angle %.0f deg] Final Separation Distance: %.2f m\n" angle_val (final_separation_distance)
            # 最終的な相対位置をHill座標系で計算
            rel_pos_hill_final = eci_to_hill_rel_pos(r_deputy_eci_final_val, r_chief_eci_final_val, v_chief_eci_final_val)
            @printf "    (Hill Coords -> R: %.2f m, T: %.2f m, N: %.2f m)\n" (rel_pos_hill_final[1]) (rel_pos_hill_final[2]) (rel_pos_hill_final[3])

        end

        if total_cost<min_cost&&isfinite(total_cost); min_cost=total_cost; optimal_angle_deg=angle_val; end
    end
    println("ループ終了")
    println("\n--- 結果 (Pert: $perturbation_setting, DragModel: $drag_model_setting, Plane: $separation_plane_setting) ---")
    if optimal_angle_deg!=-1.0; println("最適分離方向: $optimal_angle_deg deg"); println("最小コスト: $min_cost"); else; println("有効な解が見つかりませんでした。"); end
    
    plot_results(angles_plot_list, roe_data_log, perturbation_setting, drag_model_setting, separation_plane_setting)
end

# --- 実行 ---
function run_all_cases()
    main_simulation(KEPLER_ONLY, NO_DRAG, RT_PLANE)
    main_simulation(J2_ONLY, NO_DRAG, RT_PLANE)
    main_simulation(DRAG_ONLY, DENSITY_MODEL_FREE, RT_PLANE)
    main_simulation(J2_AND_DRAG, DENSITY_MODEL_FREE, RT_PLANE)
end

run_all_cases()
run_state_reconstruction_test()
debug_stm_propagation_long_term()
debug_with_inclination()
debug_stm_components()
debug_reconstruction_with_propagator()