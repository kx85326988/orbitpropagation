using LinearAlgebra
using StaticArrays
using Plots
using Printf
using Base64
using Dates
using Statistics

# --- 物理定数 ---
const mu_earth = 3.986004418e14  # 地心重力定数 (m^3/s^2)
const J2_coeff = 1.08263e-3     # J2係数
const R_E = 6378137.0            # 地球半径 (m)

# --- 主衛星の初期軌道要素 ---
a_c_stm_init = 6903137.0        # (投入予定軌道：近地点、遠地点高度：540km ± 15kmから計算)
e_c_stm_init = 0.0022           # 0~0.0022(投入予定軌道：近地点、遠地点高度：540km ± 15kmから計算)
i_c_stm_init = deg2rad(97.65)   # (投入予定軌道：軌道傾斜角：97.50 ± 0.15 degから計算)
Omega_c_stm_init = deg2rad(0.0) #
omega_c_stm_init = deg2rad(0.0) #
M_c_stm_init = deg2rad(0.0)     # 

# --- 編隊飛行関連パラメータ ---
delta_v_magnitude = 0.1 # m/s
delta_a_dot_drag_initial_normalized = -1.0e-7 # [1/s]
delta_B_initial_param = 0.01 # 仮の差動弾道係数

# --- 構造体定義 ---
struct OrbitalElementsClassical
    a::Float64; e::Float64; i::Float64; RAAN::Float64
    omega::Float64; f_true::Float64; n::Float64; M::Float64
end

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

# --- ヘルパー関数 ---
function cartesian_to_elements_matlab_style(posvel0_vec::SVector{6, Float64}, GM_e::Float64)::OrbitalElementsClassical
    r_vec = posvel0_vec[1:3]; v_vec = posvel0_vec[4:6]
    R_mag = norm(r_vec); v_sq = dot(v_vec, v_vec)
    a = GM_e / ((2 * GM_e / R_mag) - v_sq)
    if a < 0 && abs( (2 * GM_e / R_mag) - v_sq ) > 1e-9;
    elseif abs( (2 * GM_e / R_mag) - v_sq ) < 1e-9; a = Inf; end

    h_vec = cross(r_vec, v_vec); h_mag = norm(h_vec)
    if h_mag < 1e-9; h_mag = 1e-9; end
    i = acos(clamp(h_vec[3] / h_mag, -1.0, 1.0))
    node_vec = SVector(-h_vec[2], h_vec[1], 0.0); node_mag = norm(node_vec)
    RAAN = 0.0
    if node_mag > 1e-9 && abs(i) > 1e-9 && abs(i - pi) > 1e-9
        RAAN = atan(node_vec[2], node_vec[1]); if RAAN < 0.0; RAAN += 2*pi; end
    end
    e_vec = (1/GM_e) * ( (v_sq - GM_e/R_mag)*r_vec - dot(r_vec, v_vec)*v_vec )
    e = norm(e_vec); if e < 1e-10; e = 1e-10; end
    omega = 0.0
    if e > 1e-9
        if node_mag > 1e-9
            cos_omega_val = dot(node_vec, e_vec) / (node_mag * e)
            omega = acos(clamp(cos_omega_val, -1.0, 1.0))
            if e_vec[3] < 0.0; omega = 2*pi - omega; end
        else
            omega = atan(e_vec[2], e_vec[1]); if omega < 0.0; omega += 2*pi; end
        end
    end
    f_true = 0.0
    if e > 1e-9
        cos_f_val = dot(e_vec, r_vec) / (e * R_mag)
        f_true = acos(clamp(cos_f_val, -1.0, 1.0))
        if dot(r_vec, v_vec) < 0.0; f_true = 2*pi - f_true; end
    else
        if node_mag > 1e-9 && abs(i) > 1e-9 && abs(i-pi) > 1e-9
            cos_u_val = dot(normalize(node_vec), normalize(r_vec))
            u_angle = acos(clamp(cos_u_val, -1.0, 1.0))
            if dot(h_vec, cross(node_vec, r_vec)) < 0.0; u_angle = 2*pi - u_angle; end
            f_true = u_angle
        else
            f_true = atan(r_vec[2], r_vec[1]); if f_true < 0.0; f_true += 2*pi; end
        end
    end
    f_true = mod(f_true, 2*pi)
    n_val = sqrt(GM_e / abs(a)^3)
    E_anom = 0.0
    if e < 1.0 - 1e-9
        tan_f_half = tan(f_true/2.0)
        sqrt_term_val = (1.0-e)/(1.0+e); if sqrt_term_val < 0; sqrt_term_val = 0; end
        E_anom = 2.0 * atan(sqrt(sqrt_term_val) * tan_f_half)
    end
    E_anom = mod(E_anom, 2*pi); if E_anom < 0.0; E_anom += 2*pi; end
    M = 0.0
    if e < 1.0 - 1e-9; M = E_anom - e * sin(E_anom); end
    M = mod(M, 2*pi); if M < 0.0; M += 2*pi; end
    return OrbitalElementsClassical(a, e, i, RAAN, omega, f_true, n_val, M)
end

function euler_rotation(angle_rad::Float64, axis::Int)::SMatrix{3,3,Float64}
    c=cos(angle_rad); s=sin(angle_rad)
    if axis==1; return @SMatrix [1 0 0; 0 c s; 0 -s c];
    elseif axis==2; return @SMatrix [c 0 -s; 0 1 0; s 0 c];
    else return @SMatrix [c s 0; -s c 0; 0 0 1]; end
end

function elements_to_cartesian_matlab_style(a::Float64, e::Float64, i_rad::Float64, RAAN_rad::Float64, omega_rad::Float64, f_true_rad::Float64, GM_e::Float64)::SVector{6,Float64}
    if e<0.0||e>=1.0-1e-9; if e<1e-9; e=1e-9; end; if e>=1.0-1e-9; e=1.0-1e-9; end; end
    if a<=0.0&&abs(a)>1e-9&&a!=Inf; error("a must be positive. Got a=$a");
    elseif abs(a)<1e-9&&a!=Inf; error("a is too small. Got a=$a"); end
    p_val=a*(1.0-e^2); if p_val<1e-6&&a!=Inf; p_val=1e-6; end
    if isinf(a); p_val=2*norm(cross(elements_to_cartesian_matlab_style(a,e,i_rad,RAAN_rad,omega_rad,f_true_rad,GM_e)[1:3], elements_to_cartesian_matlab_style(a,e,i_rad,RAAN_rad,omega_rad,f_true_rad,GM_e)[4:6]))^2/GM_e; end
    r_norm_pqw=p_val/(1.0+e*cos(f_true_rad))
    r_pqw=SVector(r_norm_pqw*cos(f_true_rad),r_norm_pqw*sin(f_true_rad),0.0)
    v_pqw_x=-sqrt(GM_e/p_val)*sin(f_true_rad); v_pqw_y=sqrt(GM_e/p_val)*(e+cos(f_true_rad))
    v_pqw=SVector(v_pqw_x,v_pqw_y,0.0)
    Rot3_neg_w=euler_rotation(-omega_rad,3); Rot1_neg_i=euler_rotation(-i_rad,1); Rot3_neg_RAAN=euler_rotation(-RAAN_rad,3)
    DCM_pqw_to_eci=Rot3_neg_RAAN*Rot1_neg_i*Rot3_neg_w
    r_eci=DCM_pqw_to_eci*r_pqw; v_eci=DCM_pqw_to_eci*v_pqw
    return vcat(r_eci,v_eci)
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
    nd=sqrt(mu_earth/abs(ad)^3); f_true_d_approx=Md
    if abs(ed)>1e-9
        E_approx_d=Md+ed*sin(Md)
        if abs(1.0-ed)>1e-9; tan_E_half_d=tan(E_approx_d/2.0); if abs(1.0-ed)<1e-9; f_true_d_approx=E_approx_d; else; sqrt_term_d=sqrt(abs((1.0+ed)/(1.0-ed))); f_true_d_approx=2.0*atan(sqrt_term_d*tan_E_half_d); end
        else; f_true_d_approx=E_approx_d; end
        f_true_d_approx=mod(f_true_d_approx,2*pi); if f_true_d_approx<0; f_true_d_approx+=2*pi; end
    end
    return OrbitalElementsClassical(ad,ed,id,Omegad,omegad,f_true_d_approx,nd,Md)
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

function get_A_prime_qns_augmented_koenig_selectable(
    ac_val::Float64, ec_val::Float64, ic_val::Float64, omegac_val::Float64, 
    include_j2::Bool, include_drag_effects::Bool, drag_model_type::DragModelTypeForSTM
)::Tuple{SMatrix{7,7,Float64}, SMatrix{7,7,Float64}, SMatrix{7,7,Float64}}
    
    A_kep_p = @MMatrix zeros(Float64,7,7)
    A_j2_p = @MMatrix zeros(Float64,7,7)
    A_drag_p = @MMatrix zeros(Float64,7,7)
    
    n_c = sqrt(mu_earth / ac_val^3)

    # 1. ケプラー項
    A_kep_p[2,1] = -1.5 * n_c

    # 2. J2摂動項
    if include_j2
        eta_c=sqrt(1.0-ec_val^2); if eta_c<1e-9; eta_c=1e-9; end
        kappa_J2=(3.0/4.0)*J2_coeff*(R_E^2*sqrt(mu_earth))/(ac_val^(3.5)*eta_c^4)
        E_f=1.0+eta_c; F_f=4.0+3.0*eta_c; G_f=1.0/eta_c^2
        cos_i=cos(ic_val); sin_i=sin(ic_val)
        P_g=3.0*cos_i^2-1.0; Q_g=5.0*cos_i^2-1.0; S_g=sin(2.0*ic_val); T_g=sin_i^2
        ex_c=ec_val*cos(omegac_val); ey_c=ec_val*sin(omegac_val)
        
        A_j2_p[2,1] = -0.5*kappa_J2*E_f*P_g
        A_j2_p[2,3] = kappa_J2*F_f*G_f*ex_c
        A_j2_p[2,4] = kappa_J2*F_f*G_f*ey_c
        A_j2_p[2,5] = -kappa_J2*F_f*S_g
        A_j2_p[3,4] = -kappa_J2*Q_g
        A_j2_p[3,5] = kappa_J2*Q_g*G_f*ey_c
        A_j2_p[4,3] = kappa_J2*Q_g
        A_j2_p[4,5] = -kappa_J2*Q_g*G_f*ex_c
        A_j2_p[6,1] = 0.5*kappa_J2*S_g
        A_j2_p[6,3] = -kappa_J2*G_f*S_g*ex_c
        A_j2_p[6,4] = -kappa_J2*G_f*S_g*ey_c
        A_j2_p[6,5] = kappa_J2*T_g
    end

    # 3. 差動抗力項
    if include_drag_effects
        if drag_model_type == DENSITY_MODEL_FREE
            A_drag_p[1,7] = 1.0
            A_drag_p[3,7] = (1.0 - ec_val)
        elseif drag_model_type == DENSITY_MODEL_SPECIFIC
            println("警告: DENSITY_MODEL_SPECIFIC のプラント行列は未実装です。")
        end
    end
    
    return SMatrix(A_kep_p), SMatrix(A_j2_p), SMatrix(A_drag_p)
end

function get_STM_prime_qns_augmented_koenig_model_selectable(
    A_kep_prime::SMatrix{7,7,Float64},
    A_j2_prime::SMatrix{7,7,Float64},
    A_drag_prime::SMatrix{7,7,Float64},
    t_prop::Float64,
    include_drag_effects::Bool,
    drag_model_type::DragModelTypeForSTM
)::SMatrix{7,7,Float64}
    
    A_kep_J2_prime = A_kep_prime + A_j2_prime # ケプラーとJ2のプラント行列を結合

    if drag_model_type == DENSITY_MODEL_FREE && include_drag_effects
        Phi_drag_prime = SMatrix{7,7,Float64}(I) + A_drag_prime * t_prop
        Integral_Phi_drag_prime = SMatrix{7,7,Float64}(I) * t_prop + A_drag_prime * (t_prop^2 / 2.0)
        return Phi_drag_prime + A_kep_J2_prime * Integral_Phi_drag_prime
    elseif drag_model_type == DENSITY_MODEL_SPECIFIC && include_drag_effects
        println("警告: DENSITY_MODEL_SPECIFIC のSTM計算は未実装です。")
        return SMatrix{7,7,Float64}(I) + (A_kep_J2_prime + A_drag_prime) * t_prop # 仮
    else # NO_DRAG
        return SMatrix{7,7,Float64}(I) + A_kep_J2_prime * t_prop
    end
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

# --- ★★★ 状態再構成プロセスを検証するためのテスト関数 ★★★ ---
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
        deg2rad(30.0), # f_true
        sqrt(mu_earth / (7000e3)^3), # n
        0.0 # M (f_trueから別途計算)
    )
    # Mをf_trueから計算
    E_anom_c = 2.0 * atan(sqrt((1.0-oe_chief_initial.e)/(1.0+oe_chief_initial.e)) * tan(oe_chief_initial.f_true/2.0))
    M_c = E_anom_c - oe_chief_initial.e * sin(E_anom_c)
    oe_chief_initial = OrbitalElementsClassical(oe_chief_initial.a, oe_chief_initial.e, oe_chief_initial.i, oe_chief_initial.RAAN, oe_chief_initial.omega, oe_chief_initial.f_true, oe_chief_initial.n, M_c)
    
    posvel_chief_initial_eci = elements_to_cartesian_matlab_style(oe_chief_initial.a, oe_chief_initial.e, oe_chief_initial.i, oe_chief_initial.RAAN, oe_chief_initial.omega, oe_chief_initial.f_true, mu_earth)
    r_chief_initial_eci = SVector{3}(posvel_chief_initial_eci[1:3])
    v_chief_initial_eci = SVector{3}(posvel_chief_initial_eci[4:6])

    # 副衛星の初期状態 (主衛星に対して意図的にずれを持たせる)
    # 例: 軌道長半径をわずかに大きく、離心率ベクトルのx成分をずらす
    oe_deputy_initial_true = OrbitalElementsClassical(
        oe_chief_initial.a + 10.0, # a_d = a_c + 10m
        oe_chief_initial.e + 0.0001, # e_d
        oe_chief_initial.i + deg2rad(0.01), # i_d
        oe_chief_initial.RAAN + deg2rad(0.01), # RAAN_d
        oe_chief_initial.omega + deg2rad(0.02), # omega_d
        oe_chief_initial.f_true + deg2rad(0.03), # f_true_d
        0.0, 0.0 # n, M は後で計算
    )
    E_anom_d = 2.0 * atan(sqrt((1.0-oe_deputy_initial_true.e)/(1.0+oe_deputy_initial_true.e)) * tan(oe_deputy_initial_true.f_true/2.0))
    M_d = E_anom_d - oe_deputy_initial_true.e * sin(E_anom_d)
    n_d = sqrt(mu_earth / oe_deputy_initial_true.a^3)
    oe_deputy_initial_true = OrbitalElementsClassical(oe_deputy_initial_true.a, oe_deputy_initial_true.e, oe_deputy_initial_true.i, oe_deputy_initial_true.RAAN, oe_deputy_initial_true.omega, oe_deputy_initial_true.f_true, n_d, M_d)
    
    posvel_deputy_initial_eci_true = elements_to_cartesian_matlab_style(oe_deputy_initial_true.a, oe_deputy_initial_true.e, oe_deputy_initial_true.i, oe_deputy_initial_true.RAAN, oe_deputy_initial_true.omega, oe_deputy_initial_true.f_true, mu_earth)
    r_deputy_initial_eci_true = SVector{3}(posvel_deputy_initial_eci_true[1:3])
    v_deputy_initial_eci_true = SVector{3}(posvel_deputy_initial_eci_true[4:6])

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
    posvel_deputy_reconstructed_eci = elements_to_cartesian_matlab_style(
        oe_deputy_reconstructed.a, oe_deputy_reconstructed.e, oe_deputy_reconstructed.i, oe_deputy_reconstructed.RAAN,
        oe_deputy_reconstructed.omega, oe_deputy_reconstructed.f_true, mu_earth
    )
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
            push!(plot_j2_final_list, plot(angles_for_final_j2_plot, filter_outliers_iqr(roe_data_log[:final_rel_j2_t][valid_final_j2_indices]), title="J2 (Along-track Hill)"*plot_title_main_suffix, legend=false, m=:o, ms=2, xlims=xlims_plot, ylabel="a_T (m/s^2)"))
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

    f_true_c_init_calc=M_c_stm_init
    if abs(e_c_stm_init)>1e-9; E_approx_calc=M_c_stm_init+e_c_stm_init*sin(M_c_stm_init); if abs(1.0-e_c_stm_init)>1e-9; tan_E_half_calc=tan(E_approx_calc/2.0); if abs(1.0-e_c_stm_init)<1e-9; f_true_c_init_calc=E_approx_calc; else; sqrt_term_calc=sqrt(abs((1.0+e_c_stm_init)/(1.0-e_c_stm_init))); f_true_c_init_calc=2.0*atan(sqrt_term_calc*tan_E_half_calc); end; else; f_true_c_init_calc=E_approx_calc; end; f_true_c_init_calc=mod(f_true_c_init_calc,2*pi); if f_true_c_init_calc<0; f_true_c_init_calc+=2*pi; end; end
    posvel_chief_initial_eci_vec = elements_to_cartesian_matlab_style(a_c_stm_init,e_c_stm_init,i_c_stm_init,Omega_c_stm_init,omega_c_stm_init,f_true_c_init_calc,mu_earth)
    r_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[1:3]); v_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[4:6])
    oe_chief_eval=cartesian_to_elements_matlab_style(posvel_chief_initial_eci_vec,mu_earth)

    optimal_angle_deg=-1.0; min_cost=Inf; angles_plot_list=Float64[]
    roe_data_log=Dict(
        :initial_delta_a=>Float64[], :initial_delta_lambda=>Float64[], :initial_delta_ex=>Float64[], :initial_delta_ey=>Float64[], :initial_delta_ix=>Float64[], :initial_delta_iy=>Float64[],
        :final_delta_a=>Float64[], :final_delta_lambda=>Float64[], :final_delta_ex=>Float64[], :final_delta_ey=>Float64[], :final_delta_ix=>Float64[], :final_delta_iy=>Float64[],
        :cost=>Float64[], :t_final=>Float64[],
        :initial_rel_j2_r=>Float64[], :initial_rel_j2_t=>Float64[], :initial_rel_j2_n=>Float64[], :initial_rel_j2_norm=>Float64[],
        :final_rel_j2_r=>Float64[], :final_rel_j2_t=>Float64[], :final_rel_j2_n=>Float64[], :final_rel_j2_norm=>Float64[]
    )
    println("Pert: $perturbation_setting, J2: $include_j2_active, Drag STM active: $include_drag_active_stm (Model: $drag_model_setting), Separation Plane: $separation_plane_setting")

    # ★★★ 固定伝播時間の設定 (10軌道周期) ★★★
    fixed_propagation_time = 10.0 * 2.0 * pi * sqrt(oe_chief_eval.a^3 / mu_earth)
    println("Fixed propagation time set to 10 orbits: $(fixed_propagation_time) seconds")

    angle_step_val=10.0
    for angle_val in 0.0:angle_step_val:(360.0-angle_step_val)
        angle_rad_val=deg2rad(angle_val)
        dv_R_val_comp=0.0; dv_T_val_comp=0.0; dv_N_val_comp=0.0
        if separation_plane_setting==RT_PLANE; dv_R_val_comp=delta_v_magnitude*cos(angle_rad_val); dv_T_val_comp=delta_v_magnitude*sin(angle_rad_val);
        elseif separation_plane_setting==RN_PLANE; dv_R_val_comp=delta_v_magnitude*cos(angle_rad_val); dv_N_val_comp=delta_v_magnitude*sin(angle_rad_val);
        elseif separation_plane_setting==NT_PLANE; dv_T_val_comp=delta_v_magnitude*cos(angle_rad_val); dv_N_val_comp=delta_v_magnitude*sin(angle_rad_val); end
        dv_lvlh_vec=SVector(dv_R_val_comp,dv_T_val_comp,dv_N_val_comp); dr_lvlh_vec=SVector(10.0,0.0,0.0) # 初期相対位置
        state_deputy_init_eci=cw_to_eci_deputy_state(r_chief_init_eci,v_chief_init_eci,dr_lvlh_vec,dv_lvlh_vec)
        r_dep_init_eci=state_deputy_init_eci.r_vec; v_dep_init_eci=state_deputy_init_eci.v_vec
        aj2_chief_init=calculate_j2_perturbation_eci(r_chief_init_eci,mu_earth,J2_coeff,R_E)
        aj2_dep_init=calculate_j2_perturbation_eci(r_dep_init_eci,mu_earth,J2_coeff,R_E)
        aj2_rel_init_eci=aj2_dep_init-aj2_chief_init
        aj2_rel_init_hill=eci_to_hill_relative_acceleration(aj2_rel_init_eci,r_chief_init_eci,v_chief_init_eci)
        push!(roe_data_log[:initial_rel_j2_r],aj2_rel_init_hill[1]); push!(roe_data_log[:initial_rel_j2_t],aj2_rel_init_hill[2]); push!(roe_data_log[:initial_rel_j2_n],aj2_rel_init_hill[3]); push!(roe_data_log[:initial_rel_j2_norm],norm(aj2_rel_init_eci))
        posvel_dep_init_vec=vcat(r_dep_init_eci,v_dep_init_eci)
        oe_dep_init=cartesian_to_elements_matlab_style(posvel_dep_init_vec,mu_earth)
        qns_roes_init=orbital_elements_to_qns_roe_koenig(oe_chief_eval,oe_dep_init)
        push!(roe_data_log[:initial_delta_a],qns_roes_init.delta_a_norm); push!(roe_data_log[:initial_delta_lambda],rad2deg(qns_roes_init.delta_lambda)); push!(roe_data_log[:initial_delta_ex],qns_roes_init.delta_ex); push!(roe_data_log[:initial_delta_ey],qns_roes_init.delta_ey); push!(roe_data_log[:initial_delta_ix],rad2deg(qns_roes_init.delta_ix)); push!(roe_data_log[:initial_delta_iy],rad2deg(qns_roes_init.delta_iy))
        aug_param_val=0.0
        if drag_model_setting==DENSITY_MODEL_FREE; aug_param_val=delta_a_dot_drag_initial_normalized; elseif drag_model_setting==DENSITY_MODEL_SPECIFIC; aug_param_val=delta_B_initial_param; end
        roe_aug_init_vec_temp=MVector{7,Float64}(qns_roes_init.delta_a_norm,qns_roes_init.delta_lambda,qns_roes_init.delta_ex,qns_roes_init.delta_ey,qns_roes_init.delta_ix,qns_roes_init.delta_iy,aug_param_val)
        if !include_drag_active_stm; roe_aug_init_vec_temp[7]=0.0; end
        roe_aug_init_vec=SVector(roe_aug_init_vec_temp)
        
        # ★★★ 伝播時間を全てのケースで固定値に設定 ★★★
        tf_val = fixed_propagation_time
        push!(angles_plot_list,angle_val)
        push!(roe_data_log[:t_final],tf_val)
        
        omega_c_ti=oe_chief_eval.omega; omega_dot_j2,Omega_dot_j2=include_j2_active ? get_secular_j2_rates_koenig(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i) : (0.0,0.0)
        oe_chief_at_tf=OrbitalElementsClassical(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,mod(oe_chief_eval.RAAN+Omega_dot_j2*tf_val,2*pi),mod(oe_chief_eval.omega+omega_dot_j2*tf_val,2*pi),0.0,oe_chief_eval.n,mod(oe_chief_eval.M+oe_chief_eval.n*tf_val,2*pi))
        f_true_chief_tf_calc=oe_chief_at_tf.M; if abs(oe_chief_at_tf.e)>1e-9; E_approx_tf_calc=oe_chief_at_tf.M+oe_chief_at_tf.e*sin(oe_chief_at_tf.M); if abs(1.0-oe_chief_at_tf.e)>1e-9; tan_E_half_tf_calc=tan(E_approx_tf_calc/2.0); if abs(1.0-oe_chief_at_tf.e)<1e-9; f_true_chief_tf_calc=E_approx_tf_calc; else; sqrt_term_tf_calc=sqrt(abs((1.0+oe_chief_at_tf.e)/(1.0-oe_chief_at_tf.e))); f_true_chief_tf_calc=2.0*atan(sqrt_term_tf_calc*tan_E_half_tf_calc); end; else; f_true_chief_tf_calc=E_approx_tf_calc; end; f_true_chief_tf_calc=mod(f_true_chief_tf_calc,2*pi); if f_true_chief_tf_calc<0; f_true_chief_tf_calc+=2*pi; end; end
        oe_chief_at_tf=OrbitalElementsClassical(oe_chief_at_tf.a,oe_chief_at_tf.e,oe_chief_at_tf.i,oe_chief_at_tf.RAAN,oe_chief_at_tf.omega,f_true_chief_tf_calc,oe_chief_at_tf.n,oe_chief_at_tf.M)
        omega_c_tf_val=oe_chief_at_tf.omega; J_ti=get_J_qns_augmented_koenig(omega_c_ti); J_tf_inv=get_J_qns_inv_augmented_koenig(omega_c_tf_val); roe_prime_init=J_ti*roe_aug_init_vec
        
        # ★★★ 修正箇所: 不要な引数 delta_B_initial_param を削除 ★★★
        A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(
            oe_chief_eval.a, oe_chief_eval.e, oe_chief_eval.i, omega_c_ti, 
            include_j2_active, include_drag_active_stm, drag_model_setting
        )
        
        STM_prime = get_STM_prime_qns_augmented_koenig_model_selectable(
            A_kep_p, A_j2_p, A_drag_p, 
            tf_val, 
            include_drag_active_stm, drag_model_setting
        )

        roe_prime_final=STM_prime*roe_prime_init; roe_aug_final_vec=J_tf_inv*roe_prime_final
        push!(roe_data_log[:final_delta_a],roe_aug_final_vec[1]); push!(roe_data_log[:final_delta_lambda],rad2deg(roe_aug_final_vec[2])); push!(roe_data_log[:final_delta_ex],roe_aug_final_vec[3]); push!(roe_data_log[:final_delta_ey],roe_aug_final_vec[4]); push!(roe_data_log[:final_delta_ix],rad2deg(roe_aug_final_vec[5])); push!(roe_data_log[:final_delta_iy],rad2deg(roe_aug_final_vec[6]))
        
        # ★★★ 新しいコスト関数の定義 ★★★
        cost_da_penalty=1000.0*roe_aug_final_vec[1]^2 # 10軌道周期後に残っているδaへのペナルティ
        cost_others=norm(view(roe_aug_final_vec,2:6)); 
        total_cost=cost_da_penalty+cost_others; push!(roe_data_log[:cost],total_cost)

        posvel_chief_final_eci=elements_to_cartesian_matlab_style(oe_chief_at_tf.a,oe_chief_at_tf.e,oe_chief_at_tf.i,oe_chief_at_tf.RAAN,oe_chief_at_tf.omega,oe_chief_at_tf.f_true,mu_earth)
        r_chief_eci_final_val=SVector{3}(posvel_chief_final_eci[1:3]); v_chief_eci_final_val=SVector{3}(posvel_chief_final_eci[4:6])
        oe_deputy_at_tf=final_roe_to_deputy_oe(oe_chief_eval,roe_aug_final_vec)
        posvel_deputy_final_eci=elements_to_cartesian_matlab_style(oe_deputy_at_tf.a,oe_deputy_at_tf.e,oe_deputy_at_tf.i,oe_deputy_at_tf.RAAN,oe_deputy_at_tf.omega,oe_deputy_at_tf.f_true,mu_earth)
        r_deputy_eci_final_val=SVector{3}(posvel_deputy_final_eci[1:3])
        aj2_chief_final_eci=calculate_j2_perturbation_eci(r_chief_eci_final_val,mu_earth,J2_coeff,R_E); aj2_deputy_final_eci=calculate_j2_perturbation_eci(r_deputy_eci_final_val,mu_earth,J2_coeff,R_E)
        aj2_relative_final_eci=aj2_deputy_final_eci-aj2_chief_final_eci; aj2_relative_final_hill=eci_to_hill_relative_acceleration(aj2_relative_final_eci,r_chief_eci_final_val,v_chief_eci_final_val)
        push!(roe_data_log[:final_rel_j2_r],aj2_relative_final_hill[1]); push!(roe_data_log[:final_rel_j2_t],aj2_relative_final_hill[2]); push!(roe_data_log[:final_rel_j2_n],aj2_relative_final_hill[3]); push!(roe_data_log[:final_rel_j2_norm],norm(aj2_relative_final_eci))
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
run_state_reconstruction_test() # デバッグ用関数を実行
