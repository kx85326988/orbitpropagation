# ==============================================================================
# [ J2摂動下での低推力編隊形成のための最適分離マヌーバ探索プログラム ]
#
# ## 目的 (Main Purpose)
# 
# このプログラムは，J2摂動と差動抗力の影響下で，特定の相対軌道（J2不変条件を満たす
# 安定な編隊）を形成するための，最適な初期分離マヌーバ（分離方向と分離速度）を
# 探索することを目的とする．
# 燃料を消費しない差動抗力を利用しつつ，J2摂動による長期的な軌道のずれを最小化する
# バランスの取れた解を見つけ出す．
#
# ## コードの流れ (Workflow)
#
# 1.  **初期条件と目標の設定:**
#     - 主衛星の初期軌道要素，衛星の物理パラメータ（質量，面積など）を設定．
#     - 目標とする相対軌道の形状（例: 0.5km x 1kmの楕円）と，編隊維持の安定性の指標と
#       なる「J2不変条件」から，最終的に目指すべき理想的な相対軌道要素(ROE)を計算する．
#
# 2.  **パラメータ探索ループ:**
#     - 分離速度の大きさと，分離方向（0°～360°）を変化させながら，二重のループで
#       全ての組み合わせをテストする．
#
# 3.  **順伝播シミュレーション:**
#     - 各分離マヌーバに対して，まず初期ROEを計算する．
#     - Koenigらの論文に基づく状態遷移マトリックス(STM)を用いて，一定時間
#       （例: 10軌道周期）後の最終的なROEを予測計算する．
#
# 4.  **コスト計算と最適解の探索:**
#     - 計算された最終ROEが，ステップ1で設定した「J2不変条件を満たす理想のROE」から
#       どれだけずれているかを「コスト」として定量化する．
#     - 全ての分離マヌーバの中で，このコストが最小となるものを「最適解」として記録する．
#
# 5.  **結果の表示と保存:**
#     - 探索ループ終了後，見つかった最適な分離方向と分離速度をコンソールに出力する．
#     - コストが分離方向と分離速度によってどう変化するかの全体像を3Dサーフェスプロットで
#       可視化し，HTMLレポートとして保存する．
#
# ==============================================================================
using LinearAlgebra
using StaticArrays
using Plots
using Printf
using Base64
using Dates
using Statistics
using SatelliteToolbox

# ★★★ インタラクティブなバックエンドを指定 ★★★
# gr() # GRバックエンドを使用する場合
plotlyjs() # PlotlyJSバックエンドを使用する場合

# --- 物理定数 ---
const mu_earth = 3.986004418e14
const J2_coeff = 1.08263e-3
const R_E = 6378137.0

# --- 主衛星の初期軌道要素 ---
a_c_stm_init = 6903137.0
e_c_stm_init = 0.0022
i_c_stm_init = deg2rad(97.65)
Omega_c_stm_init = deg2rad(0.0)
omega_c_stm_init = deg2rad(0.0)
M_c_stm_init = deg2rad(0.0)

# --- 編隊飛行関連パラメータ ---
const dr_lvlh_init = SVector(0.0, 0.0, 0.0)
const delta_a_dot_drag = -4.6e-11 # [1/s]

const PROPAGATION_ORBITS = 10.0 # 評価を行う軌道周期数

# --- 目標軌道パラメータ ---
const TARGET_a_delta_e_norm = 500.0 # [m] (0.5km x 1km 楕円の短軸半径)
# (その他のターゲットボックス条件はコスト関数内で直接評価)

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
function sv_to_orbital_elements(state::CartesianStateECI, epoch::Float64 = 0.0)::OrbitalElementsClassical
    sv = OrbitStateVector(epoch, state.r_vec, state.v_vec)
    kep = SatelliteToolbox.sv_to_kepler(sv)
    M_val = SatelliteToolbox.true_to_mean_anomaly(kep.e, kep.f)
    n_val = sqrt(mu_earth / kep.a^3)
    return OrbitalElementsClassical(kep.a, kep.e, kep.i, kep.Ω, kep.ω, kep.f, n_val, M_val)
end

function orbital_elements_to_sv(oe::OrbitalElementsClassical, epoch::Float64 = 0.0)::SVector{6,Float64}
    f_true_val = SatelliteToolbox.mean_to_true_anomaly(oe.e, oe.M)
    keps = KeplerianElements(epoch, oe.a, oe.e, oe.i, oe.RAAN, oe.omega, f_true_val)
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

function final_roe_to_deputy_oe(oe_chief_final::OrbitalElementsClassical, final_roes::SVector{7,Float64})::OrbitalElementsClassical
    ac,ec,ic,Omegac,omegac,Mc=oe_chief_final.a,oe_chief_final.e,oe_chief_final.i,oe_chief_final.RAAN,oe_chief_final.omega,oe_chief_final.M
    delta_a_norm_val=final_roes[1]; delta_lambda_val=final_roes[2]; delta_ex_val=final_roes[3]; delta_ey_val=final_roes[4]; delta_ix_val=final_roes[5]; delta_iy_val=final_roes[6]
    ad=ac*(1.0+delta_a_norm_val); id=ic+delta_ix_val; Omegad=Omegac
    if abs(sin(ic))>1e-7; Omegad=Omegac+delta_iy_val/sin(ic); end; Omegad=mod(Omegad,2*pi)
    X=delta_ex_val+ec*cos(omegac); Y=delta_ey_val+ec*sin(omegac); ed=sqrt(X^2+Y^2); if ed<1e-10; ed=1e-10; end
    omegad=0.0; if ed>1e-9; omegad=atan(Y,X); if omegad<0.0; omegad+=2*pi; end; end
    Md=delta_lambda_val+(Mc+omegac+Omegac*cos(ic))-(omegad+Omegad*cos(id)); Md=mod(Md,2*pi); if Md<0.0; Md+=2*pi; end
    nd=sqrt(mu_earth/abs(ad)^3); f_true_d_val = SatelliteToolbox.mean_to_true_anomaly(ed, Md)
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
        A_j2_p[2,1]=-0.5*kappa_J2*E_f*P_g; A_j2_p[2,3]=kappa_J2*F_f*G_f*ex_c; A_j2_p[2,4]=kappa_J2*F_f*G_f*ey_c; A_j2_p[2,5]=-kappa_J2*F_f*S_g
        A_j2_p[3,4]=-kappa_J2*Q_g; A_j2_p[3,5]=kappa_J2*Q_g*G_f*ey_c
        A_j2_p[4,3]=kappa_J2*Q_g; A_j2_p[4,5]=-kappa_J2*G_f*Q_g*ex_c
        A_j2_p[6,1]=0.5*kappa_J2*S_g; A_j2_p[6,3]=-kappa_J2*G_f*S_g*ex_c; A_j2_p[6,4]=-kappa_J2*G_f*S_g*ey_c; A_j2_p[6,5]=kappa_J2*T_g
    end
    if include_drag_effects
        if drag_model_type==DENSITY_MODEL_FREE; A_drag_p[1,7]=1.0;
        elseif drag_model_type==DENSITY_MODEL_SPECIFIC; println("警告: DENSITY_MODEL_SPECIFIC のプラント行列は未実装です．") end
    end
    return SMatrix(A_kep_p), SMatrix(A_j2_p), SMatrix(A_drag_p)
end

function get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_prime::SMatrix{7,7,Float64}, A_j2_prime::SMatrix{7,7,Float64}, A_drag_prime::SMatrix{7,7,Float64}, t_prop::Float64, ec_val_for_drag_effect::Float64, include_drag_effects::Bool, drag_model_type::DragModelTypeForSTM)::SMatrix{7,7,Float64}
    A_kep_J2_prime=A_kep_prime+A_j2_prime
    if drag_model_type==DENSITY_MODEL_FREE && include_drag_effects
        Phi_drag_prime=SMatrix{7,7,Float64}(I)+A_drag_prime*t_prop
        Integral_Phi_drag_prime=SMatrix{7,7,Float64}(I)*t_prop+A_drag_prime*(t_prop^2/2.0)
        return Phi_drag_prime+A_kep_J2_prime*Integral_Phi_drag_prime
    else; return SMatrix{7,7,Float64}(I)+A_kep_J2_prime*t_prop; end
end

function plot_results(angles_plot_list, cost_data, perturbation_setting, drag_model_setting, separation_plane_setting)
    # 3Dサーフェスプロット用にデータを整形
    dv_mags = 0.001:0.001:0.05
    angles_deg = 0.0:10.0:350.0
        
    # cost_dataが空でないことを確認
    if isempty(cost_data)
        println("プロットするためのコストデータがありません．")
        return
    end

    cost_matrix = reshape(cost_data, (length(angles_deg), length(dv_mags)))

    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    html_filename = "j2_invariant_search_report_$(timestamp).html"
    
        # 3Dプロットの作成
    p = surface(
        angles_deg, 
        dv_mags, 
        cost_matrix', # データの向きを合わせるために転置
        xlabel="Separation Angle (deg)", 
        ylabel="Delta-V Mag (m/s)", 
        zlabel="Cost (J2-Invariant Error)", 
        camera=(30,30), 
        color=:viridis, 
        title="Cost Landscape"
    )
    
    # ★★★ ここが重要: プロットを明示的に表示する ★★★
    display(p)
    
    open(html_filename, "w") do f
        write(f, "<html><head><title>J2-Invariant Maneuver Search Report</title>")
        write(f, "<style> body { font-family: sans-serif; } h1, h2 { color: #333; } div.plot-container { page-break-inside: avoid; margin-bottom: 30px; padding-top: 20px; } img { border: 1px solid #ccc; max-width: 100%; height: auto; } table { border-collapse: collapse; width: 50%; margin-bottom: 20px; } th, td { border: 1px solid #ddd; padding: 8px; text-align: left; } th { background-color: #f2f2f2; } </style>")
        write(f, "</head><body>")
        write(f, "<h1>J2-Invariant Maneuver Search Report</h1>")
        write(f, "<h2>Cost Function (J2-Invariant Error) vs. Separation Maneuver</h2>")
        
        # Base64エンコードしてHTMLに埋め込み
        io = IOBuffer()
        show(io, MIME"image/png"(), p)
        plot_base64 = base64encode(take!(io))
        write(f, "<div><img src=\"data:image/png;base64,$(plot_base64)\"/></div>")
        
        write(f, "</body></html>")
    end
    println("HTMLレポートを保存しました: $html_filename")
end

function find_j2_invariant_maneuver(perturbation_setting::PerturbationType, separation_plane_setting::SeparationPlane)
    drag_model_setting = DENSITY_MODEL_FREE
    println("\n\n--- J2不変条件を満たす分離マヌーバの探索を開始します ---")
    println("Pert: $perturbation_setting, Plane: $separation_plane_setting, Propagation: $PROPAGATION_ORBITS orbits")
    
    include_j2_active = (perturbation_setting == J2_ONLY || perturbation_setting == J2_AND_DRAG)
    include_drag_active_stm = (perturbation_setting == DRAG_ONLY || perturbation_setting == J2_AND_DRAG)
    
    oe_chief_initial_for_sv = OrbitalElementsClassical(a_c_stm_init,e_c_stm_init,i_c_stm_init,Omega_c_stm_init,omega_c_stm_init,0.0,0.0,M_c_stm_init)
    posvel_chief_initial_eci_vec = orbital_elements_to_sv(oe_chief_initial_for_sv)
    r_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[1:3]); v_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[4:6])
    oe_chief_eval = sv_to_orbital_elements(CartesianStateECI(r_chief_init_eci, v_chief_init_eci))
    
    optimal_cost = Inf
    optimal_params = (angle=0.0, dv_mag=0.0)
    
    tf_val = PROPAGATION_ORBITS * 2.0 * pi * sqrt(oe_chief_eval.a^3 / mu_earth)
    println("伝播時間: $tf_val 秒 ($(PROPAGATION_ORBITS)軌道周期)")
    
    cost_data = Float64[] # プロット用のコストデータを保存

    for dv_mag in 0.001:0.001:0.05
        for angle_val in 0.0:10.0:350.0
            angle_rad_val = deg2rad(angle_val)
            dv_R_val_comp=dv_mag*cos(angle_rad_val); dv_T_val_comp=dv_mag*sin(angle_rad_val)
            dv_lvlh_vec = SVector(dv_R_val_comp, dv_T_val_comp, 0.0)

            state_deputy_init_eci=cw_to_eci_deputy_state(r_chief_init_eci,v_chief_init_eci,dr_lvlh_init,dv_lvlh_vec)
            oe_dep_init = sv_to_orbital_elements(CartesianStateECI(state_deputy_init_eci.r_vec, state_deputy_init_eci.v_vec))
            qns_roes_init=orbital_elements_to_qns_roe_koenig(oe_chief_eval,oe_dep_init)
            roe_aug_init_vec=SVector(qns_roes_init.delta_a_norm, qns_roes_init.delta_lambda, qns_roes_init.delta_ex, qns_roes_init.delta_ey, qns_roes_init.delta_ix, qns_roes_init.delta_iy, delta_a_dot_drag)
            
            omega_c_ti=oe_chief_eval.omega; omega_dot_j2,Omega_dot_j2=get_secular_j2_rates_koenig(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i)
            oe_chief_at_tf=OrbitalElementsClassical(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,mod(oe_chief_eval.RAAN+Omega_dot_j2*tf_val,2*pi),mod(oe_chief_eval.omega+omega_dot_j2*tf_val,2*pi),0.0,oe_chief_eval.n,mod(oe_chief_eval.M+oe_chief_eval.n*tf_val,2*pi))
            oe_chief_at_tf=OrbitalElementsClassical(oe_chief_at_tf.a,oe_chief_at_tf.e,oe_chief_at_tf.i,oe_chief_at_tf.RAAN,oe_chief_at_tf.omega,SatelliteToolbox.mean_to_true_anomaly(oe_chief_at_tf.e,oe_chief_at_tf.M),oe_chief_at_tf.n,oe_chief_at_tf.M)
            
            omega_c_tf_val=oe_chief_at_tf.omega; J_ti=get_J_qns_augmented_koenig(omega_c_ti); J_tf_inv=get_J_qns_inv_augmented_koenig(omega_c_tf_val); roe_prime_init=J_ti*roe_aug_init_vec
            A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,omega_c_ti,include_j2_active,include_drag_active_stm,drag_model_setting)
            STM_prime=get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_p,A_j2_p,A_drag_p,tf_val,oe_chief_eval.e,include_drag_active_stm,drag_model_setting)
            roe_prime_final=STM_prime*roe_prime_init; roe_aug_final_vec=J_tf_inv*roe_prime_final
            
            ac, ec, ic = oe_chief_at_tf.a, oe_chief_at_tf.e, oe_chief_at_tf.i
            final_δa_norm, _, final_δex, _, final_δix, _ = roe_aug_final_vec

            eta_c = sqrt(1-ec^2)
            delta_e_approx = final_δex 
            delta_eta_approx = (-ec / eta_c) * delta_e_approx
            C11 = (2*J2_coeff*R_E^2)/(4*ac^2*eta_c^5) * (4+3*eta_c) * (1+5*cos(ic)^2)
            target_delta_a_norm = C11 * delta_eta_approx
            actual_delta_a_norm = final_δa_norm
            cost_1 = (actual_delta_a_norm - target_delta_a_norm)^2

            C12 = (1-ec^2)*tan(ic)/(4*ec)
            target_delta_e_approx = C12 * final_δix
            actual_delta_e_approx = final_δex
            cost_2 = (actual_delta_e_approx - target_delta_e_approx)^2
            
            total_cost = cost_1 + cost_2
            push!(cost_data, total_cost)

            if total_cost < optimal_cost
                optimal_cost = total_cost
                optimal_params = (angle=angle_val, dv_mag=dv_mag)
            end
        end
    end
    println("探索ループ終了")
    
    println("\n--- 結果 ---")
    println("J2不変条件を最もよく満たす最適な分離マヌーバ:")
    @printf "  分離方向: %.1f deg\n" optimal_params.angle
    @printf "  分離速度: %.4f m/s\n" optimal_params.dv_mag
    @printf "  最小コスト（J2不変条件からの誤差の2乗和）: %.3e\n" optimal_cost

    plot_results(0.0:10.0:350.0, cost_data, perturbation_setting, drag_model_setting, separation_plane_setting)
end

# --- 実行 ---
function run_target_search()
    find_j2_invariant_maneuver(J2_AND_DRAG, RT_PLANE)
end

run_target_search()