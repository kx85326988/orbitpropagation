# ==============================================================================
# [ J2摂動と差動抗力を考慮した目標相対軌道形成のための最適分離マヌーバ探索 ]
#
# ## 目的 (Main Purpose)
#
# このプログラムは、低軌道環境で支配的なJ2摂動と差動抗力の影響下で、
# あらかじめ設計された**「目標とする相対軌道」**を形成するための、最適な初期分離マヌーバ
# （分離方向と分離速度）を探索することを目的とする。
#
# スラスタ（燃料）を消費しない差動抗力を利用して軌道エネルギー差を解消しつつ、
# J2摂動による長期的な軌道のずれも考慮に入れ、最終的に目標の編隊形状に
# 最も近づけるような、バランスの取れた解を見つけ出す。
#
# ## コードの流れ (Workflow)
#
# 1.  **初期条件と目標ROEの定義:**
#     - 主衛星の初期軌道要素や、差動抗力の効果をモデル化するパラメータを設定する。
#     - ミッションで要求される物理的な制約（例：相対軌道の大きさ、許容される最大軌道面外ずれ）
#       に基づき、最終的に目指すべき**理想的な7次元の相対軌道要素ベクトル（目標ROE）**を設計する。
#
# 2.  **分離マヌーバの全パターン探索:**
#     - 分離速度の大きさと、軌道面内での分離方向（0°～360°）を変化させながら、
#       二重のループ処理で全ての組み合わせを網羅的にテストする。
#
# 3.  **最終ROEの順伝播予測:**
#     - 各分離マヌーバに対して、まず初期の相対軌道要素（ROE）を計算する。
#     - Koenigらの論文に基づく状態遷移マトリックス（STM）を用いて、一定時間後
#       （例: 10軌道周期後）の**最終的なROE**を高速に予測計算する。
#
# 4.  **コスト計算と最適解の探索:**
#     - 予測された最終ROEが、ステップ1で設計した「目標ROE」からどれだけずれているかを、
#       **「重み付きの誤差二乗和」**としてコストを計算する。
#       （例：軌道エネルギーの誤差は厳しく、位相の誤差は許容するなど重みで調整可能）
#     - 全ての分離マヌーバの中で、このコストが最小となるものを「最適解」として記録する。
#
# 5.  **結果の可視化と保存:**
#     - 探索終了後、見つかった最適な分離方向と分離速度をコンソールに出力する。
#     - コストが分離条件（方向、速度）によってどう変化するかの全体像を3Dサーフェスプロットで
#       可視化し、HTMLレポートとして保存する。
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

# ECI座標系の相対ベクトルを、主衛星中心のRTN（LVLH）座標系に変換する
function eci_to_rtn(r_chief_eci::SVector{3,Float64}, v_chief_eci::SVector{3,Float64}, vec_eci::SVector{3,Float64})::SVector{3,Float64}
    r_hat = normalize(r_chief_eci)
    h_vec = cross(r_chief_eci, v_chief_eci)
    n_hat = normalize(h_vec)
    t_hat = cross(n_hat, r_hat)
    
    dcm_eci_to_rtn = transpose(hcat(r_hat, t_hat, n_hat))
    
    return dcm_eci_to_rtn * vec_eci
end

# ==============================================================================
# [ 状態再構成プロセスの検証用テスト関数 ]
# ==============================================================================
function run_state_reconstruction_test()
    println("\n\n--- 最終ROEからの状態再構成プロセスの検証を開始します ---")

    # --- 1. 既知の初期状態を準備 ---
    # 主衛星の初期状態 (テスト用にシンプルな値を使用)
    oe_chief_initial_test = OrbitalElementsClassical(
        7000e3, 0.01, deg2rad(50.0), deg2rad(10.0), deg2rad(20.0), 0.0, 0.0, deg2rad(30.0)
    )
    posvel_chief_initial_test = orbital_elements_to_sv(oe_chief_initial_test)
    r_chief_initial_test = SVector{3}(posvel_chief_initial_test[1:3])
    
    # 副衛星の初期状態 (主衛星に対して意図的に100m程度のずれを持たせる)
    oe_deputy_initial_true = OrbitalElementsClassical(
        oe_chief_initial_test.a + 10.0, # a_d = a_c + 10m
        oe_chief_initial_test.e + 0.0001,
        oe_chief_initial_test.i + deg2rad(0.01),
        oe_chief_initial_test.RAAN + deg2rad(0.01),
        oe_chief_initial_test.omega + deg2rad(0.02),
        0.0, 0.0,
        oe_chief_initial_test.M + deg2rad(0.03)
    )
    posvel_deputy_initial_true = orbital_elements_to_sv(oe_deputy_initial_true)
    r_deputy_initial_true = SVector{3}(posvel_deputy_initial_true[1:3])
    v_deputy_initial_true = SVector{3}(posvel_deputy_initial_true[4:6])

    println("--- 1. 検証用の「真の」状態を設定 ---")
    println("主衛星の真のECI位置: ", r_chief_initial_test)
    println("副衛星の真のECI位置: ", r_deputy_initial_true)
    @printf("真の初期相対距離: %.3f m\n", norm(r_deputy_initial_true - r_chief_initial_test))

    # --- 2. 順変換 (ECI -> ROE) ---
    println("\n--- 2. 順変換 (ECI -> ROE) を実行 ---")
    true_roes = orbital_elements_to_qns_roe_koenig(oe_chief_initial_test, oe_deputy_initial_true)
    true_roes_augmented = SVector(
        true_roes.delta_a_norm, true_roes.delta_lambda,
        true_roes.delta_ex, true_roes.delta_ey,
        true_roes.delta_ix, true_roes.delta_iy,
        0.0 # ダミーの拡張パラメータ
    )
    println("計算された「真の」ROE: ", true_roes)

    # --- 3. 逆変換 (ROE -> ECI) ---
    println("\n--- 3. 逆変換 (ROE -> ECI) を実行 ---")
    oe_deputy_reconstructed = final_roe_to_deputy_oe(oe_chief_initial_test, true_roes_augmented)
    posvel_deputy_reconstructed_eci = orbital_elements_to_sv(oe_deputy_reconstructed)
    r_deputy_reconstructed_eci = SVector{3}(posvel_deputy_reconstructed_eci[1:3])
    v_deputy_reconstructed_eci = SVector{3}(posvel_deputy_reconstructed_eci[4:6])
    
    println("再構成された副衛星のECI位置: ", r_deputy_reconstructed_eci)

    # --- 4. 比較・検証 ---
    println("\n--- 4. 比較・検証 ---")
    position_error_vec = r_deputy_initial_true - r_deputy_reconstructed_eci
    position_error_norm = norm(position_error_vec)

    @printf "位置ベクトルの誤差 (ノルム): %.4e m\n" position_error_norm

    if position_error_norm < 1e-6 # 許容誤差を1マイクロメートルに設定
        println("検証結果: 正常です。順変換と逆変換は整合しています。")
    else
        println("\n★★★★★ エラー ★★★★★")
        println("検証結果: 異常です。状態再構成プロセスに大きな誤差が存在。")
        println("バグは `final_roe_to_deputy_oe` 関数または `orbital_elements_to_sv` 関数にある可能性が非常に高いです。")
    end
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
    println("\n\n--- 目標ROEを達成するための最適分離マヌーバの探索を開始 ---")
    println("Pert: $perturbation_setting, Plane: $separation_plane_setting, Propagation: $PROPAGATION_ORBITS orbits")

    # ★★★ 1. 物理的なミッション要求から目標ROEターゲットを定義 ★★★

    # --- 目標とする物理的な軌道形状 ---
    TARGET_DELTA_E_NORM_METERS = 500.0 # [m] 相対軌道の短軸半径 (例: 0.5km)
    TARGET_Z_MAX_METERS      = 10.0  # [m] 許容される最大軌道面外ずれ

    # --- 物理要求をROEターゲットに変換 ---
    roe_target_a_norm = 0.0
    roe_target_lambda = 0.0

    # a * δe = 500 [m] より、δe を計算
    roe_target_ex     = TARGET_DELTA_E_NORM_METERS / a_c_stm_init
    roe_target_ey     = 0.0

    # δz_max ≈ a * |δiy| より、δiy を計算
    # δix は面内分離なのでゼロを目標とする
    roe_target_ix     = 0.0
    roe_target_iy     = TARGET_Z_MAX_METERS / a_c_stm_init # ここでは正の値を目標とする

    # 7次元の目標ROEベクトルを作成
    roe_target_vec = SVector{7,Float64}(
        roe_target_a_norm,
        roe_target_lambda,
        roe_target_ex,
        roe_target_ey,
        roe_target_ix,
        roe_target_iy,
        0.0 # 拡張パラメータ
    )
    println("物理要求に基づき、以下の目標ROEターゲットを設定しました。")
    @printf " - 目標δex: %.3e (a*δe = %.1f m)\n" roe_target_vec[3] TARGET_DELTA_E_NORM_METERS
    @printf " - 目標δiy: %.3e (max Z = %.1f m)\n" roe_target_vec[6] TARGET_Z_MAX_METERS
    println("目標ROE全体: ", roe_target_vec)

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
            
            # ★★★ 新しいコスト関数の定義 (目標ROEとの誤差の重み付き二乗和) ★★★

            # 誤差ベクトルを計算
            error_vec = roe_aug_final_vec - roe_target_vec

            # 各ROE要素の重要度に応じた重み付け行列を定義
            # δaやδex/iyの誤差は厳しく、δλの誤差は許容するなど調整可能
            W = Diagonal(SVector{7,Float64}(
                1.0e6,   # δa_norm の重み (エネルギー差は非常に重要)
                1.0,     # δlambda の重み (位相の重要度が低い場合)
                1000.0,  # δex の重み (軌道形状)
                1000.0,  # δey の重み (軌道形状)
                1000.0,  # δix の重み (面外ずれ)
                1000.0,  # δiy の重み (面外ずれ)
                0.0      # 拡張パラメータはコストに含めない
            ))

            # 重み付きの誤差二乗和をコストとする (error_vec' * W * error_vec)
            total_cost = dot(error_vec, W * error_vec)
            push!(cost_data, total_cost)

            if total_cost < optimal_cost
                optimal_cost = total_cost
                optimal_params = (angle=angle_val, dv_mag=dv_mag)
            end
        end
    end
    println("探索ループ終了")
    
    println("\n--- 結果 ---")
    println("最も目標ROEに近い最適な分離マヌーバ:")
    @printf "  分離方向: %.1f deg\n" optimal_params.angle
    @printf "  分離速度: %.4f m/s\n" optimal_params.dv_mag
    @printf "  最小コスト（目標ROEからの誤差の2乗和）: %.3e\n" optimal_cost

    plot_results(0.0:10.0:350.0, cost_data, perturbation_setting, drag_model_setting, separation_plane_setting)
    return optimal_params
end

# ==============================================================================
# [ 最適解の物理的状態を分析する関数 ]
# ==============================================================================
function analyze_optimal_result(optimal_angle_deg::Float64, optimal_dv_mag::Float64)
    # println("\n\n--- 最適解の詳細分析を開始します ---")
    @printf "入力: 分離方向 %.1f deg, 分離速度 %.4f m/s\n\n" optimal_angle_deg optimal_dv_mag

    # --- 1. 最適マヌーバによる最終状態の再計算 ---
    # (この部分は変更なし)
    oe_chief_initial_for_sv = OrbitalElementsClassical(a_c_stm_init,e_c_stm_init,i_c_stm_init,Omega_c_stm_init,omega_c_stm_init,0.0,0.0,M_c_stm_init)
    posvel_chief_initial_eci_vec = orbital_elements_to_sv(oe_chief_initial_for_sv)
    r_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[1:3]); v_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[4:6])
    oe_chief_eval = sv_to_orbital_elements(CartesianStateECI(r_chief_init_eci, v_chief_init_eci))
    tf_val = PROPAGATION_ORBITS * 2.0 * pi * sqrt(oe_chief_eval.a^3 / mu_earth)
    
    angle_rad_val = deg2rad(optimal_angle_deg)
    dv_R = optimal_dv_mag * cos(angle_rad_val)
    dv_T = optimal_dv_mag * sin(angle_rad_val)
    dv_lvlh_vec = SVector(dv_R, dv_T, 0.0)
    
    state_deputy_init_eci = cw_to_eci_deputy_state(r_chief_init_eci, v_chief_init_eci, dr_lvlh_init, dv_lvlh_vec)
    oe_dep_init = sv_to_orbital_elements(state_deputy_init_eci)
    qns_roes_init = orbital_elements_to_qns_roe_koenig(oe_chief_eval, oe_dep_init)
    roe_aug_init_vec = SVector(qns_roes_init.delta_a_norm, qns_roes_init.delta_lambda, qns_roes_init.delta_ex, qns_roes_init.delta_ey, qns_roes_init.delta_ix, qns_roes_init.delta_iy, delta_a_dot_drag)

    omega_dot_j2, Omega_dot_j2 = get_secular_j2_rates_koenig(oe_chief_eval.a, oe_chief_eval.e, oe_chief_eval.i)
    oe_chief_at_tf = OrbitalElementsClassical(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,mod(oe_chief_eval.RAAN+Omega_dot_j2*tf_val,2*pi),mod(oe_chief_eval.omega+omega_dot_j2*tf_val,2*pi),0.0,oe_chief_eval.n,mod(oe_chief_eval.M+oe_chief_eval.n*tf_val,2*pi))
    
    omega_c_ti=oe_chief_eval.omega; omega_c_tf_val=oe_chief_at_tf.omega; J_ti=get_J_qns_augmented_koenig(omega_c_ti); J_tf_inv=get_J_qns_inv_augmented_koenig(omega_c_tf_val);
    roe_prime_init=J_ti*roe_aug_init_vec
    A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,omega_c_ti,true,true,DENSITY_MODEL_FREE)
    STM_prime=get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_p,A_j2_p,A_drag_p,tf_val,oe_chief_eval.e,true,DENSITY_MODEL_FREE)
    roe_prime_final=STM_prime*roe_prime_init;
    
    # ★★★ ここが重要 ★★★
    # 最終ROEベクトルは、この行で計算されています
    roe_aug_final_vec=J_tf_inv*roe_prime_final

    # --- 2. 最終的な状態の表示 ---
    oe_deputy_at_tf = final_roe_to_deputy_oe(oe_chief_at_tf, roe_aug_final_vec)
    
    posvel_chief_final = orbital_elements_to_sv(oe_chief_at_tf)
    r_chief_final = SVector{3}(posvel_chief_final[1:3])
    v_chief_final = SVector{3}(posvel_chief_final[4:6])

    posvel_deputy_final = orbital_elements_to_sv(oe_deputy_at_tf)
    r_deputy_final = SVector{3}(posvel_deputy_final[1:3])

    relative_pos_eci = r_deputy_final - r_chief_final
    relative_pos_rtn = eci_to_rtn(r_chief_final, v_chief_final, relative_pos_eci)

    println("\n--- 編隊形成完了時の状態 ---")
    println("■ 最終的な相対軌道要素 (ROE):")
    @printf "  δa (規格化軌道長半径差): %.3e\n" roe_aug_final_vec[1]
    @printf "  δλ (平均経度差):        %+.3f deg\n" rad2deg(roe_aug_final_vec[2])
    @printf "  δex (離心率ベクトルx):    %.3e\n" roe_aug_final_vec[3]
    @printf "  δey (離心率ベクトルy):    %.3e\n" roe_aug_final_vec[4]
    @printf "  δix (軌道傾斜角ベクトルx):  %.3e\n" roe_aug_final_vec[5]
    @printf "  δiy (軌道傾斜角ベクトルy):  %.3e\n" roe_aug_final_vec[6]
    
    println("\n■ 最終的な相対位置 (RTN座標系):")
    @printf "  R (動径):    %+.2f m\n" relative_pos_rtn[1]
    @printf "  T (進行):    %+.2f m\n" relative_pos_rtn[2]
    @printf "  N (面外):    %+.2f m\n" relative_pos_rtn[3]
    println("\t(注意：10軌道周期後の、ある瞬間の相対位置)")

    # --- 3. 摂動の比較 (このセクションを以下のように変更) ---
    
    # J2摂動の計算用関数
    function calculate_j2_accel(r_vec::SVector{3,Float64})::SVector{3,Float64}
        x,y,z=r_vec; r_sq=dot(r_vec,r_vec); r=sqrt(r_sq);
        term_common=-1.5*mu_earth*J2_coeff*R_E^2/(r^5); z_sq_r_sq=(z^2)/r_sq
        ax=term_common*x*(1.0-5.0*z_sq_r_sq); ay=term_common*y*(1.0-5.0*z_sq_r_sq); az=term_common*z*(3.0-5.0*z_sq_r_sq)
        return SVector(ax,ay,az)
    end
    j2_accel_chief = calculate_j2_accel(r_chief_final)
    j2_accel_deputy = calculate_j2_accel(r_deputy_final)
    
    # ECI座標系での相対J2摂動ベクトル
    relative_j2_accel_eci = j2_accel_deputy - j2_accel_chief
    
    # ★★★ ECIからRTN座標系へ変換 ★★★
    relative_j2_accel_rtn = eci_to_rtn(r_chief_final, v_chief_final, relative_j2_accel_eci)
    
    # 各成分の大きさを取得
    relative_j2_accel_norm = norm(relative_j2_accel_eci)
    relative_j2_accel_n_comp = relative_j2_accel_rtn[3] # N方向成分

    println("\n--- 摂動の比較 ---")
    @printf "相対J2摂動の総量 (ノルム):      %.3e m/s^2\n" relative_j2_accel_norm
    @printf "  └ 軌道面外(N)方向の成分:     %+.3e m/s^2\n" relative_j2_accel_n_comp

    # --- 4. 相対的な太陽輻射圧の推定 (変更なし) ---
    P_srp = 4.56e-6 
    mass = 100.0 
    C_r = 1.0 
    A = 1.0 
    delta_A_over_m = 0.1
    
    relative_srp_accel_mag_est = P_srp * C_r * delta_A_over_m

    @printf "相対太陽輻射圧の大きさ (推定値): %.3e m/s^2\n" relative_srp_accel_mag_est
    
    # # --- 5. 比較と考察 ---
    # if relative_j2_accel_norm > relative_srp_accel_mag_est
    #     ratio = relative_j2_accel_norm / relative_srp_accel_mag_est
    #     @printf "\n 相対J2摂動の総量は、相対太陽輻射圧の約 %.1f 倍の大きさ\n" ratio
    # else
    #     ratio = relative_srp_accel_mag_est / relative_j2_accel_norm
    #     @printf "\n 相対太陽輻射圧は、相対J2摂動の約 %.1f 倍の大きさ\n" ratio
    # end
end

# --- 実行 ---
function run_target_search()
    # 最適解の探索
    optimal_params = find_j2_invariant_maneuver(J2_AND_DRAG, RT_PLANE)
    
    # 最適解の結果を分析
    if optimal_params !== nothing
        analyze_optimal_result(optimal_params.angle, optimal_params.dv_mag)
    end
end

run_target_search()

# 検証テストを実行
run_state_reconstruction_test()