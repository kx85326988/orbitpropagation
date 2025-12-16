# ==============================================================================
# [ 衛星の分離前回転を利用した無推薬編隊飛行の統合設計・解析コード ]
#
# ## 目的 (Main Purpose)
#
# このプログラムは、低軌道環境におけるJ2摂動と差動抗力の影響下で、
# 目標とする相対軌道（編隊）を**無推薬（スラスタレス）**で構築するための設計手法を確立することを目的とする。
#
# 分離メカニズムとして、より現実的な**衛星の分離前回転**を仮定し、分離マヌーバのパラメータ（分離速度`Δv`、
# 分離方向`θ`）を、衛星の姿勢運動のパラメータ（角速度`ω`、回転位相`φ`）に直接結びつける。
# なお、本コードでの分離マヌーバは、軌道面に垂直な成分を持たない**軌道面内分離**に限定してモデル化している。
#
# 本コードは、最終的な編隊の精度、短中期的な衝突回避の安全性、そして燃料消費に相当する
# 分離速度`Δv`の最小化という、複数の相反する要求を同時に満たす、包括的な最適設計解
# （最適な`Δv`, `θ`, および分離を行う最適な軌道位相`M`）を導出するためのツール群を提供する。
#
# ## 主要な手法と参考文献 (Key Methodologies / References)
#
# 本コードの中核をなす軌道伝播計算は、以下の論文で提案された状態遷移マトリックス（STM）に基づいている。
# このSTMは、J2摂動と差動抗力を含む、近円軌道における相対運動を高精度かつ高速に予測するものである。
#
# - **参考文献:** A. W. Koenig, T. Guffanti, and S. D'Amico, "New State Transition Matrices for
#   Spacecraft Relative Motion in Perturbed Orbits," Journal of Guidance, Control, and
#   Dynamics, Vol. 40, No. 5, 2017, pp. 1749-1768.
#   (https://arc.aiaa.org/doi/10.2514/1.G002409)
#
# ## コードの機能と流れ (Functions and Workflow)
#
# 本プログラムは、以下の3つの主要な実行関数を提供する。
#
# ### 1. `run_target_search()` - 2次元最適マヌーバ探索
#
# - **機能:** 分離を行う軌道位相`M`を固定した上で、目標ROEとの誤差を最小化する
#   最適な分離マヌーバ（`Δv`, `θ`）を探索する。
# - **流れ:** 各マヌーバ候補に対し、上記の**参考文献に基づく状態遷移マトリックス（STM）**を用いて
#   最終的なROEを高速に予測する。予測された最終ROEと目標ROEとの差を、各要素の重要度に応じて
#   ペナルティを課す**「重み付き誤差二乗和」**としてコストを計算し、このコストが最小となる解を探索する。
#
# ### 2. `run_reachable_set_analysis()` - 設計限界とトレードオフの可視化
#
# - **機能:** `Δv`や`M`といった特定のパラメータを固定し、分離方向`θ`を0°～360°まで変化させた場合に、
#   物理的に到達可能な最終ROEの全集合（Reachable Set）を計算し、プロットする。
# - **用途:** 提案手法で形成可能な編隊の**物理的な限界（デザインスペース）**を可視化する。
#   軌道面内の形状（δex, δey）と面外のずれ（δiy）の間に存在する**トレードオフ関係**を解明する。
#
# ### 3. `run_global_optimization()` - 3次元包括的最適化
#
# - **機能:** `Δv`, `θ`, `M`の3変数を同時に変化させ、**衝突回避**や**最大`Δv`**といった
#   物理的な制約条件を満たす解の中から、最終的な編隊精度（コスト）が最も高い（低い）解を、
#   3次元のグリッドサーチにより探索する。
# - **用途:** 本研究の最終的な結論である、**「全ての制約を満たす、最も効率的な
#   包括的最適設計解（最適な`Δv`, `θ`, `M`）」**を導出する。
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
using Symbolics
using StatsPlots
using Distributions

# インタラクティブなバックエンドを指定
# gr() # GRバックエンドを使用する場合
plotlyjs() # PlotlyJSバックエンドを使用する場合

# --- 物理定数 ---
const mu_earth = 3.986004418e14
const J2_coeff = 1.08263e-3
const R_E = 6378137.0
const RHO_LEO = 4.71e-13           # 空気モデルspecific高度約525kmの平均的な大気密度 [kg/m^3]
const J2 = 1.08262668e-3
const BC_CHIEF = 0.02             # 主衛星の弾道係数 [m^2/kg]
const DELTA_B_INIT = 0.1          # 初期ΔBの例 (副衛星が10%大きい)

# --- 軌道補正制御関連パラメータ ---
const DELTA_B_MAX = 1.0    # 制御可能な最大の |(B_d - B_c) / B_c|
const DELTA_B_MIN = -0.5   # 制御可能な最小の |(B_d - B_c) / B_c|
const ATTITUDE_CHANGE_TIMECONSTANT_SEC = 60.0 # 姿勢変更の時定数 [s]
const CONTROL_GAIN = -50.0  # 軌道補正のための制御ゲイン(未使用)
const SIM_SEGMENT_ORBITS = 0.1 # シミュレーションを分割するセグメント長（軌道周期）

# --- 主衛星の初期軌道要素 ---
a_c_stm_init = 6903137.0
e_c_stm_init = 0.0022
i_c_stm_init = deg2rad(97.65)
Omega_c_stm_init = deg2rad(0.0)
omega_c_stm_init = deg2rad(270.0)
# omega_c_stm_init = deg2rad(180.0)
M_c_stm_init = deg2rad(180.0)
# omega_c_stm_init = deg2rad(0.0) #比較用
# M_c_stm_init = deg2rad(0.0) #比較用

# --- 編隊飛行関連パラメータ ---
const dr_lvlh_init = SVector(0.0, 0.0, 0.0)
const delta_a_dot_drag = -4.6e-11  # 空気モデルフリー[1/s]

const PROPAGATION_ORBITS = 10.0 # 評価を行う軌道周期数

# --- 目標軌道パラメータ ---
const TARGET_a_delta_e_norm = 500.0 # [m] (0.5km x 1km 楕円の短軸半径)
# (その他のターゲットボックス条件はコスト関数内で直接評価)

# --- 衝突リスク評価パラメータ ---
const SAFETY_THRESHOLD_M = 50.0 # 衝突判定距離 (m)
const NUM_SAMPLES = 500        # 試行回数 (動作確認用。本番は1000~10000推奨)
# 分離機構の誤差設定 (標準偏差)
const SIGMA_MAG = 0.05         # 速度の大きさ誤差 (5%)
const SIGMA_ANG = deg2rad(2.0) # 放出方向誤差 (2度)
# シミュレーション時間設定
const ORBIT_PERIOD_SEC = 5700.0 # 約95分
const SIM_DURATION_ORBITS = 5.0 # 戻ってくるのを見るため長めに (5周回)
const DT_SEC = 30.0             # 時間刻み (秒)

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
    # R軸 (衛星の半径方向) を定義
    r_c_hat=normalize(r_chief_eci)
    # N軸 (軌道面に垂直な方向) を定義
    h_c_vec=cross(r_chief_eci,v_chief_eci)
    h_c_hat_val=normalize(h_c_vec)
    if norm(h_c_vec)<1e-9; t_c_hat_temp=normalize(v_chief_eci); if abs(dot(r_c_hat,t_c_hat_temp))>1.0-1e-6; temp_axis=abs(t_c_hat_temp[1])<0.9 ? SVector(1.0,0,0) : SVector(0,1.0,0); h_c_hat_val=normalize(cross(t_c_hat_temp,temp_axis)); else; h_c_hat_val=normalize(cross(r_c_hat,t_c_hat_temp)); end; end
    # T軸 (衛星の進行方向) を定義
    t_c_hat_final=normalize(cross(h_c_hat_val,r_c_hat))
    # RTN座標系からECI座標系への変換行列
    dcm_lvlh_to_eci=hcat(r_c_hat,t_c_hat_final,h_c_hat_val)

    # 副衛星の相対位置ベクトルを、LVLH系からECI系に回転
    dr_eci=dcm_lvlh_to_eci*dr_lvlh
    # 主衛星の絶対位置に、回転させた相対位置を足す
    r_deputy_eci=r_chief_eci+dr_eci

    # まず、LVLH座標系の回転角速度ωを計算
    omega_lvlh_scalar=dot(h_c_vec,h_c_hat_val)/(norm(r_chief_eci)^2)
    omega_vector_lvlh_frame=SVector(0.0,0.0,omega_lvlh_scalar)
    # 次に、輸送定理の後半部分 (V_rel_rotating + ω × r_rel) を計算しECI座標系に回転
    dv_eci_relative=dcm_lvlh_to_eci*(dv_lvlh+cross(omega_vector_lvlh_frame,dr_lvlh))
    # 最後に、主衛星の速度に足し合わせ
    v_deputy_eci=v_chief_eci+dv_eci_relative
    return CartesianStateECI(r_deputy_eci,v_deputy_eci)
end

# 角度の差を正しく計算するヘルパー関数
function shortest_angle_diff(a1, a2)
    return mod(a2 - a1 + pi, 2*pi) - pi
end
# ROE (Koenigの準非特異的ROE) への変換
function orbital_elements_to_qns_roe_koenig(oe_c::OrbitalElementsClassical, oe_d::OrbitalElementsClassical)::QuasiNonsingularROEsKoenig
    # println("\n  --- δλ 計算デバッグ ---")
    # @printf "  [Chief]    M: %8.3f, ω: %8.3f, Ω: %8.3f\n" rad2deg(oe_c.M) rad2deg(oe_c.omega) rad2deg(oe_c.RAAN)
    # @printf "  [Deputy]   M: %8.3f, ω: %8.3f, Ω: %8.3f\n" rad2deg(oe_d.M) rad2deg(oe_d.omega) rad2deg(oe_d.RAAN)
    # flush(stdout)

    delta_a_norm_val=(oe_d.a-oe_c.a)/oe_c.a

    delta_M = shortest_angle_diff(oe_c.M, oe_d.M)
    delta_omega = shortest_angle_diff(oe_c.omega, oe_d.omega)
    delta_RAAN = shortest_angle_diff(oe_c.RAAN, oe_d.RAAN)

    delta_lambda_val = delta_M + delta_omega + delta_RAAN * cos(oe_c.i)
    
    delta_ex_val=oe_d.e*cos(oe_d.omega)-oe_c.e*cos(oe_c.omega); delta_ey_val=oe_d.e*sin(oe_d.omega)-oe_c.e*sin(oe_c.omega)
    delta_ix_val=oe_d.i-oe_c.i; delta_Omega_val=mod(oe_d.RAAN-oe_c.RAAN+pi,2*pi)-pi; delta_iy_val=delta_Omega_val*sin(oe_c.i)
    return QuasiNonsingularROEsKoenig(delta_a_norm_val,delta_lambda_val,delta_ex_val,delta_ey_val,delta_ix_val,delta_iy_val)
end
# # 逆変換: 最終ROEから副衛星の軌道要素
# function final_roe_to_deputy_oe(oe_chief_final::OrbitalElementsClassical, final_roes::SVector{7,Float64})::OrbitalElementsClassical
#     ac,ec,ic,Omegac,omegac,Mc=oe_chief_final.a,oe_chief_final.e,oe_chief_final.i,oe_chief_final.RAAN,oe_chief_final.omega,oe_chief_final.M
#     delta_a_norm_val=final_roes[1]; delta_lambda_val=final_roes[2]; delta_ex_val=final_roes[3]; delta_ey_val=final_roes[4]; delta_ix_val=final_roes[5]; delta_iy_val=final_roes[6]
#     ad=ac*(1.0+delta_a_norm_val); id=ic+delta_ix_val; Omegad=Omegac
#     if abs(sin(ic))>1e-7; Omegad=Omegac+delta_iy_val/sin(ic); end; Omegad=mod(Omegad,2*pi)
#     X=delta_ex_val+ec*cos(omegac); Y=delta_ey_val+ec*sin(omegac); ed=sqrt(X^2+Y^2); if ed<1e-10; ed=1e-10; end
#     omegad=0.0; if ed>1e-9; omegad=atan(Y,X); if omegad<0.0; omegad+=2*pi; end; end
#     Md=delta_lambda_val+(Mc+omegac+Omegac*cos(ic))-(omegad+Omegad*cos(ic)); Md=mod(Md,2*pi); if Md<0.0; Md+=2*pi; end
#     nd=sqrt(mu_earth/abs(ad)^3); f_true_d_val = SatelliteToolbox.mean_to_true_anomaly(ed, Md)
#     return OrbitalElementsClassical(ad,ed,id,Omegad,omegad,f_true_d_val,nd,Md)
# end
# ==============================================================================
# 修正版: final_roe_to_deputy_oe
# (引数 final_roes の型制約を SVector から AbstractVector に緩和)
# ==============================================================================
function final_roe_to_deputy_oe(oe_chief_final::OrbitalElementsClassical, final_roes::AbstractVector)::OrbitalElementsClassical
    ac,ec,ic,Omegac,omegac,Mc=oe_chief_final.a,oe_chief_final.e,oe_chief_final.i,oe_chief_final.RAAN,oe_chief_final.omega,oe_chief_final.M
    
    # Vector{Any}の場合もあるため、念のためFloat64にキャストして取り出す
    delta_a_norm_val = Float64(final_roes[1])
    delta_lambda_val = Float64(final_roes[2])
    delta_ex_val     = Float64(final_roes[3])
    delta_ey_val     = Float64(final_roes[4])
    delta_ix_val     = Float64(final_roes[5])
    delta_iy_val     = Float64(final_roes[6])
    
    ad=ac*(1.0+delta_a_norm_val); id=ic+delta_ix_val; Omegad=Omegac
    if abs(sin(ic))>1e-7; Omegad=Omegac+delta_iy_val/sin(ic); end; Omegad=mod(Omegad,2*pi)
    X=delta_ex_val+ec*cos(omegac); Y=delta_ey_val+ec*sin(omegac); ed=sqrt(X^2+Y^2); if ed<1e-10; ed=1e-10; end
    omegad=0.0; if ed>1e-9; omegad=atan(Y,X); if omegad<0.0; omegad+=2*pi; end; end
    Md=delta_lambda_val+(Mc+omegac+Omegac*cos(ic))-(omegad+Omegad*cos(ic)); Md=mod(Md,2*pi); if Md<0.0; Md+=2*pi; end
    nd=sqrt(mu_earth/abs(ad)^3); f_true_d_val = SatelliteToolbox.mean_to_true_anomaly(ed, Md)
    return OrbitalElementsClassical(ad,ed,id,Omegad,omegad,f_true_d_val,nd,Md)
end

# 数値計算用
# function get_secular_j2_rates_koenig(ac::Float64, ec::Float64, ic::Float64)::Tuple{Float64,Float64}
#     n_c=sqrt(mu_earth/ac^3); eta_c=sqrt(1.0-ec^2); if eta_c<1e-9; eta_c=1e-9; end
#     common_factor=(3.0/4.0)*J2_coeff*(R_E/ac)^2*n_c/(eta_c^4)
#     omega_dot=common_factor*(5.0*cos(ic)^2-1.0); Omega_dot=common_factor*(-2.0*cos(ic))
#     return omega_dot,Omega_dot
# end

# function get_J_qns_augmented_koenig(omega_c_val::Float64)::SMatrix{7,7,Float64}
#     J_aug=@MMatrix fill(0.0,7,7); J_aug[1,1]=1.0; J_aug[2,2]=1.0; cos_wc=cos(omega_c_val); sin_wc=sin(omega_c_val)
#     J_aug[3,3]=cos_wc; J_aug[3,4]=sin_wc; J_aug[4,3]=-sin_wc; J_aug[4,4]=cos_wc
#     J_aug[5,5]=1.0; J_aug[6,6]=1.0; J_aug[7,7]=1.0
#     return SMatrix(J_aug)
# end

# function get_J_qns_inv_augmented_koenig(omega_c_val::Float64)::SMatrix{7,7,Float64}
#     J_inv_aug=@MMatrix fill(0.0,7,7); J_inv_aug[1,1]=1.0; J_inv_aug[2,2]=1.0; cos_wc=cos(omega_c_val); sin_wc=sin(omega_c_val)
#     J_inv_aug[3,3]=cos_wc; J_inv_aug[3,4]=-sin_wc; J_inv_aug[4,3]=sin_wc; J_inv_aug[4,4]=cos_wc
#     J_inv_aug[5,5]=1.0; J_inv_aug[6,6]=1.0; J_inv_aug[7,7]=1.0
#     return SMatrix(J_inv_aug)
# end

# function get_A_prime_qns_augmented_koenig_selectable(ac_val::Float64, ec_val::Float64, ic_val::Float64, omegac_val::Float64, include_j2::Bool, include_drag_effects::Bool, drag_model_type::DragModelTypeForSTM, rho_val::Float64, Bc_val::Float64)::Tuple{SMatrix{7,7,Float64}, SMatrix{7,7,Float64}, SMatrix{7,7,Float64}}
#     A_kep_p=@MMatrix zeros(Float64,7,7); A_j2_p=@MMatrix zeros(Float64,7,7); A_drag_p=@MMatrix zeros(Float64,7,7)
#     n_c=sqrt(mu_earth/ac_val^3); A_kep_p[2,1]=-1.5*n_c
#     if include_j2
#         eta_c=sqrt(1.0-ec_val^2); if eta_c<1e-9; eta_c=1e-9; end
#         kappa_J2=(3.0/4.0)*J2_coeff*(R_E^2*sqrt(mu_earth))/(ac_val^(3.5)*eta_c^4)
#         E_f=1.0+eta_c; F_f=4.0+3.0*eta_c; G_f=1.0/eta_c^2
#         cos_i=cos(ic_val); sin_i=sin(ic_val)
#         P_g=3.0*cos_i^2-1.0; Q_g=5.0*cos_i^2-1.0; S_g=sin(2.0*ic_val); T_g=sin_i^2
#         # ex_c=ec_val*cos(omegac_val); ey_c=ec_val*sin(omegac_val)
#         A_j2_p[2,1]=-3.5*kappa_J2*E_f*P_g; A_j2_p[2,3]=kappa_J2*ec_val*F_f*G_f*P_g; A_j2_p[2,5]=-kappa_J2*F_f*S_g
#         A_j2_p[4,1]=-3.5*kappa_J2*ec_val*Q_g; A_j2_p[4,3]=4*kappa_J2*ec_val^2*G_f*Q_g; A_j2_p[4,5]=-5*kappa_J2*ec_val*S_g
#         A_j2_p[6,1]=3.5*kappa_J2*S_g; A_j2_p[6,3]=-4*kappa_J2*ec_val*G_f*S_g; A_j2_p[6,5]=2*kappa_J2*T_g
#     end
#     if include_drag_effects
#         if drag_model_type==DENSITY_MODEL_FREE; A_drag_p[1,7]=1.0; A_drag_p[3,7]=1-ec_val;
#         elseif drag_model_type==DENSITY_MODEL_SPECIFIC
#             # このモデルでは、aug_param (状態ベクトルの7番目の要素) が
#             # 無次元化された差動弾道係数 ΔB = (B_d - B_c) / B_c を表すと仮定
            
#             # 差動抗力が主にδaとδeに与える永年的な影響をモデル化
            
#             # 物理的な係数を計算
#             K_drag = -rho_val * n_c * ac_val * Bc_val

#             # d(δa_norm)/dt の δB に対する係数
#             A_drag_p[1,7] = K_drag
            
#             # d(δex')/dt の δB に対する係数 (近円軌道近似)
#             # この項は、抗力によって軌道がより円に近づく効果（円軌道化）を表す
#             A_drag_p[3,7] = K_drag
#         end
#     end
#     return SMatrix(A_kep_p), SMatrix(A_j2_p), SMatrix(A_drag_p)
# end

# function get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_prime::SMatrix{7,7,Float64}, A_j2_prime::SMatrix{7,7,Float64}, A_drag_prime::SMatrix{7,7,Float64}, t_prop::Float64, ec_val_for_drag_effect::Float64, include_drag_effects::Bool, drag_model_type::DragModelTypeForSTM)::SMatrix{7,7,Float64}
#     A_kep_J2_prime=A_kep_prime+A_j2_prime
#     # `include_drag_effects`がtrueであれば、モデルの種類を問わず、
#     # J2摂動と差動抗力のカップリングを考慮したSTMを計算する
#     if include_drag_effects
#         # 差動抗力のみによるSTM（Φ_drag'）
#         # exp(A*t) の1次近似: I + A*t
#         Phi_drag_prime = SMatrix{7,7,Float64}(I) + A_drag_prime * t_prop

#         # Φ_drag' の時間積分（∫Φ_drag' dt）
#         # I*t + 0.5*A*t^2
#         Integral_Phi_drag_prime = SMatrix{7,7,Float64}(I) * t_prop + A_drag_prime * (t_prop^2 / 2.0)
        
#         # 論文 式(64)および(67)に基づき、J2と抗力のカップリング項を計算し、最終的なSTM'を返す
#         # Φ' = Φ_drag' + A_kep_j2' * ∫Φ_drag' dt
#         return Phi_drag_prime + A_kep_J2_prime * Integral_Phi_drag_prime
    
#     # 抗力の影響がない場合は、ケプラーとJ2の線形STMを返す
#     else
#         return SMatrix{7,7,Float64}(I) + A_kep_J2_prime * t_prop
#     end
# end

# 感度解析用(型指定削除)
# 型指定 (::Float64) を削除し、SMatrixの型も指定しないか、あるいは型パラメータ T を使う
function get_secular_j2_rates_koenig(ac, ec, ic)
    n_c = sqrt(mu_earth / ac^3)
    # シンボリック変数の場合、条件分岐はエラーになることがあるため、eが数値の時のみチェックする等の工夫か、
    # 単に数式として分母に eta^4 を置く（eta=0にならない前提）
    eta_c = sqrt(1.0 - ec^2) 
    common_factor = (3.0/4.0) * J2_coeff * (R_E/ac)^2 * n_c / (eta_c^4)
    omega_dot = common_factor * (5.0 * cos(ic)^2 - 1.0)
    Omega_dot = common_factor * (-2.0 * cos(ic))
    return omega_dot, Omega_dot
end

function get_J_qns_augmented_koenig(omega_c_val)
    # 配列要素に数式が入るため、型を特定しない行列、またはAny型の配列を使う
    # Symbolicsを使う場合、Num型の行列を作る必要がある
    J = Matrix{Any}(undef, 7, 7)
    fill!(J, 0)
    J[1,1]=1; J[2,2]=1; J[5,5]=1; J[6,6]=1; J[7,7]=1
    cω = cos(omega_c_val); sω = sin(omega_c_val)
    J[3,3] = cω; J[3,4] = sω
    J[4,3] = -sω; J[4,4] = cω
    return J
end

function get_J_qns_inv_augmented_koenig(omega_c_val)
    J_inv = Matrix{Any}(undef, 7, 7)
    fill!(J_inv, 0)
    J_inv[1,1]=1; J_inv[2,2]=1; J_inv[5,5]=1; J_inv[6,6]=1; J_inv[7,7]=1
    cω = cos(omega_c_val); sω = sin(omega_c_val)
    J_inv[3,3] = cω; J_inv[3,4] = -sω
    J_inv[4,3] = sω; J_inv[4,4] = cω
    return J_inv
end

# 引数の型指定を削除
function get_A_prime_qns_augmented_koenig_selectable(ac_val, ec_val, ic_val, omegac_val, include_j2, include_drag_effects, drag_model_type, rho_val, Bc_val)
    # シンボリック演算用に Any 型の行列を初期化
    A_kep_p = Matrix{Any}(undef, 7, 7); fill!(A_kep_p, 0)
    A_j2_p  = Matrix{Any}(undef, 7, 7); fill!(A_j2_p, 0)
    A_drag_p = Matrix{Any}(undef, 7, 7); fill!(A_drag_p, 0)

    n_c = sqrt(mu_earth / ac_val^3)
    A_kep_p[2,1] = -1.5 * n_c
    
    if include_j2
        eta_c = sqrt(1.0 - ec_val^2)
        kappa_J2 = (3.0/4.0) * J2_coeff * (R_E^2 * sqrt(mu_earth)) / (ac_val^(3.5) * eta_c^4)
        E_f = 1.0 + eta_c; F_f = 4.0 + 3.0 * eta_c; G_f = 1.0 / eta_c^2
        cos_i = cos(ic_val); sin_i = sin(ic_val)
        P_g = 3.0 * cos_i^2 - 1.0; Q_g = 5.0 * cos_i^2 - 1.0; S_g = sin(2.0 * ic_val); T_g = sin_i^2
        
        # ex_c, ey_c はここでは使わず、論文通りの係数定義を使用
        A_j2_p[2,1] = -3.5 * kappa_J2 * E_f * P_g
        A_j2_p[2,3] = kappa_J2 * ec_val * F_f * G_f * P_g
        A_j2_p[2,5] = -kappa_J2 * F_f * S_g
        A_j2_p[4,1] = -3.5 * kappa_J2 * ec_val * Q_g
        A_j2_p[4,3] = 4 * kappa_J2 * ec_val^2 * G_f * Q_g
        A_j2_p[4,5] = -5 * kappa_J2 * ec_val * S_g
        A_j2_p[6,1] = 3.5 * kappa_J2 * S_g
        A_j2_p[6,3] = -4 * kappa_J2 * ec_val * G_f * S_g
        A_j2_p[6,5] = 2 * kappa_J2 * T_g
    end

    if include_drag_effects
        if drag_model_type == DENSITY_MODEL_FREE
            A_drag_p[1,7] = 1.0
            A_drag_p[3,7] = 1 - ec_val
        elseif drag_model_type == DENSITY_MODEL_SPECIFIC
            K_drag = -rho_val * n_c * ac_val * Bc_val
            A_drag_p[1,7] = K_drag
            A_drag_p[3,7] = K_drag
        end
    end
    return A_kep_p, A_j2_p, A_drag_p
end

function get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_prime, A_j2_prime, A_drag_prime, t_prop, ec_val_for_drag_effect, include_drag_effects, drag_model_type)
    A_kep_J2_prime = A_kep_prime + A_j2_prime
    
    # 単位行列 I も Any 型で用意する
    I_mat = Matrix{Any}(undef, 7, 7)
    fill!(I_mat, 0)
    for i in 1:7
        I_mat[i, i] = 1
    end

    if include_drag_effects
        Phi_drag_prime = I_mat + A_drag_prime * t_prop
        Integral_Phi_drag_prime = I_mat * t_prop + A_drag_prime * (t_prop^2 / 2.0)
        CouplingTerm = A_kep_J2_prime * Integral_Phi_drag_prime
        return Phi_drag_prime + CouplingTerm
    else
        return I_mat + A_kep_J2_prime * t_prop
    end
end


# ECI座標系の相対ベクトルを、主衛星中心のRTN座標系に変換する
function eci_to_rtn(r_chief_eci::SVector{3,Float64}, v_chief_eci::SVector{3,Float64}, vec_eci::SVector{3,Float64})::SVector{3,Float64}
    r_hat = normalize(r_chief_eci)
    h_vec = cross(r_chief_eci, v_chief_eci)
    n_hat = normalize(h_vec)
    t_hat = cross(n_hat, r_hat)
    
    dcm_eci_to_rtn = transpose(hcat(r_hat, t_hat, n_hat))
    
    return dcm_eci_to_rtn * vec_eci
end

# 任意軸周りの回転行列を計算 (ロドリゲスの回転公式)
function rotation_matrix_around_axis(axis::SVector{3,Float64}, angle_rad::Float64)
    c = cos(angle_rad); s = sin(angle_rad); C = 1 - c
    x, y, z = axis[1], axis[2], axis[3]
    return @SMatrix [
        x*x*C + c    x*y*C - z*s  x*z*C + y*s;
        y*x*C + z*s  y*y*C + c    y*z*C - x*s;
        z*x*C - y*s  z*y*C + x*s  z*z*C + c
    ]
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

function find_j2_invariant_maneuver(perturbation_setting::PerturbationType, separation_plane_setting::SeparationPlane, drag_model_setting::DragModelTypeForSTM, enable_drag_correction::Bool)
    println("\n\n--- 目標ROEを達成するための最適分離マヌーバの探索を開始 ---")
    println("Pert: $perturbation_setting, Plane: $separation_plane_setting, DragM: $drag_model_setting, Propagation: $PROPAGATION_ORBITS orbits")
    println("軌道補正制御(Drag Correction): $(enable_drag_correction ? "有効" : "無効")")

    #  1. 物理的なミッション要求から目標ROEターゲットを定義 

    # --- 目標とする物理的な軌道形状 ---
    TARGET_DELTA_E_NORM_METERS = 500.0 # [m] 相対軌道の短軸半径
    TARGET_Z_MAX_METERS      = 1.0  # [m] 許容される最大軌道面外ずれ

    # --- 物理要求をROEターゲットに変換 ---
    roe_target_a_norm = 0.0
    roe_target_lambda = 0.0

    # a * δe = 500 [m] より、δe を計算
    roe_target_ex     = TARGET_DELTA_E_NORM_METERS / a_c_stm_init
    roe_target_ey     = 0.0

    # δz_max ≈ a * |δiy| より、δiy を計算
    # δix は面内分離なのでゼロを目標とする。
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
    # 各ROE要素の重要度に応じた重み付け行列を定義
    W = Diagonal(SVector{7,Float64}(
        1.0e6,   # δa_norm の重み (エネルギー差)
        1.0,     # δlambda の重み (位相差)
        1.0e3,  # δex の重み (軌道形状)
        1.0e3,  # δey の重み (軌道形状)
        1.0e3,  # δix の重み (面外ずれ)
        1.0e3,  # δiy の重み (面外ずれ)
        0.0      # 拡張パラメータ(delta_a_dot_drag)はコストに含めない
    ))
    println("物理要求に基づき、以下の目標ROEターゲットを設定。")
    @printf " - 目標δex: %.3e (a*δe = %.1f m)\n" roe_target_vec[3] TARGET_DELTA_E_NORM_METERS
    @printf " - 目標δiy: %.3e (max Z = %.1f m)\n" roe_target_vec[6] TARGET_Z_MAX_METERS
    println("目標ROE全体: ", roe_target_vec)

    include_j2_active = (perturbation_setting == J2_ONLY || perturbation_setting == J2_AND_DRAG)
    # include_drag_active_stm = (perturbation_setting == DRAG_ONLY || perturbation_setting == J2_AND_DRAG)
    
    oe_chief_initial_for_sv = OrbitalElementsClassical(a_c_stm_init,e_c_stm_init,i_c_stm_init,Omega_c_stm_init,omega_c_stm_init,0.0,0.0,M_c_stm_init)
    posvel_chief_initial_eci_vec = orbital_elements_to_sv(oe_chief_initial_for_sv)
    r_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[1:3]); v_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[4:6])
    oe_chief_eval = sv_to_orbital_elements(CartesianStateECI(r_chief_init_eci, v_chief_init_eci))
    
    optimal_cost = Inf
    optimal_params = (angle=0.0, dv_mag=0.0)
    
    tf_val = PROPAGATION_ORBITS * 2.0 * pi * sqrt(oe_chief_eval.a^3 / mu_earth)
    println("伝播時間: $tf_val 秒 ($(PROPAGATION_ORBITS)軌道周期)")
    
    cost_data = Float64[] # プロット用のコストデータを保存

    for dv_mag in 0.01:0.01:0.5
        for angle_val in 0.0:10.0:350.0
            angle_rad_val = deg2rad(angle_val)
            dv_R_val_comp=dv_mag*cos(angle_rad_val); dv_T_val_comp=dv_mag*sin(angle_rad_val)
            dv_lvlh_vec = SVector(dv_R_val_comp, dv_T_val_comp, 0.0)

            state_deputy_init_eci=cw_to_eci_deputy_state(r_chief_init_eci,v_chief_init_eci,dr_lvlh_init,dv_lvlh_vec)
            oe_dep_init = sv_to_orbital_elements(CartesianStateECI(state_deputy_init_eci.r_vec, state_deputy_init_eci.v_vec))
            qns_roes_init=orbital_elements_to_qns_roe_koenig(oe_chief_eval,oe_dep_init)

            aug_param = 0.0
            if drag_model_setting == DENSITY_MODEL_FREE
                aug_param = delta_a_dot_drag
            elseif drag_model_setting == DENSITY_MODEL_SPECIFIC
                aug_param = DELTA_B_INIT
            end
            roe_aug_init_vec = SVector(
                qns_roes_init.delta_a_norm, qns_roes_init.delta_lambda, 
                qns_roes_init.delta_ex, qns_roes_init.delta_ey, 
                qns_roes_init.delta_ix, qns_roes_init.delta_iy, 
                aug_param
            )
            
            total_cost = 0.0
            #  制御のON/OFFに応じて処理を分岐
            if enable_drag_correction
                # 軌道補正ありシミュレーション
                total_cost = run_drag_correction_simulation(roe_aug_init_vec, oe_chief_eval, roe_target_vec, W)
            else
                include_drag_active_stm = (perturbation_setting == DRAG_ONLY || perturbation_setting == J2_AND_DRAG)
                        
                omega_c_ti=oe_chief_eval.omega; omega_dot_j2,Omega_dot_j2=get_secular_j2_rates_koenig(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i)
                oe_chief_at_tf=OrbitalElementsClassical(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,mod(oe_chief_eval.RAAN+Omega_dot_j2*tf_val,2*pi),mod(oe_chief_eval.omega+omega_dot_j2*tf_val,2*pi),0.0,oe_chief_eval.n,mod(oe_chief_eval.M+oe_chief_eval.n*tf_val,2*pi))
                # oe_chief_at_tf=OrbitalElementsClassical(oe_chief_at_tf.a,oe_chief_at_tf.e,oe_chief_at_tf.i,oe_chief_at_tf.RAAN,oe_chief_at_tf.omega,SatelliteToolbox.mean_to_true_anomaly(oe_chief_at_tf.e,oe_chief_at_tf.M),oe_chief_at_tf.n,oe_chief_at_tf.M)
            
                omega_c_tf_val=oe_chief_at_tf.omega; J_ti=get_J_qns_augmented_koenig(omega_c_ti); J_tf_inv=get_J_qns_inv_augmented_koenig(omega_c_tf_val); roe_prime_init=J_ti*roe_aug_init_vec
                A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,omega_c_ti,include_j2_active,include_drag_active_stm,drag_model_setting, RHO_LEO, BC_CHIEF)
                STM_prime=get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_p,A_j2_p,A_drag_p,tf_val,oe_chief_eval.e,include_drag_active_stm,drag_model_setting)
                roe_prime_final=STM_prime*roe_prime_init; roe_aug_final_vec=J_tf_inv*roe_prime_final
            
                # コスト関数定義 (目標ROEとの誤差の重み付き二乗和) 
                # 誤差ベクトルを計算
                error_vec = roe_aug_final_vec - roe_target_vec

                total_cost = dot(error_vec, W * error_vec)
            end

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

    println("\n--- 最適解の最終ROEを計算 ---")
    
    # 最適パラメータで初期ROEを再計算
    angle_rad_val = deg2rad(optimal_params.angle)
    dv_R_val_comp = optimal_params.dv_mag * cos(angle_rad_val)
    dv_T_val_comp = optimal_params.dv_mag * sin(angle_rad_val)
    dv_lvlh_vec = SVector(dv_R_val_comp, dv_T_val_comp, 0.0)
    state_deputy_init_eci = cw_to_eci_deputy_state(r_chief_init_eci, v_chief_init_eci, dr_lvlh_init, dv_lvlh_vec)
    oe_dep_init = sv_to_orbital_elements(CartesianStateECI(state_deputy_init_eci.r_vec, state_deputy_init_eci.v_vec))
    qns_roes_init = orbital_elements_to_qns_roe_koenig(oe_chief_eval, oe_dep_init)
    
    aug_param = (drag_model_setting == DENSITY_MODEL_SPECIFIC) ? DELTA_B_INIT : delta_a_dot_drag
    optimal_initial_roe = SVector(
        qns_roes_init.delta_a_norm, qns_roes_init.delta_lambda, qns_roes_init.delta_ex,
        qns_roes_init.delta_ey, qns_roes_init.delta_ix, qns_roes_init.delta_iy, aug_param
    )

    if enable_drag_correction
        # 軌道補正ありの場合：詳細ログモードで再実行
        run_drag_correction_simulation(optimal_initial_roe, oe_chief_eval, roe_target_vec, W, true)
    else
        # 軌道補正なしの場合：単純な伝播計算で最終ROEを表示
        println("--- [軌道補正なし（撃ちっぱなし）の最終ROE] ---")
        # 伝播開始・終了時の主衛星の状態を計算
        tf_val = PROPAGATION_ORBITS * 2.0 * pi * sqrt(oe_chief_eval.a^3 / mu_earth)
        omega_c_ti = oe_chief_eval.omega
        omega_dot_j2, Omega_dot_j2 = get_secular_j2_rates_koenig(oe_chief_eval.a, oe_chief_eval.e, oe_chief_eval.i)
        oe_chief_at_tf = OrbitalElementsClassical(
            oe_chief_eval.a, oe_chief_eval.e, oe_chief_eval.i,
            mod(oe_chief_eval.RAAN + Omega_dot_j2 * tf_val, 2*pi),
            mod(oe_chief_eval.omega + omega_dot_j2 * tf_val, 2*pi),
            0.0, oe_chief_eval.n, mod(oe_chief_eval.M + oe_chief_eval.n * tf_val, 2*pi)
        )
        omega_c_tf_val = oe_chief_at_tf.omega

        # 変換行列 J(t_i) と J_inv(t_f) を計算
        J_ti = get_J_qns_augmented_koenig(omega_c_ti)
        J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf_val)
        roe_prime_init = J_ti * optimal_initial_roe
        A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,oe_chief_eval.omega,true,true,drag_model_setting, RHO_LEO, BC_CHIEF)
        STM_prime = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_p,A_j2_p,A_drag_p,tf_val,oe_chief_eval.e,true,drag_model_setting)
        
        roe_prime_final = STM_prime * roe_prime_init
        final_roe_no_correction = J_tf_inv * roe_prime_final
        
        println("  最終ROEベクトル: ")
        @printf "    δa_norm: %.3e\n" final_roe_no_correction[1]
        @printf "    δλ:     %.3e\n" final_roe_no_correction[2]
        @printf "    δex:    %.3e\n" final_roe_no_correction[3]
        @printf "    δey:    %.3e\n" final_roe_no_correction[4]
        @printf "    δix:    %.3e\n" final_roe_no_correction[5]
        @printf "    δiy:    %.3e\n" final_roe_no_correction[6]
    end
    return optimal_params, roe_target_vec
end

# # ==============================================================================
# # 差動抗力による軌道補正シミュレーション関数
# # ==============================================================================
# function run_drag_correction_simulation(
#     initial_roe_vec::SVector{7,Float64}, 
#     chief_oe_initial::OrbitalElementsClassical, 
#     target_roe_vec::SVector{7,Float64},
#     W::Diagonal,
#     is_debug_run::Bool = false # 最適解を見つけた後の最終確認用フラグ
#     )::Float64
    
#     num_segments = Int(floor(PROPAGATION_ORBITS / SIM_SEGMENT_ORBITS))
#     segment_time = SIM_SEGMENT_ORBITS * 2.0 * pi * sqrt(chief_oe_initial.a^3 / mu_earth)
    
#     current_roe = initial_roe_vec
#     current_chief_oe = chief_oe_initial
    
#     # 現在のδBを保持する変数を初期化
#     current_delta_B = initial_roe_vec[7] # 初期分離時のδBから開始

#     # 1次遅れフィルタの係数を計算
#     alpha = 1.0 - exp(-segment_time / ATTITUDE_CHANGE_TIMECONSTANT_SEC)
    
#     for i in 1:num_segments

#         #残り時間を使って、無制御の場合の最終状態を予測
#         time_to_go = (PROPAGATION_ORBITS - (i-1)*SIM_SEGMENT_ORBITS) * 2.0 * pi * sqrt(current_chief_oe.a^3 / mu_earth)
        
#         # 予測時にはδB=0と仮定
#         roe_for_prediction = SVector(
#             current_roe[1], current_roe[2], current_roe[3], current_roe[4],
#             current_roe[5], current_roe[6], 0.0 # 無制御なのでδB=0
#         )

#         # 予測用のSTMを計算（抗力の影響はゼロとして計算）
#         omega_c_ti_pred = current_chief_oe.omega
#         omega_dot_j2_pred, Omega_dot_j2_pred = get_secular_j2_rates_koenig(current_chief_oe.a, current_chief_oe.e, current_chief_oe.i)
#         oe_chief_at_tf_pred = OrbitalElementsClassical(current_chief_oe.a,current_chief_oe.e,current_chief_oe.i,mod(current_chief_oe.RAAN+Omega_dot_j2_pred*time_to_go,2*pi),mod(current_chief_oe.omega+omega_dot_j2_pred*time_to_go,2*pi),0.0,current_chief_oe.n,mod(current_chief_oe.M+current_chief_oe.n*time_to_go,2*pi))
        
#         omega_c_tf_val_pred = oe_chief_at_tf_pred.omega
#         J_ti_pred = get_J_qns_augmented_koenig(omega_c_ti_pred)
#         J_tf_inv_pred = get_J_qns_inv_augmented_koenig(omega_c_tf_val_pred)
        
#         # 無制御なので、A_drag_pはゼロ行列を使用
#         A_kep_p_pred, A_j2_p_pred, _ = get_A_prime_qns_augmented_koenig_selectable(current_chief_oe.a,current_chief_oe.e,current_chief_oe.i,omega_c_ti_pred,true,false,NO_DRAG, 0.0, 0.0)
#         STM_prime_pred = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_p_pred,A_j2_p_pred,SMatrix{7,7,Float64}(zeros(7,7)),time_to_go,current_chief_oe.e,false,NO_DRAG)
        
#         # 現在の状態（current_roe）を無制御で最後まで伝播させて、最終状態を予測
#         roe_prime_init_pred = J_ti_pred * roe_for_prediction
#         roe_prime_final_pred = STM_prime_pred * roe_prime_init_pred
#         predicted_final_roe = J_tf_inv_pred * roe_prime_final_pred
        
#         # 制御指令値の決定（物理量ベース）
#         predicted_final_error_da_norm = predicted_final_roe[1] - target_roe_vec[1]
        
#         # 物理的な誤差（単位:m）で指令値を計算
#         # ゲインのスケールも調整（物理量に合わせる）
#         PHYSICAL_CONTROL_GAIN = 1.0 
#         delta_B_command = clamp(PHYSICAL_CONTROL_GAIN * (predicted_final_error_da_norm * current_chief_oe.a), DELTA_B_MIN, DELTA_B_MAX)

#         # 決定したδBを使って、1セグメント分だけ「実際に」伝播させる
#         roe_aug_init_segment = SVector(
#             current_roe[1], current_roe[2], current_roe[3], current_roe[4],
#             current_roe[5], current_roe[6],
#             current_delta_B
#         )
        
#         omega_c_ti_actual=current_chief_oe.omega; omega_dot_j2_actual,Omega_dot_j2_actual=get_secular_j2_rates_koenig(current_chief_oe.a,current_chief_oe.e,current_chief_oe.i)
#         oe_chief_at_tf_segment=OrbitalElementsClassical(current_chief_oe.a,current_chief_oe.e,current_chief_oe.i,mod(current_chief_oe.RAAN+Omega_dot_j2_actual*segment_time,2*pi),mod(current_chief_oe.omega+omega_dot_j2_actual*segment_time,2*pi),0.0,current_chief_oe.n,mod(current_chief_oe.M+current_chief_oe.n*segment_time,2*pi))
        
#         omega_c_tf_val_segment=oe_chief_at_tf_segment.omega; J_ti_actual=get_J_qns_augmented_koenig(omega_c_ti_actual); J_tf_inv_actual=get_J_qns_inv_augmented_koenig(omega_c_tf_val_segment); roe_prime_init_actual=J_ti_actual*roe_aug_init_segment
        
#         A_kep_p_actual, A_j2_p_actual, A_drag_p_actual = get_A_prime_qns_augmented_koenig_selectable(current_chief_oe.a,current_chief_oe.e,current_chief_oe.i,omega_c_ti_actual,true,true,DENSITY_MODEL_SPECIFIC, RHO_LEO, BC_CHIEF)
#         STM_prime_actual=get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_p_actual,A_j2_p_actual,A_drag_p_actual,segment_time,current_chief_oe.e,true,DENSITY_MODEL_SPECIFIC)
        
#         roe_prime_final_actual=STM_prime_actual*roe_prime_init_actual; roe_aug_final_actual=J_tf_inv_actual*roe_prime_final_actual

#         # if is_debug_run
#         #     println("\n" * "-"^30 * " セグメント $i " * "-"^30)
#         #     @printf "  [開始時] δa: %8.2fm (norm: %.3e) | current_δB: %.4f\n" (current_roe[1]*current_chief_oe.a) current_roe[1] current_delta_B
#         #     @printf "  [予測]   最終誤差: %8.2fm -> δB指令: %.4f\n" (predicted_final_error_da_norm*current_chief_oe.a) delta_B_command
#         #     @printf "  [実行]   STM'[1,7] = %.3e | STM'[2,7] = %.3e\n" STM_prime_actual[1,7] STM_prime_actual[2,7]
#         #     @printf "  [結果]   δa: %8.2fm (norm: %.3e)\n" (roe_aug_final_actual[1]*current_chief_oe.a) roe_aug_final_actual[1]
#         # end
  
#         # 状態の更新
#         current_roe = roe_aug_final_actual
#         current_chief_oe = oe_chief_at_tf_segment

#         # 次のステップのδBを、1次遅れモデルで更新
#         current_delta_B = alpha * delta_B_command + (1.0 - alpha) * current_delta_B
#     end
    
#     # 最終的なコストを計算して返す
#     final_cost = dot(current_roe - target_roe_vec, W * (current_roe - target_roe_vec))
#     if is_debug_run
#         println("\n" * "─"^70)
#         @printf "  最終コスト: %.4e\n" final_cost
#         println("  最終ROEベクトル: ")
#         @printf "    δa_norm: %.3e\n" current_roe[1]
#         @printf "    δλ:     %.3e\n" current_roe[2]
#         @printf "    δex:    %.3e\n" current_roe[3]
#         @printf "    δey:    %.3e\n" current_roe[4]
#         @printf "    δix:    %.3e\n" current_roe[5]
#         @printf "    δiy:    %.3e\n" current_roe[6]
#         println("─"^70)
#     end
#     return final_cost
# end
# ==============================================================================
# 差動抗力による軌道補正シミュレーション関数 (修正版: 符号修正)
# ==============================================================================
function run_drag_correction_simulation(
    initial_roe_vec::SVector{7,Float64}, 
    chief_oe_initial::OrbitalElementsClassical, 
    target_roe_vec::SVector{7,Float64},
    W::Diagonal,
    is_debug_run::Bool = false,
    duration_orbits::Float64 = PROPAGATION_ORBITS # デフォルトは元の定数
    )::Float64

    initial_da_meters = initial_roe_vec[1] * chief_oe_initial.a

    if is_debug_run
        println("\n" * "─"^70)
        println("★★★ 軌道補正シミュレーション詳細ログ（符号修正版） ★★★")
        println("─"^70)
        println("  更新周期: $(SIM_SEGMENT_ORBITS) 軌道周期 (約$(round(SIM_SEGMENT_ORBITS*93, digits=1))分)")
    end
    
    num_segments = Int(floor(duration_orbits / SIM_SEGMENT_ORBITS))      # 変更後
    segment_time = SIM_SEGMENT_ORBITS * 2.0 * pi * sqrt(chief_oe_initial.a^3 / mu_earth)
    
    current_roe = initial_roe_vec
    current_chief_oe = chief_oe_initial
    current_delta_B = initial_roe_vec[7]
    alpha = 1.0 - exp(-segment_time / ATTITUDE_CHANGE_TIMECONSTANT_SEC)
    
    time_history = Float64[]
    da_history = Float64[]
    db_history = Float64[]
    
    push!(time_history, 0.0)
    push!(da_history, current_roe[1] * current_chief_oe.a)
    push!(db_history, current_delta_B)

    for i in 1:num_segments
        time_to_go = (PROPAGATION_ORBITS - (i-1)*SIM_SEGMENT_ORBITS) * 2.0 * pi * sqrt(current_chief_oe.a^3 / mu_earth)
        omega_c_ti_pred = current_chief_oe.omega; omega_dot_j2_pred, _ = get_secular_j2_rates_koenig(current_chief_oe.a, current_chief_oe.e, current_chief_oe.i)
        oe_chief_at_tf_pred = OrbitalElementsClassical(current_chief_oe.a,current_chief_oe.e,current_chief_oe.i,0.0,mod(current_chief_oe.omega+omega_dot_j2_pred*time_to_go,2*pi),0.0,current_chief_oe.n,0.0)
        
        A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(current_chief_oe.a,current_chief_oe.e,current_chief_oe.i,current_chief_oe.omega,true,true,DENSITY_MODEL_SPECIFIC, RHO_LEO, BC_CHIEF)
        K_drag_val = A_drag[1,7] # これは通常マイナスの値 (密度係数)
        
        # 簡易予測: 何もしないとどうなるか
        # (現在のδBではなく、中立状態(δB=0)で進んだ場合との差分を見るべきだが、
        #  ここでは単純なP制御として「現在の偏差」を見る)
        current_da_norm = current_roe[1]
        error_da_norm = current_da_norm - target_roe_vec[1]
        error_da_meters = error_da_norm * current_chief_oe.a
        
        # ★★★ 符号修正 ★★★
        # δa が高い(プラス) -> 下げたい -> 抗力(δB)を増やす
        # δa が低い(マイナス) -> 上げたい(減りを抑えたい) -> 抗力(δB)を減らす
        # K_drag_val はマイナスの値なので、
        # d(δa)/dt = K * δB
        # 減らしたいなら d(δa)/dt < 0 にしたい -> K(-) * δB(+) -> δBはプラス
        # つまり、誤差(error)がプラスなら、δBもプラスにしたい。
        
        PHYSICAL_CONTROL_GAIN = 1.5 # ゲイン調整 (大きくする)
        
        # 以前はマイナスを掛けていたが、プラスにする
        delta_B_req = error_da_meters * PHYSICAL_CONTROL_GAIN
        
        delta_B_command = clamp(delta_B_req, DELTA_B_MIN, DELTA_B_MAX)

        # --- 実伝播 ---
        roe_aug_init = SVector(current_roe[1], current_roe[2], current_roe[3], current_roe[4], current_roe[5], current_roe[6], current_delta_B)
        omega_dot, Omega_dot = get_secular_j2_rates_koenig(current_chief_oe.a, current_chief_oe.e, current_chief_oe.i)
        omega_c_tf = current_chief_oe.omega + omega_dot * segment_time
        J_t0 = get_J_qns_augmented_koenig(current_chief_oe.omega)
        J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)
        STM = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, segment_time, current_chief_oe.e, true, DENSITY_MODEL_SPECIFIC)
        roe_aug_final = J_tf_inv * STM * J_t0 * roe_aug_init
        
        current_roe = roe_aug_final
        new_M = mod(current_chief_oe.M + current_chief_oe.n * segment_time, 2*pi)
        new_omega = mod(current_chief_oe.omega + omega_dot * segment_time, 2*pi)
        new_RAAN = mod(current_chief_oe.RAAN + Omega_dot * segment_time, 2*pi)
        current_chief_oe = OrbitalElementsClassical(current_chief_oe.a, current_chief_oe.e, current_chief_oe.i, new_RAAN, new_omega, 0.0, current_chief_oe.n, new_M)
        current_delta_B = alpha * delta_B_command + (1.0 - alpha) * current_delta_B
        
        push!(time_history, i * segment_time)
        push!(da_history, current_roe[1] * current_chief_oe.a)
        push!(db_history, current_delta_B)
    end
    
    final_cost = dot(current_roe - target_roe_vec, W * (current_roe - target_roe_vec))
    
    if is_debug_run
        final_da_meters = current_roe[1] * current_chief_oe.a
        println("\n  [結果] 初期δa: $(initial_da_meters) m -> 最終δa: $(final_da_meters) m")
        
        default(dpi=300, guidefontsize=12, tickfontsize=10, legendfontsize=10, titlefontsize=12, margin=10Plots.mm)
        time_orbits = time_history ./ (2.0 * pi * sqrt(chief_oe_initial.a^3 / mu_earth))
        p_da = plot(time_orbits, da_history, label="δa [m]", xlabel="Time [orbits]", ylabel="δa [m]", title="Evolution of δa under Drag Control", lw=3, color=:blue, yformatter=:scientific)
        p_db = plot(time_orbits, db_history, label="δB (Input)", xlabel="Time [orbits]", ylabel="δB [-]", title="Control Input History", lw=3, color=:red)
        p_combined = plot(p_da, p_db, layout=(2,1), size=(1000, 800))
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        filename = "delta_a_history_corrected_$(timestamp).png"
        savefig(p_combined, filename)
        println("グラフを保存しました: $filename")
        display(p_combined)
    end

    return final_cost
end

# ==============================================================================
# 単一ケースでの軌道補正デバッグ関数 (時間短縮版)
# ==============================================================================
function debug_single_correction_case()
    println("\n" * "="^60)
    println("単一ケースでの軌道補正デバッグ (2.5軌道周期)")
    println("="^60)
    
    TARGET_DELTA_E_NORM_METERS = 500.0 
    TARGET_Z_MAX_METERS      = 1.0

    # 検証する初期分離マヌーバ
    dv_mag_case = 0.5
    angle_deg_case = 180.0
    println("検証ケース: 分離速度 = $(dv_mag_case) m/s, 分離方向 = $(angle_deg_case) deg")

    # 初期設定
    target_roe_vec = SVector{7,Float64}(
        0.0, 0.0, TARGET_DELTA_E_NORM_METERS / a_c_stm_init, 0.0,
        0.0, TARGET_Z_MAX_METERS / a_c_stm_init, 0.0
    )
    W = Diagonal(SVector{7,Float64}(1.0e6, 1.0, 1.0e3, 1.0e3, 1.0e3, 1.0e3, 0.0))
    
    n_init = sqrt(mu_earth / a_c_stm_init^3)
    oe_chief_initial_for_sv = OrbitalElementsClassical(a_c_stm_init,e_c_stm_init,i_c_stm_init,Omega_c_stm_init,omega_c_stm_init,0.0,n_init,M_c_stm_init)
    
    posvel_chief_initial_eci_vec = orbital_elements_to_sv(oe_chief_initial_for_sv)
    r_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[1:3]); v_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[4:6])
    oe_chief_eval = sv_to_orbital_elements(CartesianStateECI(r_chief_init_eci, v_chief_init_eci))

    # 初期ROEの計算
    angle_rad_val = deg2rad(angle_deg_case)
    dv_R_val_comp=dv_mag_case*cos(angle_rad_val); dv_T_val_comp=dv_mag_case*sin(angle_rad_val)
    dv_lvlh_vec = SVector(dv_R_val_comp, dv_T_val_comp, 0.0)
    state_deputy_init_eci=cw_to_eci_deputy_state(r_chief_init_eci,v_chief_init_eci,dr_lvlh_init,dv_lvlh_vec)
    oe_dep_init = sv_to_orbital_elements(CartesianStateECI(state_deputy_init_eci.r_vec, state_deputy_init_eci.v_vec))
    qns_roes_init=orbital_elements_to_qns_roe_koenig(oe_chief_eval,oe_dep_init)

    initial_roe_vec = SVector(
        qns_roes_init.delta_a_norm, qns_roes_init.delta_lambda, 
        qns_roes_init.delta_ex, qns_roes_init.delta_ey, 
        qns_roes_init.delta_ix, qns_roes_init.delta_iy, 
        DELTA_B_INIT 
    )

    # ★★★ 修正: シミュレーション時間を一時的に変更するためのハック ★★★
    # run_drag_correction_simulation はグローバル定数 PROPAGATION_ORBITS を参照するため、
    # ここで一時的に上書きするか、関数の引数に時間を追加するのが正攻法ですが、
    # 簡単のためグローバル定数の値を変更して呼び出します。
    # (Juliaの定数変更は警告が出ますが、デバッグ用途なら許容範囲です)
    
    # ※ ただし、Juliaではconstの再定義はエラーになる場合があるため、
    # 関数に引数 `duration_orbits` を追加する修正を行います。
    
    final_cost = run_drag_correction_simulation(initial_roe_vec, oe_chief_eval, target_roe_vec, W, true, 2.5) # 2.5軌道周期を指定

    println("\n" * "="^60)
    @printf "デバッグ実行完了。最終コスト: %.3e\n" final_cost
    println("="^60)
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
    A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,omega_c_ti,true,true,DENSITY_MODEL_FREE, RHO_LEO, BC_CHIEF)
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

# ==============================================================================
# [ 到達可能集合のデータ点を計算する関数 ] 
# ==============================================================================
"""
指定された分離速度で全方位に分離した場合の、最終的なROEの集合を計算。

戻り値:
- Tupleの配列: 各要素は (分離角度[deg], 最終ROEベクトル)
"""
function calculate_reachable_set_data(dv_for_set::Float64, propagation_orbits::Float64)
    println("\n--- dv = $(dv_for_set) m/s における到達可能集合のデータ計算を開始 ---")

    # 結果を (角度, ROE) のタプルで格納する配列
    results_list = Tuple{Float64, SVector{7,Float64}}[] 

    # --- (既存コードから初期設定を流用) ---
    oe_chief_initial_for_sv = OrbitalElementsClassical(a_c_stm_init,e_c_stm_init,i_c_stm_init,Omega_c_stm_init,omega_c_stm_init,0.0,0.0,M_c_stm_init)
    posvel_chief_initial_eci_vec = orbital_elements_to_sv(oe_chief_initial_for_sv)
    r_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[1:3]); v_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[4:6])
    oe_chief_eval = sv_to_orbital_elements(CartesianStateECI(r_chief_init_eci, v_chief_init_eci))
    tf_val = propagation_orbits * 2.0 * pi * sqrt(oe_chief_eval.a^3 / mu_earth)

    for angle_val in 0.0:2.0:358.0
        angle_rad_val = deg2rad(angle_val)
        dv_R_val_comp=dv_for_set*cos(angle_rad_val); dv_T_val_comp=dv_for_set*sin(angle_rad_val)
        dv_lvlh_vec = SVector(dv_R_val_comp, dv_T_val_comp, 0.0)

        # --- (既存の順伝播計算はそのまま) ---
        state_deputy_init_eci=cw_to_eci_deputy_state(r_chief_init_eci,v_chief_init_eci,dr_lvlh_init,dv_lvlh_vec)
        oe_dep_init = sv_to_orbital_elements(CartesianStateECI(state_deputy_init_eci.r_vec, state_deputy_init_eci.v_vec))
        qns_roes_init=orbital_elements_to_qns_roe_koenig(oe_chief_eval,oe_dep_init)
        roe_aug_init_vec=SVector(qns_roes_init.delta_a_norm, qns_roes_init.delta_lambda, qns_roes_init.delta_ex, qns_roes_init.delta_ey, qns_roes_init.delta_ix, qns_roes_init.delta_iy, delta_a_dot_drag)
        omega_c_ti=oe_chief_eval.omega
        omega_dot_j2,Omega_dot_j2=get_secular_j2_rates_koenig(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i)
        omega_c_tf_val=mod(oe_chief_eval.omega+omega_dot_j2*tf_val,2*pi)
        J_ti=get_J_qns_augmented_koenig(omega_c_ti); J_tf_inv=get_J_qns_inv_augmented_koenig(omega_c_tf_val)
        roe_prime_init=J_ti*roe_aug_init_vec
        A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,omega_c_ti,true,true,DENSITY_MODEL_FREE, RHO_LEO, BC_CHIEF)
        STM_prime=get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_p,A_j2_p,A_drag_p,tf_val,oe_chief_eval.e,true,DENSITY_MODEL_FREE)
        roe_prime_final=STM_prime*roe_prime_init
        roe_aug_final_vec=J_tf_inv*roe_prime_final
        
        # ★★★ 修正点: (角度, ROE) のタプルとして結果を保存 ★★★
        push!(results_list, (angle_val, roe_aug_final_vec))
    end
    
    return results_list
end

# ==============================================================================
# [ 到達可能集合をプロットする関数 ] 
# ==============================================================================
"""
最終ROEのリストと理想点をプロット。
"""
function plot_reachable_set_2d(results_list, dv_mag, target_roe_vec, best_point_info)
    # println("--- 到達可能集合と理想点のプロットを作成中 ---")
    
    # --- 物理量に変換 ---
    a_c = a_c_stm_init
    valid_results = [item for item in results_list if !any(isnan, item[2])]
    if isempty(valid_results)
        println("警告: 有効なデータポイントがありませんでした。プロットをスキップします。")
        return
    end
    roe_list = [item[2] for item in valid_results] # ROEデータだけを抽出

    # 到達可能集合の点
    final_a_da_m = [a_c * roe[1] for roe in roe_list]
    final_a_dlambda_m = [a_c * roe[2] for roe in roe_list]
    # ... (他のROE要素も同様に抽出) ...
    final_a_dex_m = [a_c * roe[3] for roe in roe_list]
    final_a_dey_m = [a_c * roe[4] for roe in roe_list]
    final_a_dix_m = [a_c * roe[5] for roe in roe_list]
    final_a_diy_m = [a_c * roe[6] for roe in roe_list]

    # 目標点
    target_a_dex_m = a_c * target_roe_vec[3]; target_a_dey_m = a_c * target_roe_vec[4]
    target_a_dix_m = a_c * target_roe_vec[5]; target_a_diy_m = a_c * target_roe_vec[6]
    
    # ★★★ 追加: δa ≈ 0 となる理想点の物理量を計算 ★★★
    best_roe = best_point_info.roe
    best_a_da_m = a_c * best_roe[1]
    best_a_dlambda_m = a_c * best_roe[2]
    best_a_dex_m = a_c * best_roe[3]; best_a_dey_m = a_c * best_roe[4]
    best_a_dix_m = a_c * best_roe[5]; best_a_diy_m = a_c * best_roe[6]

    @printf "【プロット関数内デバッグ】best_a_dlambda_m の値: %.3f [m]\n" best_a_dlambda_m
    flush(stdout) # 表示を確実にする

    # --- プロットを作成 ---
    p1 = scatter(final_a_dlambda_m, final_a_da_m, xlabel="a * δλ [m]", ylabel="a * δa [m]",
        legend=false, aspect_ratio=:equal, m=:o, ms=2, markerstrokewidth=0)
    # ★★★ 追加: 理想点をプロット ★★★
    scatter!(p1, [best_a_dlambda_m], [best_a_da_m], marker=:xcross, markersize=8, markerstrokecolor=:green)

    p2 = scatter(final_a_dex_m, final_a_dey_m, xlabel="a * δe_x [m]", ylabel="a * δe_y [m]",
        aspect_ratio=:equal, m=:o, ms=2, markerstrokewidth=0, label="Reachable Set")
    scatter!(p2, [target_a_dex_m], [target_a_dey_m], marker=:diamond, markersize=8, label="Target")
    # ★★★ 追加: 理想点をプロット ★★★
    scatter!(p2, [best_a_dex_m], [best_a_dey_m], marker=:xcross, markersize=8, markerstrokecolor=:green, label="Ideal Point (δa ≈ 0)")

    p3 = scatter(final_a_dix_m, final_a_diy_m, xlabel="a * δi_x [m]", ylabel="a * δi_y [m]",
        aspect_ratio=:equal, m=:o, ms=2, markerstrokewidth=0, label="") 
    scatter!(p3, [target_a_dix_m], [target_a_diy_m], marker=:diamond, markersize=8, label="")
    # ★★★ 追加: 理想点をプロット ★★★
    scatter!(p3, [best_a_dix_m], [best_a_diy_m], marker=:xcross, markersize=8, markerstrokecolor=:green)

    final_plot = plot(p1, p2, p3, layout=(1,3), size=(1800, 600), 
        plot_title="Reachable Set (Δv=$(dv_mag)m/s) and Ideal Point (Angle=$(round(best_point_info.angle, digits=1))°)",
        legend=:outerright)
    display(final_plot)
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    html_filename = "reachable_set_report_$(timestamp).html"
    savefig(final_plot, html_filename)
    println("プロットを保存しました: $html_filename")
end

# ==============================================================================
# [ 到達可能集合のデータ点を計算+プロット実行ブロック ] 
# ==============================================================================
function run_reachable_set_analysis()
    # 1. 可視化したい分離速度を設定
    dv_to_visualize = 0.09 # [m/s]

    # 2. 到達可能集合のデータを計算 (角度情報も含む)
    results_list = calculate_reachable_set_data(dv_to_visualize, PROPAGATION_ORBITS)
    
    if isempty(results_list)
        println("データが計算されませんでした。")
        return
    end

    # δaが最も0に近くなる点を探す
    min_abs_da = Inf
    best_index = -1
    for (i, result) in enumerate(results_list)
        final_roe = result[2]
        abs_da = abs(final_roe[1]) # δa_norm はROEベクトルの1番目の要素
        if abs_da < min_abs_da
            min_abs_da = abs_da
            best_index = i
        end
    end
    
    # 見つかった理想点の情報を取得
    best_angle = results_list[best_index][1]
    best_roe_vec = results_list[best_index][2]
    best_point_info = (angle=best_angle, roe=best_roe_vec)
    
    println("\n" * "="^40)
    @printf "解析結果: δaが最も0に近づく理想的な分離方向は %.1f 度\n" best_angle
    @printf "その時の最終的な軌道長半径差 (δa_norm) は %.3e \n" best_roe_vec[1]
    println("="^40 * "\n")


    # 3. 目標ROEベクトルを定義
    TARGET_a_delta_e_norm = 500.0
    TARGET_Z_MAX_METERS = 1.0
    roe_target_vec = SVector{7,Float64}(0.0, 0.0,
        TARGET_a_delta_e_norm / a_c_stm_init, 0.0,
        0.0, TARGET_Z_MAX_METERS / a_c_stm_init, 0.0)

    # 4. プロット関数に理想点の情報を渡して呼び出し
    plot_reachable_set_2d(results_list, dv_to_visualize, roe_target_vec, best_point_info)
end

# ==============================================================================
# [ 感度分析：分離マヌーバ誤差の影響を評価する関数 ]
# ==============================================================================
function run_sensitivity_analysis(base_angle_deg::Float64, base_dv_mag::Float64, target_roe_vec::SVector{7,Float64})
    println("\n\n--- 分離マヌーバ誤差に対する感度分析を開始します ---")
    
    # --- 分析範囲の設定 ---
    angle_error_range_deg = -10.0:1.0:10.0  # ±10度を1度刻みで評価
    dv_error_range_percent = -10.0:1.0:10.0 # ±10%を1%刻みで評価

    costs = zeros(length(angle_error_range_deg), length(dv_error_range_percent))

    # --- 初期条件のセットアップ (find_j2_invariant_maneuverから流用) ---
    oe_chief_initial_for_sv = OrbitalElementsClassical(a_c_stm_init,e_c_stm_init,i_c_stm_init,Omega_c_stm_init,omega_c_stm_init,0.0,0.0,M_c_stm_init)
    posvel_chief_initial_eci_vec = orbital_elements_to_sv(oe_chief_initial_for_sv)
    r_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[1:3]); v_chief_init_eci=SVector{3}(posvel_chief_initial_eci_vec[4:6])
    oe_chief_eval = sv_to_orbital_elements(CartesianStateECI(r_chief_init_eci, v_chief_init_eci))
    tf_val = PROPAGATION_ORBITS * 2.0 * pi * sqrt(oe_chief_eval.a^3 / mu_earth)

    # --- STMの計算 (ループ外で一度だけ実行) ---
    omega_dot_j2, Omega_dot_j2 = get_secular_j2_rates_koenig(oe_chief_eval.a, oe_chief_eval.e, oe_chief_eval.i)
    oe_chief_at_tf = OrbitalElementsClassical(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,mod(oe_chief_eval.RAAN+Omega_dot_j2*tf_val,2*pi),mod(oe_chief_eval.omega+omega_dot_j2*tf_val,2*pi),0.0,oe_chief_eval.n,mod(oe_chief_eval.M+oe_chief_eval.n*tf_val,2*pi))
    omega_c_ti=oe_chief_eval.omega; omega_c_tf_val=oe_chief_at_tf.omega; 
    J_ti=get_J_qns_augmented_koenig(omega_c_ti); J_tf_inv=get_J_qns_inv_augmented_koenig(omega_c_tf_val);
    A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(oe_chief_eval.a,oe_chief_eval.e,oe_chief_eval.i,omega_c_ti,true,true,DENSITY_MODEL_FREE, RHO_LEO, BC_CHIEF)
    STM_prime=get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_p,A_j2_p,A_drag_p,tf_val,oe_chief_eval.e,true,DENSITY_MODEL_FREE)

    # --- 誤差ループ ---
    for (i, angle_err) in enumerate(angle_error_range_deg)
        for (j, dv_err_percent) in enumerate(dv_error_range_percent)
            
            current_angle_deg = base_angle_deg + angle_err
            current_dv_mag = base_dv_mag * (1.0 + dv_err_percent / 100.0)

            # 初期ROEの計算
            angle_rad_val = deg2rad(current_angle_deg)
            dv_R = current_dv_mag * cos(angle_rad_val); dv_T = current_dv_mag * sin(angle_rad_val)
            dv_lvlh_vec = SVector(dv_R, dv_T, 0.0)
            state_deputy_init_eci = cw_to_eci_deputy_state(r_chief_init_eci, v_chief_init_eci, dr_lvlh_init, dv_lvlh_vec)
            oe_dep_init = sv_to_orbital_elements(state_deputy_init_eci)
            qns_roes_init = orbital_elements_to_qns_roe_koenig(oe_chief_eval, oe_dep_init)
            roe_aug_init_vec = SVector(qns_roes_init.delta_a_norm, qns_roes_init.delta_lambda, qns_roes_init.delta_ex, qns_roes_init.delta_ey, qns_roes_init.delta_ix, qns_roes_init.delta_iy, delta_a_dot_drag)

            # 最終ROEの計算
            roe_prime_init = J_ti * roe_aug_init_vec
            roe_prime_final = STM_prime * roe_prime_init
            roe_aug_final_vec = J_tf_inv * roe_prime_final

            # コスト計算
            error_vec = roe_aug_final_vec - target_roe_vec
            W = Diagonal(SVector{7,Float64}(1.0e6, 1.0, 1.0e3, 1.0e3, 1.0e3, 1.0e3, 0.0))
            costs[i, j] = dot(error_vec, W * error_vec)
        end
    end
    
    println("感度分析の計算が完了しました。")

    # --- 結果のプロット ---
    p_sensitivity = heatmap(
        dv_error_range_percent,
        angle_error_range_deg,
        costs,
        xlabel="Delta-V Error (%)",
        ylabel="Separation Angle Error (deg)",
        title="Sensitivity of Final Cost to Maneuver Errors",
        color=:viridis,
        colorbar_title="Cost (Target ROE Error)"
    )
    display(p_sensitivity)
    
    # HTMLレポートに追記
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    html_filename = "sensitivity_report_$(timestamp).html"
    open(html_filename, "w") do f
        write(f, "<html><head><title>Sensitivity Analysis Report</title></head><body>")
        write(f, "<h1>Sensitivity Analysis Report</h1>")
        write(f, "<h2>Base Optimal Maneuver: $(base_angle_deg) deg, $(base_dv_mag) m/s</h2>")
        
        io = IOBuffer()
        show(io, MIME"image/png"(), p_sensitivity)
        plot_base64 = base64encode(take!(io))
        write(f, "<div><img src=\"data:image/png;base64,$(plot_base64)\"/></div>")
        
        write(f, "</body></html>")
    end
    println("感度分析レポートを保存しました: $html_filename")
end

# ==============================================================================
# [ NaN発生箇所を特定するためのデバッグ専用関数]
# ==============================================================================
function debug_single_angle(angle_deg::Float64, dv_mag::Float64)
    
    println("\n" * "="^50)
    @printf "デバッグ実行: 分離角度 = %.1f [deg], 分離速度 = %.3f [m/s]\n" angle_deg dv_mag
    println("="^50)

    # ... (ステップ1の部分は変更なし) ...
    oe_chief_initial_for_sv = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, 0.0, M_c_stm_init)
    posvel_chief_initial_eci_vec = orbital_elements_to_sv(oe_chief_initial_for_sv)
    r_chief_init_eci = SVector{3}(posvel_chief_initial_eci_vec[1:3])
    v_chief_init_eci = SVector{3}(posvel_chief_initial_eci_vec[4:6])
    oe_chief_eval = sv_to_orbital_elements(CartesianStateECI(r_chief_init_eci, v_chief_init_eci))
    angle_rad_val = deg2rad(angle_deg)
    dv_R = dv_mag * cos(angle_rad_val); dv_T = dv_mag * sin(angle_rad_val)
    dv_lvlh_vec = SVector(dv_R, dv_T, 0.0); dr_lvlh_init = SVector(0.0, 0.0, 0.0)
    state_deputy_init_eci = cw_to_eci_deputy_state(r_chief_init_eci, v_chief_init_eci, dr_lvlh_init, dv_lvlh_vec)
    
    println("\n[ステップ1] 副衛星の初期ECI状態ベクトル:")
    println("  r_vec: ", state_deputy_init_eci.r_vec)
    println("  v_vec: ", state_deputy_init_eci.v_vec)
    if any(isnan, state_deputy_init_eci.r_vec) || any(isnan, state_deputy_init_eci.v_vec)
        println("  ==> ⚠️ この段階でNaNが発生しました！")
        return
    end

    # --- 3. 副衛星の古典軌道要素(COE)を計算 ---
    try
        # ★【修正】oe_dep_init の中身を表示する前に、sv_to_keplerの結果を直接見る
        sv_dep = OrbitStateVector(0.0, state_deputy_init_eci.r_vec, state_deputy_init_eci.v_vec)
        kep_dep = SatelliteToolbox.sv_to_kepler(sv_dep)

        println("\n[ステップ2] 副衛星の初期古典軌道要素(COE):")
        println("  a: ", kep_dep.a)
        println("  e: ", kep_dep.e)
        println("  i: ", rad2deg(kep_dep.i))
        println("  Ω: ", rad2deg(kep_dep.Ω))
        println("  ω: ", rad2deg(kep_dep.ω)) # ★ここでNaNが出るか？
        
        if isnan(kep_dep.ω)
             println("  ==> ⚠️ この段階でωがNaNになりました！(e ≈ 0 が原因の可能性大)")
        end

        # --- 4. 相対軌道要素(ROE)を計算 ---
        # oe_dep_init を正しく作成
        M_val = SatelliteToolbox.true_to_mean_anomaly(kep_dep.e, kep_dep.f)
        n_val = sqrt(mu_earth / kep_dep.a^3)
        # 注意：あなたのOrbitalElementsClassicalの定義に合わせてください
        # Mだけを保存する場合:
        oe_dep_init = OrbitalElementsClassical(kep_dep.a, kep_dep.e, kep_dep.i, kep_dep.Ω, kep_dep.ω, 0.0, 0.0, M_val)


        qns_roes_init = orbital_elements_to_qns_roe_koenig(oe_chief_eval, oe_dep_init)
        
        println("\n[ステップ3] 副衛星の初期相対軌道要素(ROE):")
        println(qns_roes_init)
        if any(isnan, [qns_roes_init.delta_a_norm, qns_roes_init.delta_lambda, qns_roes_init.delta_ex, qns_roes_init.delta_ey, qns_roes_init.delta_ix, qns_roes_init.delta_iy])
            println("  ==> ⚠️ ROEの計算結果にNaNが含まれています。")
        end

    catch e
        println("\n[ステップ2の途中でエラー発生]: ", e)
    end
    
    println("\n" * "="^50, "\nデバッグ実行完了。", "\n", "="^50)
end

# ------------------------------------------------------------------------------
# 1. 目的関数 
# ------------------------------------------------------------------------------
function objective_function(params::Vector{Float64}, target_roe_vec::SVector{7,Float64}, W::Diagonal)
    dv_mag, theta_deg, M_deg = params[1], params[2], params[3]
    
    try
        M_rad = deg2rad(M_deg)
        n_init = sqrt(mu_earth / a_c_stm_init^3)
        oe_chief_initial_for_sv = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, n_init, M_rad)
        posvel_chief_initial_eci_vec = orbital_elements_to_sv(oe_chief_initial_for_sv)
        r_chief_init_eci = SVector{3}(posvel_chief_initial_eci_vec[1:3]); v_chief_init_eci = SVector{3}(posvel_chief_initial_eci_vec[4:6])
        oe_chief_eval = sv_to_orbital_elements(CartesianStateECI(r_chief_init_eci, v_chief_init_eci))
        
        tf_val = PROPAGATION_ORBITS * 2.0 * pi * sqrt(oe_chief_eval.a^3 / mu_earth)
        omega_dot_j2, Omega_dot_j2 = get_secular_j2_rates_koenig(oe_chief_eval.a, oe_chief_eval.e, oe_chief_eval.i)
        oe_chief_at_tf = OrbitalElementsClassical(oe_chief_eval.a, oe_chief_eval.e, oe_chief_eval.i, mod(oe_chief_eval.RAAN + Omega_dot_j2 * tf_val, 2*pi), mod(oe_chief_eval.omega + omega_dot_j2 * tf_val, 2*pi), 0.0, oe_chief_eval.n, mod(oe_chief_eval.M + oe_chief_eval.n * tf_val, 2*pi))

        omega_c_ti = oe_chief_eval.omega; omega_c_tf_val = oe_chief_at_tf.omega
        J_ti = get_J_qns_augmented_koenig(omega_c_ti); J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf_val)
        
        A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_chief_eval.a, oe_chief_eval.e, oe_chief_eval.i, omega_c_ti, true, true, DENSITY_MODEL_FREE, RHO_LEO, BC_CHIEF)
        A_kep_f = SMatrix{7,7,Float64}(Float64.(A_kep)); A_j2_f = SMatrix{7,7,Float64}(Float64.(A_j2)); A_drag_f = SMatrix{7,7,Float64}(Float64.(A_drag))
        STM_prime_any = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_f, A_j2_f, A_drag_f, tf_val, oe_chief_eval.e, true, DENSITY_MODEL_FREE)
        STM_prime = SMatrix{7,7,Float64}(Float64.(STM_prime_any))
        J_ti_f = SMatrix{7,7,Float64}(Float64.(J_ti)); J_tf_inv_f = SMatrix{7,7,Float64}(Float64.(J_tf_inv))

        theta_rad = deg2rad(theta_deg)
        psi_rad = 0.0 

        # ★★★ 修正: θ=90度を接線方向にする ★★★
        dv_R = dv_mag * cos(psi_rad) * cos(theta_rad)
        dv_T = dv_mag * cos(psi_rad) * sin(theta_rad)
        dv_N = dv_mag * sin(psi_rad)

        dv_lvlh_vec = SVector(dv_R, dv_T, dv_N)
        dr_lvlh_init = SVector(0.0, 0.0, 0.0)
        state_deputy_init_eci = cw_to_eci_deputy_state(r_chief_init_eci, v_chief_init_eci, dr_lvlh_init, dv_lvlh_vec)
        oe_dep_init = sv_to_orbital_elements(state_deputy_init_eci)
        qns_roes_init = orbital_elements_to_qns_roe_koenig(oe_chief_eval, oe_dep_init)
        roe_aug_init_vec = SVector(qns_roes_init.delta_a_norm, qns_roes_init.delta_lambda, qns_roes_init.delta_ex, qns_roes_init.delta_ey, qns_roes_init.delta_ix, qns_roes_init.delta_iy, delta_a_dot_drag)

        roe_prime_init = J_ti_f * roe_aug_init_vec
        roe_prime_final = STM_prime * roe_prime_init
        roe_aug_final_vec = J_tf_inv_f * roe_prime_final

        error_vec = SVector{7,Float64}(roe_aug_final_vec[1:7]) - target_roe_vec
        cost = dot(error_vec, W * error_vec)

        return isnan(cost) ? Inf : cost
    catch e
        return Inf
    end
end

# ------------------------------------------------------------------------------
# 2. 制約関数 
# ------------------------------------------------------------------------------
function constraint_function(params::Vector{Float64})
    dv_mag, theta_deg, M_deg = params[1], params[2], params[3]
    
    SAFE_DISTANCE_METERS = 100.0 
    MAX_DV_MAG_MPS = 0.5
    
    try
        if dv_mag > MAX_DV_MAG_MPS; return false, :dv_limit; end

        M_rad = deg2rad(M_deg)
        n_init = sqrt(mu_earth / a_c_stm_init^3)
        oe_chief_initial = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, n_init, M_rad)
        sv_chief = orbital_elements_to_sv(oe_chief_initial)
        r_c = SVector{3}(sv_chief[1:3]); v_c = SVector{3}(sv_chief[4:6])

        theta_rad = deg2rad(theta_deg)
        psi_rad = 0.0

        # ★★★ 修正: θ=90度を接線方向にする ★★★
        dv_R = dv_mag * cos(psi_rad) * cos(theta_rad)
        dv_T = dv_mag * cos(psi_rad) * sin(theta_rad)
        dv_N = dv_mag * sin(psi_rad)
        
        dv_lvlh = SVector(dv_R, dv_T, dv_N)
        dr_lvlh = SVector(0.0, 0.0, 0.0)
        
        state_deputy = cw_to_eci_deputy_state(r_c, v_c, dr_lvlh, dv_lvlh)
        
        T_orbit = 2.0 * pi * sqrt(a_c_stm_init^3 / mu_earth)
        
        sv_c_struct = OrbitStateVector(0.0, r_c, v_c)
        sv_d_struct = OrbitStateVector(0.0, state_deputy.r_vec, state_deputy.v_vec)
        kep_c = sv_to_kepler(sv_c_struct); kep_d = sv_to_kepler(sv_d_struct)
        j2d_c = j2_init(kep_c); r_c_1, _ = j2!(j2d_c, T_orbit)
        j2d_d = j2_init(kep_d); r_d_1, _ = j2!(j2d_d, T_orbit)

        dist_1 = norm(r_d_1 - r_c_1)
        
        if dist_1 > SAFE_DISTANCE_METERS
            return true, :ok
        else
            return false, :collision_risk
        end
    catch e
        return false, :error
    end
end

# ------------------------------------------------------------------------------
# 3. 3Dグリッドサーチ実行関数 (修正版: NG理由集計・エラー回避)
# ------------------------------------------------------------------------------
function run_global_optimization()
    # --- 探索グリッド ---
    dv_range = 0.01:0.01:1.0
    theta_range = 0:10:360
    M_range = 0:10:360

    TARGET_a_delta_e_norm = 500.0
    TARGET_Z_MAX_METERS = 1.0
    target_roe_vec = SVector{7,Float64}(0.0, 0.0,
        TARGET_a_delta_e_norm / a_c_stm_init, 0.0,
        0.0, TARGET_Z_MAX_METERS / a_c_stm_init, 0.0)
    
    W = Diagonal(SVector{7,Float64}(1.0e6, 1.0, 1000.0, 1000.0, 1000.0, 1000.0, 0.0))

    min_cost = Inf
    optimal_params = nothing
    
    # マップデータ
    cost_map = fill(NaN, length(M_range), length(theta_range))
    safety_map = fill(0, length(M_range), length(theta_range))
    
    # NG理由のカウンター
    ng_counts = Dict(:dv_limit => 0, :collision_risk => 0, :error => 0)
    total_safe_count = 0
    
    total_iterations = length(dv_range) * length(theta_range) * length(M_range)
    println("\n包括的最適化を開始します... (Target: $(total_iterations) cases)")
    println("  - 安全距離制約: 100.0 m")
    println("  - Δv上限制約:   1.0 m/s")
    
    counter = 0

    for (i, M) in enumerate(M_range)
        for (j, theta) in enumerate(theta_range)
            
            best_cost_in_cell = Inf
            is_cell_safe = false

            for dv in dv_range
                counter += 1
                if counter % 10000 == 0 
                    @printf "  進捗: %d / %d (%.1f %%)\n" counter total_iterations (counter/total_iterations*100)
                end

                params = [dv, theta, M]
                
                # ★ 制約チェック (理由も受け取る)
                is_ok, reason = constraint_function(params)
                
                if is_ok
                    is_cell_safe = true
                    total_safe_count += 1
                    cost = objective_function(params, target_roe_vec, W)
                    
                    if cost < best_cost_in_cell
                        best_cost_in_cell = cost
                    end
                    if cost < min_cost
                        min_cost = cost
                        optimal_params = (dv=dv, theta=theta, M=M)
                    end
                else
                    ng_counts[reason] += 1
                end
            end

            if is_cell_safe
                cost_map[i, j] = best_cost_in_cell
                safety_map[i, j] = 1
            else
                cost_map[i, j] = NaN 
                safety_map[i, j] = 0
            end
        end
    end

    println("\n=== 最適化完了 ===")
    println("[制約チェック結果]")
    println("  安全なケース数: $total_safe_count")
    println("  NG (Δv上限):    $(ng_counts[:dv_limit])")
    println("  NG (衝突危険):  $(ng_counts[:collision_risk])")
    println("  NG (計算エラー): $(ng_counts[:error])")

    if optimal_params !== nothing
        println("\n【包括的最適解】")
        @printf "  最小コスト: %.3e\n" min_cost
        @printf "  分離位相 (M): %.1f deg\n" optimal_params.M
        @printf "  分離方向 (θ): %.1f deg\n" optimal_params.theta
        @printf "  分離速度 (Δv): %.3f m/s\n" optimal_params.dv
    else
        println("\n制約を満たす解が見つかりませんでした。")
    end

    # --- マップのプロット ---
    # データが空でないか確認
    valid_costs = filter(!isnan, vec(cost_map))
    
    if isempty(valid_costs)
        println("有効なコストデータがないため、マップ作成をスキップします。")
        return
    end

    println("安全性マップを作成中...")
    
    # 対数コストで見やすくする
    log_costs = log10.(valid_costs)
    # 色の範囲をデータの5%~95%に設定してコントラストを確保
    c_min, c_max = quantile(log_costs, [0.05, 0.95])
    
    # NaNを含む元のマップを対数化 (NaNはそのままNaNになる)
    log_cost_map = log10.(cost_map)

    p = heatmap(
        M_range, 
        theta_range, 
        log_cost_map', 
        xlabel="Separation Phase M [deg] (0=Perigee)",
        ylabel="Separation Angle θ [deg] (90=Tangential)",
        title="Safety & Cost Landscape (log10 Cost)",
        color=:viridis,
        clims=(c_min, c_max),
        size=(900, 700)
    )

    if optimal_params !== nothing
        scatter!(p, [optimal_params.M], [optimal_params.theta], 
            marker=:star, color=:red, markersize=10, label="Global Optimum")
    end

    # ターゲット（極・遠地点）の確認マーカー
    # omega=270 (90の逆) の場合、遠地点は M=180
    # u=90(北極) も M=180 に相当
    target_M = 180.0 
    target_theta = 90.0 
    scatter!(p, [target_M], [target_theta], 
        marker=:xcross, color=:magenta, markersize=10, label="Target (Apogee/Polar)")

    display(p)
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "optimization_landscape_$(timestamp).png"
    savefig(p, filename)
    println("マップを保存しました: $filename")
end

# ==============================================================================
# [最終比較] 最適解(Optimum) vs 推奨解(Robust) の詳細比較
# ==============================================================================
function compare_optimal_and_robust()
    println("\n" * "="^60)
    println(" 数値的最適解 vs ロバスト推奨解")
    println("="^60)
    
    # 共通設定
    TARGET_a_delta_e_norm = 500.0
    TARGET_Z_MAX_METERS = 1.0
    target_roe_vec = SVector{7,Float64}(0.0, 0.0,
        TARGET_a_delta_e_norm / a_c_stm_init, 0.0,
        0.0, TARGET_Z_MAX_METERS / a_c_stm_init, 0.0)
    W = Diagonal(SVector{7,Float64}(1.0e6, 1.0, 1000.0, 1000.0, 1000.0, 1000.0, 0.0))

    # --- 1. 数値的最適解 (Global Optimum) ---
    # ※先ほどのログの値を入力してください
    opt_M = 20.0
    opt_theta = 170.0
    opt_dv = 0.060
    
    # --- 2. ロバスト推奨解 (Robust Solution) ---
    # 遠地点かつ接線方向
    rob_M = 180.0
    rob_theta = 90.0
    
    # 推奨解のΔvを探索 (制約を満たす最小のΔvを探す)
    # ※手動で見つけるのが面倒なので、ここで簡易探索します
    println("推奨解(M=180, θ=90)の必要Δvを探索中...")
    rob_dv = 0.0
    min_rob_cost = Inf
    
    for dv in 0.01:0.001:0.5
        params = [dv, rob_theta, rob_M]
        is_ok, _ = constraint_function(params)
        if is_ok
            cost = objective_function(params, target_roe_vec, W)
            if cost < min_rob_cost
                min_rob_cost = cost
                rob_dv = dv
            end
        end
    end
    
    # --- 比較出力 ---
    function print_case(name, dv, theta, M)
        println("\n--- $name ---")
        params = [dv, theta, M]
        
        # 安全性チェック (再計算)
        is_safe, reason = constraint_function(params)
        
        # コスト計算
        cost = objective_function(params, target_roe_vec, W)
        
        # 物理量の取得 (calculate_final_stateを利用)
        # ※ psi, phi_tilt は 0 とする
        params_full = [dv, theta, 0.0, 0.0, M]
        final_roe, d_1orbit = calculate_final_state(params_full)
        
        final_da = final_roe[1] * a_c_stm_init
        final_de = norm([final_roe[3], final_roe[4]]) * a_c_stm_init
        
        @printf "  入力: Δv=%.3f m/s, θ=%.1f deg, M=%.1f deg\n" dv theta M
        @printf "  評価: Cost=%.3e, 安全性=%s (1周後距離: %.1f m)\n" cost (is_safe ? "OK" : "NG ($reason)") d_1orbit
        @printf "  結果: δa=%.3f m, δe=%.1f m\n" final_da final_de
    end

    print_case("1. 数値的最適解 ", opt_dv, opt_theta, opt_M)
    print_case("2. ロバスト推奨解 ", rob_dv, rob_theta, rob_M)
    
    println("\n" * "="^60)
end

# ==============================================================================
# [ 最終状態計算関数]
# ==============================================================================
function calculate_final_state(params::Vector{Float64})
    # --- 0. 入力と準備 ---
    # パラメータに「軸ブレの方向」を追加
    dv_mag, theta_deg, psi_deg, phi_tilt_deg, M_deg = params[1], params[2], params[3], params[4], params[5]

        try
        # --- 1. 初期状態の計算 ---
        M_rad = deg2rad(M_deg)
        oe_chief_initial_for_sv = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, 0.0, M_rad)
        sv_chief_init_vec = orbital_elements_to_sv(oe_chief_initial_for_sv)
        r_chief_init_eci = SVector{3}(sv_chief_init_vec[1:3]); v_chief_init_eci = SVector{3}(sv_chief_init_vec[4:6])
        oe_chief_eval = sv_to_orbital_elements(CartesianStateECI(r_chief_init_eci, v_chief_init_eci))
        
        # ★★★【ここからが厳密な3D分離ベクトルの計算】★★★
        theta_rad = deg2rad(theta_deg)         # 分離位相
        psi_rad = deg2rad(psi_deg)             # 軸ブレの大きさ
        phi_tilt_rad = deg2rad(phi_tilt_deg)   # 軸ブレの方向

        # 1. 理想状態（ブレなし）の分離速度ベクトルをRT平面内に定義
        #    (注意: ご自身のコードの R,T の定義に合わせて sin/cos を確認してください)
        v_ideal = SVector(
            dv_mag * cos(theta_rad), # R成分 (cos)
            dv_mag * sin(theta_rad), # T成分 (sin) - θ=90で最大
            0.0                      # N成分
        )

        # 2. 「軸ブレ」を表す回転を定義
        #    ブレの方向(phi_tilt)に垂直な軸の周りで、ブレの大きさ(psi)だけ回転させる
        rotation_axis = SVector(cos(phi_tilt_rad), -sin(phi_tilt_rad), 0.0)
        R_tilt = rotation_matrix_around_axis(rotation_axis, psi_rad)

        # 3. 理想ベクトルに回転を適用して、現実の分離速度ベクトルを計算
        dv_lvlh_vec = R_tilt * v_ideal
        dr_lvlh_init = SVector(0.0, 0.0, 0.0)
        
        state_deputy_init_eci = cw_to_eci_deputy_state(r_chief_init_eci, v_chief_init_eci, dr_lvlh_init, dv_lvlh_vec)
        sv_deputy_init_struct = OrbitStateVector(0.0, state_deputy_init_eci.r_vec, state_deputy_init_eci.v_vec)
        sv_chief_init_struct = OrbitStateVector(0.0, r_chief_init_eci, v_chief_init_eci)

        # --- 2. 最終ROEの計算 (10周期後) ---
        tf_val = PROPAGATION_ORBITS * 2.0 * pi * sqrt(oe_chief_eval.a^3 / mu_earth)
        omega_dot_j2, Omega_dot_j2 = get_secular_j2_rates_koenig(oe_chief_eval.a, oe_chief_eval.e, oe_chief_eval.i)
        oe_chief_at_tf = OrbitalElementsClassical(oe_chief_eval.a, oe_chief_eval.e, oe_chief_eval.i, mod(oe_chief_eval.RAAN + Omega_dot_j2 * tf_val, 2*pi), mod(oe_chief_eval.omega + omega_dot_j2 * tf_val, 2*pi), 0.0, oe_chief_eval.n, mod(oe_chief_eval.M + oe_chief_eval.n * tf_val, 2*pi))
        omega_c_ti = oe_chief_eval.omega; omega_c_tf_val = oe_chief_at_tf.omega
        J_ti = get_J_qns_augmented_koenig(omega_c_ti); J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf_val)
        A_kep_p, A_j2_p, A_drag_p = get_A_prime_qns_augmented_koenig_selectable(oe_chief_eval.a, oe_chief_eval.e, oe_chief_eval.i, omega_c_ti, true, true, DENSITY_MODEL_FREE, RHO_LEO, BC_CHIEF)
        STM_prime = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep_p, A_j2_p, A_drag_p, tf_val, oe_chief_eval.e, true, DENSITY_MODEL_FREE)
        oe_dep_init = sv_to_orbital_elements(state_deputy_init_eci)
        qns_roes_init = orbital_elements_to_qns_roe_koenig(oe_chief_eval, oe_dep_init)
        roe_aug_init_vec = SVector(qns_roes_init.delta_a_norm, qns_roes_init.delta_lambda, qns_roes_init.delta_ex, qns_roes_init.delta_ey, qns_roes_init.delta_ix, qns_roes_init.delta_iy, delta_a_dot_drag)
        roe_prime_init = J_ti * roe_aug_init_vec
        roe_prime_final = STM_prime * roe_prime_init
        final_roe = J_tf_inv * roe_prime_final

        # --- 3. 1周期後の距離の計算 ---
        T_orbit = 2.0 * pi * sqrt(a_c_stm_init^3 / mu_earth)
        
        # ★★★【ここを修正】★★★
        # j2.jl のAPIを正しく使用して軌道伝播を行う
        kep_chief_init = sv_to_kepler(sv_chief_init_struct)
        j2d_chief = j2_init(kep_chief_init)
        r_chief_1_orbit, v_chief_1_orbit = j2!(j2d_chief, T_orbit)

        kep_deputy_init = sv_to_kepler(sv_deputy_init_struct)
        j2d_deputy = j2_init(kep_deputy_init)
        r_deputy_1_orbit, v_deputy_1_orbit = j2!(j2d_deputy, T_orbit)
        # ★★★【ここまで】★★★
        
        d_1orbit = norm(r_deputy_1_orbit - r_chief_1_orbit)

        return final_roe, d_1orbit

    catch e
        # 予期せぬエラーを捕捉
        # @printf "  [計算エラー発生] %s\n" e
        return (SVector{7,Float64}(fill(NaN, 7)), NaN)
    end
end

# ==============================================================================
# [ 姿勢系要求分析の実行関数]
# 最初の計算エラーを発見したら、その場で処理を中断
# ==============================================================================
function run_attitude_requirement_analysis()
    # 探索範囲
    dv_range = 0.01:0.01:0.5
    theta_range = 0:20:340
    psi_range = -1:0.1:1
    phi_tilt_range = 0:45:315 # 軸ブレの方向 0=T軸、90=R軸、180=-T軸、270=-R軸
    M_range = 0:30:330

    results = []
    total_iterations = length(dv_range) * length(theta_range) * length(psi_range) * length(M_range)
    println("姿勢系要求分析のための大規模シミュレーションを開始します... (合計: $(total_iterations) ケース)")

    for M in M_range, phi_tilt in phi_tilt_range, psi in psi_range, dv in dv_range, theta in theta_range
        
        params = [dv, theta, psi, phi_tilt, M]
        
        # 1. 最終状態を計算
        final_roe, d_1orbit = calculate_final_state(params)

        # 2. 計算が失敗したか(NaNが返されたか)をチェック
        if isnan(final_roe[1])
            # 3. 失敗していたら、メッセージを表示して即座に関数を終了する
            println("\n最初の計算エラーが検出されたため、処理を中断します。")
            println("上記のエラーメッセージが根本原因です。")
            return [] # 空の結果を返して終了
        end

        # 計算が成功した場合のみ結果を保存
        push!(results, (params=params, final_roe=final_roe, d_1orbit=d_1orbit))
    end
    
    println("シミュレーションが完了しました。")
    return results
end

# ==============================================================================
# [ 姿勢系要求分析の実行と結果表示を行う関数]
# ==============================================================================
function analyze_attitude_requirements()
    
    # --- 1. 全てのパラメータの組み合わせについてシミュレーションを実行 ---
    println("姿勢系要求分析のための大規模シミュレーションを開始します...")
    all_simulation_results = run_attitude_requirement_analysis()
    println("シミュレーションが完了しました。")

    if isempty(all_simulation_results)
        println("シミュレーション結果が空です。処理を中断します。")
        return
    end

    # --- 2. シミュレーション結果の中から「成功ケース」をフィルタリング ---
    println("成功ケースをフィルタリング中...")
    successful_cases = filter(
        r -> abs(r.final_roe[6] * a_c_stm_init) <= 1.0 && r.d_1orbit > 100.0,
        all_simulation_results
    )

    # --- 3. フィルタリング結果の集計と表示 ---
    total_cases = length(all_simulation_results)
    success_count = length(successful_cases)
    
    println("\n" * "="^50)
    println("分析サマリー")
    println("="^50)
    @printf "総計算ケース数: %d\n" total_cases
    @printf "要求達成ケース数: %d (成功率: %.2f %%)\n" success_count (success_count / total_cases * 100)

    # --- 4. 成功ケースの簡単な分析と考察 ---
    if success_count > 0
        println("\n--- 成功ケースの詳細分析 ---")

        min_dv_success = minimum(r.params[1] for r in successful_cases)
        @printf "要求を達成した最小分離速度 (Δv_min): %.3f m/s\n" min_dv_success

        max_psi_error_success = maximum(abs(r.params[3]) for r in successful_cases)
        @printf "要求を達成した最大回転軸ブレ (ψ_max): %.1f deg\n" max_psi_error_success

        # ★★★【ここを修正】★★★
        # println を @printf に変更し、引数の間にカンマを追加
        @printf("\nこの結果は、要求を達成するには最低でも %.3f m/s の分離速度が必要であり、その際の回転軸のブレは最大でも %.1f 度まで許容されることを示唆しています。\n", min_dv_success, max_psi_error_success)
        # ★★★【ここまで】★★★
    end
    println("="^50)
    
    if success_count > 0
        # plot_successful_distribution_3d(successful_cases)
        create_analysis_plots(successful_cases)
    end
end

# ==============================================================================
# [ 成功ケースの分布を可視化する3D散布図関数 ]
# ==============================================================================
function plot_successful_distribution_3d(successful_cases)
    
    if isempty(successful_cases)
        println("プロットする成功ケースがありません。")
        return
    end

    println("成功ケースの3D分布プロットを作成中...")

    # --- 1. プロット用データを準備 ---
    # successful_casesから各パラメータの値を抽出
    dv_vals    = [r.params[1] for r in successful_cases]
    theta_vals = [r.params[2] for r in successful_cases]
    psi_vals   = [r.params[3] for r in successful_cases]
    M_vals     = [r.params[4] for r in successful_cases]

    # --- 2. 3D散布図を作成 ---
    # 注：このプロットはインタラクティブな plotlyjs() バックエンドで見るのが最適です
    # ファイルの冒頭で gr() をコメントアウトし、plotlyjs() を有効にしてください。
    p3d = scatter(
        M_vals,
        theta_vals,
        psi_vals,
        marker_z=dv_vals,    # 点の色をΔvの値で変化させる
        xlabel="Orbital Phase M [deg]",
        ylabel="Separation Angle θ [deg]",
        zlabel="Axis Error ψ [deg]",
        title="Distribution of Successful Maneuvers (N = $(length(successful_cases)))",
        markersize=2,
        markerstrokewidth=0,
        label="",
        color=:viridis,
        colorbar_title="Separation Δv [m/s]"
    )
    
    display(p3d)

    # --- 3. プロットをHTMLとして保存 ---
    # 3Dプロットはインタラクティブなので、HTMLで保存するのが最適
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    html_filename = "successful_distribution_$(timestamp).html"
    savefig(p3d, html_filename)
    println("3D分布プロットを保存しました: $(html_filename)")
end

# ==============================================================================
# [ 成功ケースの分析グラフ作成関数]
# ==============================================================================
function create_analysis_plots(successful_cases)
    if isempty(successful_cases); println("プロットする成功ケースがありません。"); return; end
    println("成功ケースの分析グラフを作成中...")
    
    # --- データの準備 ---
    params_list = [r.params for r in successful_cases]
    dv_vals    = [p[1] for p in params_list]; theta_vals = [p[2] for p in params_list]
    psi_vals   = [p[3] for p in params_list]; phi_tilt_vals = [p[4] for p in params_list]
    M_vals     = [p[5] for p in params_list]

    # --- グラフA: 各傾き方向における、成功可能なΔvの範囲 ---
    phi_tilt_unique = sort(unique(phi_tilt_vals))
    # ★★★【修正】型を Float64 と明示的に指定 ★★★
    min_dv_per_phi = Float64[]; max_dv_per_phi = Float64[]
    for phi in phi_tilt_unique
        dvs_for_this_phi = [dv_vals[i] for i in 1:length(phi_tilt_vals) if phi_tilt_vals[i] == phi]
        if !isempty(dvs_for_this_phi)
            push!(min_dv_per_phi, minimum(dvs_for_this_phi))
            push!(max_dv_per_phi, maximum(dvs_for_this_phi))
        end
    end
    
    p_a = plot(phi_tilt_unique, min_dv_per_phi, fillrange = max_dv_per_phi, fillalpha = 0.3,
        label="Success Range", xlabel="Axis Tilt Direction φ_tilt [deg]", ylabel="Allowed Δv [m/s]",
        title="Δv Robustness to Tilt Direction", xticks=0:45:315)
    plot!(p_a, phi_tilt_unique, min_dv_per_phi, seriestype=:line, marker=:circle, label="", color=:blue)
    plot!(p_a, phi_tilt_unique, max_dv_per_phi, seriestype=:line, marker=:circle, label="", color=:blue)

    # --- グラフB: 各傾きサイズにおける、成功可能なΔvの範囲 ---
    psi_unique = sort(unique(psi_vals))
    # ★★★【修正】型を Float64 と明示的に指定 ★★★
    min_dv_per_psi = Float64[]; max_dv_per_psi = Float64[]
    for psi in psi_unique
        dvs_for_this_psi = [dv_vals[i] for i in 1:length(psi_vals) if psi_vals[i] == psi]
        if !isempty(dvs_for_this_psi)
            push!(min_dv_per_psi, minimum(dvs_for_this_psi))
            push!(max_dv_per_psi, maximum(dvs_for_this_psi))
        end
    end

    p_b = plot(psi_unique, min_dv_per_psi, fillrange = max_dv_per_psi, fillalpha = 0.3,
        label="Success Range", xlabel="Allowed Axis Tilt ψ [deg]", ylabel="Allowed Δv [m/s]",
        title="Trade-off: Attitude Error vs. Δv")
    plot!(p_b, psi_unique, min_dv_per_psi, seriestype=:line, marker=:circle, label="", color=:blue)
    plot!(p_b, psi_unique, max_dv_per_psi, seriestype=:line, marker=:circle, label="", color=:blue)

    # --- グラフD: 最もロバストな設計点(M, θ)の探索 ---
    # ★★★【修正】Dictと配列の型を明示的に指定 ★★★
    robustness_map = Dict{Tuple{Float64, Float64}, Vector{Float64}}()
    for i in 1:length(M_vals)
        key = (M_vals[i], theta_vals[i])
        if !haskey(robustness_map, key)
            robustness_map[key] = Float64[]
        end
        push!(robustness_map[key], dv_vals[i])
    end

    M_robust = Float64[]; theta_robust = Float64[]; dv_range_width = Float64[]
    for (key, dvs) in robustness_map
        push!(M_robust, key[1])
        push!(theta_robust, key[2])
        push!(dv_range_width, isempty(dvs) ? 0.0 : maximum(dvs) - minimum(dvs))
    end

    p_d = scatter(
        M_robust, theta_robust, marker_z=dv_range_width,
        xlabel="Orbital Phase M [deg]", ylabel="Separation Angle θ [deg]",
        title="Most Robust Design Points (Widest Δv Range)",
        markersize=4, markerstrokewidth=0, label="", color=:inferno,
        colorbar_title="Δv Range Width [m/s]"
    )
    
    # --- グラフC (変更なし) ---
    p_c = scatter(
        M_vals, theta_vals, marker_z=dv_vals, xlabel="Orbital Phase M [deg]",
        ylabel="Separation Angle θ [deg]", title="Sweet Spot for Low-Δv Maneuvers",
        markersize=4, markerstrokewidth=0, label="", color=:viridis, colorbar_title="Required Δv [m/s]")

    # --- グラフの表示と保存 ---
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    final_plot = plot(p_a, p_b, p_c, p_d, layout=(2, 2), size=(1200, 1000))
    display(final_plot)
    
    # 各グラフを個別のファイルとして保存
    savefig(p_a, "plot_A_tilt_robustness_range_$(timestamp).png")
    savefig(p_b, "plot_B_tradeoff_dv_psi_range_$(timestamp).png")
    savefig(p_c, "plot_C_sweet_spot_min_dv_$(timestamp).png")
    savefig(p_d, "plot_D_robust_design_points_$(timestamp).png")
    println("4種類の分析グラフをPNGファイルとして保存しました。")
end

function run_target_search()
    # 最適解の探索
    # 使用したい空気抵抗モデルをここで指定する
    # DENSITY_MODEL_SPECIFIC: 物理パラメータ(大気密度, 弾道係数)に基づくモデル
    # DENSITY_MODEL_FREE:     推定された軌道長半径の変化率に基づくモデ
    drag_model_to_use = DENSITY_MODEL_SPECIFIC
    optimal_params, target_roe_vec = find_j2_invariant_maneuver(J2_AND_DRAG, RT_PLANE, drag_model_to_use)
    
    # 最適解の結果を分析
    if optimal_params !== nothing
        analyze_optimal_result(optimal_params.angle, optimal_params.dv_mag)
        run_sensitivity_analysis(optimal_params.angle, optimal_params.dv_mag, target_roe_vec)
    end
end

function run_comparison_analysis()
    println("\n" * "="^60)
    println("比較解析")
    println("="^60)

    # ケース1：軌道補正なし（従来の「撃ちっぱなし」）
    find_j2_invariant_maneuver(J2_AND_DRAG, RT_PLANE, DENSITY_MODEL_SPECIFIC, false)

    # ケース2：軌道補正あり
    find_j2_invariant_maneuver(J2_AND_DRAG, RT_PLANE, DENSITY_MODEL_SPECIFIC, true)
    
    println("\n" * "="^60)
    println("比較解析完了")
    println("="^60)
end

# ==============================================================================
# 目標時間でδa=0となる分離の検証
# ==============================================================================
function verify_drag_cancellation_separation()
    println("\n" * "="^60)
    println("目標時間でδa=0となる分離マヌーバの検証")
    println("="^60)

    # --- 1. パラメータ設定 ---
    n_init = sqrt(mu_earth / a_c_stm_init^3)
    oe_chief = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, n_init, M_c_stm_init)
    
    target_orbits = PROPAGATION_ORBITS
    t_target = target_orbits * 2.0 * pi * sqrt(oe_chief.a^3 / mu_earth)
    
    rho = RHO_LEO
    Bc = BC_CHIEF
    delta_B = DELTA_B_INIT

    # --- 2. 必要な初期δaの逆算 ---
    K_drag = -rho * oe_chief.n * oe_chief.a * Bc
    adot_drag_norm = K_drag * delta_B 
    req_da_norm_0 = -adot_drag_norm * t_target
    req_da_meters_0 = req_da_norm_0 * oe_chief.a

    println("条件:")
    @printf "  目標時間: %.1f orbits (%.1f sec)\n" target_orbits t_target
    @printf "  差動弾道係数(δB): %.2f\n" delta_B
    
    # --- 3. 必要な接線方向分離速度(ΔvT) ---
    req_dv_T = (oe_chief.n * oe_chief.a / 2.0) * req_da_norm_0
    @printf "  必要な接線方向分離速度(ΔvT): %.5f [m/s]\n" req_dv_T

    # --- 4. STMシミュレーションによる検証 ---
    dv_lvlh_vec = SVector(0.0, req_dv_T, 0.0)
    
    posvel_c = orbital_elements_to_sv(oe_chief)
    r_c = SVector{3}(posvel_c[1:3]); v_c = SVector{3}(posvel_c[4:6])
    state_d = cw_to_eci_deputy_state(r_c, v_c, dr_lvlh_init, dv_lvlh_vec)
    oe_d = sv_to_orbital_elements(CartesianStateECI(state_d.r_vec, state_d.v_vec))
    
    qns_roe_0 = orbital_elements_to_qns_roe_koenig(oe_chief, oe_d)
    roe_vec_0 = SVector(qns_roe_0.delta_a_norm, qns_roe_0.delta_lambda, qns_roe_0.delta_ex, qns_roe_0.delta_ey, qns_roe_0.delta_ix, qns_roe_0.delta_iy, delta_B)

    # 伝播
    omega_dot, Omega_dot = get_secular_j2_rates_koenig(oe_chief.a, oe_chief.e, oe_chief.i)
    omega_c_tf = oe_chief.omega + omega_dot * t_target
    
    J_t0 = get_J_qns_augmented_koenig(oe_chief.omega)
    J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)
    
    A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_chief.a, oe_chief.e, oe_chief.i, oe_chief.omega, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
    STM = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, t_target, oe_chief.e, true, DENSITY_MODEL_SPECIFIC)
    
    roe_vec_f = J_tf_inv * (STM * (J_t0 * roe_vec_0))

    # ★★★ 修正箇所：全ROEの表示 ★★★
    println("\n検証結果 (STM伝播後の最終状態):")
    roe_labels = ["δa", "δλ", "δex", "δey", "δix", "δiy"]
    for i in 1:6
        val_m = roe_vec_f[i] * oe_chief.a
        @printf "  %s : %10.3f [m]\n" roe_labels[i] val_m
    end
    
    if abs(roe_vec_f[1] * oe_chief.a) < 1.0
        println("\n  >> 成功: 目標時間でδa≈0を達成しました。")
    else
        println("\n  >> 注意: δaに残差があります。")
    end
    
    return req_dv_T
end

# ==============================================================================
#  Symbolics.jl による誤差伝播・感度解析 
# ==============================================================================
# function run_symbolic_sensitivity_analysis(nominal_dv_T::Float64)
#     println("\n" * "="^60)
#     println("Symbolics.jl による誤差伝播・感度解析 (Rigorous STM Model)")
#     println("="^60)

#     # --- 1. 誤差の設定 ---
#     sigma_dv    = 0.001 * abs(nominal_dv_T) # 0.1%
#     sigma_theta = deg2rad(1.0)
#     sigma_psi   = deg2rad(1.0)
#     sigma_u     = deg2rad(0.1)

#     println("[設定された誤差標準偏差 (1σ)]")
#     @printf "  Δv誤差: %.3e [m/s], 位相: %.3f [deg], 軸: %.3f [deg], 位置: %.3f [deg]\n" sigma_dv rad2deg(sigma_theta) rad2deg(sigma_psi) rad2deg(sigma_u)

#     # -------------------------------------------------------
#     # 2. 変数定義
#     # -------------------------------------------------------
#     # 軌道パラメータ (数式に残したいもの)
#     @variables a_c e_c i_c omega_c Omega_c M_c # 主衛星要素
#     @variables t_tgt                           # ターゲット時間
#     @variables rho_sym Bc_sym dB_sym           # 抗力パラメータ (rho, Bc, δB)

#     # 誤差変数
#     @variables d_dv d_theta d_psi d_u

#     # ノミナル値変数
#     @variables dv_nom theta_nom u_nom

#     # -------------------------------------------------------
#     # 3. 初期ROEの構築 (GVE)
#     # -------------------------------------------------------
#     # 平均運動 n_c は a_c からの関数として定義
#     n_c_sym = sqrt(mu_earth / a_c^3)

#     # 分離ベクトルのモデル化 (RTN)
#     v_mag = dv_nom + d_dv
#     th    = theta_nom + d_theta
#     ps    = d_psi 
    
#     dv_R = v_mag * cos(ps) * cos(th)
#     dv_T = v_mag * cos(ps) * sin(th)
#     dv_N = v_mag * sin(ps)
    
#     # GVE (円軌道近似)
#     u_sep = u_nom + d_u
#     coef = 1 / (n_c_sym * a_c)
    
#     da0  = 2 * coef * dv_T
#     dl0  = -2 * coef * dv_R
#     dex0 = coef * (dv_R * sin(u_sep) + 2 * dv_T * cos(u_sep))
#     dey0 = coef * (-dv_R * cos(u_sep) + 2 * dv_T * sin(u_sep))
#     dix0 = coef * (dv_N * cos(u_sep))
#     diy0 = coef * (dv_N * sin(u_sep))
    
#     # 初期ROEベクトル (拡張状態: 7要素目は δB)
#     roe_vec_0 = [da0, dl0, dex0, dey0, dix0, diy0, dB_sym]

#     # -------------------------------------------------------
#     # 4. STMによる伝播 (既存関数を利用)
#     # -------------------------------------------------------
#     println("STMをシンボリックに構築中...")

#     # J2レート計算 (シンボリック)
#     omega_dot_sym, Omega_dot_sym = get_secular_j2_rates_koenig(a_c, e_c, i_c)
    
#     # 終了時刻の角度
#     omega_c_tf = omega_c + omega_dot_sym * t_tgt
    
#     # 座標変換行列 J (シンボリック)
#     J_t0 = get_J_qns_augmented_koenig(omega_c)
#     J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)

#     # プラント行列 A (シンボリック)
#     # SPECIFICモデルを使用
#     A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(
#         a_c, e_c, i_c, omega_c, true, true, DENSITY_MODEL_SPECIFIC, rho_sym, Bc_sym
#     )
    
#     # STM (シンボリック)
#     STM = get_STM_prime_qns_augmented_koenig_model_selectable(
#         A_kep, A_j2, A_drag, t_tgt, e_c, true, DENSITY_MODEL_SPECIFIC
#     )
    
#     # 最終ROEの計算
#     # roe_f = J_tf_inv * STM * J_t0 * roe_0
#     roe_prime_0 = J_t0 * roe_vec_0
#     roe_prime_f = STM * roe_prime_0
#     roe_vec_f = J_tf_inv * roe_prime_f

#     roe_names = ["δa", "δλ", "δex", "δey", "δix", "δiy"]

#     # -------------------------------------------------------
#     # 5. 評価用の数値辞書作成
#     # -------------------------------------------------------
#     # 実際の数値を代入するための辞書
#     n_val = sqrt(mu_earth / a_c_stm_init^3)
#     t_tgt_val = PROPAGATION_ORBITS * 2 * pi / n_val
    
#     val_dict = Dict(
#         dv_nom => abs(nominal_dv_T),
#         theta_nom => pi/2,
#         u_nom => 0.0,
#         a_c => a_c_stm_init,
#         e_c => e_c_stm_init,
#         i_c => i_c_stm_init,
#         omega_c => omega_c_stm_init,
#         t_tgt => t_tgt_val,
#         rho_sym => RHO_LEO,
#         Bc_sym => BC_CHIEF,
#         dB_sym => DELTA_B_INIT,
#         d_dv => 0, d_theta => 0, d_psi => 0, d_u => 0
#     )

#     # -------------------------------------------------------
#     # 6. 感度解析実行
#     # -------------------------------------------------------
#     error_vars = [d_dv, d_theta, d_psi, d_u]
#     sigmas     = [sigma_dv, sigma_theta, sigma_psi, sigma_u]
#     error_labels = ["Δv誤差", "位相誤差", "軸倒れ", "位置誤差"]

#     println("\n[感度解析結果]")
#     for i in 1:6
#         println("\n" * "-"^40)
#         println("■ 最終 $(roe_names[i]) への影響:")
        
#         expr = roe_vec_f[i]
#         total_variance = 0.0
#         contributions = []

#         for (j, err_var) in enumerate(error_vars)
#             # 偏微分 (ここが少し重い計算になる可能性があります)
#             diff_expr = Symbolics.derivative(expr, err_var)
            
#             # 数値評価
#             sens_val = Symbolics.value(substitute(diff_expr, val_dict))
            
#             # 寄与度計算
#             contribution = abs(sens_val) * sigmas[j]
#             total_variance += contribution^2
            
#             push!(contributions, (error_labels[j], sens_val, contribution, diff_expr))
#         end
        
#         total_sigma = sqrt(total_variance)
#         @printf "  予測される総誤差 (1σ): %.3e [m]\n" (total_sigma * a_c_stm_init)

#         # 寄与の大きい順に表示
#         sort!(contributions, by = x -> x[3], rev=true)
#         for (label, sens, cont, raw_expr) in contributions
#             if cont > 1e-12
#                 ratio = (cont^2 / total_variance) * 100
#                 @printf "  ・%s:\n" label
#                 @printf "      感度係数: %.2e\n" sens
#                 @printf "      寄与(1σ): %.2e [m] (寄与率: %4.1f%%)\n" (cont * a_c_stm_init) ratio
                
#                 # 数式表示（長くなりすぎる場合は簡略化辞書を適用）
#                 simple_dict = Dict(d_dv=>0, d_theta=>0, d_psi=>0, d_u=>0, theta_nom=>pi/2, u_nom=>0, e_c=>0) # e_c=0とするとかなりスッキリする
#                 simplified_form = substitute(raw_expr, simple_dict)
#                 println("      数式(簡易): ", simplified_form)
#             end
#         end
#     end
#     println("\n" * "="^60)
# end

# ==============================================================================
# Symbolics.jl による誤差伝播・感度解析 (堅牢な数値評価版)
# ==============================================================================
# function run_symbolic_sensitivity_analysis(nominal_dv_T::Float64)
#     println("\n" * "="^60)
#     println("Symbolics.jl による誤差伝播・感度解析 (Rigorous STM Model)")
#     println("="^60)

#     # --- 1. 誤差の設定 ---
#     sigma_dv    = 0.001 * abs(nominal_dv_T) # 分離速度誤差: 0.1%
#     sigma_theta = deg2rad(1.0)              # 分離方向誤差: 1.0 deg
#     sigma_psi   = deg2rad(1.0)              # 軸倒れ誤差:   1.0 deg
#     sigma_u     = deg2rad(0.1)              # 分離位置誤差: 0.1 deg

#     println("[設定された誤差標準偏差 (1σ)]")
#     @printf "  Δv誤差: %.3e [m/s], 位相: %.3f [deg], 軸: %.3f [deg], 位置: %.3f [deg]\n" sigma_dv rad2deg(sigma_theta) rad2deg(sigma_psi) rad2deg(sigma_u)

#     # -------------------------------------------------------
#     # 2. 変数定義
#     # -------------------------------------------------------
#     @variables a_c e_c i_c omega_c Omega_c M_c t_tgt rho_sym Bc_sym dB_sym
#     @variables d_dv d_theta d_psi d_u
#     @variables dv_nom theta_nom u_nom

#     # -------------------------------------------------------
#     # 3. 初期ROEの構築 (GVE)
#     # -------------------------------------------------------
#     n_c_sym = sqrt(mu_earth / a_c^3)
    
#     v_mag = dv_nom + d_dv
#     th    = theta_nom + d_theta
#     ps    = d_psi 
    
#     dv_R = v_mag * cos(ps) * cos(th)
#     dv_T = v_mag * cos(ps) * sin(th)
#     dv_N = v_mag * sin(ps)
    
#     u_sep = u_nom + d_u
#     coef = 1 / (n_c_sym * a_c)
    
#     da0  = 2 * coef * dv_T
#     dl0  = -2 * coef * dv_R
#     dex0 = coef * (dv_R * sin(u_sep) + 2 * dv_T * cos(u_sep))
#     dey0 = coef * (-dv_R * cos(u_sep) + 2 * dv_T * sin(u_sep))
#     dix0 = coef * (dv_N * cos(u_sep))
#     diy0 = coef * (dv_N * sin(u_sep))
    
#     roe_vec_0 = [da0, dl0, dex0, dey0, dix0, diy0, dB_sym]

#     # -------------------------------------------------------
#     # 4. STMによる伝播 (既存関数を利用)
#     # -------------------------------------------------------
#     println("STMをシンボリックに構築中...")

#     omega_dot_sym, Omega_dot_sym = get_secular_j2_rates_koenig(a_c, e_c, i_c)
#     omega_c_tf = omega_c + omega_dot_sym * t_tgt
    
#     J_t0 = get_J_qns_augmented_koenig(omega_c)
#     J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)

#     A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(
#         a_c, e_c, i_c, omega_c, true, true, DENSITY_MODEL_SPECIFIC, rho_sym, Bc_sym
#     )
    
#     STM = get_STM_prime_qns_augmented_koenig_model_selectable(
#         A_kep, A_j2, A_drag, t_tgt, e_c, true, DENSITY_MODEL_SPECIFIC
#     )
    
#     roe_prime_0 = J_t0 * roe_vec_0
#     roe_prime_f = STM * roe_prime_0
#     roe_vec_f = J_tf_inv * roe_prime_f

#     roe_names = ["δa", "δλ", "δex", "δey", "δix", "δiy"]

#     # -------------------------------------------------------
#     # 5. 評価用の数値辞書作成
#     # -------------------------------------------------------
#     n_val = sqrt(mu_earth / a_c_stm_init^3)
#     t_tgt_val = PROPAGATION_ORBITS * 2 * pi / n_val
    
#     val_dict = Dict(
#         dv_nom => abs(nominal_dv_T),
#         theta_nom => pi/2,
#         u_nom => 0.0,
#         a_c => a_c_stm_init,
#         e_c => e_c_stm_init,
#         i_c => i_c_stm_init,
#         omega_c => omega_c_stm_init,
#         t_tgt => t_tgt_val,
#         rho_sym => RHO_LEO,
#         Bc_sym => BC_CHIEF,
#         dB_sym => DELTA_B_INIT,
#         d_dv => 0, d_theta => 0, d_psi => 0, d_u => 0
#     )

#     # -------------------------------------------------------
#     # ★★★ 修正: ノミナル最終ROEの表示 (安全な数値化) ★★★
#     # -------------------------------------------------------
#     println("\n[ノミナル最終ROE (誤差なし)]")
    
#     # 辞書を使って代入
#     nominal_roe_sym = substitute(roe_vec_f, val_dict)
    
#     for i in 1:6
#         # 数値化を試みる
#         val_sym = nominal_roe_sym[i]
#         try
#             # Symbolics.value で中身を取り出し、Float64に変換
#             val_num = Symbolics.value(val_sym)
#             val_float = Float64(val_num)
            
#             @printf "  %s : %10.3f [m]\n" roe_names[i] (val_float * a_c_stm_init)
#         catch e
#             println("  $(roe_names[i]) : 数値化に失敗しました。残存変数: $(Symbolics.get_variables(val_sym))")
#             # エラー詳細が必要なら以下をコメントアウト解除
#             # println(e)
#         end
#     end

#     # -------------------------------------------------------
#     # 6. 感度解析実行
#     # -------------------------------------------------------
#     error_vars = [d_dv, d_theta, d_psi, d_u]
#     sigmas     = [sigma_dv, sigma_theta, sigma_psi, sigma_u]
#     error_labels = ["Δv誤差", "位相誤差", "軸倒れ", "位置誤差"]
    
#     final_sigma_da = 0.0

#     println("\n[感度解析結果 (1σ)]")
#     for i in 1:6
#         println("-"^40)
#         expr = roe_vec_f[i]
#         total_variance = 0.0
#         contributions = []

#         for (j, err_var) in enumerate(error_vars)
#             # 偏微分
#             diff_expr = Symbolics.derivative(expr, err_var)
            
#             # 数値評価 (ここも安全に)
#             sens_sym = substitute(diff_expr, val_dict)
#             sens_val = 0.0
#             try
#                 sens_val = Float64(Symbolics.value(sens_sym))
#             catch
#                 # 偏微分の結果が複雑で数値化できない場合のフォールバック（通常は起きないはず）
#                 sens_val = 0.0
#             end
            
#             contribution = abs(sens_val) * sigmas[j]
#             total_variance += contribution^2
            
#             push!(contributions, (error_labels[j], sens_val, contribution))
#         end
        
#         total_sigma = sqrt(total_variance)
#         if i == 1; final_sigma_da = total_sigma; end 

#         @printf "■ 最終 %s 誤差 (1σ): %.3e [m]\n" roe_names[i] (total_sigma * a_c_stm_init)
        
#         sort!(contributions, by = x -> x[3], rev=true)
#         for (label, sens, cont) in contributions
#             if cont > 1e-12
#                 ratio = (cont^2 / total_variance) * 100
#                 @printf "  - %s: 寄与 %.2e [m] (%.1f%%)\n" label (cont * a_c_stm_init) ratio
#             end
#         end
#     end

#     # -------------------------------------------------------
#     # 5. リカバリー判定
#     # -------------------------------------------------------
#     println("\n" * "="^60)
#     println("リカバリー判定 (Correctability Check)")
#     println("="^60)
    
#     max_da_rate_norm = abs(-RHO_LEO * n_val * a_c_stm_init * BC_CHIEF * DELTA_B_MAX)
#     max_da_rate_meters = max_da_rate_norm * a_c_stm_init 

#     recovery_period_orbits = 1.0
#     recovery_time = recovery_period_orbits * (2 * pi / n_val)
#     recoverable_da = max_da_rate_meters * recovery_time
#     predicted_error_da = final_sigma_da * a_c_stm_init

#     println("条件:")
#     @printf "  最大制御能力(δa変化率): %.3e [m/s] (ΔB_max=%.1f)\n" max_da_rate_meters DELTA_B_MAX
#     @printf "  リカバリー期間:         %.1f orbits (%.1f sec)\n" recovery_period_orbits recovery_time
#     @printf "  修正可能な最大δa量:    %.3f [m]\n" recoverable_da
#     println("-"^40)
#     @printf "  発生が予測されるδa誤差(1σ): %.3f [m]\n" predicted_error_da

#     println("\n判定:")
#     if predicted_error_da <= recoverable_da
#         margin = recoverable_da / predicted_error_da
#         println("  [OK] 1σ誤差は、1軌道周期以内の補正で十分に修正可能です。")
#         @printf "       (マージン: %.1f倍)\n" margin
#     else
#         shortage = predicted_error_da - recoverable_da
#         println("  [WARNING] 1σ誤差が、1軌道周期での修正能力を超えています。")
#         @printf "            (不足分: %.3f m)\n" shortage
#         req_orbits = predicted_error_da / (max_da_rate_meters * (2 * pi / n_val))
#         @printf "            -> 修正には %.1f 軌道周期が必要です。\n" req_orbits
#     end
#     println("="^60)
# end

# ==============================================================================
# Symbolics.jl による誤差伝播・感度解析 (GVE(Section 2.4) & Rigorous STM)
# ==============================================================================
function run_symbolic_sensitivity_analysis(nominal_dv_T::Float64)
    println("\n" * "="^60)
    println("GVE(Section 2.4) & Rigorous STM による感度解析・項別分析")
    println("="^60)

    # --- 1. 誤差の設定 ---
    sigma_dv    = 0.001 * abs(nominal_dv_T)
    sigma_theta = deg2rad(1.0)
    sigma_psi   = deg2rad(1.0)
    sigma_u     = deg2rad(0.1)

    println("[設定された誤差標準偏差 (1σ)]")
    @printf "  Δv誤差: %.3e [m/s], 位相: %.3f [deg], 軸: %.3f [deg], 位置: %.3f [deg]\n" sigma_dv rad2deg(sigma_theta) rad2deg(sigma_psi) rad2deg(sigma_u)

    # -------------------------------------------------------
    # 2. 変数定義
    # -------------------------------------------------------
    @variables a_c e_c i_c omega_c Omega_c M_c t_tgt rho_sym Bc_sym dB_sym
    @variables d_dv d_theta d_psi d_u
    @variables dv_nom theta_nom u_nom # u_nom は mean anomaly ではなく argument of latitude 等の基準

    # 物理定数 (項別分析で数値化するため代入用辞書で管理)
    # ここでは数式に残すためにシンボルとして扱うことも可能だが、
    # 係数が複雑になりすぎるのを防ぐため、定数は展開されることが多い。
    
    # -------------------------------------------------------
    # 3. 分離ベクトルのモデル化 (RTN)
    # -------------------------------------------------------
    v_mag = dv_nom + d_dv
    th    = theta_nom + d_theta
    ps    = d_psi 
    
    # RTN成分
    dv_R = v_mag * cos(ps) * cos(th)
    dv_T = v_mag * cos(ps) * sin(th)
    dv_N = v_mag * sin(ps)
    
    # -------------------------------------------------------
    # 4. GVE (ガウスの変分方程式) による Δα の計算
    # -------------------------------------------------------
    # PDF 2.4節 / 標準的なGVEに基づく
    # ※ PDFの式は dα/dt なので、インパルス近似として Δα ≈ (dα/dt / 力) * Δv を用いる
    # 具体的には、各式の 力項 (dR, dT, dN) を 速度項 (dvR, dvT, dvN) に置き換える
    
    # 必要な補助変数
    # u_sep: 分離位置の緯度引数 (u = omega + f)
    # ここでは u_nom を基準とし、誤差 d_u が乗るとする
    u_sep = u_nom + d_u
    
    # 真近点離角 f の計算
    # u = ω + f => f = u - ω
    f_sep = u_sep - omega_c
    
    # 平均運動 n, 動径 r, 角運動量 h, パラメータ p
    n_sym = sqrt(mu_earth / a_c^3)
    p_sym = a_c * (1 - e_c^2)
    r_sym = p_sym / (1 + e_c * cos(f_sep))
    h_sym = sqrt(mu_earth * p_sym) # または n*a^2*sqrt(1-e^2)
    
    # (1) 半長径 a
    # da/dt = (2/n) * (e sin f * dR + (1 + e cos f) * dT)
    # -> Δa = (2 a^2 / h) * ... と同等
    delta_a_gve = (2 / n_sym) * (e_c * sin(f_sep) * dv_R + (1 + e_c * cos(f_sep)) * dv_T)
    
    # (2) 離心率 e
    # PDF Eq(5) / Standard GVE
    # de/dt = (1/na) * (sin f * dR + (2 cos f + e + e cos^2 f)? * dT)
    # Standard: Δe = (1/h) * (p sin f * dvR + ( (p+r) cos f + r e ) * dvT)
    delta_e_gve = (1 / h_sym) * (p_sym * sin(f_sep) * dv_R + ((p_sym + r_sym) * cos(f_sep) + r_sym * e_c) * dv_T)
    
    # (3) 傾斜角 i
    # di/dt = (r cos theta / h) * dN
    delta_i_gve = (r_sym * cos(u_sep) / h_sym) * dv_N
    
    # (4) 昇交点赤経 Ω
    # dO/dt = (r sin theta / (h sin i)) * dN
    delta_Om_gve = (r_sym * sin(u_sep) / (h_sym * sin(i_c))) * dv_N
    
    # (5) 近地点引数 ω
    # dw/dt = (1/he) * (-p cos f * dR + (p+r) sin f * dT) - (r sin theta cos i / (h sin i)) * dN
    delta_w_gve = (1 / (h_sym * e_c)) * (-p_sym * cos(f_sep) * dv_R + (p_sym + r_sym) * sin(f_sep) * dv_T) - (r_sym * sin(u_sep) * cos(i_c) / (h_sym * sin(i_c))) * dv_N
    
    # (6) 平均近点離角 M
    # dM/dt = n + (1/nae) * ((cos f - 2e) * dR - (2 - e cos f) sin f * dT) ? (PDF Eq 5 approx)
    # Standard Impulsive: ΔM = (1/h) * ( p * cos f - 2 r e ) / e * dvR - (p+r) sin f / e * dvT
    # 注: PDFの式(5)の dM/dt の第2項以降を使用
    term_M_R = (cos(f_sep) - 2 * e_c * r_sym / p_sym) # 近似形: (cos f - 2e)
    # より厳密な一般形を使用
    delta_M_gve = (sqrt(1-e_c^2) / (n_sym * a_c * e_c)) * ( (cos(f_sep) - 2*e_c) * dv_R - (1 + e_c * cos(f_sep) / (1 + e_c * cos(f_sep))) * sin(f_sep) * dv_T )
    # 簡易的にPDFの形に近いもの:
    # delta_M_gve = (1 / (n_sym * a_c * e_c)) * ( (cos(f_sep) - 2*e_c) * dv_R - (2 - e_c * cos(f_sep)) * sin(f_sep) * dv_T )

    # -------------------------------------------------------
    # 4. Deputyの軌道要素 & 初期ROE計算
    # -------------------------------------------------------
    # Chiefの状態 (分離時)
    ac0, ec0, ic0, wc0, Omc0, Mc0 = a_c, e_c, i_c, omega_c, Omega_c, M_c
    
    # Deputyの状態 = Chief + Δα
    ad0 = ac0 + delta_a_gve
    ed0 = ec0 + delta_e_gve
    id0 = ic0 + delta_i_gve
    wd0 = wc0 + delta_w_gve
    Omd0 = Omc0 + delta_Om_gve
    Md0 = Mc0 + delta_M_gve
    
    # ROEへの変換 (Koenig定義)
    # δa = (ad - ac) / ac
    roe_da = (ad0 - ac0) / ac0
    
    # 角度差の計算
    dM = Md0 - Mc0
    dw = wd0 - wc0
    dOm = Omd0 - Omc0
    
    # δλ = δM + δω + δΩ cos i
    roe_dl = dM + dw + dOm * cos(ic0)
    
    # δex = ed cos wd - ec cos wc (近似: e * δe_vec)
    # 厳密な定義: δex = e_d cos(w_d) - e_c cos(w_c)
    # しかしここでは微小量なので展開形で記述
    # δex ≈ δe * cos(wc) - ec * δw * sin(wc)
    roe_dex = ed0 * cos(wd0) - ec0 * cos(wc0)
    roe_dey = ed0 * sin(wd0) - ec0 * sin(wc0)
    
    roe_dix = id0 - ic0
    roe_diy = (Omd0 - Omc0) * sin(ic0)
    
    roe_vec_0 = [roe_da, roe_dl, roe_dex, roe_dey, roe_dix, roe_diy, dB_sym]

    # -------------------------------------------------------
    # 5. STMによる伝播 (既存関数利用)
    # -------------------------------------------------------
    println("STMと伝播式を構築中...")
    
    # J2レート
    omega_dot_sym, Omega_dot_sym = get_secular_j2_rates_koenig(a_c, e_c, i_c)
    omega_c_tf = omega_c + omega_dot_sym * t_tgt
    
    # 座標変換行列 J
    J_t0 = get_J_qns_augmented_koenig(omega_c)
    J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)

    # STM (Specific Model)
    A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(
        a_c, e_c, i_c, omega_c, true, true, DENSITY_MODEL_SPECIFIC, rho_sym, Bc_sym
    )
    STM = get_STM_prime_qns_augmented_koenig_model_selectable(
        A_kep, A_j2, A_drag, t_tgt, e_c, true, DENSITY_MODEL_SPECIFIC
    )
    
    # 最終ROE
    roe_vec_f = J_tf_inv * STM * J_t0 * roe_vec_0
    roe_names = ["δa", "δλ", "δex", "δey", "δix", "δiy"]

    # -------------------------------------------------------
    # 6. 数値評価用辞書
    # -------------------------------------------------------
    n_val = sqrt(mu_earth / a_c_stm_init^3)
    t_tgt_val = PROPAGATION_ORBITS * 2 * pi / n_val
    
    # 分離位置 u_nom: ここでは 近地点 (u=0) または 接線方向分離ならどこでも
    # 指定がない場合は 0 (近地点/昇交点) とする
    
    val_dict = Dict(
        dv_nom => abs(nominal_dv_T),
        theta_nom => pi/2,
        u_nom => deg2rad(90.0), # 必要に応じて変更 (例: pi (遠地点))
        # u_nom => deg2rad(0.0), # 比較用
        a_c => a_c_stm_init,
        e_c => e_c_stm_init,
        i_c => i_c_stm_init,
        omega_c => omega_c_stm_init,
        M_c => M_c_stm_init, # δλ計算の基準
        t_tgt => t_tgt_val,
        rho_sym => RHO_LEO,
        Bc_sym => BC_CHIEF,
        dB_sym => DELTA_B_INIT,
        d_dv => 0, d_theta => 0, d_psi => 0, d_u => 0
    )

    # -------------------------------------------------------
    # 7. 感度解析 & 項別分析 (Dominant Term Analysis)
    # -------------------------------------------------------
    error_vars = [d_dv, d_theta, d_psi, d_u]
    sigmas     = [sigma_dv, sigma_theta, sigma_psi, sigma_u]
    error_labels = ["Δv誤差", "位相誤差", "軸倒れ", "位置誤差"]

    println("\n[感度解析結果 (GVEベース)]")
    
    for i in 1:6
        println("\n" * "-"^60)
        println("■ 最終 $(roe_names[i]) への影響:")
        
        expr = roe_vec_f[i]
        total_variance = 0.0
        contributions = []

        for (j, err_var) in enumerate(error_vars)
            # 1. 偏微分
            diff_expr = Symbolics.derivative(expr, err_var)
            
            # 2. 感度係数の数値評価
            sens_val = 0.0
            try
                sens_val = Float64(Symbolics.value(substitute(diff_expr, val_dict)))
            catch
                sens_val = 0.0
            end
            
            # 3. 寄与度
            contribution = abs(sens_val) * sigmas[j]
            total_variance += contribution^2
            
            # 4. ★項別分析★ (Dominant Term Analysis)
            # 偏微分の式を展開し、加算で構成される項に分解する
            expanded_diff = Symbolics.expand(diff_expr)
            terms = []
            
            # 式が足し算 (Add) かどうかで分岐
            if istree(expanded_diff) && operation(expanded_diff) == +
                args = arguments(expanded_diff)
            else
                args = [expanded_diff]
            end
            
            # 各項の数値を計算
            for term in args
                try
                    t_val = Float64(Symbolics.value(substitute(term, val_dict)))
                    # 簡易化された数式表現 (定数や係数をまとめる)
                    # ここでは見やすくするために、変数d_*=0などを代入した形を作る
                    # ただし term 自体には d_* は含まれていないはず（偏微分後なので）
                    
                    # 物理定数を代入せず記号のまま残したい場合は substitute の辞書を工夫するが、
                    # ここでは構造を示すために、あえて e_c=0 等の近似を入れた式を表示
                    simple_dict = Dict(d_dv=>0, d_theta=>0, d_psi=>0, d_u=>0, theta_nom=>pi/2, u_nom=>0, e_c=>0)
                    term_sym = substitute(term, simple_dict)
                    
                    push!(terms, (abs(t_val), t_val, term_sym))
                catch
                end
            end
            
            # 絶対値の大きい順にソート
            sort!(terms, by = x -> x[1], rev=true)
            
            # 最も支配的な項（トップ1）を取得
            dominant_term_str = length(terms) > 0 ? string(terms[1][3]) : "N/A"
            
            push!(contributions, (error_labels[j], sens_val, contribution, dominant_term_str, terms))
        end
        
        total_sigma = sqrt(total_variance)
        @printf "  予測総誤差 (1σ): %.3e [m]\n" (total_sigma * a_c_stm_init)
        
        sort!(contributions, by = x -> x[3], rev=true)
        for (label, sens, cont, dom_str, term_list) in contributions
            if cont > 1e-12
                ratio = (cont^2 / total_variance) * 100
                @printf "  - %s: 寄与 %.2e [m] (%.1f%%)\n" label (cont * a_c_stm_init) ratio
                println("      支配項(簡易): ", dom_str)
                
                # 詳細分析: もし複数の項が拮抗している場合は内訳を表示
                if length(term_list) > 1 && term_list[2][1] > term_list[1][1] * 0.1
                    println("      [詳細内訳]")
                    for k in 1:min(3, length(term_list))
                        val = term_list[k][2]
                        expr_s = term_list[k][3]
                        if abs(val) > 1e-15
                            @printf "        val=%.2e : %s\n" val expr_s
                        end
                    end
                end
            end
        end
    end
    println("\n" * "="^60)
end

# ==============================================================================
# 感度解析結果の可視化（積み上げ棒グラフ）
# ==============================================================================
function plot_sensitivity_contribution(contributions_data)
    println("\n--- 感度解析結果のグラフを作成中 ---")
    
    roe_names = ["δa", "δλ", "δex", "δey", "δix", "δiy"]
    error_labels = ["Δrω", "Δθ", "Δs", "Δf"]
    
    # データの整形
    n_roe = length(roe_names)
    n_err = length(error_labels)
    data_matrix = zeros(Float64, n_err, n_roe)
    
    for (i, roe_name) in enumerate(roe_names)
        res_list = contributions_data[i]
        for item in res_list
            label, val_m = item[1], item[3] # item[3] はメートル単位に変換済み
            
            row_idx = 0
            if occursin("Δv", label); row_idx = 1
            elseif occursin("位相", label); row_idx = 2
            elseif occursin("軸", label); row_idx = 3
            elseif occursin("位置", label); row_idx = 4
            end
            
            if row_idx > 0
                data_matrix[row_idx, i] = val_m
            end
        end
    end

    # --- グラフ設定 (高画質化) ---
    # DPIを300に設定し、フォントサイズも調整
    default(dpi=300, guidefontsize=18, tickfontsize=14, legendfontsize=18, titlefontsize=20)

    # --- グラフ1: 絶対量（メートル） ---
    p1 = groupedbar(data_matrix', 
        bar_position = :stack,
        bar_width = 0.6,
        xticks = (1:n_roe, roe_names),
        label = reshape(error_labels, 1, :),
        title = "Error Contribution (Absolute)",
        ylabel = "1σ Error [m]",
        # xlabel = "ROE Component",
        legend = :outertopright,
        palette = :tab10,
        left_margin = 15Plots.mm,
        bottom_margin = 20Plots.mm,
        top_margin = 20Plots.mm,
        right_margin = 10Plots.mm
    )

    # --- グラフ2: 構成比（％） ---
    data_percent = zeros(Float64, n_err, n_roe)
    for i in 1:n_roe
        total = sum(data_matrix[:, i])
        if total > 1e-15
            data_percent[:, i] = data_matrix[:, i] ./ total * 100.0
        end
    end

    p2 = groupedbar(data_percent', 
        bar_position = :stack,
        bar_width = 0.6,
        xticks = (1:n_roe, roe_names),
        label = reshape(error_labels, 1, :),
        title = "Error Contribution (Percentage)",
        ylabel = "Contribution [%]",
        # xlabel = "ROE Component",
        legend = :outertopright,
        palette = :tab10,
        ylims = (0, 105),
        left_margin = 15Plots.mm,
        bottom_margin = 20Plots.mm,
        top_margin = 20Plots.mm,
        right_margin = 10Plots.mm
    )

    # まとめて表示・保存
    # サイズを大きめに確保
    final_plot = plot(p1, p2, layout=(2,1), size=(1200, 1400))
    display(final_plot)
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "sensitivity_contribution_$(timestamp).png"
    
    # savefigでもdpiが適用されますが、念のため確認
    savefig(final_plot, filename)
    println("高解像度グラフを保存しました: $filename")
end


# # ==============================================================================
# # 「地球周回低軌道における超小型スターシェード衛星システムの編隊維持に必要な速度調整量」式(5)の近似GVEを用いた感度解析
# # ==============================================================================
# function run_symbolic_sensitivity_analysis_pdf_eq5(nominal_dv_T::Float64)
#     println("\n" * "="^60)
#     println("PDF式(5) [近似GVE] による感度解析 & 可視化")
#     println("="^60)

#     # --- 1. 誤差の設定 ---
#     sigma_dv    = 0.001 * abs(nominal_dv_T)
#     sigma_theta = deg2rad(1.0)
#     sigma_psi   = deg2rad(1.0)
#     sigma_u     = deg2rad(0.1)

#     println("[設定誤差標準偏差 (1σ)]")
#     @printf "  Δv誤差: %.3e [m/s], 位相: %.3f [deg], 軸: %.3f [deg], 位置: %.3f [deg]\n" sigma_dv rad2deg(sigma_theta) rad2deg(sigma_psi) rad2deg(sigma_u)

#     # -------------------------------------------------------
#     # 2. 変数定義
#     # -------------------------------------------------------
#     @variables a_c e_c i_c omega_c Omega_c M_c t_tgt rho_sym Bc_sym dB_sym
#     @variables d_dv d_theta d_psi d_u
#     @variables dv_nom theta_nom u_nom

#     # -------------------------------------------------------
#     # 3. 分離ベクトルのモデル化 (RTN)
#     # -------------------------------------------------------
#     v_mag = dv_nom + d_dv
#     th    = theta_nom + d_theta
#     ps    = d_psi 
    
#     dv_R = v_mag * cos(ps) * cos(th)
#     dv_T = v_mag * cos(ps) * sin(th)
#     dv_N = v_mag * sin(ps)
    
#     # -------------------------------------------------------
#     # 4. PDF式(5) による Δα の計算 (近似GVE)
#     # -------------------------------------------------------
#     u_sep = u_nom + d_u
#     f_sep = u_sep - omega_c
#     n_sym = sqrt(mu_earth / a_c^3)
    
#     delta_a_pdf = (2 / n_sym) * (e_c * sin(f_sep) * dv_R + (1 + e_c * cos(f_sep)) * dv_T)
#     delta_e_pdf = (1 / (n_sym * a_c)) * (sin(f_sep) * dv_R + ((2 - e_c * cos(f_sep)) + e_c) * dv_T)
#     delta_i_pdf = (1 / (n_sym * a_c)) * (1 - e_c * cos(f_sep)) * cos(u_sep) * dv_N
#     delta_Om_pdf = (1 / (n_sym * a_c * sin(i_c))) * (1 - e_c * cos(f_sep)) * sin(u_sep) * dv_N
#     term_w_inplane = (1 / e_c) * (-cos(f_sep) * dv_R + (2 - e_c * cos(f_sep)) * sin(f_sep) * dv_T)
#     term_w_outplane = (1 / sin(i_c)) * (1 - e_c * cos(f_sep)) * sin(u_sep) * cos(i_c) * dv_N
#     delta_w_pdf = (1 / (n_sym * a_c)) * (term_w_inplane - term_w_outplane)
#     delta_M_pdf = (1 / (n_sym * a_c * e_c)) * ((cos(f_sep) - 2*e_c) * dv_R - (2 - e_c * cos(f_sep)) * sin(f_sep) * dv_T)

#     # -------------------------------------------------------
#     # 5. Deputyの軌道要素 & 初期ROE計算
#     # -------------------------------------------------------
#     ac0, ec0, ic0, wc0, Omc0, Mc0 = a_c, e_c, i_c, omega_c, Omega_c, M_c
#     ad0, ed0, id0, wd0, Omd0, Md0 = ac0 + delta_a_pdf, ec0 + delta_e_pdf, ic0 + delta_i_pdf, wc0 + delta_w_pdf, Omc0 + delta_Om_pdf, Mc0 + delta_M_pdf
    
#     roe_da = (ad0 - ac0) / ac0
#     dM, dw, dOm = Md0 - Mc0, wd0 - wc0, Omd0 - Omc0
#     roe_dl = dM + dw + dOm * cos(ic0)
#     roe_dex = ed0 * cos(wd0) - ec0 * cos(wc0)
#     roe_dey = ed0 * sin(wd0) - ec0 * sin(wc0)
#     roe_dix = id0 - ic0
#     roe_diy = dOm * sin(ic0)
    
#     roe_vec_0 = [roe_da, roe_dl, roe_dex, roe_dey, roe_dix, roe_diy, dB_sym]

#     # -------------------------------------------------------
#     # 6. STMによる伝播
#     # -------------------------------------------------------
#     omega_dot_sym, Omega_dot_sym = get_secular_j2_rates_koenig(a_c, e_c, i_c)
#     omega_c_tf = omega_c + omega_dot_sym * t_tgt
#     J_t0 = get_J_qns_augmented_koenig(omega_c)
#     J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)
#     A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(a_c, e_c, i_c, omega_c, true, true, DENSITY_MODEL_SPECIFIC, rho_sym, Bc_sym)
#     STM = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, t_tgt, e_c, true, DENSITY_MODEL_SPECIFIC)
    
#     roe_vec_f = J_tf_inv * STM * J_t0 * roe_vec_0
#     roe_names = ["δa", "δλ", "δex", "δey", "δix", "δiy"]

#     # -------------------------------------------------------
#     # 7. 数値評価用辞書
#     # -------------------------------------------------------
#     n_val = sqrt(mu_earth / a_c_stm_init^3)
#     t_tgt_val = PROPAGATION_ORBITS * 2 * pi / n_val
    
#     val_dict = Dict(
#         dv_nom => abs(nominal_dv_T), theta_nom => pi/2, u_nom => deg2rad(90.0), # u=90度(極)　
#         # theta_nom => 5*pi/8, u_nom => deg2rad(0.0),# 比較用　
#         a_c => a_c_stm_init, e_c => e_c_stm_init, i_c => i_c_stm_init, omega_c => omega_c_stm_init, M_c => M_c_stm_init,
#         t_tgt => t_tgt_val, rho_sym => RHO_LEO, Bc_sym => BC_CHIEF, dB_sym => DELTA_B_INIT,
#         d_dv => 0, d_theta => 0, d_psi => 0, d_u => 0
#     )

#     # -------------------------------------------------------
#     # 8. 感度解析実行 & データ収集
#     # -------------------------------------------------------
#     error_vars = [d_dv, d_theta, d_psi, d_u]
#     sigmas     = [sigma_dv, sigma_theta, sigma_psi, sigma_u]
#     error_labels = ["Δv誤差", "位相誤差", "軸倒れ", "位置誤差"]
    
#     all_contributions = []

#     println("\n[感度解析結果 (PDF式(5)ベース)]")
#     for i in 1:6
#         println("-"^40)
#         expr = roe_vec_f[i]
#         total_variance = 0.0
#         contributions = []

#         for (j, err_var) in enumerate(error_vars)
#             diff_expr = Symbolics.derivative(expr, err_var)
#             sens_sym = substitute(diff_expr, val_dict)
            
#             sens_val = 0.0
#             try
#                 sens_val = Float64(Symbolics.value(sens_sym))
#             catch
#                 sens_val = 0.0
#             end
            
#             # ★★★ 修正: ここで a_c を掛けてメートル単位に変換する ★★★
#             contribution_meters = abs(sens_val) * sigmas[j] * a_c_stm_init
            
#             total_variance += contribution_meters^2
            
#             push!(contributions, (error_labels[j], sens_val, contribution_meters))
#         end
        
#         push!(all_contributions, contributions)
        
#         total_sigma = sqrt(total_variance)
#         @printf "■ 最終 %s 誤差 (1σ): %.3e [m]\n" roe_names[i] total_sigma
        
#         sort!(contributions, by = x -> x[3], rev=true)
#         for (label, sens, cont) in contributions
#             if cont > 1e-12
#                 ratio = (cont^2 / total_variance) * 100
#                 @printf "  - %s: 寄与 %.2e [m] (%.1f%%)\n" label cont ratio
#             end
#         end
#     end
#     println("\n" * "="^60)
    
#     # --- 9. グラフ作成 ---
#     plot_sensitivity_contribution(all_contributions)
# end

# ==============================================================================
# [修正版] 感度解析: 数式の簡略化と表示改善 (強制表示版)
# ==============================================================================
function run_symbolic_sensitivity_analysis_pdf_eq5(nominal_dv_T::Float64)
    println("\n" * "="^60)
    println("感度解析: 時間依存性(t)の抽出と数値感度係数の算出 (表示改善版)")
    println("="^60)

    # --- 1. 誤差の設定 ---
    sigma_dv    = 0.001 * abs(nominal_dv_T)
    sigma_theta = deg2rad(1.0)
    sigma_psi   = deg2rad(1.0)
    sigma_u     = deg2rad(0.1)

    # -------------------------------------------------------
    # 2. 変数定義
    # -------------------------------------------------------
    @variables a_c e_c i_c omega_c Omega_c M_c t_tgt rho_sym Bc_sym dB_sym
    @variables d_dv d_theta d_psi d_u
    @variables dv_nom theta_nom u_nom

    # -------------------------------------------------------
    # 3. 分離ベクトルのモデル化 (RTN)
    # -------------------------------------------------------
    v_mag = dv_nom + d_dv
    th    = theta_nom + d_theta
    ps    = d_psi 
    
    dv_R = v_mag * cos(ps) * cos(th)
    dv_T = v_mag * cos(ps) * sin(th)
    dv_N = v_mag * sin(ps)
    
    # -------------------------------------------------------
    # 4. 近似GVE (PDF式5)
    # -------------------------------------------------------
    u_sep = u_nom + d_u
    f_sep = u_sep - omega_c
    n_sym = sqrt(mu_earth / a_c^3)
    
    delta_a_pdf = (2.0 / n_sym) * (e_c * sin(f_sep) * dv_R + (1.0 + e_c * cos(f_sep)) * dv_T)
    delta_e_pdf = (1.0 / (n_sym * a_c)) * (sin(f_sep) * dv_R + ((2.0 - e_c * cos(f_sep)) + e_c) * dv_T)
    delta_i_pdf = (1.0 / (n_sym * a_c)) * (1.0 - e_c * cos(f_sep)) * cos(u_sep) * dv_N
    delta_Om_pdf = (1.0 / (n_sym * a_c * sin(i_c))) * (1.0 - e_c * cos(f_sep)) * sin(u_sep) * dv_N
    
    term_w_inplane = (1.0 / e_c) * (-cos(f_sep) * dv_R + (2.0 - e_c * cos(f_sep)) * sin(f_sep) * dv_T)
    term_w_outplane = (1.0 / sin(i_c)) * (1.0 - e_c * cos(f_sep)) * sin(u_sep) * cos(i_c) * dv_N
    delta_w_pdf = (1.0 / (n_sym * a_c)) * (term_w_inplane - term_w_outplane)
    
    delta_M_pdf = (1.0 / (n_sym * a_c * e_c)) * ((cos(f_sep) - 2.0*e_c) * dv_R - (2.0 - e_c * cos(f_sep)) * sin(f_sep) * dv_T)

    # -------------------------------------------------------
    # 5. 初期ROE計算
    # -------------------------------------------------------
    ac0, ec0, ic0, wc0, Omc0, Mc0 = a_c, e_c, i_c, omega_c, Omega_c, M_c
    ad0, ed0, id0, wd0, Omd0, Md0 = ac0 + delta_a_pdf, ec0 + delta_e_pdf, ic0 + delta_i_pdf, wc0 + delta_w_pdf, Omc0 + delta_Om_pdf, Mc0 + delta_M_pdf
    
    roe_da = (ad0 - ac0) / ac0
    dM, dw, dOm = Md0 - Mc0, wd0 - wc0, Omd0 - Omc0
    roe_dl = dM + dw + dOm * cos(ic0)
    roe_dex = ed0 * cos(wd0) - ec0 * cos(wc0)
    roe_dey = ed0 * sin(wd0) - ec0 * sin(wc0)
    roe_dix = id0 - ic0
    roe_diy = dOm * sin(ic0)
    
    roe_vec_0 = [roe_da, roe_dl, roe_dex, roe_dey, roe_dix, roe_diy, dB_sym]

    # -------------------------------------------------------
    # 6. STMによる伝播
    # -------------------------------------------------------
    omega_dot_sym, Omega_dot_sym = get_secular_j2_rates_koenig(a_c, e_c, i_c)
    omega_c_tf = omega_c + omega_dot_sym * t_tgt
    J_t0 = get_J_qns_augmented_koenig(omega_c)
    J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)
    A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(a_c, e_c, i_c, omega_c, true, true, DENSITY_MODEL_SPECIFIC, rho_sym, Bc_sym)
    STM = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, t_tgt, e_c, true, DENSITY_MODEL_SPECIFIC)
    
    roe_vec_f = J_tf_inv * STM * J_t0 * roe_vec_0
    roe_names = ["δa", "δλ", "δex", "δey", "δix", "δiy"]

    # -------------------------------------------------------
    # 7. 数値評価用辞書
    # -------------------------------------------------------
    n_val = sqrt(mu_earth / a_c_stm_init^3)
    t_tgt_val = PROPAGATION_ORBITS * 2 * pi / n_val
    
    # 定数辞書 (t_tgt と 誤差変数 は除く)
    const_dict = Dict(
        dv_nom => abs(nominal_dv_T), theta_nom => pi/2, u_nom => deg2rad(90.0), 
        a_c => a_c_stm_init, e_c => e_c_stm_init, i_c => i_c_stm_init, omega_c => omega_c_stm_init, M_c => M_c_stm_init,
        rho_sym => RHO_LEO, Bc_sym => BC_CHIEF, dB_sym => DELTA_B_INIT
    )
    
    full_dict = merge(const_dict, Dict(t_tgt => t_tgt_val, d_dv => 0.0, d_theta => 0.0, d_psi => 0.0, d_u => 0.0))

    # -------------------------------------------------------
    # 8. 感度解析 & 表示
    # -------------------------------------------------------
    error_vars = [d_dv, d_theta, d_psi, d_u]
    sigmas     = [sigma_dv, sigma_theta, sigma_psi, sigma_u]
    error_labels = ["Δv誤差", "位相誤差", "軸倒れ", "位置誤差"]
    error_units = ["[m/s]", "[rad]", "[rad]", "[rad]"]
    
    all_contributions = []

    println("\n[感度解析結果: 数値係数と時間依存性]")
    
    for i in 1:6
        println("-"^40)
        println("■ 最終 $(roe_names[i])")
        expr = roe_vec_f[i]
        
        contributions = []
        total_variance = 0.0

        for (j, err_var) in enumerate(error_vars)
            diff_expr = Symbolics.derivative(expr, err_var)
            
            # 数値感度
            sens_val = 0.0
            try
                sens_sym = substitute(diff_expr, full_dict)
                sens_val = Float64(Symbolics.value(sens_sym))
            catch
                sens_val = 0.0
            end
            
            slope_meters = sens_val * a_c_stm_init
            contribution_meters = abs(slope_meters) * sigmas[j]
            total_variance += contribution_meters^2
            
            # 時間依存性の表示
            time_func_str = "N/A"
            try
                # 定数代入
                zero_error_dict = Dict(d_dv => 0.0, d_theta => 0.0, d_psi => 0.0, d_u => 0.0)
                combined_dict = merge(const_dict, zero_error_dict)
                time_func_sym = substitute(diff_expr, combined_dict)
                
                # 数値を丸めて文字列化するための簡易処理
                # (正規表現などで長い小数を短くする)
                raw_str = string(time_func_sym)
                
                # 非常に長い式の場合、主要項だけを取り出すのは難しいが、
                # そのまま表示してユーザーに判断させる
                time_func_str = raw_str
            catch
            end
            
            push!(contributions, (error_labels[j], slope_meters, contribution_meters, time_func_str))
        end
        
        sort!(contributions, by = x -> x[3], rev=true)
        
        for (label, slope, cont, time_func) in contributions
            if cont > 1e-12
                @printf "  ・%s:\n" label
                @printf "      数値感度: %.2e [m / %s]\n" slope (error_units[findfirst(==(label), error_labels)])
                @printf "      寄与(1σ): %.2e [m]\n" cont
                
                # ★修正: 長さ制限を撤廃し、改行して表示
                println("      時間依存式:")
                # 長すぎる場合は適当に改行を入れるなどの処理があってもいいが、まずは生で表示
                if length(time_func) > 1000
                     println("      (式が極めて長いため先頭のみ表示): ", time_func[1:200], "...")
                else
                     println("      ", time_func)
                end
            end
        end
        
        # グラフ用データ
        graph_data = []
        for (label, slope, cont, _) in contributions
             push!(graph_data, (label, 0.0, cont, "")) 
        end
        push!(all_contributions, graph_data)
    end
    println("\n" * "="^60)
    
    plot_sensitivity_contribution(all_contributions)
end

# # ==============================================================================
# # [修正版] 指定ドリフト量達成のためのΔv逆算と安全性確認 (レイアウト調整版)
# # ==============================================================================
# function run_drift_design_simulation()
#     println("\n" * "="^60)
#     println("指定ドリフト量達成のためのΔv逆算と時刻歴シミュレーション (レイアウト調整版)")
#     println("="^60)

#     # --- 1. 目標の設定 ---
#     target_dlambda_meters = 50.0   
#     target_time_days = 1.0         
    
#     # 安全確保のための半径方向分離速度
#     dv_R_input = 0.00 # [m/s] 
    
#     target_time_sec = target_time_days * 86400.0
#     println("条件:")
#     println("  到達時間: $target_time_days [day] ($target_time_sec sec)")
#     println("  目標δλ  : $target_dlambda_meters [m]")
#     println("  追加ΔvR : $dv_R_input [m/s]")

#     # --- 2. パラメータ準備 ---
#     n_init = sqrt(mu_earth / a_c_stm_init^3)
#     oe_c = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, n_init, M_c_stm_init)
#     rho = RHO_LEO; Bc = BC_CHIEF; dB = DELTA_B_INIT

#     # --- 3. STMを用いたΔvTの逆算 ---
#     omega_c_t0 = oe_c.omega
#     omega_dot, Omega_dot = get_secular_j2_rates_koenig(oe_c.a, oe_c.e, oe_c.i)
#     omega_c_tf = omega_c_t0 + omega_dot * target_time_sec
    
#     J_t0 = get_J_qns_augmented_koenig(omega_c_t0)
#     J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)
#     A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c.a, oe_c.e, oe_c.i, omega_c_t0, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
#     STM = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, target_time_sec, oe_c.e, true, DENSITY_MODEL_SPECIFIC)
#     M_trans = J_tf_inv * STM * J_t0

#     u_0 = oe_c.M + oe_c.omega 
#     C_gve = 1.0 / (oe_c.n * oe_c.a)
    
#     da_from_R  = 0.0
#     dl_from_R  = -2 * C_gve * dv_R_input
#     dex_from_R = C_gve * sin(u_0) * dv_R_input
#     dey_from_R = -C_gve * cos(u_0) * dv_R_input
    
#     term_drift_drag = M_trans[2,7] * dB
#     term_drift_R = M_trans[2,2]*dl_from_R + M_trans[2,3]*dex_from_R + M_trans[2,4]*dey_from_R
#     term_coeff_T = M_trans[2,1]*(2*C_gve) + M_trans[2,3]*(2*C_gve*cos(u_0)) + M_trans[2,4]*(2*C_gve*sin(u_0))
    
#     target_dl_rad = target_dlambda_meters / oe_c.a
#     req_dv_T = (target_dl_rad - term_drift_R - term_drift_drag) / term_coeff_T
    
#     println("\n逆算結果:")
#     println("  必要な分離速度 ΔvT: $(req_dv_T) [m/s] ($(req_dv_T*1000) mm/s)")

#     # --- 4. 時刻歴シミュレーション ---
#     println("\nシミュレーション実行中...")
    
#     da_0  = 2*C_gve*req_dv_T
#     dl_0  = -2*C_gve*dv_R_input
#     dex_0 = C_gve*(sin(u_0)*dv_R_input + 2*cos(u_0)*req_dv_T)
#     dey_0 = C_gve*(-cos(u_0)*dv_R_input + 2*sin(u_0)*req_dv_T)
#     roe_0 = SVector(da_0, dl_0, dex_0, dey_0, 0.0, 0.0, dB)
    
#     times = 0:60:target_time_sec
#     hist_t = Float64[]
#     hist_dl = Float64[]
#     hist_dist = Float64[] 
#     hist_x = Float64[]
#     hist_y = Float64[]
#     hist_z = Float64[]
    
#     min_dist = Inf
#     min_dist_time = 0.0
    
#     for t in times
#         omega_c_curr = omega_c_t0 + omega_dot * t
#         J_inv_curr = get_J_qns_inv_augmented_koenig(omega_c_curr)
#         STM_curr = get_STM_prime_qns_augmented_koenig_model_selectable(
#             A_kep, A_j2, A_drag, t, oe_c.e, true, DENSITY_MODEL_SPECIFIC
#         )
#         roe_curr = J_inv_curr * STM_curr * J_t0 * roe_0
        
#         current_RAAN = mod(oe_c.RAAN + Omega_dot * t, 2*pi)
#         current_omega = mod(oe_c.omega + omega_dot * t, 2*pi)
#         current_M     = mod(oe_c.M + oe_c.n * t, 2*pi)
        
#         oe_c_curr = OrbitalElementsClassical(oe_c.a, oe_c.e, oe_c.i, current_RAAN, current_omega, 0.0, oe_c.n, current_M)
#         oe_d_curr = final_roe_to_deputy_oe(oe_c_curr, roe_curr)
        
#         sv_c_vec = orbital_elements_to_sv(oe_c_curr)
#         sv_d_vec = orbital_elements_to_sv(oe_d_curr)
        
#         r_c_eci = SVector(sv_c_vec[1], sv_c_vec[2], sv_c_vec[3])
#         v_c_eci = SVector(sv_c_vec[4], sv_c_vec[5], sv_c_vec[6])
#         r_d_eci = SVector(sv_d_vec[1], sv_d_vec[2], sv_d_vec[3])
        
#         dr_eci = r_d_eci - r_c_eci
#         dr_rtn = eci_to_rtn(r_c_eci, v_c_eci, dr_eci)
        
#         dist = norm(dr_rtn)
        
#         push!(hist_t, t/86400.0)
#         push!(hist_dl, roe_curr[2] * oe_c.a)
#         push!(hist_dist, dist)
#         push!(hist_x, dr_rtn[1])
#         push!(hist_y, dr_rtn[2])
#         push!(hist_z, dr_rtn[3])
        
#         if dist < min_dist
#             min_dist = dist
#             min_dist_time = t
#         end
#     end
    
#     println("\n安全性評価:")
#     println("  最小接近距離: $(min_dist) [m] (at $(min_dist_time/60) min)")

#     # --- 5. プロット作成 ---
#     # レイアウト調整設定
#     default(
#         dpi=300, 
#         guidefontsize=10, 
#         tickfontsize=8, 
#         legendfontsize=8, 
#         titlefontsize=10,
#         margin=10Plots.mm # 全体の余白を確保
#     )
    
#     # グラフ1: δλ
#     p1 = plot(hist_t, hist_dl, label="δλ", 
#         xlabel="Time [days]", ylabel="δλ [m]", title="Along-Track Drift", lw=2, legend=:topleft)
#     scatter!(p1, [target_time_days], [target_dlambda_meters], label="Target", marker=:diamond, color=:red)
    
#     # グラフ2: 相対距離 (ライン削除済み)
#     p2 = plot(hist_t, hist_dist, label="3D Distance", 
#         xlabel="Time [days]", ylabel="Distance [m]", title="Relative Distance", lw=2, color=:green, legend=:topleft)

#     # グラフ3: RT面内軌跡 (全期間プロットへ変更)
#     p3 = plot(hist_y, hist_x, 
#         xlabel="Along-Track (y) [m]", ylabel="Radial (x) [m]", 
#         title="RT-Plane Trajectory", aspect_ratio=:equal, lw=1.5, legend=:topright)
#     # Chiefマーカー削除済み
    
#     # グラフ4: Z軸 (全期間プロットへ変更)
#     p4 = plot(hist_t, hist_z,
#         xlabel="Time [days]", ylabel="Cross-Track (z) [m]",
#         title="Cross-Track Motion", lw=1.5, color=:purple, legend=:topright)

#     # レイアウト配置
#     p_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
    
#     timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
#     filename = "drift_safety_check_layout_$(timestamp).png"
#     savefig(p_combined, filename)
#     println("グラフを保存しました: $filename")
#     display(p_combined)
# end

# ==============================================================================
# [修正版] 指定ドリフト量シミュレーション (バイアス補正付き完全版)
# ==============================================================================
function run_drift_design_simulation()
    println("\n" * "="^60)
    println("指定ドリフト量達成のためのΔv逆算と時刻歴シミュレーション (バイアス補正版)")
    println("="^60)

    # --- 1. 目標の設定 ---
    target_dlambda_meters = -50.0   
    target_time_days = 1.0         
    dv_R_input = 0.0 # [m/s] 
    
    target_time_sec = target_time_days * 86400.0

    # --- 2. パラメータ準備 ---
    n_init = sqrt(mu_earth / a_c_stm_init^3)
    oe_c = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, n_init, M_c_stm_init)
    rho = RHO_LEO; Bc = BC_CHIEF; dB = DELTA_B_INIT

    # --- 3. STMを用いたΔvTの逆算 ---
    omega_c_t0 = oe_c.omega
    omega_dot, Omega_dot = get_secular_j2_rates_koenig(oe_c.a, oe_c.e, oe_c.i)
    omega_c_tf = omega_c_t0 + omega_dot * target_time_sec
    
    J_t0 = get_J_qns_augmented_koenig(omega_c_t0)
    J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)
    A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c.a, oe_c.e, oe_c.i, omega_c_t0, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
    STM = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, target_time_sec, oe_c.e, true, DENSITY_MODEL_SPECIFIC)
    M_trans = J_tf_inv * STM * J_t0

    u_0 = oe_c.M + oe_c.omega 
    C_gve = 1.0 / (oe_c.n * oe_c.a)
    
    da_from_R  = 0.0
    dl_from_R  = -2 * C_gve * dv_R_input
    dex_from_R = C_gve * sin(u_0) * dv_R_input
    dey_from_R = -C_gve * cos(u_0) * dv_R_input
    
    term_drift_drag = M_trans[2,7] * dB
    term_drift_R = M_trans[2,2]*dl_from_R + M_trans[2,3]*dex_from_R + M_trans[2,4]*dey_from_R
    term_coeff_T = M_trans[2,1]*(2*C_gve) + M_trans[2,3]*(2*C_gve*cos(u_0)) + M_trans[2,4]*(2*C_gve*sin(u_0))
    
    target_dl_rad = target_dlambda_meters / oe_c.a
    req_dv_T = (target_dl_rad - term_drift_R - term_drift_drag) / term_coeff_T
    
    println("\n[逆算結果]")
    println("  必要な分離速度 ΔvT: $(req_dv_T) [m/s]")

    # --- 初期状態 (t=0) の計算 ---
    da_0  = 2*C_gve*req_dv_T
    dl_0  = -2*C_gve*dv_R_input
    dex_0 = C_gve*(sin(u_0)*dv_R_input + 2*cos(u_0)*req_dv_T)
    dey_0 = C_gve*(-cos(u_0)*dv_R_input + 2*sin(u_0)*req_dv_T)
    roe_0 = SVector(da_0, dl_0, dex_0, dey_0, 0.0, 0.0, dB)

    oe_d_0 = final_roe_to_deputy_oe(oe_c, roe_0)
    
    sv_c_0 = orbital_elements_to_sv(oe_c)
    sv_d_0 = orbital_elements_to_sv(oe_d_0)
    
    r_c_0 = SVector(sv_c_0[1], sv_c_0[2], sv_c_0[3])
    v_c_0 = SVector(sv_c_0[4], sv_c_0[5], sv_c_0[6])
    r_d_0 = SVector(sv_d_0[1], sv_d_0[2], sv_d_0[3])
    v_d_0 = SVector(sv_d_0[4], sv_d_0[5], sv_d_0[6])
    
    dr_eci_0 = r_d_0 - r_c_0
    dv_eci_0 = v_d_0 - v_c_0
    
    dr_rtn_0 = eci_to_rtn(r_c_0, v_c_0, dr_eci_0)
    
    # ★バイアス補正用: 初期位置ズレを記録
    bias_offset = dr_rtn_0
    
    dv_rtn_0_inertial = eci_to_rtn(r_c_0, v_c_0, dv_eci_0)

    # Hill速度
    h_vec_0 = cross(r_c_0, v_c_0)
    omega_vec_eci = h_vec_0 / dot(r_c_0, r_c_0)
    omega_norm = norm(omega_vec_eci)
    vx_hill_0 = dv_rtn_0_inertial[1] + omega_norm * dr_rtn_0[2]
    vy_hill_0 = dv_rtn_0_inertial[2] - omega_norm * dr_rtn_0[1]
    vz_hill_0 = dv_rtn_0_inertial[3]

    println("\n--- [Start] 初期状態 (t = 0.0 days) ---")
    # 表示上はバイアス補正後の値（理論値 0）を表示
    println("  相対位置 (RTN, Bias Corrected) [m]:")
    @printf "    R (x): %10.4f\n" (dr_rtn_0[1] - bias_offset[1])
    @printf "    T (y): %10.4f\n" (dr_rtn_0[2] - bias_offset[2])
    @printf "    N (z): %10.4f\n" (dr_rtn_0[3] - bias_offset[3])
    println("  (補正前生データ: $(dr_rtn_0))")
    println("  相対速度 (RTN: 慣性速度差) [m/s]:")
    @printf "    vR:    %10.6f\n" dv_rtn_0_inertial[1]
    @printf "    vT:    %10.6f\n" dv_rtn_0_inertial[2]
    @printf "    vN:    %10.6f\n" dv_rtn_0_inertial[3]

    # --- 4. 時刻歴シミュレーション ---
    times = 0:60:target_time_sec
    hist_t = Float64[]
    hist_dl = Float64[]
    hist_dist = Float64[] 
    hist_x = Float64[]; hist_y = Float64[]; hist_z = Float64[]
    
    for t in times
        omega_c_curr = omega_c_t0 + omega_dot * t
        J_inv_curr = get_J_qns_inv_augmented_koenig(omega_c_curr)
        STM_curr = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, t, oe_c.e, true, DENSITY_MODEL_SPECIFIC)
        roe_curr = J_inv_curr * STM_curr * J_t0 * roe_0
        
        current_RAAN = mod(oe_c.RAAN + Omega_dot * t, 2*pi)
        current_omega = mod(oe_c.omega + omega_dot * t, 2*pi)
        current_M     = mod(oe_c.M + oe_c.n * t, 2*pi)
        
        oe_c_curr = OrbitalElementsClassical(oe_c.a, oe_c.e, oe_c.i, current_RAAN, current_omega, 0.0, oe_c.n, current_M)
        oe_d_curr = final_roe_to_deputy_oe(oe_c_curr, roe_curr)
        
        sv_c_vec = orbital_elements_to_sv(oe_c_curr)
        sv_d_vec = orbital_elements_to_sv(oe_d_curr)
        
        r_c_eci = SVector(sv_c_vec[1], sv_c_vec[2], sv_c_vec[3])
        v_c_eci = SVector(sv_c_vec[4], sv_c_vec[5], sv_c_vec[6])
        r_d_eci = SVector(sv_d_vec[1], sv_d_vec[2], sv_d_vec[3])
        
        dr_eci = r_d_eci - r_c_eci
        dr_rtn_raw = eci_to_rtn(r_c_eci, v_c_eci, dr_eci)
        
        # ★バイアス補正: 初期ズレ分を差し引く
        dr_rtn = dr_rtn_raw - bias_offset
        
        dist = norm(dr_rtn)
        
        push!(hist_t, t/86400.0)
        push!(hist_dl, roe_curr[2] * oe_c.a)
        push!(hist_dist, dist)
        push!(hist_x, dr_rtn[1]); push!(hist_y, dr_rtn[2]); push!(hist_z, dr_rtn[3])
    end
    
    # --- 終了状態 ---
    t_final = times[end]
    omega_c_final = omega_c_t0 + omega_dot * t_final
    J_inv_final = get_J_qns_inv_augmented_koenig(omega_c_final)
    STM_final = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, t_final, oe_c.e, true, DENSITY_MODEL_SPECIFIC)
    roe_final = J_inv_final * STM_final * J_t0 * roe_0

    current_RAAN_f = mod(oe_c.RAAN + Omega_dot * t_final, 2*pi)
    current_omega_f = mod(oe_c.omega + omega_dot * t_final, 2*pi)
    current_M_f     = mod(oe_c.M + oe_c.n * t_final, 2*pi)
    oe_c_f = OrbitalElementsClassical(oe_c.a, oe_c.e, oe_c.i, current_RAAN_f, current_omega_f, 0.0, oe_c.n, current_M_f)
    oe_d_f = final_roe_to_deputy_oe(oe_c_f, roe_final)
    
    sv_c_f = orbital_elements_to_sv(oe_c_f); sv_d_f = orbital_elements_to_sv(oe_d_f)
    
    r_c_f = SVector(sv_c_f[1], sv_c_f[2], sv_c_f[3])
    v_c_f = SVector(sv_c_f[4], sv_c_f[5], sv_c_f[6])
    r_d_f = SVector(sv_d_f[1], sv_d_f[2], sv_d_f[3])
    v_d_f = SVector(sv_d_f[4], sv_d_f[5], sv_d_f[6])
    
    dr_f = r_d_f - r_c_f
    dv_f = v_d_f - v_c_f
    dr_rtn_f_raw = eci_to_rtn(r_c_f, v_c_f, dr_f)
    
    # ★ここでも補正
    dr_rtn_f = dr_rtn_f_raw - bias_offset
    
    dv_rtn_f_inertial = eci_to_rtn(r_c_f, v_c_f, dv_f)
    
    h_vec_f = cross(r_c_f, v_c_f)
    omega_norm_f = norm(h_vec_f) / dot(r_c_f, r_c_f)
    vx_hill_f = dv_rtn_f_inertial[1] + omega_norm_f * dr_rtn_f[2]
    vy_hill_f = dv_rtn_f_inertial[2] - omega_norm_f * dr_rtn_f[1]
    vz_hill_f = dv_rtn_f_inertial[3]

    println("\n--- [End] 終了状態 (t = $(round(t_final/86400.0, digits=2)) days) ---")
    println("  相対位置 (RTN, Bias Corrected) [m]:")
    @printf "    R (x): %10.4f\n" dr_rtn_f[1]
    @printf "    T (y): %10.4f\n" dr_rtn_f[2]
    @printf "    N (z): %10.4f\n" dr_rtn_f[3]
    println("  相対速度 (RTN: 慣性速度差) [m/s]:")
    @printf "    vR:    %10.6f\n" dv_rtn_f_inertial[1]
    @printf "    vT:    %10.6f\n" dv_rtn_f_inertial[2]
    @printf "    vN:    %10.6f\n" dv_rtn_f_inertial[3]
    println("="^60)

    # --- プロット作成 (変更なし) ---
    default(dpi=300, guidefontsize=10, tickfontsize=8, legendfontsize=8, titlefontsize=10, margin=10Plots.mm)
    p1 = plot(hist_t, hist_dl, label="δλ", xlabel="Time [days]", ylabel="δλ [m]", title="Along-Track Drift", lw=2, legend=:topleft)
    scatter!(p1, [target_time_days], [target_dlambda_meters], label="Target", marker=:diamond, color=:red)
    p2 = plot(hist_t, hist_dist, label="3D Distance", xlabel="Time [days]", ylabel="Distance [m]", title="Relative Distance", lw=2, color=:green, legend=:topleft)
    p3 = plot(hist_y, hist_x, xlabel="Along-Track (y) [m]", ylabel="Radial (x) [m]", title="RT-Plane Trajectory", aspect_ratio=:equal, lw=1.5, legend=:topright)
    p4 = plot(hist_t, hist_z, xlabel="Time [days]", ylabel="Cross-Track (z) [m]", title="Cross-Track Motion", lw=1.5, color=:purple, legend=:topright)
    p_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "drift_simulation_strict_corrected_$(timestamp).png"
    savefig(p_combined, filename)
    println("グラフを保存しました: $filename")
    display(p_combined)
end

# ==============================================================================
# 実行ブロック (既存の呼び出しの後に追加)
# ==============================================================================

# 1. 目標δa=0となる分離マヌーバを計算・検証し、その公称Δvを取得
nominal_dv_T = verify_drag_cancellation_separation()

# 2. その公称値周りでの感度解析を実行
run_symbolic_sensitivity_analysis(nominal_dv_T)

# --- 実行 ---
run_symbolic_sensitivity_analysis_pdf_eq5(nominal_dv_T)

run_target_search()

run_reachable_set_analysis()

run_global_optimization()

# 実行
compare_optimal_and_robust()

analyze_attitude_requirements()

run_comparison_analysis()

debug_single_correction_case()

# 依頼用
run_drift_design_simulation()

# 以下依頼用
# ヘルパー関数: 状態保存
function save_state!(ht, hdl, hda, hdist, hx, hy, hz, hphase, t, roe, oe_c, bias, phase_sym)
    oe_d = final_roe_to_deputy_oe(oe_c, roe)
    sv_c = orbital_elements_to_sv(oe_c); sv_d = orbital_elements_to_sv(oe_d)
    r_c = SVector(sv_c[1], sv_c[2], sv_c[3]); v_c = SVector(sv_c[4], sv_c[5], sv_c[6])
    r_d = SVector(sv_d[1], sv_d[2], sv_d[3])
    dr_rtn = eci_to_rtn(r_c, v_c, r_d - r_c) - bias
    
    push!(ht, t)
    push!(hdl, roe[2] * oe_c.a)
    push!(hda, roe[1] * oe_c.a)
    push!(hdist, norm(dr_rtn))
    push!(hx, dr_rtn[1]); push!(hy, dr_rtn[2]); push!(hz, dr_rtn[3])
    push!(hphase, phase_sym)
end

# ==============================================================================
#  分離からδa制御収束まで(δiyの振動が(よく見ると)発散)
# ==============================================================================
function run_station_keeping_simulation()
    println("\n" * "="^60)
    println("分離から制御収束までの全行程シミュレーション (グラフ追加版)")
    println("="^60)

    # --- 0. 設定 ---
    target_dlambda_meters = -60.0   
    drift_duration_days = 1.0         
    dv_R_input = 0.0 
    
    drift_duration_sec = drift_duration_days * 86400.0
    control_duration_orbits = 3.0

    # --- 1. 初期パラメータとΔv設計 ---
    n_init = sqrt(mu_earth / a_c_stm_init^3)
    oe_c_init = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, n_init, M_c_stm_init)
    rho = RHO_LEO; Bc = BC_CHIEF; dB_natural = DELTA_B_INIT

    # ΔvTの逆算
    omega_c_t0 = oe_c_init.omega
    omega_dot, Omega_dot = get_secular_j2_rates_koenig(oe_c_init.a, oe_c_init.e, oe_c_init.i)
    omega_c_tf = omega_c_t0 + omega_dot * drift_duration_sec
    
    J_t0 = get_J_qns_augmented_koenig(omega_c_t0)
    J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)
    A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_init.a, oe_c_init.e, oe_c_init.i, omega_c_t0, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
    STM_drift = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, drift_duration_sec, oe_c_init.e, true, DENSITY_MODEL_SPECIFIC)
    M_trans = J_tf_inv * STM_drift * J_t0
    u_0 = oe_c_init.M + oe_c_init.omega 
    C_gve = 1.0 / (oe_c_init.n * oe_c_init.a)
    
    da_from_R, dl_from_R = 0.0, -2*C_gve*dv_R_input
    dex_from_R, dey_from_R = C_gve*sin(u_0)*dv_R_input, -C_gve*cos(u_0)*dv_R_input
    
    term_drift_drag = M_trans[2,7] * dB_natural
    term_drift_R = M_trans[2,2]*dl_from_R + M_trans[2,3]*dex_from_R + M_trans[2,4]*dey_from_R
    term_coeff_T = M_trans[2,1]*(2*C_gve) + M_trans[2,3]*(2*C_gve*cos(u_0)) + M_trans[2,4]*(2*C_gve*sin(u_0))
    
    req_dv_T = (target_dlambda_meters / oe_c_init.a - term_drift_R - term_drift_drag) / term_coeff_T
    
    println("設計結果:")
    println("  必要分離速度 ΔvT: $(req_dv_T) [m/s]")

    # --- 2. シミュレーション準備 ---
    da_0  = 2*C_gve*req_dv_T
    dl_0  = -2*C_gve*dv_R_input
    dex_0 = C_gve*(sin(u_0)*dv_R_input + 2*cos(u_0)*req_dv_T)
    dey_0 = C_gve*(-cos(u_0)*dv_R_input + 2*sin(u_0)*req_dv_T)
    # roe_curr = SVector(da_0, dl_0, dex_0, dey_0, 0.0, 0.0, dB_natural)
    
    # δix試しにいれる
    dix_input = 0.0 / oe_c_init.a
    roe_curr = SVector(da_0, dl_0, dex_0, dey_0, dix_input, 0.0, dB_natural)

    oe_c_curr = oe_c_init
    current_time = 0.0

    hist_t = Float64[]
    hist_dl = Float64[]
    hist_da = Float64[]
    hist_dist = Float64[]
    hist_x = Float64[]; hist_y = Float64[]; hist_z = Float64[]
    hist_db = Float64[] # ★追加: δB履歴
    hist_phase = Symbol[] 

    # バイアス補正用の初期位置計算
    oe_d_0 = final_roe_to_deputy_oe(oe_c_curr, roe_curr)
    sv_c_0 = orbital_elements_to_sv(oe_c_curr)
    sv_d_0 = orbital_elements_to_sv(oe_d_0)
    
    r_c_0 = SVector(sv_c_0[1], sv_c_0[2], sv_c_0[3])
    v_c_0 = SVector(sv_c_0[4], sv_c_0[5], sv_c_0[6])
    r_d_0 = SVector(sv_d_0[1], sv_d_0[2], sv_d_0[3])
    
    dr_eci_0 = r_d_0 - r_c_0
    dr_rtn_0_raw = eci_to_rtn(r_c_0, v_c_0, dr_eci_0)
    bias_offset = dr_rtn_0_raw

    # ヘルパー関数: 状態保存 (δB引数を追加)
    function record_state!(t, roe, oe_c, bias, phase, db_val)
        oe_d = final_roe_to_deputy_oe(oe_c, roe)
        sv_c = orbital_elements_to_sv(oe_c); sv_d = orbital_elements_to_sv(oe_d)
        r_c = SVector(sv_c[1], sv_c[2], sv_c[3]); v_c = SVector(sv_c[4], sv_c[5], sv_c[6])
        r_d = SVector(sv_d[1], sv_d[2], sv_d[3])
        dr_rtn = eci_to_rtn(r_c, v_c, r_d - r_c) - bias
        
        push!(hist_t, t)
        push!(hist_dl, roe[2] * oe_c.a)
        push!(hist_da, roe[1] * oe_c.a)
        push!(hist_dist, norm(dr_rtn))
        push!(hist_x, dr_rtn[1]); push!(hist_y, dr_rtn[2]); push!(hist_z, dr_rtn[3])
        push!(hist_db, db_val) # ★追加
        push!(hist_phase, phase)
    end

    # --- 3. フェーズ1: ドリフト期間 ---
    println("Phase 1: ドリフトシミュレーション実行中...")
    dt_drift = 600.0
    steps_drift = Int(drift_duration_sec / dt_drift)
    
    for i in 1:steps_drift
        record_state!(current_time, roe_curr, oe_c_curr, bias_offset, :Drift, roe_curr[7])

        omega_c = oe_c_curr.omega
        omega_dot_step, Omega_dot_step = get_secular_j2_rates_koenig(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i)
        omega_c_next = omega_c + omega_dot_step * dt_drift
        J_t0_step = get_J_qns_augmented_koenig(omega_c)
        J_tf_inv_step = get_J_qns_inv_augmented_koenig(omega_c_next)
        A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, omega_c, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
        STM_step = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, dt_drift, oe_c_curr.e, true, DENSITY_MODEL_SPECIFIC)
        roe_next = J_tf_inv_step * STM_step * J_t0_step * roe_curr
        
        roe_curr = SVector{7,Float64}(Float64.(roe_next))
        new_M = mod(oe_c_curr.M + oe_c_curr.n * dt_drift, 2*pi)
        new_omega = mod(oe_c_curr.omega + omega_dot_step * dt_drift, 2*pi)
        new_RAAN = mod(oe_c_curr.RAAN + Omega_dot_step * dt_drift, 2*pi)
        oe_c_curr = OrbitalElementsClassical(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, new_RAAN, new_omega, 0.0, oe_c_curr.n, new_M)
        current_time += dt_drift
    end

    println("  -> 1日後の到達地点 δλ: $(roe_curr[2] * oe_c_curr.a) [m]")

    # --- 4. フェーズ2: 制御期間 ---
    println("Phase 2: 定点保持制御シミュレーション実行中...")
    
    control_duration_sec = control_duration_orbits * (2*pi/oe_c_curr.n)
    dt_control = SIM_SEGMENT_ORBITS * (2*pi/oe_c_curr.n)
    steps_control = Int(control_duration_sec / dt_control)
    
    target_roe_control = SVector(0.0, target_dlambda_meters/oe_c_init.a, roe_curr[3], roe_curr[4], roe_curr[5], roe_curr[6], 0.0)
    current_delta_B = roe_curr[7]
    alpha = 1.0 - exp(-dt_control / ATTITUDE_CHANGE_TIMECONSTANT_SEC)
    
    for i in 1:steps_control
        record_state!(current_time, roe_curr, oe_c_curr, bias_offset, :Control, current_delta_B)

        A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, oe_c_curr.omega, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
        error_da_meters = (roe_curr[1] - target_roe_control[1]) * oe_c_curr.a
        GAIN = 1.5
        delta_B_req = error_da_meters * GAIN
        delta_B_command = clamp(delta_B_req, DELTA_B_MIN, DELTA_B_MAX)
        
        omega_dot_step, Omega_dot_step = get_secular_j2_rates_koenig(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i)
        omega_c_next = oe_c_curr.omega + omega_dot_step * dt_control
        J_t0_step = get_J_qns_augmented_koenig(oe_c_curr.omega)
        J_tf_inv_step = get_J_qns_inv_augmented_koenig(omega_c_next)
        STM_step = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, dt_control, oe_c_curr.e, true, DENSITY_MODEL_SPECIFIC)
        
        # 入力δBを反映して伝播
        roe_vec_input = SVector(roe_curr[1], roe_curr[2], roe_curr[3], roe_curr[4], roe_curr[5], roe_curr[6], current_delta_B)
        roe_next = J_tf_inv_step * STM_step * J_t0_step * roe_vec_input
        
        roe_curr = SVector{7,Float64}(Float64.(roe_next))
        new_M = mod(oe_c_curr.M + oe_c_curr.n * dt_control, 2*pi)
        new_omega = mod(oe_c_curr.omega + omega_dot_step * dt_control, 2*pi)
        new_RAAN = mod(oe_c_curr.RAAN + Omega_dot_step * dt_control, 2*pi)
        oe_c_curr = OrbitalElementsClassical(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, new_RAAN, new_omega, 0.0, oe_c_curr.n, new_M)
        current_delta_B = alpha * delta_B_command + (1.0 - alpha) * current_delta_B
        current_time += dt_control
    end
    record_state!(current_time, roe_curr, oe_c_curr, bias_offset, :Control, current_delta_B)

    println("  -> 最終到達地点 δλ: $(roe_curr[2] * oe_c_curr.a) [m]")

    # --- 5. プロット作成 (6分割) ---
    default(dpi=300, guidefontsize=9, tickfontsize=7, legendfontsize=7, titlefontsize=10, margin=3Plots.mm)
    
    t_days = hist_t ./ 86400.0

    # P1: δλ
    p1 = plot(t_days, hist_dl, label="δλ", xlabel="Time [days]", ylabel="δλ [m]", title="Along-Track Drift", lw=2, legend=:topleft, bottom_margin=20Plots.mm, right_margin=20Plots.mm)
    vline!(p1, [drift_duration_days], label="Control", color=:gray, ls=:dash)
    
    # P2: δa
    p2 = plot(t_days, hist_da, label="δa", xlabel="Time [days]", ylabel="δa [m]", title="Semi-Major Axis Diff", lw=2, color=:blue, legend=:outertopright, bottom_margin=20Plots.mm)
    vline!(p2, [drift_duration_days], label=false, color=:gray, ls=:dash)

    # P3: RT面内軌跡
    p3 = plot(hist_y, hist_x, label=false, xlabel="y [m]", ylabel="x [m]", title="RT-Plane Trajectory", aspect_ratio=:equal, lw=0.5, color=:gray, legend=:outertopright, top_margin=20Plots.mm, bottom_margin=20Plots.mm, right_margin=20Plots.mm)
    drift_idx = findall(x -> x == :Drift, hist_phase)
    ctrl_idx = findall(x -> x == :Control, hist_phase)
    if !isempty(ctrl_idx); push!(drift_idx, ctrl_idx[1]); end
    plot!(p3, hist_y[drift_idx], hist_x[drift_idx], label="Drift", lw=1.5, color=:blue)
    plot!(p3, hist_y[ctrl_idx], hist_x[ctrl_idx], label="Control", lw=2.0, color=:blue)
    scatter!(p3, [0], [0], label="Chief", marker=:circle, color=:black)
    
    # P4: 相対距離
    p4 = plot(t_days, hist_dist, label="Distance", xlabel="Time [days]", ylabel="Dist [m]", title="Relative Distance", lw=2, color=:green, legend=:topleft, top_margin=20Plots.mm, bottom_margin=20Plots.mm)
    vline!(p4, [drift_duration_days], label=false, color=:gray, ls=:dash)

    # P5: δB (制御入力) ★追加
    p5 = plot(t_days, hist_db, label="δB", xlabel="Time [days]", ylabel="δB [-]", title="Control Input", lw=2, color=:orange, legend=:topright, top_margin=20Plots.mm, right_margin=20Plots.mm)
    vline!(p5, [drift_duration_days], label=false, color=:gray, ls=:dash)

    # P6: Z軸 (面外位置) ★追加
    p6 = plot(t_days, hist_z, label="z", xlabel="Time [days]", ylabel="z [m]", title="Cross-Track Motion", lw=1.5, color=:purple, legend=:outertopright, top_margin=20Plots.mm)
    vline!(p6, [drift_duration_days], label=false, color=:gray, ls=:dash)

    # 3x2 レイアウト
    p_combined = plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(1100, 1200)) # 横幅も少し拡大    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "full_mission_6panel_$(timestamp).png"
    savefig(p_combined, filename)
    println("グラフを保存しました: $filename")
    display(p_combined)
end

# # ==============================================================================
# #  分離からδaδiy制御収束まで(δiyは収束(怪しい)するがδλは拡大)
# # ==============================================================================
# function run_station_keeping_simulation()
#     println("\n" * "="^60)
#     println("定点保持制御シミュレーション (δiyドリフト抑制・Coupled Control)")
#     println("="^60)

#     # --- 0. 設定 ---
#     target_dlambda_meters = -60.0 # ユーザー設定
#     drift_duration_days = 1.0         
#     dv_R_input = 0.00 
    
#     drift_duration_sec = drift_duration_days * 86400.0
#     control_duration_orbits = 100.0 # 長期制御で効果を見る

#     # --- 1. 初期パラメータとΔv設計 ---
#     n_init = sqrt(mu_earth / a_c_stm_init^3)
#     oe_c_init = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, n_init, M_c_stm_init)
#     rho = RHO_LEO; Bc = BC_CHIEF; dB_natural = DELTA_B_INIT

#     omega_c_t0 = oe_c_init.omega
#     omega_dot, Omega_dot = get_secular_j2_rates_koenig(oe_c_init.a, oe_c_init.e, oe_c_init.i)
#     omega_c_tf = omega_c_t0 + omega_dot * drift_duration_sec
    
#     J_t0 = get_J_qns_augmented_koenig(omega_c_t0)
#     J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)
#     A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_init.a, oe_c_init.e, oe_c_init.i, omega_c_t0, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
#     STM_drift = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, drift_duration_sec, oe_c_init.e, true, DENSITY_MODEL_SPECIFIC)
#     M_trans = J_tf_inv * STM_drift * J_t0
#     u_0 = oe_c_init.M + oe_c_init.omega 
#     C_gve = 1.0 / (oe_c_init.n * oe_c_init.a)
    
#     da_from_R, dl_from_R = 0.0, -2*C_gve*dv_R_input
#     dex_from_R, dey_from_R = C_gve*sin(u_0)*dv_R_input, -C_gve*cos(u_0)*dv_R_input
#     term_drift_drag = M_trans[2,7] * dB_natural
#     term_drift_R = M_trans[2,2]*dl_from_R + M_trans[2,3]*dex_from_R + M_trans[2,4]*dey_from_R
#     term_coeff_T = M_trans[2,1]*(2*C_gve) + M_trans[2,3]*(2*C_gve*cos(u_0)) + M_trans[2,4]*(2*C_gve*sin(u_0))
#     req_dv_T = (target_dlambda_meters / oe_c_init.a - term_drift_R - term_drift_drag) / term_coeff_T
    
#     println("設計結果:")
#     println("  必要分離速度 ΔvT: $(req_dv_T) [m/s]")

#     # --- 2. シミュレーション準備 ---
#     da_0  = 2*C_gve*req_dv_T
#     dl_0  = -2*C_gve*dv_R_input
#     dex_0 = C_gve*(sin(u_0)*dv_R_input + 2*cos(u_0)*req_dv_T)
#     dey_0 = C_gve*(-cos(u_0)*dv_R_input + 2*sin(u_0)*req_dv_T)
#     roe_curr = SVector(da_0, dl_0, dex_0, dey_0, 0.0, 0.0, dB_natural)
#     oe_c_curr = oe_c_init
#     current_time = 0.0

#     hist_t = Float64[]
#     hist_dl = Float64[]
#     hist_da = Float64[]
#     hist_diy = Float64[] # ★追加: δiyの履歴
#     hist_dist = Float64[]
#     hist_x = Float64[]; hist_y = Float64[]; hist_z = Float64[]
#     hist_db = Float64[]
#     hist_phase = Symbol[] 

#     oe_d_0 = final_roe_to_deputy_oe(oe_c_curr, roe_curr)
#     sv_c_0 = orbital_elements_to_sv(oe_c_curr); sv_d_0 = orbital_elements_to_sv(oe_d_0)
#     r_c_0 = SVector(sv_c_0[1], sv_c_0[2], sv_c_0[3]); v_c_0 = SVector(sv_c_0[4], sv_c_0[5], sv_c_0[6])
#     r_d_0 = SVector(sv_d_0[1], sv_d_0[2], sv_d_0[3])
#     bias_offset = eci_to_rtn(r_c_0, v_c_0, r_d_0 - r_c_0)

#     function record_state!(t, roe, oe_c, bias, phase, db_val)
#         oe_d = final_roe_to_deputy_oe(oe_c, roe)
#         sv_c = orbital_elements_to_sv(oe_c); sv_d = orbital_elements_to_sv(oe_d)
#         r_c = SVector(sv_c[1], sv_c[2], sv_c[3]); v_c = SVector(sv_c[4], sv_c[5], sv_c[6])
#         r_d = SVector(sv_d[1], sv_d[2], sv_d[3])
#         dr_rtn = eci_to_rtn(r_c, v_c, r_d - r_c) - bias
        
#         push!(hist_t, t)
#         push!(hist_dl, roe[2] * oe_c.a)
#         push!(hist_da, roe[1] * oe_c.a)
#         push!(hist_diy, roe[6] * oe_c.a) # δiy
#         push!(hist_dist, norm(dr_rtn))
#         push!(hist_x, dr_rtn[1]); push!(hist_y, dr_rtn[2]); push!(hist_z, dr_rtn[3])
#         push!(hist_db, db_val)
#         push!(hist_phase, phase)
#     end

#     # --- 3. フェーズ1: ドリフト期間 ---
#     println("Phase 1: ドリフト...")
#     dt_drift = 600.0
#     steps_drift = Int(drift_duration_sec / dt_drift)
    
#     for i in 1:steps_drift
#         record_state!(current_time, roe_curr, oe_c_curr, bias_offset, :Drift, roe_curr[7])
#         # (伝播処理: 簡略化のため中身は省略せず書くのが安全だが、長くなるので前と同じロジック)
#         omega_c = oe_c_curr.omega
#         omega_dot_step, _ = get_secular_j2_rates_koenig(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i)
#         omega_c_next = omega_c + omega_dot_step * dt_drift
#         J_t0_step = get_J_qns_augmented_koenig(omega_c)
#         J_tf_inv_step = get_J_qns_inv_augmented_koenig(omega_c_next)
#         A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, omega_c, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
#         STM_step = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, dt_drift, oe_c_curr.e, true, DENSITY_MODEL_SPECIFIC)
#         roe_next = J_tf_inv_step * STM_step * J_t0_step * roe_curr
#         roe_curr = SVector{7,Float64}(Float64.(roe_next))
#         new_M = mod(oe_c_curr.M + oe_c_curr.n * dt_drift, 2*pi)
#         new_omega = mod(oe_c_curr.omega + omega_dot_step * dt_drift, 2*pi)
#         new_RAAN = mod(oe_c_curr.RAAN + 0.0, 2*pi)
#         oe_c_curr = OrbitalElementsClassical(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, new_RAAN, new_omega, 0.0, oe_c_curr.n, new_M)
#         current_time += dt_drift
#     end

#     # --- 4. フェーズ2: 制御期間 (δiy制御追加) ---
#     println("Phase 2: 制御開始 (δiyフィードバック付き)...")
    
#     control_duration_sec = control_duration_orbits * (2*pi/oe_c_curr.n)
#     dt_control = SIM_SEGMENT_ORBITS * (2*pi/oe_c_curr.n)
#     steps_control = Int(control_duration_sec / dt_control)
    
#     # 制御パラメータ
#     GAIN_DA = 1.5  # δa制御ゲイン (Inner Loop)
#     GAIN_IY = 0.5   # δiy制御ゲイン (Outer Loop: 小さめにする)
    
#     # 目標
#     target_dl = target_dlambda_meters / oe_c_init.a
#     target_diy = 0.0 # δiyをゼロにしたい
    
#     current_delta_B = roe_curr[7]
#     alpha = 1.0 - exp(-dt_control / ATTITUDE_CHANGE_TIMECONSTANT_SEC)
    
#     for i in 1:steps_control
#         record_state!(current_time, roe_curr, oe_c_curr, bias_offset, :Control, current_delta_B)

#         # --- A. Outer Loop: δiyを直すための目標δaを決める ---
#         # δiyの誤差
#         err_diy = roe_curr[6] - target_diy
        
#         # 感度係数 (∂Ωdot / ∂a) の概算
#         # Ωdot ∝ a^-3.5 なので、感度は -3.5 * Ωdot / a
#         # δiy_dot ∝ -3.5 * Ωdot * sin(i) / a * δa
#         # 非常に小さい値なので、単純なP制御で方向を決める
        
#         # 必要なδa (符号に注意: δa下げるとΩdot速くなる→相対的に追いつく？)
#         # J2係数(負)を考慮すると、符号関係は複雑だが、シミュレーションで調整
#         # 通常: δiy > 0 なら δa > 0 にして修正
        
#         # δiyの制御入力をδaの目標値に上乗せ
#         # ただし、δλへの影響が甚大なので、リミッターをかける
#         da_ref_from_diy = GAIN_IY * err_diy 
#         da_ref_from_diy = clamp(da_ref_from_diy, -50.0/oe_c_curr.a, 50.0/oe_c_curr.a) # 最大50m程度のδaに制限
        
#         # --- B. Inner Loop: δaを目標値にするためのδBを決める ---
#         # 目標δa = 0 (静止) + δiy補正分
#         target_da_total = 0.0 + da_ref_from_diy
        
#         err_da_meters = (roe_curr[1] - target_da_total) * oe_c_curr.a
        
#         delta_B_req = err_da_meters * GAIN_DA
#         delta_B_command = clamp(delta_B_req, DELTA_B_MIN, DELTA_B_MAX)
        
#         # --- C. 伝播 ---
#         omega_dot_step, _ = get_secular_j2_rates_koenig(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i)
#         omega_c_next = oe_c_curr.omega + omega_dot_step * dt_control
#         J_t0_step = get_J_qns_augmented_koenig(oe_c_curr.omega)
#         J_tf_inv_step = get_J_qns_inv_augmented_koenig(omega_c_next)
#         A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, oe_c_curr.omega, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
#         STM_step = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, dt_control, oe_c_curr.e, true, DENSITY_MODEL_SPECIFIC)
        
#         roe_vec_input = SVector(roe_curr[1], roe_curr[2], roe_curr[3], roe_curr[4], roe_curr[5], roe_curr[6], current_delta_B)
#         roe_next = J_tf_inv_step * STM_step * J_t0_step * roe_vec_input
#         roe_curr = SVector{7,Float64}(Float64.(roe_next))
        
#         new_M = mod(oe_c_curr.M + oe_c_curr.n * dt_control, 2*pi)
#         new_omega = mod(oe_c_curr.omega + omega_dot_step * dt_control, 2*pi)
#         new_RAAN = mod(oe_c_curr.RAAN + 0.0, 2*pi)
#         oe_c_curr = OrbitalElementsClassical(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, new_RAAN, new_omega, 0.0, oe_c_curr.n, new_M)
#         current_delta_B = alpha * delta_B_command + (1.0 - alpha) * current_delta_B
#         current_time += dt_control
#     end
#     record_state!(current_time, roe_curr, oe_c_curr, bias_offset, :Control, current_delta_B)

#     # --- 5. プロット ---
#     default(dpi=300, guidefontsize=9, tickfontsize=7, legendfontsize=7, titlefontsize=10, margin=3Plots.mm, right_margin=20Plots.mm)
#     t_days = hist_t ./ 86400.0
    
#     p1 = plot(t_days, hist_dl, label="δλ", xlabel="Time [days]", ylabel="δλ [m]", title="Along-Track Drift", lw=2, legend=:topleft)
#     vline!(p1, [drift_duration_days], label="Control", color=:gray, ls=:dash)
    
#     p2 = plot(t_days, hist_da, label="δa", xlabel="Time [days]", ylabel="δa [m]", title="Semi-Major Axis Diff", lw=2, color=:blue, legend=:outertopright)
#     vline!(p2, [drift_duration_days], label=false, color=:gray, ls=:dash)

#     p3 = plot(hist_y, hist_x, label=false, xlabel="y [m]", ylabel="x [m]", title="RT-Plane", aspect_ratio=:equal, lw=0.5, color=:gray)
#     drift_idx = findall(x -> x == :Drift, hist_phase); ctrl_idx = findall(x -> x == :Control, hist_phase)
#     if !isempty(ctrl_idx); push!(drift_idx, ctrl_idx[1]); end
#     plot!(p3, hist_y[drift_idx], hist_x[drift_idx], label="Drift", lw=1.5, color=:blue)
#     plot!(p3, hist_y[ctrl_idx], hist_x[ctrl_idx], label="Control", lw=2.0, color=:red)
    
#     p4 = plot(t_days, hist_dist, label="Distance", xlabel="Time [days]", ylabel="Dist [m]", title="Rel Dist", lw=2, color=:green, legend=:topright)
    
#     p5 = plot(t_days, hist_db, label="δB", xlabel="Time [days]", ylabel="δB [-]", title="Control Input", lw=2, color=:orange, legend=:topright)
    
#     # ★追加: δiyの推移も確認
#     p6 = plot(t_days, hist_z, label="z", xlabel="Time [days]", ylabel="z [m]", title="Cross-Track (z)", lw=1.5, color=:purple, legend=:outertopright)
#     vline!(p6, [drift_duration_days], label=false, color=:gray, ls=:dash)

#     p_combined = plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(1100, 1200))
    
#     timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
#     filename = "station_keeping_coupled_$(timestamp).png"
#     savefig(p_combined, filename)
#     println("グラフを保存しました: $filename")
#     display(p_combined)
# end
# 実行
run_station_keeping_simulation()

# ==============================================================================
#  シーケンシャル制御 (Step 1: δiy修正 -> Step 2: δλ修正)(上手くいってない)
# ==============================================================================
function run_sequential_control_simulation()
    println("\n" * "="^60)
    println("シーケンシャル軌道制御シミュレーション")
    println(" (優先度: 面外δiy -> 面内δλ)")
    println("="^60)

    # --- 0. 設定 ---
    target_dlambda_meters = -60.0 # 最終目標位置
    drift_duration_days = 1.0     # 初期ドリフト期間
    
    # 制御期間を長めに設定 (J2の効きが悪いため時間がかかる)
    control_duration_orbits = 300.0 
    
    # --- 1. 初期条件計算 (前回のコードと同様) ---
    n_init = sqrt(mu_earth / a_c_stm_init^3)
    oe_c_init = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, n_init, M_c_stm_init)
    rho = RHO_LEO; Bc = BC_CHIEF; dB_natural = DELTA_B_INIT

    # ドリフト後の状態計算
    drift_duration_sec = drift_duration_days * 86400.0
    omega_dot, Omega_dot = get_secular_j2_rates_koenig(oe_c_init.a, oe_c_init.e, oe_c_init.i)
    
    # 簡単のため、ドリフト計算は省略し、t=1.0日後の状態からスタートさせる
    # (実際には前のコード同様にSTMで飛ばすが、ここでは制御ロジックに集中するため初期値を設定)
    # ※前のシミュレーション結果に近い値を模擬的に初期値とします
    
    # 想定される1日後の状態 (ドリフトでδλが流れ、δiyも少しズレているとする)
    # δa=0 (ドリフト中は減衰するが、制御開始時に一度リセットして考える)
    roe_current = SVector(
        0.0,                # δa (一旦静止)
        -100.0/oe_c_init.a, # δλ (目標-60mに対して、現在は-100mにいるとする)
        0.0, 0.0, 0.0,      # δe, δix
        20.0/oe_c_init.a,   # δiy (20mの面外ズレが発生していると仮定)
        dB_natural
    )
    
    oe_c_curr = oe_c_init
    current_time = 0.0
    
    # 履歴用
    hist_t = Float64[]
    hist_dl = Float64[]
    hist_da = Float64[]
    hist_diy = Float64[]
    hist_z = Float64[]
    hist_mode = Float64[] # 制御モード記録 (1: δiy, 2: δλ, 3: Keep)

    # 制御パラメータ
    GAIN_DA = 1.5  # インナーループ (δa追従用)
    
    # モード判定しきい値
    TOL_DIY = 1.0   # [m] δiyの許容誤差
    TOL_DL  = 5.0   # [m] δλの許容誤差
    
    # 目標δaの設定値 (m)
    # δiyを直すときは大きく動かす必要があるが、空気抵抗の限界がある
    # ここでは、空気抵抗で作れる現実的なδaとして ±100m 程度を上限とする
    DA_FOR_DIY = 80.0 
    DA_FOR_DL  = 30.0 

    println("制御開始: 初期誤差 δλ=-100m, δiy=20m")
    println("目標: δλ=$(target_dlambda_meters)m, δiy=0m")

    # --- シミュレーションループ ---
    dt_control = SIM_SEGMENT_ORBITS * (2*pi/oe_c_curr.n)
    steps = Int(control_duration_orbits / SIM_SEGMENT_ORBITS)
    
    # 状態変数の初期化
    current_delta_B = roe_current[7]
    alpha = 1.0 - exp(-dt_control / ATTITUDE_CHANGE_TIMECONSTANT_SEC)
    
    mode = 1 # 初期モード: δiy修正
    
    for i in 1:steps
        # 現在の誤差 (メートル換算)
        val_dl  = roe_current[2] * oe_c_curr.a
        val_da  = roe_current[1] * oe_c_curr.a
        val_diy = roe_current[6] * oe_c_curr.a
        
        err_dl  = val_dl - target_dlambda_meters
        err_diy = val_diy - 0.0
        
        # --- モード切替ロジック (ステートマシン) ---
        target_da_meter = 0.0
        
        if mode == 1
            # 【Phase 1: δiy 修正モード】
            # 面外ズレが大きいうちは、δλを無視してδiyを直す
            if abs(err_diy) < TOL_DIY
                mode = 2 # 修正完了 -> 次へ
            else
                # δiyを減らすためのδaを設定
                # (符号の関係: 軌道解析に基づく。通常、高度を下げるとΩdotが速くなり追いつく等)
                # ここでは仮に、err_diyと逆符号のδaを入れると収束すると仮定して実装
                # ※厳密にはJ2係数と傾斜角に依存。シミュレーションで挙動確認して符号反転が必要かも。
                target_da_meter = sign(err_diy) * DA_FOR_DIY
            end
            
        elseif mode == 2
            # 【Phase 2: δλ 修正モード】
            # δiyはOKなので、ズレてしまったδλを急いで戻す
            if abs(err_dl) < TOL_DL
                mode = 3 # 修正完了 -> 維持へ
            else
                # δλを戻すためのδaを設定
                # δλ_dot ≈ -1.5 n δa なので、
                # δλをプラスにしたい(戻したい)なら、δaはマイナスにする必要あり
                # target_da ∝ - (target - current) = - (-err) = err
                # つまり、err_dlと同じ符号のδaを作ればよい
                target_da_meter = sign(err_dl) * DA_FOR_DL
            end
            
        elseif mode == 3
            # 【Phase 3: 維持モード】
            # 両方OKなので、δa=0にしてその場に止まる
            # (微修正のためP制御を入れても良いが、今回は完全停止を狙う)
            target_da_meter = 0.0
            
            # もし誤差が再拡大したらモードを戻すリセットを入れることも可能
            if abs(err_diy) > TOL_DIY * 2.0; mode = 1; end
            if abs(err_dl)  > TOL_DL * 2.0;  mode = 2; end
        end
        
        # --- δa制御 (インナーループ) ---
        # 決定した target_da_meter になるように空気抵抗(δB)を操作
        err_da_ctrl = val_da - target_da_meter
        delta_B_req = err_da_ctrl * GAIN_DA
        delta_B_command = clamp(delta_B_req, DELTA_B_MIN, DELTA_B_MAX)
        
        # --- 状態保存 ---
        push!(hist_t, current_time)
        push!(hist_dl, val_dl)
        push!(hist_da, val_da)
        push!(hist_diy, val_diy)
        push!(hist_mode, Float64(mode))
        
        # z位置の概算 (δiy * a * sin(theta)) 振幅として記録
        push!(hist_z, abs(val_diy)) 

        # --- 伝播 (1ステップ) ---
        omega_dot_step, _ = get_secular_j2_rates_koenig(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i)
        omega_c_next = oe_c_curr.omega + omega_dot_step * dt_control
        J_t0_step = get_J_qns_augmented_koenig(oe_c_curr.omega)
        J_tf_inv_step = get_J_qns_inv_augmented_koenig(omega_c_next)
        A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, oe_c_curr.omega, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
        STM_step = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, dt_control, oe_c_curr.e, true, DENSITY_MODEL_SPECIFIC)
        
        roe_vec_input = SVector(roe_current[1], roe_current[2], roe_current[3], roe_current[4], roe_current[5], roe_current[6], current_delta_B)
        roe_next = J_tf_inv_step * STM_step * J_t0_step * roe_vec_input
        
        roe_current = SVector{7,Float64}(Float64.(roe_next))
        new_M = mod(oe_c_curr.M + oe_c_curr.n * dt_control, 2*pi)
        new_omega = mod(oe_c_curr.omega + omega_dot_step * dt_control, 2*pi)
        oe_c_curr = OrbitalElementsClassical(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, oe_c_curr.RAAN, new_omega, 0.0, oe_c_curr.n, new_M)
        
        current_delta_B = alpha * delta_B_command + (1.0 - alpha) * current_delta_B
        current_time += dt_control
    end

    # --- プロット ---
    default(dpi=300, guidefontsize=9, tickfontsize=7, legendfontsize=7, margin=5Plots.mm)
    t_days = hist_t ./ 86400.0
    
    # モードの可視化用背景
    function plot_background!(p)
        # 背景色を変える等は難しいので、モード値を別グラフで表示
    end

    p1 = plot(t_days, hist_dl, label="δλ", ylabel="[m]", title="Along-Track (δλ)", lw=2)
    hline!(p1, [target_dlambda_meters], label="Target", ls=:dash, color=:black)
    
    p2 = plot(t_days, hist_da, label="δa", ylabel="[m]", title="Semi-Major Axis (δa)", lw=2, color=:blue)
    
    p3 = plot(t_days, hist_diy, label="δiy", ylabel="[m]", title="Cross-Track Drift (δiy)", lw=2, color=:purple)
    hline!(p3, [0.0], label="Target", ls=:dash, color=:black)
    
    p4 = plot(t_days, hist_mode, label="Mode", ylabel="1:FixIY, 2:FixDL, 3:Keep", title="Control Phase", lw=2, color=:orange, yticks=[1,2,3])

    p_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "sequential_control_$(timestamp).png"
    savefig(p_combined, filename)
    println("グラフを保存しました: $filename")
    display(p_combined)
end
# 実行
run_sequential_control_simulation()

# ==============================================================================
#  δiyとδλの同時制御 (δλとδiyの両方の要求δaが拮抗して目標値からずれて固まる)
# ==============================================================================
function run_coordinated_control_simulation()
    println("\n" * "="^60)
    println("協調制御シミュレーション (修正版)")
    println("="^60)

    # --- 設定 ---
    target_dlambda_meters = -60.0
    control_duration_orbits = 300.0 
    
    # --- 初期化 ---
    n_init = sqrt(mu_earth / a_c_stm_init^3)
    oe_c_init = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, n_init, M_c_stm_init)
    rho = RHO_LEO; Bc = BC_CHIEF; dB_natural = DELTA_B_INIT
    
    roe_current = SVector(
        0.0,                
        -100.0/oe_c_init.a, 
        0.0, 0.0, 0.0,      
        20.0/oe_c_init.a,   
        dB_natural
    )
    oe_c_curr = oe_c_init
    current_time = 0.0
    
    hist_t = Float64[]; hist_dl = Float64[]; hist_da = Float64[]; hist_diy = Float64[]

    # --- 制御ゲイン (再調整) ---
    # δiyは非常に遅いのでゲイン大。
    # δλは非常に速いのでゲイン小。
    # バランス: δiyの修正(da)が、δλの修正(da)を邪魔しない程度にする。
    
    # K_IY: 誤差[m]あたり、どれだけのδa[m]を作るか
    K_IY = 20.0   # 1mズレたら、δaを20m作って直す
    
    # K_DL: 誤差[m]あたり、どれだけのδa[m]を作るか
    # δλ_dot = -1.5 * n * δa  => 1日で約 1000 * δa 進む
    # 早く戻したいが、K_IYと喧嘩しないようにする
    K_DL = 0.02 # 100mズレてたら、δaを0.5m作る (ゆっくり戻す)

    # インナーループ (δa追従)
    GAIN_DA = 50.0 

    println("制御開始: K_IY=$K_IY, K_DL=$K_DL")

    dt_control = SIM_SEGMENT_ORBITS * (2*pi/oe_c_curr.n)
    steps = Int(control_duration_orbits / SIM_SEGMENT_ORBITS)
    alpha = 1.0 - exp(-dt_control / ATTITUDE_CHANGE_TIMECONSTANT_SEC)
    current_delta_B = roe_current[7]

    for i in 1:steps
        val_dl  = roe_current[2] * oe_c_curr.a
        val_da  = roe_current[1] * oe_c_curr.a
        val_diy = roe_current[6] * oe_c_curr.a
        
        err_dl  = val_dl - target_dlambda_meters
        err_diy = val_diy - 0.0
        
        # --- 協調制御則 ---
        # 符号の確認:
        # δiy > 0 のとき、δa > 0 にすると J2効果で Ωdot_d < Ωdot_c (遅れる) -> δiy減る？
        # 軌道解析上、δiyを減らすには「逆符号のδa」が必要なケースが多いので、まずはマイナスで試す
        req_da_from_diy = +K_IY * err_diy
        
        # δλ > 目標 のとき(進みすぎ)、戻すには速度を落とす(δa < 0)必要はない？
        # δa < 0 (低い) -> n_d > n_c (速い) -> 前に進む (δλ増える)
        # 逆に δa > 0 (高い) -> n_d < n_c (遅い) -> 後ろに下がる (δλ減る)
        # よって、δλを減らしたい(err_dl > 0)なら、δaをプラスにする必要あり
        # target_da ∝ + err_dl
        req_da_from_dl = +K_DL * err_dl
        
        target_da_total = req_da_from_diy + req_da_from_dl
        target_da_total = clamp(target_da_total, -500.0, 500.0) # メートル単位で制限
        
        # --- δa制御 ---
        # 目標δa(m) を 無次元化して比較
        target_da_nd = target_da_total / oe_c_curr.a
        
        # 誤差
        err_da_nd = roe_current[1] - target_da_nd
        
        # 指令 (P制御)
        # δaを下げたい(err>0) -> 抵抗増やす(dB>0)
        delta_B_req = err_da_nd * oe_c_curr.a * GAIN_DA
        delta_B_command = clamp(delta_B_req, DELTA_B_MIN, DELTA_B_MAX)
        
        # デバッグ表示 (最初の数ステップだけ)
        if i < 5 || i % 1000 == 0
            @printf "Step %d: err_diy=%.1f, req_da_iy=%.1f | err_dl=%.1f, req_da_dl=%.1f | Tgt_da=%.1f -> dB=%.3f\n" i err_diy req_da_from_diy err_dl req_da_from_dl target_da_total delta_B_command
        end

        push!(hist_t, current_time); push!(hist_dl, val_dl); push!(hist_da, val_da); push!(hist_diy, val_diy)
        
        # 伝播
        omega_dot_step, _ = get_secular_j2_rates_koenig(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i)
        omega_c_next = oe_c_curr.omega + omega_dot_step * dt_control
        J_t0_step = get_J_qns_augmented_koenig(oe_c_curr.omega)
        J_tf_inv_step = get_J_qns_inv_augmented_koenig(omega_c_next)
        A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, oe_c_curr.omega, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
        STM_step = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, dt_control, oe_c_curr.e, true, DENSITY_MODEL_SPECIFIC)
        
        roe_vec_input = SVector(roe_current[1], roe_current[2], roe_current[3], roe_current[4], roe_current[5], roe_current[6], current_delta_B)
        roe_next = J_tf_inv_step * STM_step * J_t0_step * roe_vec_input
        
        roe_current = SVector{7,Float64}(Float64.(roe_next))
        new_M = mod(oe_c_curr.M + oe_c_curr.n * dt_control, 2*pi)
        new_omega = mod(oe_c_curr.omega + omega_dot_step * dt_control, 2*pi)
        oe_c_curr = OrbitalElementsClassical(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, oe_c_curr.RAAN, new_omega, 0.0, oe_c_curr.n, new_M)
        
        current_delta_B = alpha * delta_B_command + (1.0 - alpha) * current_delta_B
        current_time += dt_control
    end

    default(dpi=300, guidefontsize=9, tickfontsize=7, legendfontsize=7, margin=5Plots.mm)
    t_days = hist_t ./ 86400.0
    
    p1 = plot(t_days, hist_dl, label="δλ", ylabel="[m]", title="Along-Track", lw=2)
    hline!(p1, [target_dlambda_meters], label="Target", ls=:dash, color=:black)
    
    p2 = plot(t_days, hist_da, label="δa", ylabel="[m]", title="Semi-Major Axis", lw=2, color=:blue)
    
    p3 = plot(t_days, hist_diy, label="δiy", ylabel="[m]", title="Cross-Track Drift", lw=2, color=:purple)
    hline!(p3, [0.0], label="Target", ls=:dash, color=:black)
    
    p_combined = plot(p1, p2, p3, layout=(3,1), size=(800, 800))
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "coordinated_control_debug_$(timestamp).png"
    savefig(p_combined, filename)
    display(p_combined)
end

run_coordinated_control_simulation()

# ==============================================================================
# タイムシェアリング制御 
# ==============================================================================
function run_timesharing_control_simulation()
    println("\n" * "="^60)
    println("タイムシェアリング制御シミュレーション (完成版)")
    println("="^60)

    target_dlambda_meters = -60.0
    control_duration_orbits = 300.0 
    
    n_init = sqrt(mu_earth / a_c_stm_init^3)
    oe_c_init = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, n_init, M_c_stm_init)
    rho = RHO_LEO; Bc = BC_CHIEF; dB_natural = DELTA_B_INIT
    
    # 初期条件
    roe_current = SVector(0.0, -100.0/oe_c_init.a, 0.0, 0.0, 0.0, 20.0/oe_c_init.a, dB_natural)
    oe_c_curr = oe_c_init
    current_time = 0.0
    
    hist_t = Float64[]; hist_dl = Float64[]; hist_da = Float64[]; hist_diy = Float64[]

    dt_control = SIM_SEGMENT_ORBITS * (2*pi/oe_c_curr.n)
    steps = Int(control_duration_orbits / SIM_SEGMENT_ORBITS)
    alpha = 1.0 - exp(-dt_control / ATTITUDE_CHANGE_TIMECONSTANT_SEC)
    current_delta_B = roe_current[7]

    # ★調整: サイクルを短くし、バランスを五分五分にする
    CYCLE_STEPS = 500   # 2000 -> 500 (こまめに切り替え)
    SWITCH_RATIO = 0.5  # 0.8 -> 0.5 (公平に時間を配分)

    for i in 1:steps
        val_dl  = roe_current[2] * oe_c_curr.a
        val_da  = roe_current[1] * oe_c_curr.a
        val_diy = roe_current[6] * oe_c_curr.a
        
        err_dl  = val_dl - target_dlambda_meters
        err_diy = val_diy - 0.0
        
        cycle_phase = (i % CYCLE_STEPS) / CYCLE_STEPS
        target_da_total = 0.0
        
        # 収束判定を少し厳しく
        if abs(err_diy) < 0.1 && abs(err_dl) < 1.0
            target_da_total = 0.0
        elseif cycle_phase < SWITCH_RATIO
            # Mode A: δiy優先
            target_da_total = 20.0 * err_diy 
        else
            # Mode B: δλ優先
            target_da_total = 0.1 * err_dl 
        end
        
        target_da_total = clamp(target_da_total, -200.0, 200.0)
        
        target_da_nd = target_da_total / oe_c_curr.a
        err_da_nd = roe_current[1] - target_da_nd
        delta_B_req = err_da_nd * oe_c_curr.a * 50.0
        delta_B_command = clamp(delta_B_req, DELTA_B_MIN, DELTA_B_MAX)
        
        push!(hist_t, current_time); push!(hist_dl, val_dl); push!(hist_da, val_da); push!(hist_diy, val_diy)
        
        omega_dot_step, _ = get_secular_j2_rates_koenig(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i)
        omega_c_next = oe_c_curr.omega + omega_dot_step * dt_control
        J_t0_step = get_J_qns_augmented_koenig(oe_c_curr.omega)
        J_tf_inv_step = get_J_qns_inv_augmented_koenig(omega_c_next)
        A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, oe_c_curr.omega, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
        STM_step = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, dt_control, oe_c_curr.e, true, DENSITY_MODEL_SPECIFIC)
        roe_vec_input = SVector(roe_current[1], roe_current[2], roe_current[3], roe_current[4], roe_current[5], roe_current[6], current_delta_B)
        roe_next = J_tf_inv_step * STM_step * J_t0_step * roe_vec_input
        roe_current = SVector{7,Float64}(Float64.(roe_next))
        new_M = mod(oe_c_curr.M + oe_c_curr.n * dt_control, 2*pi)
        new_omega = mod(oe_c_curr.omega + omega_dot_step * dt_control, 2*pi)
        oe_c_curr = OrbitalElementsClassical(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, oe_c_curr.RAAN, new_omega, 0.0, oe_c_curr.n, new_M)
        current_delta_B = alpha * delta_B_command + (1.0 - alpha) * current_delta_B
        current_time += dt_control
    end

    default(dpi=300, guidefontsize=9, tickfontsize=7, legendfontsize=7, margin=5Plots.mm)
    t_days = hist_t ./ 86400.0
    # タイトル修正
    p1 = plot(t_days, hist_dl, label="δλ", ylabel="[m]", title="Time History of δλ (Along-Track)", lw=2); hline!(p1, [target_dlambda_meters], ls=:dash, color=:black)
    p2 = plot(t_days, hist_da, label="δa", ylabel="[m]", title="Time History of δa (Semi-Major Axis)", lw=2, color=:blue)
    p3 = plot(t_days, hist_diy, label="δiy", ylabel="[m]", title="Time History of δiy (Cross-Track Drift)", lw=2, color=:purple); hline!(p3, [0.0], ls=:dash, color=:black)
    p_combined = plot(p1, p2, p3, layout=(3,1), size=(800, 800))
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "timesharing_control_final_$(timestamp).png"
    savefig(p_combined, filename)
    println("グラフを保存しました: $filename")
    display(p_combined)
end

run_timesharing_control_simulation()

# # ==============================================================================
# #  全行程: 分離 -> ドリフト -> タイムシェアリング制御(δiyが徐々に目標値から離れていってしまう)
# # ==============================================================================
# function run_full_mission_timesharing_simulation()
#     println("\n" * "="^60)
#     println("全行程シミュレーション: 分離 -> ドリフト -> タイムシェアリング制御 (修正版)")
#     println("="^60)

#     # --- 0. 全体設定 ---
#     target_dlambda_meters = -60.0   # 目標分離位置
#     drift_duration_days = 1.0       # ドリフト期間
    
#     control_duration_orbits = 300.0 # 制御期間 (約20日)
    
#     dv_R_input = 0.0 

#     # --- 1. 初期パラメータとΔv設計 ---
#     n_init = sqrt(mu_earth / a_c_stm_init^3)
#     oe_c_init = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, n_init, M_c_stm_init)
#     rho = RHO_LEO; Bc = BC_CHIEF; dB_natural = DELTA_B_INIT

#     drift_duration_sec = drift_duration_days * 86400.0
#     omega_c_t0 = oe_c_init.omega
#     omega_dot, Omega_dot = get_secular_j2_rates_koenig(oe_c_init.a, oe_c_init.e, oe_c_init.i)
#     omega_c_tf = omega_c_t0 + omega_dot * drift_duration_sec
    
#     J_t0 = get_J_qns_augmented_koenig(omega_c_t0)
#     J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)
#     A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_init.a, oe_c_init.e, oe_c_init.i, omega_c_t0, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
#     STM_drift = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, drift_duration_sec, oe_c_init.e, true, DENSITY_MODEL_SPECIFIC)
#     M_trans = J_tf_inv * STM_drift * J_t0
    
#     u_0 = oe_c_init.M + oe_c_init.omega 
#     C_gve = 1.0 / (oe_c_init.n * oe_c_init.a)
#     da_from_R, dl_from_R = 0.0, -2*C_gve*dv_R_input
#     dex_from_R, dey_from_R = C_gve*sin(u_0)*dv_R_input, -C_gve*cos(u_0)*dv_R_input
    
#     term_drift_drag = M_trans[2,7] * dB_natural
#     term_drift_R = M_trans[2,2]*dl_from_R + M_trans[2,3]*dex_from_R + M_trans[2,4]*dey_from_R
#     term_coeff_T = M_trans[2,1]*(2*C_gve) + M_trans[2,3]*(2*C_gve*cos(u_0)) + M_trans[2,4]*(2*C_gve*sin(u_0))
#     req_dv_T = (target_dlambda_meters / oe_c_init.a - term_drift_R - term_drift_drag) / term_coeff_T
    
#     println("[Phase 1] ドリフト開始 (目標δλ: $(target_dlambda_meters)m, ΔvT: $(req_dv_T) m/s)")

#     # --- 2. シミュレーション開始 ---
#     da_0  = 2*C_gve*req_dv_T
#     dl_0  = -2*C_gve*dv_R_input
#     dex_0 = C_gve*(sin(u_0)*dv_R_input + 2*cos(u_0)*req_dv_T)
#     dey_0 = C_gve*(-cos(u_0)*dv_R_input + 2*sin(u_0)*req_dv_T)
    
#     roe_curr = SVector(da_0, dl_0, dex_0, dey_0, 0.0, 0.0, dB_natural)
#     oe_c_curr = oe_c_init
#     current_time = 0.0

#     hist_t = Float64[]; hist_dl = Float64[]; hist_da = Float64[]; hist_diy = Float64[]
    
#     # ----------------------------------------------------------
#     # Phase 1: ドリフト期間
#     # ----------------------------------------------------------
#     dt_drift = 600.0
#     steps_drift = Int(drift_duration_sec / dt_drift)
    
#     for i in 1:steps_drift
#         push!(hist_t, current_time)
#         push!(hist_dl, roe_curr[2] * oe_c_curr.a)
#         push!(hist_da, roe_curr[1] * oe_c_curr.a)
#         push!(hist_diy, roe_curr[6] * oe_c_curr.a)

#         # 修正: Omega_dot_step を受け取るように変更
#         omega_c = oe_c_curr.omega
#         omega_dot_step, Omega_dot_step = get_secular_j2_rates_koenig(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i)
        
#         omega_c_next = omega_c + omega_dot_step * dt_drift
#         J_t0_step = get_J_qns_augmented_koenig(omega_c)
#         J_tf_inv_step = get_J_qns_inv_augmented_koenig(omega_c_next)
#         A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, omega_c, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
#         STM_step = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, dt_drift, oe_c_curr.e, true, DENSITY_MODEL_SPECIFIC)
#         roe_next = J_tf_inv_step * STM_step * J_t0_step * roe_curr
        
#         roe_curr = SVector{7,Float64}(Float64.(roe_next))
#         new_M = mod(oe_c_curr.M + oe_c_curr.n * dt_drift, 2*pi)
#         new_omega = mod(oe_c_curr.omega + omega_dot_step * dt_drift, 2*pi)
#         # 修正: Omega_dot_step を使用
#         new_RAAN = mod(oe_c_curr.RAAN + Omega_dot_step * dt_drift, 2*pi)
#         oe_c_curr = OrbitalElementsClassical(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, new_RAAN, new_omega, 0.0, oe_c_curr.n, new_M)
#         current_time += dt_drift
#     end
    
#     println("[Phase 1] 終了時の状態: δλ=$(roe_curr[2]*oe_c_curr.a)m, δiy=$(roe_curr[6]*oe_c_curr.a)m")

#     # ----------------------------------------------------------
#     # Phase 2: タイムシェアリング制御期間
#     # ----------------------------------------------------------
#     println("[Phase 2] タイムシェアリング制御開始...")
    
#     dt_control = SIM_SEGMENT_ORBITS * (2*pi/oe_c_curr.n)
#     steps_control = Int((control_duration_orbits * 2*pi/oe_c_curr.n) / dt_control)
    
#     current_delta_B = roe_curr[7]
#     alpha = 1.0 - exp(-dt_control / ATTITUDE_CHANGE_TIMECONSTANT_SEC)
    
#     CYCLE_STEPS = 1000
#     SWITCH_RATIO = 0.5

#     for i in 1:steps_control
#         val_dl  = roe_curr[2] * oe_c_curr.a
#         val_da  = roe_curr[1] * oe_c_curr.a
#         val_diy = roe_curr[6] * oe_c_curr.a
        
#         err_dl  = val_dl - target_dlambda_meters
#         err_diy = val_diy - 0.0
        
#         cycle_phase = (i % CYCLE_STEPS) / CYCLE_STEPS
#         target_da_total = 0.0
        
#         if abs(err_diy) < 0.1 && abs(err_dl) < 1.0
#             target_da_total = 0.0
#         elseif cycle_phase < SWITCH_RATIO
#             target_da_total = 10.0 * err_diy 
#         else
#             target_da_total = 0.1 * err_dl 
#         end
#         target_da_total = clamp(target_da_total, -200.0, 200.0)
        
#         target_da_nd = target_da_total / oe_c_curr.a
#         err_da_nd = roe_curr[1] - target_da_nd
#         delta_B_req = err_da_nd * oe_c_curr.a * 50.0
#         delta_B_command = clamp(delta_B_req, DELTA_B_MIN, DELTA_B_MAX)
        
#         push!(hist_t, current_time); push!(hist_dl, val_dl); push!(hist_da, val_da); push!(hist_diy, val_diy)
        
#         # 修正: Omega_dot_step を受け取るように変更
#         omega_dot_step, Omega_dot_step = get_secular_j2_rates_koenig(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i)
        
#         omega_c_next = oe_c_curr.omega + omega_dot_step * dt_control
#         J_t0_step = get_J_qns_augmented_koenig(oe_c_curr.omega)
#         J_tf_inv_step = get_J_qns_inv_augmented_koenig(omega_c_next)
#         A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, oe_c_curr.omega, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
#         STM_step = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, dt_control, oe_c_curr.e, true, DENSITY_MODEL_SPECIFIC)
#         roe_vec_input = SVector(roe_curr[1], roe_curr[2], roe_curr[3], roe_curr[4], roe_curr[5], roe_curr[6], current_delta_B)
#         roe_next = J_tf_inv_step * STM_step * J_t0_step * roe_vec_input
#         roe_curr = SVector{7,Float64}(Float64.(roe_next))
        
#         new_M = mod(oe_c_curr.M + oe_c_curr.n * dt_control, 2*pi)
#         new_omega = mod(oe_c_curr.omega + omega_dot_step * dt_control, 2*pi)
#         # 修正: Omega_dot_step を使用
#         new_RAAN = mod(oe_c_curr.RAAN + Omega_dot_step * dt_control, 2*pi)
#         oe_c_curr = OrbitalElementsClassical(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, new_RAAN, new_omega, 0.0, oe_c_curr.n, new_M)
#         current_delta_B = alpha * delta_B_command + (1.0 - alpha) * current_delta_B
#         current_time += dt_control
#     end

#     default(dpi=300, guidefontsize=9, tickfontsize=7, legendfontsize=7, margin=5Plots.mm)
#     t_days = hist_t ./ 86400.0
#     p1 = plot(t_days, hist_dl, label="δλ", ylabel="[m]", title="Time History of δλ (Along-Track)", lw=2); hline!(p1, [target_dlambda_meters], ls=:dash, color=:black)
#     vline!(p1, [drift_duration_days], color=:gray, ls=:dash, label="Control Start")
#     p2 = plot(t_days, hist_da, label="δa", ylabel="[m]", title="Time History of δa (Semi-Major Axis)", lw=2, color=:blue)
#     vline!(p2, [drift_duration_days], color=:gray, ls=:dash, label=false)
#     p3 = plot(t_days, hist_diy, label="δiy", ylabel="[m]", title="Time History of δiy (Cross-Track Drift)", lw=2, color=:purple); hline!(p3, [0.0], ls=:dash, color=:black)
#     vline!(p3, [drift_duration_days], color=:gray, ls=:dash, label=false)
#     p_combined = plot(p1, p2, p3, layout=(3,1), size=(800, 800))
    
#     timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
#     filename = "full_mission_timesharing_final_$(timestamp).png"
#     savefig(p_combined, filename)
#     println("グラフを保存しました: $filename")
#     display(p_combined)
# end
# run_full_mission_timesharing_simulation()

# ==============================================================================
#  全行程 (積分制御あり)(δiyが徐々に目標値から離れていってしまう)
# ==============================================================================
function run_full_mission_timesharing_simulation_final_v2()
    println("\n" * "="^60)
    println("全行程シミュレーション: 不感帯撤廃・積分制御強化版")
    println("="^60)

    # --- 設定 ---
    target_dlambda_meters = -60.0
    drift_duration_days = 1.0
    control_duration_orbits = 500.0
    
    dv_R_input = 0.0 

    # --- 初期化 ---
    n_init = sqrt(mu_earth / a_c_stm_init^3)
    oe_c_init = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, n_init, M_c_stm_init)
    rho = RHO_LEO; Bc = BC_CHIEF; dB_natural = DELTA_B_INIT

    drift_duration_sec = drift_duration_days * 86400.0
    omega_c_t0 = oe_c_init.omega
    omega_dot, Omega_dot = get_secular_j2_rates_koenig(oe_c_init.a, oe_c_init.e, oe_c_init.i)
    omega_c_tf = omega_c_t0 + omega_dot * drift_duration_sec
    
    J_t0 = get_J_qns_augmented_koenig(omega_c_t0)
    J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)
    A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_init.a, oe_c_init.e, oe_c_init.i, omega_c_t0, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
    STM_drift = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, drift_duration_sec, oe_c_init.e, true, DENSITY_MODEL_SPECIFIC)
    M_trans = J_tf_inv * STM_drift * J_t0
    
    u_0 = oe_c_init.M + oe_c_init.omega 
    C_gve = 1.0 / (oe_c_init.n * oe_c_init.a)
    da_from_R, dl_from_R = 0.0, -2*C_gve*dv_R_input
    dex_from_R, dey_from_R = C_gve*sin(u_0)*dv_R_input, -C_gve*cos(u_0)*dv_R_input
    term_drift_drag = M_trans[2,7] * dB_natural
    term_drift_R = M_trans[2,2]*dl_from_R + M_trans[2,3]*dex_from_R + M_trans[2,4]*dey_from_R
    term_coeff_T = M_trans[2,1]*(2*C_gve) + M_trans[2,3]*(2*C_gve*cos(u_0)) + M_trans[2,4]*(2*C_gve*sin(u_0))
    req_dv_T = (target_dlambda_meters / oe_c_init.a - term_drift_R - term_drift_drag) / term_coeff_T
    
    da_0  = 2*C_gve*req_dv_T
    dl_0  = -2*C_gve*dv_R_input
    dex_0 = C_gve*(sin(u_0)*dv_R_input + 2*cos(u_0)*req_dv_T)
    dey_0 = C_gve*(-cos(u_0)*dv_R_input + 2*sin(u_0)*req_dv_T)
    roe_curr = SVector(da_0, dl_0, dex_0, dey_0, 0.0, 0.0, dB_natural)
    oe_c_curr = oe_c_init
    current_time = 0.0

    hist_t = Float64[]; hist_dl = Float64[]; hist_da = Float64[]; hist_diy = Float64[]
    
    # Phase 1: ドリフト
    dt_drift = 600.0
    steps_drift = Int(drift_duration_sec / dt_drift)
    for i in 1:steps_drift
        push!(hist_t, current_time); push!(hist_dl, roe_curr[2]*oe_c_curr.a); push!(hist_da, roe_curr[1]*oe_c_curr.a); push!(hist_diy, roe_curr[6]*oe_c_curr.a)
        omega_c = oe_c_curr.omega
        omega_dot_step, Omega_dot_step = get_secular_j2_rates_koenig(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i)
        omega_c_next = omega_c + omega_dot_step * dt_drift
        J_t0_step = get_J_qns_augmented_koenig(omega_c)
        J_tf_inv_step = get_J_qns_inv_augmented_koenig(omega_c_next)
        A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, omega_c, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
        STM_step = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, dt_drift, oe_c_curr.e, true, DENSITY_MODEL_SPECIFIC)
        roe_next = J_tf_inv_step * STM_step * J_t0_step * roe_curr
        roe_curr = SVector{7,Float64}(Float64.(roe_next))
        new_M = mod(oe_c_curr.M + oe_c_curr.n * dt_drift, 2*pi)
        new_omega = mod(oe_c_curr.omega + omega_dot_step * dt_drift, 2*pi)
        new_RAAN = mod(oe_c_curr.RAAN + Omega_dot_step * dt_drift, 2*pi)
        oe_c_curr = OrbitalElementsClassical(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, new_RAAN, new_omega, 0.0, oe_c_curr.n, new_M)
        current_time += dt_drift
    end
    
    println("[Phase 1] 終了時の状態: δλ=$(roe_curr[2]*oe_c_curr.a)m, δiy=$(roe_curr[6]*oe_c_curr.a)m")

    # Phase 2: 制御
    println("[Phase 2] 制御開始 (不感帯なし)")
    
    dt_control = SIM_SEGMENT_ORBITS * (2*pi/oe_c_curr.n)
    steps_control = Int((control_duration_orbits * 2*pi/oe_c_curr.n) / dt_control)
    current_delta_B = roe_curr[7]
    alpha = 1.0 - exp(-dt_control / ATTITUDE_CHANGE_TIMECONSTANT_SEC)
    
    CYCLE_STEPS = 1000
    SWITCH_RATIO = 0.5
    
    integ_diy = 0.0 

    for i in 1:steps_control
        val_dl  = roe_curr[2] * oe_c_curr.a
        val_diy = roe_curr[6] * oe_c_curr.a
        
        err_dl  = val_dl - target_dlambda_meters
        err_diy = val_diy - 0.0
        
        # 積分器更新 (常に計算)
        # ゲインを少し強める: 1e-5 -> 5e-5
        integ_diy += err_diy * dt_control
        # target_diy_bias = -0.00005 * integ_diy 
        
        # 積分器は廃止 (不安定要因になるため)
        target_diy_bias = 0.0 
        
        cycle_phase = (i % CYCLE_STEPS) / CYCLE_STEPS
        target_da_total = 0.0
        
        # 不感帯なしの制御則
        if cycle_phase < SWITCH_RATIO
            # Mode A: δiy優先
            target_da_total = 2.0 * err_diy
        else
            # Mode B: δλ優先
            target_da_total = 0.05 * err_dl
        end
        
        target_da_total = clamp(target_da_total, -200.0, 200.0)
        target_da_nd = target_da_total / oe_c_curr.a
        err_da_nd = roe_curr[1] - target_da_nd
        delta_B_req = err_da_nd * oe_c_curr.a * 50.0
        delta_B_command = clamp(delta_B_req, DELTA_B_MIN, DELTA_B_MAX)
        
        push!(hist_t, current_time); push!(hist_dl, val_dl); push!(hist_da, roe_curr[1]*oe_c_curr.a); push!(hist_diy, val_diy)
        
        omega_dot_step, Omega_dot_step = get_secular_j2_rates_koenig(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i)
        omega_c_next = oe_c_curr.omega + omega_dot_step * dt_control
        J_t0_step = get_J_qns_augmented_koenig(oe_c_curr.omega)
        J_tf_inv_step = get_J_qns_inv_augmented_koenig(omega_c_next)
        A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, oe_c_curr.omega, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
        STM_step = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, dt_control, oe_c_curr.e, true, DENSITY_MODEL_SPECIFIC)
        roe_vec_input = SVector(roe_curr[1], roe_curr[2], roe_curr[3], roe_curr[4], roe_curr[5], roe_curr[6], current_delta_B)
        roe_next = J_tf_inv_step * STM_step * J_t0_step * roe_vec_input
        roe_curr = SVector{7,Float64}(Float64.(roe_next))
        new_M = mod(oe_c_curr.M + oe_c_curr.n * dt_control, 2*pi)
        new_omega = mod(oe_c_curr.omega + omega_dot_step * dt_control, 2*pi)
        new_RAAN = mod(oe_c_curr.RAAN + Omega_dot_step * dt_control, 2*pi)
        oe_c_curr = OrbitalElementsClassical(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, new_RAAN, new_omega, 0.0, oe_c_curr.n, new_M)
        current_delta_B = alpha * delta_B_command + (1.0 - alpha) * current_delta_B
        current_time += dt_control
    end

    println("[Phase 2] 終了時の状態: δλ=$(roe_curr[2]*oe_c_curr.a)m, δiy=$(roe_curr[6]*oe_c_curr.a)m")

    default(dpi=300, guidefontsize=9, tickfontsize=7, legendfontsize=7, margin=5Plots.mm)
    t_days = hist_t ./ 86400.0
    p1 = plot(t_days, hist_dl, label="δλ", ylabel="[m]", title="Time History of δλ (Along-Track)", lw=2); hline!(p1, [target_dlambda_meters], ls=:dash, color=:black)
    vline!(p1, [drift_duration_days], color=:gray, ls=:dash, label="Control")
    p2 = plot(t_days, hist_da, label="δa", ylabel="[m]", title="Time History of δa (Semi-Major Axis)", lw=2, color=:blue)
    vline!(p2, [drift_duration_days], color=:gray, ls=:dash, label=false)
    p3 = plot(t_days, hist_diy, label="δiy", ylabel="[m]", title="Time History of δiy (Cross-Track Drift)", lw=2, color=:purple); hline!(p3, [0.0], ls=:dash, color=:black)
    vline!(p3, [drift_duration_days], color=:gray, ls=:dash, label=false)
    p_combined = plot(p1, p2, p3, layout=(3,1), size=(800, 800))
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "full_mission_timesharing_v2_$(timestamp).png"
    savefig(p_combined, filename)
    display(p_combined)
end

run_full_mission_timesharing_simulation_final_v2()

# ==============================================================================
# δiy優先制御 (δλのドリフトは許容)(P制御のみだとオフセットあり、PI制御でδiyは収束)
# ==============================================================================
function run_diy_priority_simulation()
    println("\n" * "="^60)
    println("δiy優先制御シミュレーション")
    println("方針: δiyをゼロに保つことを最優先し、δλが流れるのは許容する")
    println("="^60)

    # --- 設定 ---
    drift_duration_days = 1.0
    control_duration_orbits = 200.0
    dv_R_input = 0.0 

    # --- 初期化 ---
    n_init = sqrt(mu_earth / a_c_stm_init^3)
    oe_c_init = OrbitalElementsClassical(a_c_stm_init, e_c_stm_init, i_c_stm_init, Omega_c_stm_init, omega_c_stm_init, 0.0, n_init, M_c_stm_init)
    rho = RHO_LEO; Bc = BC_CHIEF; dB_natural = DELTA_B_INIT

    # 初期ドリフト計算などの準備 (省略せず記述)
    drift_duration_sec = drift_duration_days * 86400.0
    omega_c_t0 = oe_c_init.omega
    omega_dot, Omega_dot = get_secular_j2_rates_koenig(oe_c_init.a, oe_c_init.e, oe_c_init.i)
    omega_c_tf = omega_c_t0 + omega_dot * drift_duration_sec
    J_t0 = get_J_qns_augmented_koenig(omega_c_t0)
    J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_tf)
    A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_init.a, oe_c_init.e, oe_c_init.i, omega_c_t0, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
    STM_drift = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, drift_duration_sec, oe_c_init.e, true, DENSITY_MODEL_SPECIFIC)
    M_trans = J_tf_inv * STM_drift * J_t0
    u_0 = oe_c_init.M + oe_c_init.omega 
    C_gve = 1.0 / (oe_c_init.n * oe_c_init.a)
    da_from_R, dl_from_R = 0.0, -2*C_gve*dv_R_input
    dex_from_R, dey_from_R = C_gve*sin(u_0)*dv_R_input, -C_gve*cos(u_0)*dv_R_input
    term_drift_drag = M_trans[2,7] * dB_natural
    term_drift_R = M_trans[2,2]*dl_from_R + M_trans[2,3]*dex_from_R + M_trans[2,4]*dey_from_R
    term_coeff_T = M_trans[2,1]*(2*C_gve) + M_trans[2,3]*(2*C_gve*cos(u_0)) + M_trans[2,4]*(2*C_gve*sin(u_0))
    target_dlambda_meters = -60.0 # 初期ターゲット
    req_dv_T = (target_dlambda_meters / oe_c_init.a - term_drift_R - term_drift_drag) / term_coeff_T
    
    da_0  = 2*C_gve*req_dv_T
    dl_0  = -2*C_gve*dv_R_input
    dex_0 = C_gve*(sin(u_0)*dv_R_input + 2*cos(u_0)*req_dv_T)
    dey_0 = C_gve*(-cos(u_0)*dv_R_input + 2*sin(u_0)*req_dv_T)
    roe_curr = SVector(da_0, dl_0, dex_0, dey_0, 0.0, 0.0, dB_natural)
    oe_c_curr = oe_c_init
    current_time = 0.0

    hist_t = Float64[]; hist_dl = Float64[]; hist_da = Float64[]; hist_diy = Float64[]
    
    # Phase 1: ドリフト
    dt_drift = 600.0
    steps_drift = Int(drift_duration_sec / dt_drift)
    for i in 1:steps_drift
        push!(hist_t, current_time); push!(hist_dl, roe_curr[2]*oe_c_curr.a); push!(hist_da, roe_curr[1]*oe_c_curr.a); push!(hist_diy, roe_curr[6]*oe_c_curr.a)
        omega_c = oe_c_curr.omega
        omega_dot_step, Omega_dot_step = get_secular_j2_rates_koenig(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i)
        omega_c_next = omega_c + omega_dot_step * dt_drift
        J_t0_step = get_J_qns_augmented_koenig(omega_c)
        J_tf_inv_step = get_J_qns_inv_augmented_koenig(omega_c_next)
        A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, omega_c, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
        STM_step = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, dt_drift, oe_c_curr.e, true, DENSITY_MODEL_SPECIFIC)
        roe_next = J_tf_inv_step * STM_step * J_t0_step * roe_curr
        roe_curr = SVector{7,Float64}(Float64.(roe_next))
        new_M = mod(oe_c_curr.M + oe_c_curr.n * dt_drift, 2*pi)
        new_omega = mod(oe_c_curr.omega + omega_dot_step * dt_drift, 2*pi)
        new_RAAN = mod(oe_c_curr.RAAN + Omega_dot_step * dt_drift, 2*pi)
        oe_c_curr = OrbitalElementsClassical(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, new_RAAN, new_omega, 0.0, oe_c_curr.n, new_M)
        current_time += dt_drift
    end
    
    # Phase 2: 制御 (δiy優先)
    println("[Phase 2] δiy優先制御開始")
    
    dt_control = SIM_SEGMENT_ORBITS * (2*pi/oe_c_curr.n)
    steps_control = Int((control_duration_orbits * 2*pi/oe_c_curr.n) / dt_control)
    current_delta_B = roe_curr[7]
    alpha = 1.0 - exp(-dt_control / ATTITUDE_CHANGE_TIMECONSTANT_SEC)
    # 積分器の初期化
    integ_diy = 0.0
    KI_IY = 0.001 # 積分ゲイン (小さめに)
    for i in 1:steps_control
        val_diy = roe_curr[6] * oe_c_curr.a
        err_diy = val_diy - 0.0 # 目標は常に0
        
        # # δiyをゼロにするためのδaだけを計算 (δλは無視)
        # # 符号は以前の成功例 (プラス) を採用
        # target_da_total = 20.0 * err_diy 
        # target_da_total = clamp(target_da_total, -200.0, 200.0)

        # 積分更新
        integ_diy += err_diy * dt_control        
        # PI制御: P項 + I項
        target_da_total = (20.0 * err_diy) + (KI_IY * integ_diy)
        
        target_da_nd = target_da_total / oe_c_curr.a
        err_da_nd = roe_curr[1] - target_da_nd
        delta_B_req = err_da_nd * oe_c_curr.a * 1.5
        delta_B_command = clamp(delta_B_req, DELTA_B_MIN, DELTA_B_MAX)
        
        push!(hist_t, current_time); push!(hist_dl, roe_curr[2]*oe_c_curr.a); push!(hist_da, roe_curr[1]*oe_c_curr.a); push!(hist_diy, val_diy)
        
        omega_dot_step, Omega_dot_step = get_secular_j2_rates_koenig(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i)
        omega_c_next = oe_c_curr.omega + omega_dot_step * dt_control
        J_t0_step = get_J_qns_augmented_koenig(oe_c_curr.omega)
        J_tf_inv_step = get_J_qns_inv_augmented_koenig(omega_c_next)
        A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, oe_c_curr.omega, true, true, DENSITY_MODEL_SPECIFIC, rho, Bc)
        STM_step = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, dt_control, oe_c_curr.e, true, DENSITY_MODEL_SPECIFIC)
        roe_vec_input = SVector(roe_curr[1], roe_curr[2], roe_curr[3], roe_curr[4], roe_curr[5], roe_curr[6], current_delta_B)
        roe_next = J_tf_inv_step * STM_step * J_t0_step * roe_vec_input
        roe_curr = SVector{7,Float64}(Float64.(roe_next))
        new_M = mod(oe_c_curr.M + oe_c_curr.n * dt_control, 2*pi)
        new_omega = mod(oe_c_curr.omega + omega_dot_step * dt_control, 2*pi)
        new_RAAN = mod(oe_c_curr.RAAN + Omega_dot_step * dt_control, 2*pi)
        oe_c_curr = OrbitalElementsClassical(oe_c_curr.a, oe_c_curr.e, oe_c_curr.i, new_RAAN, new_omega, 0.0, oe_c_curr.n, new_M)
        current_delta_B = alpha * delta_B_command + (1.0 - alpha) * current_delta_B
        current_time += dt_control
    end

    default(dpi=300, guidefontsize=9, tickfontsize=7, legendfontsize=7, margin=5Plots.mm)
    t_days = hist_t ./ 86400.0
    p1 = plot(t_days, hist_dl, label="δλ", ylabel="[m]", title="Time History of δλ (Drifting)", lw=2)
    vline!(p1, [drift_duration_days], color=:gray, ls=:dash, label="Control")
    p2 = plot(t_days, hist_da, label="δa", ylabel="[m]", title="Time History of δa (Control Input)", lw=2, color=:blue)
    vline!(p2, [drift_duration_days], color=:gray, ls=:dash, label=false)
    p3 = plot(t_days, hist_diy, label="δiy", ylabel="[m]", title="Time History of δiy (Stabilized)", lw=2, color=:purple); hline!(p3, [0.0], ls=:dash, color=:black)
    vline!(p3, [drift_duration_days], color=:gray, ls=:dash, label=false)
    p_combined = plot(p1, p2, p3, layout=(3,1), size=(800, 800))
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "diy_priority_simulation_$(timestamp).png"
    savefig(p_combined, filename)
    display(p_combined)
end

run_diy_priority_simulation()



# 誤差生成
function apply_separation_error(dv_nominal::Vector)
    mag = norm(dv_nominal)
    if mag == 0.0; return dv_nominal; end
    
    mag_noisy = mag * (1.0 + rand(Normal(0, SIGMA_MAG)))
    
    direction = dv_nominal / mag
    perturbation = randn(3)
    perturbation -= dot(perturbation, direction) * direction 
    if norm(perturbation) > 1e-9
        perturbation = normalize(perturbation) * tan(rand(Normal(0, SIGMA_ANG)))
        direction_noisy = normalize(direction + perturbation)
    else
        direction_noisy = direction
    end
    return mag_noisy * direction_noisy
end

# 1回の試行計算
function propagate_stm_scenario(oe_c0, dv_rtn, time_vec)
    # --- 1. 分離Δv を 初期ROE に変換 (GVE) ---
    n = oe_c0.n
    a = oe_c0.a
    u_0 = oe_c0.M + oe_c0.omega # 緯度引数
    C_gve = 1.0 / (n * a) # 正規化係数 (近似)
    
    dv_R, dv_T, dv_N = dv_rtn[1], dv_rtn[2], dv_rtn[3]
    
    # GVE (Impulse -> ROE)
    da_0  = 2 * C_gve * dv_T
    dl_0  = -2 * C_gve * dv_R
    dex_0 = C_gve * (sin(u_0) * dv_R + 2 * cos(u_0) * dv_T)
    dey_0 = C_gve * (-cos(u_0) * dv_R + 2 * sin(u_0) * dv_T)
    dix_0 = C_gve * cos(u_0) * dv_N
    diy_0 = C_gve * sin(u_0) * dv_N
    
    # ROEベクトル (7要素: δa, δλ, δex, δey, δix, δiy, δB)
    # ここで空気抵抗差 DELTA_B_INIT をセット
    roe_vec_init = [da_0, dl_0, dex_0, dey_0, dix_0, diy_0, DELTA_B_INIT]
    
    # --- 2. STMによる伝播 ---
    r_rtn_history = zeros(3, length(time_vec))
    
    # ChiefのJ2レート
    omega_dot, Omega_dot = get_secular_j2_rates_koenig(oe_c0.a, oe_c0.e, oe_c0.i)
    
    # 伝播に必要な行列 (t=0時点)
    J_t0 = get_J_qns_augmented_koenig(oe_c0.omega)
    
    # STM計算用のA行列 (定数とみなす)
    A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(
        oe_c0.a, oe_c0.e, oe_c0.i, oe_c0.omega, 
        true, true, DENSITY_MODEL_SPECIFIC, RHO_LEO, BC_CHIEF
    )
    
    for (k, t) in enumerate(time_vec)
        # 現在のChiefの状態 (J2考慮)
        omega_c_curr = oe_c0.omega + omega_dot * t
        
        # STM計算 (ユーザー関数利用)
        STM = get_STM_prime_qns_augmented_koenig_model_selectable(
            A_kep, A_j2, A_drag, t, oe_c0.e, 
            true, DENSITY_MODEL_SPECIFIC
        )
        
        # 逆回転行列 (現在時刻)
        J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_curr)
        
        # ROE伝播: roe(t) = J(t)^-1 * STM(t,0) * J(0) * roe(0)
        roe_curr_vec = J_tf_inv * STM * J_t0 * roe_vec_init
        
        # --- 3. ROE -> RTN位置変換 ---
        # Chiefの現在位置
        M_c_curr = oe_c0.M + oe_c0.n * t # 平均運動のみ更新(簡易)
        oe_c_curr = OrbitalElementsClassical(oe_c0.a, oe_c0.e, oe_c0.i, oe_c0.RAAN, omega_c_curr, 0.0, oe_c0.n, M_c_curr)
        
        # DeputyのOE復元
        oe_d_curr = final_roe_to_deputy_oe(oe_c_curr, roe_curr_vec)
        
        # 位置ベクトル計算
        sv_c = orbital_elements_to_sv(oe_c_curr)
        sv_d = orbital_elements_to_sv(oe_d_curr)
        r_c = SVector(sv_c[1], sv_c[2], sv_c[3]); v_c = SVector(sv_c[4], sv_c[5], sv_c[6])
        r_d = SVector(sv_d[1], sv_d[2], sv_d[3])
        
        # 相対位置 (RTN)
        dr_rtn = eci_to_rtn(r_c, v_c, r_d - r_c) # ユーザー定義関数使用
        
        r_rtn_history[:, k] = dr_rtn
    end
    
    return r_rtn_history
end

# モンテカルロ実行
function run_safety_analysis(scenario_name, dv_nominal, t_vec, oe_init)
    println("Simulating: $scenario_name ...")
    min_distances = Float64[]
    collision_count = 0
    trajectories = [] 
    
    for i in 1:NUM_SAMPLES
        dv_noisy = apply_separation_error(dv_nominal)
        r_hist = propagate_stm_scenario(oe_init, dv_noisy, t_vec)
        
        dists = [norm(r_hist[:, k]) for k in 1:size(r_hist, 2)]
        min_dist = minimum(dists)
        push!(min_distances, min_dist)
        
        if min_dist < SAFETY_THRESHOLD_M
            collision_count += 1
        end
        if i <= 50; push!(trajectories, r_hist); end
    end
    
    prob = (collision_count / NUM_SAMPLES) * 100.0
    println("  Collision Prob: $prob %, Mean Min Dist: $(mean(min_distances)) m")
    return min_distances, prob, trajectories
end

# =========================================================================
# 4. メイン実行部
# =========================================================================
function run_main()
    println("\n" * "="^60)
    println("分離安全性評価 (STM + 空気抵抗 + ΔB)")
    println("="^60)
    
    # 初期条件 (高度500km, 太陽同期)
    a0 = R_E + 500.0*1000 # m
    e0 = 0.001
    i0 = deg2rad(97.4)
    # 北極分離を想定 (u = 90 deg -> M+w = 90)
    w0 = deg2rad(0.0)
    M0 = deg2rad(90.0) 
    n0 = sqrt(mu_earth / a0^3)
    oe_c0 = OrbitalElementsClassical(a0, e0, i0, 0.0, w0, 0.0, n0, M0)
    
    # 時間ベクトル
    t_vec = collect(0.0 : DT_SEC : SIM_DURATION_ORBITS*ORBIT_PERIOD_SEC)
    
    # --- シナリオ設定 ---
    
    # Case 1: Along-Track (空気抵抗で戻ってくる設定)
    # 前方(+T)に放出 -> aが増加 -> 周期が遅くなる -> 後ろへ下がる(y < 0)
    # しかし ΔB > 0 (副衛星が抵抗大) -> aが減少 -> 加速 -> 前へ進む力
    # バランスにより、ある地点で折り返して戻ってくるはず
    dv1 = [0.0, 0.05, 0.0] # m/s
    
    # Case 2: E/I Helix (螺旋)
    dv2 = [0.03, 0.0, 0.04] # m/s
    
    # 実行
    res1 = run_safety_analysis("Case 1: Along-Track w/ Drag", dv1, t_vec, oe_c0)
    res2 = run_safety_analysis("Case 2: E/I Helix", dv2, t_vec, oe_c0)
    
    # プロット
    default(dpi=300, guidefontsize=9, tickfontsize=7, legendfontsize=7, margin=5Plots.mm)
    
    # 1. Along-Track (y-x面: 進行-半径)
    p1 = plot(title="Case 1: Along-Track (y-x)", xlabel="Along-Track [m]", ylabel="Radial [m]")
    for t in res1[3]; plot!(p1, t[2,:], t[1,:], alpha=0.3, color=:blue, label=""); end
    scatter!(p1, [0], [0], label="Chief", color=:red, markersize=3)
    
    # 2. Helix (x-z面: 半径-面外)
    p2 = plot(title="Case 2: Helix (x-z)", xlabel="Radial [m]", ylabel="Cross-Track [m]", aspect_ratio=:equal)
    for t in res2[3]; plot!(p2, t[1,:], t[3,:], alpha=0.3, color=:green, label=""); end
    # 安全円
    th = 0:0.1:2pi+0.1
    plot!(p2, SAFETY_THRESHOLD_M.*cos.(th), SAFETY_THRESHOLD_M.*sin.(th), color=:red, lw=2, label="50m", ls=:dash)
    
    # 3. ヒストグラム
    p3 = histogram(res1[1], alpha=0.5, label="Along", xlabel="Min Dist [m]", ylabel="Count", bin=20, color=:blue)
    histogram!(p3, res2[1], alpha=0.5, label="Helix", color=:green)
    
    display(plot(p1, p2, p3, layout=(3,1), size=(600, 900)))
end

run_main()

# =========================================================================
# 5. メイン実行部 (v11: ステップ監視・完全浄化版)
# =========================================================================

function normalize_angle_diff(angle)
    return mod(angle + pi, 2*pi) - pi
end

# 1回の試行計算 (異常検知付き)
function propagate_stm_flow_v11(oe_c0, dv_rtn, time_vec)
    # --- 初期化 ---
    n = oe_c0.n; a = oe_c0.a
    u_0 = oe_c0.M + oe_c0.omega 
    C_gve = 1.0 / (n * a)
    
    dv_R, dv_T, dv_N = dv_rtn[1], dv_rtn[2], dv_rtn[3]
    
    da_0  = 2 * C_gve * dv_T
    dl_0  = -2 * C_gve * dv_R
    dex_0 = C_gve * (sin(u_0) * dv_R + 2 * cos(u_0) * dv_T)
    dey_0 = C_gve * (-cos(u_0) * dv_R + 2 * sin(u_0) * dv_T)
    dix_0 = C_gve * cos(u_0) * dv_N
    diy_0 = C_gve * sin(u_0) * dv_N
    
    roe_vec_init = [da_0, dl_0, dex_0, dey_0, dix_0, diy_0, DELTA_B_INIT]
    
    omega_dot, Omega_dot = get_secular_j2_rates_koenig(oe_c0.a, oe_c0.e, oe_c0.i)
    J_t0 = get_J_qns_augmented_koenig(oe_c0.omega)
    A_kep, A_j2, A_drag = get_A_prime_qns_augmented_koenig_selectable(oe_c0.a, oe_c0.e, oe_c0.i, oe_c0.omega, true, true, DENSITY_MODEL_SPECIFIC, RHO_LEO, BC_CHIEF)
    
    r_rtn_history = zeros(3, length(time_vec))
    
    # 異常検知用の閾値 (前回位置からのジャンプ許容量: 100km/step, 絶対距離: 5000km)
    JUMP_THRESHOLD = 100000.0 
    ABS_THRESHOLD  = 5000000.0 
    
    is_valid = true # 成功フラグ
    # # --- デバッグ挿入コード 1 ---
    # println("\n=== DEBUG: t=0 Check ===")

    # # 1. 主衛星の初期OE
    # println("Chief Init OE:")
    # println("  a=$(oe_c0.a), e=$(oe_c0.e), i=$(oe_c0.i), M=$(oe_c0.M)")

    # # 2. 誤差なしのΔv (入力)
    # println("Input Delta V (RTN): $dv_rtn")

    # # 3. ROE変換直後の値 (GVEの結果)
    # println("Initial ROE (from GVE):")
    # println("  da=$da_0")
    # println("  dl=$dl_0")
    # println("  dex=$dex_0, dey=$dey_0") 
    # println("  dix=$dix_0, diy=$diy_0")

    # # --- デバッグ挿入コード 2 ---
    # # 4. t=0 での復元 (STMを通さず、初期ROEを直接変換)
    # # Chief (t=0)
    # oe_c_0 = oe_c0 
    # # Deputy (t=0) 復元
    # oe_d_0_debug = final_roe_to_deputy_oe(oe_c_0, roe_vec_init)

    # println("Restored Deputy OE (t=0):")
    # println("  a=$(oe_d_0_debug.a)")
    # println("  M=$(oe_d_0_debug.M) (Chief M: $(oe_c_0.M))")
    # println("  Ω=$(oe_d_0_debug.RAAN) (Chief Ω: $(oe_c_0.RAAN))")
    # println("  ω=$(oe_d_0_debug.omega) (Chief ω: $(oe_c_0.omega))")

    # # 角度差チェック
    # diff_M_debug = normalize_angle_diff(oe_d_0_debug.M - oe_c_0.M)
    # println("  Diff M (rad): $diff_M_debug")

    # # --- デバッグ挿入コード 3 ---
    # # 5. ECI位置計算
    # sv_c_0 = orbital_elements_to_sv(oe_c_0)
    # sv_d_0 = orbital_elements_to_sv(oe_d_0_debug)
    # r_c_0 = SVector(sv_c_0[1], sv_c_0[2], sv_c_0[3])
    # r_d_0 = SVector(sv_d_0[1], sv_d_0[2], sv_d_0[3])

    # # 6. RTN位置計算
    # dr_rtn_0 = eci_to_rtn(r_c_0, SVector(sv_c_0[4], sv_c_0[5], sv_c_0[6]), r_d_0 - r_c_0)

    # println("Initial Position Error (RTN) at t=0:")
    # println("  R (x): $(dr_rtn_0[1]) m")
    # println("  T (y): $(dr_rtn_0[2]) m")
    # println("  N (z): $(dr_rtn_0[3]) m")
    # println("========================\n")
    
    for (k, t) in enumerate(time_vec)
        # 1. ROE伝播
        omega_c_curr = oe_c0.omega + omega_dot * t
        STM = get_STM_prime_qns_augmented_koenig_model_selectable(A_kep, A_j2, A_drag, t, oe_c0.e, true, DENSITY_MODEL_SPECIFIC)
        J_tf_inv = get_J_qns_inv_augmented_koenig(omega_c_curr)
        roe_curr_vec = J_tf_inv * STM * J_t0 * roe_vec_init
        
        # 2. OE復元 & 補正
        M_c_curr = mod(oe_c0.M + oe_c0.n * t, 2*pi)
        oe_c_curr = OrbitalElementsClassical(oe_c0.a, oe_c0.e, oe_c0.i, oe_c0.RAAN, omega_c_curr, 0.0, oe_c0.n, M_c_curr)
        oe_d_curr = final_roe_to_deputy_oe(oe_c_curr, roe_curr_vec)
        
        diff_M = normalize_angle_diff(oe_d_curr.M - oe_c_curr.M)
        oe_d_adjusted = OrbitalElementsClassical(
            oe_d_curr.a, oe_d_curr.e, oe_d_curr.i, oe_d_curr.RAAN, oe_d_curr.omega, 0.0, oe_d_curr.n, 
            oe_c_curr.M + diff_M 
        )
        
        # 3. 位置計算
        sv_c = orbital_elements_to_sv(oe_c_curr)
        sv_d = orbital_elements_to_sv(oe_d_adjusted)
        r_c = SVector(sv_c[1], sv_c[2], sv_c[3]); v_c = SVector(sv_c[4], sv_c[5], sv_c[6])
        r_d = SVector(sv_d[1], sv_d[2], sv_d[3])
        
        dr_rtn = eci_to_rtn(r_c, v_c, r_d - r_c)
        r_rtn_history[:, k] = dr_rtn
        
        # ★サニティチェック (異常値検出)
        current_dist = norm(dr_rtn)
        
        # チェック1: 絶対距離が異常に大きい
        if current_dist > ABS_THRESHOLD
            # println("Warning: Divergence detected at step $k (t=$t). Dist: $current_dist")
            is_valid = false
            break
        end
        
        # チェック2: 1ステップで異常に飛んだ (k>1のみ)
        if k > 1
            prev_dist = norm(r_rtn_history[:, k-1])
            jump = norm(dr_rtn - r_rtn_history[:, k-1])
            if jump > JUMP_THRESHOLD
                # println("Warning: Position Jump detected at step $k. Jump: $jump")
                is_valid = false
                break
            end
        end
    end
    
    return r_rtn_history, is_valid
end

function run_separation_risk_assessment_v13(sample_num::Int)
    println("\n" * "="^60)
    println("分離安全性評価 (v13: プロット範囲修正版)")
    println("Helixも t=0 から描画し、初期位置ズレがないことを確認する")
    println("="^60)
    
    # --- 設定 ---
    a0 = R_E + 500.0*1000 
    n0 = sqrt(mu_earth / a0^3)
    oe_c0 = OrbitalElementsClassical(a0, 0.001, deg2rad(97.4), 0.0, 0.0, 0.0, n0, deg2rad(90.0))
    
    sim_duration = 3.0 * ORBIT_PERIOD_SEC 
    t_vec = collect(0.0 : 30.0 : sim_duration)
    JUDGE_START_TIME = 10.0 * 60.0 
    
    # デバッグ用に大きなΔvを入れて確認する場合はここを変更
    # dv2 = [1.0, 0.0, 1.0] 
    dv1 = [0.0, 0.02, 0.0] 
    dv2 = [0.15, 0.0, 0.15] 
    
    # --- 評価関数 ---
    function evaluate_risk(label, dv_nom)
        println("Simulating: $label ...")
        min_dists = Float64[]
        collision_count = 0
        trajs = []
        
        for i in 1:sample_num
            dv_noisy = apply_separation_error(dv_nom)
            # v12で作った「オフセット補正付き」の伝播関数を使用推奨
            # (もしv11以前を使うなら初期位置ズレが数m残りますが、20mにはなりません)
            r_hist, is_valid = propagate_stm_flow_v11(oe_c0, dv_noisy, t_vec)
            
            if !is_valid; continue; end
            
            # 判定 (ここは10分後からでOK)
            valid_indices = findall(t -> t > JUDGE_START_TIME, t_vec)
            if isempty(valid_indices)
                current_min = 999999.9
            else
                dists = [norm(r_hist[:, k]) for k in valid_indices]
                current_min = minimum(dists)
            end
            
            push!(min_dists, current_min)
            if current_min < SAFETY_THRESHOLD_M
                collision_count += 1
            end
            push!(trajs, r_hist)
        end
        
        valid_num = length(min_dists)
        prob = valid_num > 0 ? (collision_count / valid_num) * 100.0 : 0.0
        println("  -> 成功: $valid_num / $sample_num")
        println("  -> 衝突確率: $(round(prob, digits=1)) %")
        
        return trajs
    end

    trajs1 = evaluate_risk("Case 1: Along-Track", dv1)
    trajs2 = evaluate_risk("Case 2: Helix", dv2)

    # --- プロット修正 ---
    default(dpi=300, guidefontsize=9, tickfontsize=7, legendfontsize=7, margin=5Plots.mm)
    
    # P1: Along-Track
    p1 = plot(title="Case 1: Along-Track", xlabel="Along-Track [m]", ylabel="Radial [m]")
    for t in trajs1
        plot!(p1, t[2,:], t[1,:], color=:blue, alpha=0.3, label="")
    end
    scatter!(p1, [0], [0], label="Chief", color=:red)

    # P2: Zoom
    p2 = plot(title="Case 1: Zoom-in", xlabel="Along-Track [m]", ylabel="Radial [m]")
    for t in trajs1
        plot!(p2, t[2,:], t[1,:], color=:blue, alpha=0.3, label="")
    end
    plot!(p2, xlims=(-300, 300), ylims=(-150, 150))
    th = 0:0.1:2pi+0.1
    plot!(p2, SAFETY_THRESHOLD_M.*cos.(th), SAFETY_THRESHOLD_M.*sin.(th), color=:red, lw=2, label="50m Zone")

    # P3: Helix (x-z)
    p3 = plot(title="Case 2: Helix (x-z)", xlabel="Radial [m]", ylabel="Cross-Track [m]", aspect_ratio=:equal)
    
    # ★ここを修正しました★
    # valid_idx を使わず、全ての時間ステップ (:) を描画します
    for t in trajs2
        plot!(p3, t[1,:], t[3,:], color=:green, alpha=0.3, label="")
    end
    
    plot!(p3, SAFETY_THRESHOLD_M.*cos.(th), SAFETY_THRESHOLD_M.*sin.(th), color=:red, lw=2, label="50m Zone")

    l = @layout [a; b c]
    p_combined = plot(p1, p2, p3, layout=l, size=(1000, 800))
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "montecarlo_result_v13_$(timestamp).png"
    savefig(p_combined, filename)
    println("グラフを保存しました: $filename")
    display(p_combined)
end

# 実行
run_separation_risk_assessment_v13(20)