using LinearAlgebra
using StaticArrays
using Symbolics

println("--- シンボリックな初期ROE導出プログラムを開始 ---")

# --- ステップ1: シンボル変数の定義 ---
println("\nステップ1: シンボル変数を定義")

# 時間と主衛星の基本パラメータ
@variables t n_c a_c e_c i_c
@variables mu J2 RE

# 主衛星の角度要素（初期と最終）
@variables ω_ci ω_cf Ω_ci Ω_cf

# 分離パラメータ (CW座標系での分離速度成分)
@variables dv_R dv_T dv_N

# J2摂動の置換変数
@variables κ η P Q S T G F

println("完了。")


# --- ステップ2: 初期ROEをΔvの関数として解析的に表現 ---
# ガウスの惑星方程式の線形化版(円軌道近似)を用いる
println("\nステップ2: 初期ROEをΔvの関数として解析的に定義...")

# 主衛星がほぼ円軌道(e_c -> 0)で、軌道上の特定の位置(f_c, u_c)にいると仮定
# f_c (真近点離角) と u_c (緯度引数) もシンボルとして扱う
@variables f_c u_c

# 各初期ROEをdv_R, dv_T, dv_Nの関数として定義
# (出典: JAXA「軌道設計の応用」式(4.2-1)~(4.2-6)等に基づく一般的な線形関係)
δa_0 = (2 / (n_c * a_c)) * dv_T
δex_0 = (sin(f_c) / (n_c * a_c)) * dv_R + (2*cos(f_c) / (n_c * a_c)) * dv_T
δey_0 = (-cos(f_c) / (n_c * a_c)) * dv_R + (2*sin(f_c) / (n_c * a_c)) * dv_T
δλ_0 = (-2 / (n_c * a_c)) * dv_R # 円軌道近似
δix_0 = (cos(u_c) / (n_c * a_c)) * dv_N
δiy_0 = (sin(u_c) / (n_c * a_c)) * dv_N

# 7次元の初期ROEベクトルを作成 (aug_paramは0)
δα_initial_vec = [δa_0, δλ_0, δex_0, δey_0, δix_0, δiy_0, 0]
println("初期ROEベクトルをΔvの関数として構築")


# --- ステップ3: STMの構築と伝播 ---
println("\nステップ3: STMをシンボリックに構築し、ROEを伝播...")

# 変換後の離心率ベクトル成分をシンボルで定義 (t_iで評価)
ex_prime = e_c * cos(ω_ci)
ey_prime = e_c * sin(ω_ci)

# J2とケプラー運動のプラント行列 A' を構築 (論文 式(24) + 式(11))
A_kep_J2_prime = Num.(zeros(7,7))
# ケプラー項
A_kep_J2_prime[2,1] = -1.5*n_c
# J2項
A_kep_J2_prime[2,1] += -3.5*κ*E*P
A_kep_J2_prime[2,3] = κ*e_c*F*G*P
A_kep_J2_prime[2,5] = -κ*F*S
A_kep_J2_prime[4,1] = -3.5*e_c*Q
A_kep_J2_prime[4,3] = 4.0*κ*e_c^2*G*Q
A_kep_J2_prime[4,5] = -5.0*κ*e_c^2*S 
A_kep_J2_prime[6,1] = 3.5*κ*S
A_kep_J2_prime[6,3] = -4.0*κ*e_c^2*G*S 
A_kep_J2_prime[6,5] = 2.0*κ*T             

println("プラント行列をシンボリックに構築")

# STMを計算 (抗力なしなので、単純な形式)
STM_prime = I(7) + A_kep_J2_prime * t
println("STM' をシンボリックに計算完了")

# 変換行列Jもシンボルで定義
function build_symbolic_J(ω)
    J = Num.(zeros(7,7)); J[1,1]=J[2,2]=J[5,5]=J[6,6]=J[7,7]=1
    cω=cos(ω); sω=sin(ω); J[3,3]=cω; J[3,4]=sω; J[4,3]=-sω; J[4,4]=cω
    return J
end
J_ti = build_symbolic_J(ω_ci)
J_tf_inv = build_symbolic_J(-ω_cf)

# 最終ROEを計算
δα_prime_initial = J_ti * δα_initial_vec
δα_prime_final = STM_prime * δα_prime_initial
δα_final_vec = J_tf_inv * δα_prime_final
println("最終ROEの数式を導出しました。")

# --- 4. 最終δiyの解析的な式を抽出・整理 ---
println("\n--- 4. 最終δiyの解析的な式を抽出・整理 ---")

# 式を展開
final_delta_iy_expanded = expand(δα_final_vec[6])

# Symbolics.coeff を使って各係数を抽出
coeff_dv_R = Symbolics.coeff(final_delta_iy_expanded, dv_R)
coeff_dv_T = Symbolics.coeff(final_delta_iy_expanded, dv_T)
coeff_dv_N = Symbolics.coeff(final_delta_iy_expanded, dv_N)

println("\nFinal δiy = (C_R) * dv_R + (C_T) * dv_T + (C_N) * dv_N の形で整理")

println("\n--- C_R (半径方向Δvの係数) ---")
println(simplify(coeff_dv_R))

println("\n--- C_T (軌道速度方向Δvの係数) ---")
println(simplify(coeff_dv_T))

println("\n--- C_N (軌道法線方向Δvの係数) ---")
println(simplify(coeff_dv_N))