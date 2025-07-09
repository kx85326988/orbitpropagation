using LinearAlgebra
using StaticArrays
using Symbolics

"""
指定された数式を、変数 var に関して整理し、係数を表示する
"""
function analyze_coefficient(expression, var_name::String)
    println("\n--- $(var_name) の係数の分析 ---")
    
    # 式を展開して、t の多項式として扱いやすくする
    expanded_expr = expand(expression)
    
    # t^1 の係数 (永年項) を抽出
    secular_coeff = simplify(Symbolics.coeff(expanded_expr, t^1))
    
    # t^0 の係数 (定数項) を抽出
    constant_coeff = simplify(substitute(expanded_expr, Dict(t => 0)))

    println("永年項 (tに比例):")
    println(secular_coeff)
    
    println("\n定数項 (tに非依存):")
    println(constant_coeff)
end


function run_full_symbolic_analysis()
    println("--- 全ての最終ROEの分離速度依存性の解析を開始します ---")

    # --- ステップ1: シンボル変数の定義 ---
    println("\nステップ1: シンボル変数を定義します...")
    @variables t n_c a_c e_c i_c
    @variables ω_ci ω_cf
    @variables dv_R dv_T dv_N
    @variables κ E P F G S T Q
    @variables ex_prime ey_prime
    @variables u_c
    println("完了。")


    # --- ステップ2: 初期ROEをΔvの関数として解析的に表現 ---
    println("\nステップ2: 初期ROEをΔvの関数として定義...")
    δa_0 = (2 / (n_c * a_c)) * dv_T
    δλ_0 = (-2 / (n_c * a_c)) * dv_R
    δex_0 = (sin(u_c - ω_ci) / (n_c * a_c)) * dv_R + (2*cos(u_c - ω_ci) / (n_c * a_c)) * dv_T
    δey_0 = (-cos(u_c - ω_ci) / (n_c * a_c)) * dv_R + (2*sin(u_c - ω_ci) / (n_c * a_c)) * dv_T
    δix_0 = (cos(u_c) / (n_c * a_c)) * dv_N
    δiy_0 = (sin(u_c) / (n_c * a_c)) * dv_N
    δα_initial_vec = [δa_0, δλ_0, δex_0, δey_0, δix_0, δiy_0, 0]
    println("初期ROEベクトルをΔvの関数として構築しました。")


    # --- ステップ3: STMの構築と伝播 ---
    println("\nステップ3: STMをシンボリックに構築し、ROEを伝播...")

    A_kep_J2_prime = Num.(zeros(7,7))
    # ケプラー項
    A_kep_J2_prime[2,1] = -1.5*n_c

    # J2項
    A_kep_J2_prime[2,1] += -3.5*κ*E*P
    A_kep_J2_prime[2,3] = κ*e_c*F*G*P
    A_kep_J2_prime[2,5] = -κ*F*S
    A_kep_J2_prime[4,1] = -3.5*κ*e_c*Q
    A_kep_J2_prime[4,3] = 4.0*κ*e_c^2*G*Q
    A_kep_J2_prime[4,5] = -5.0*κ*e_c^2*S 
    A_kep_J2_prime[6,1] = 3.5*κ*S
    A_kep_J2_prime[6,3] = -4.0*κ*e_c^2*G*S 
    A_kep_J2_prime[6,5] = 2.0*κ*T 

    STM_prime = I(7) + A_kep_J2_prime * t
    
    function build_symbolic_J(ω)
        J = Num.(zeros(7,7)); J[1,1]=J[2,2]=J[5,5]=J[6,6]=J[7,7]=1
        cω=cos(ω); sω=sin(ω); J[3,3]=cω; J[3,4]=sω; J[4,3]=-sω; J[4,4]=cω
        return J
    end
    J_ti = build_symbolic_J(ω_ci)
    J_tf_inv = build_symbolic_J(-ω_cf)

    δα_prime_initial = J_ti * δα_initial_vec
    δα_prime_final = STM_prime * δα_prime_initial
    δα_final_vec = J_tf_inv * δα_prime_final
    println("最終ROEの数式を導出しました。")

    # --- ステップ4: 全ての最終ROEの解析的な式を抽出・整理 ---
    println("\n--- 4. 全ての最終ROEの解析的な式を抽出・整理 ---")

    roe_names = ["δa", "δλ", "δex", "δey", "δix", "δiy"]
    for i in 1:6
        println("\n" * "="^40)
        println("■ Final $(roe_names[i]) の解析")
        println("="^40)
        
        final_roe_expanded = expand(δα_final_vec[i])

        coeff_dv_R = Symbolics.coeff(final_roe_expanded, dv_R)
        coeff_dv_T = Symbolics.coeff(final_roe_expanded, dv_T)
        coeff_dv_N = Symbolics.coeff(final_roe_expanded, dv_N)

        println("\nFinal $(roe_names[i]) = (C_R) * dv_R + (C_T) * dv_T + (C_N) * dv_N の形で整理します。")
        
        analyze_coefficient(coeff_dv_R, "C_R (半径方向Δvの係数)")
        analyze_coefficient(coeff_dv_T, "C_T (軌道速度方向Δvの係数)")
        analyze_coefficient(coeff_dv_N, "C_N (軌道法線方向Δvの係数)")
    end
end

# --- 実行 ---
run_full_symbolic_analysis()