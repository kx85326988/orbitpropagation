# 物理定数
const P_SRP = 4.56e-6  # Solar Radiation Pressure [N/m^2]
const mu_earth = 3.986004418e14
const J2_coeff = 1.08263e-3
const R_E = 6378137.0
const a_c = 6903137.0

# 衛星パラメータ
m1 = 1.5  # kg
m2 = 3.0  # kg
Cr1 = 1.4
Cr2 = 1.4
S1 = 0.1 * 0.1 # m^2
S2_min = 0.1 * 0.1 # m^2
S2_max = 0.1 * 0.2 # m^2

# 弾道係数
B1 = Cr1 * S1 / m1
B2_min = Cr2 * S2_min / m2
B2_max = Cr2 * S2_max / m2

# 最大の相対加速度
delta_B_max = abs(B1 - B2_min)
delta_g_srp_max = P_SRP * delta_B_max
println("太陽輻射圧で生成可能な最大の相対加速度: ", delta_g_srp_max, " m/s^2")

# 相対J2摂動に対抗可能な距離
# 絶対的なJ2摂動の概算
abs_a_j2 = (3/2) * mu_earth * J2_coeff * R_E^2 / a_c^4
println("軌道上での絶対的なJ2摂動力のオーダー: ", abs_a_j2, " m/s^2")

max_controllable_distance = (delta_g_srp_max * a_c) / abs_a_j2
println("太陽輻射圧で相対J2摂動を打ち消せる最大分離距離: ", max_controllable_distance, " m")