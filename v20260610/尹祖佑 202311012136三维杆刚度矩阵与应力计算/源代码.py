import numpy as np

def truss3d_element_stiffness(x1, x2, E, A):
    """
    计算三维杆单元长度、方向余弦、全局刚度矩阵Ke
    :param x1: 节点1坐标 [x1, y1, z1]
    :param x2: 节点2坐标 [x2, y2, z2]
    :param E: 弹性模量 (Pa)
    :param A: 横截面积 (m^2)
    :return: L, (cx, cy, cz), Ke
    """
    # 坐标差
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]

    # 单元长度
    L = np.sqrt(dx**2 + dy**2 + dz**2)

    # 检查退化单元（两节点重合）
    if np.isclose(L, 0.0):
        raise ValueError("错误：两个节点坐标重合，单元退化，无法计算！")

    # 方向余弦
    cx = dx / L
    cy = dy / L
    cz = dz / L
    dir_cos = (cx, cy, cz)

    # 构造6×6刚度矩阵 Ke
    cxx = cx * cx
    cxy = cx * cy
    cxz = cx * cz
    cyy = cy * cy
    cyz = cy * cz
    czz = cz * cz

    Ke = (E * A / L) * np.array([
        [cxx, cxy, cxz, -cxx, -cxy, -cxz],
        [cxy, cyy, cyz, -cxy, -cyy, -cyz],
        [cxz, cyz, czz, -cxz, -cyz, -czz],
        [-cxx, -cxy, -cxz, cxx, cxy, cxz],
        [-cxy, -cyy, -cyz, cxy, cyy, cyz],
        [-cxz, -cyz, -czz, cxz, cyz, czz]
    ])

    return L, dir_cos, Ke


def truss3d_element_stress(x1, x2, E, A, de):
    """
    根据节点位移计算应变、应力、轴力
    :param x1: 节点1坐标 [x,y,z]
    :param x2: 节点2坐标 [x,y,z]
    :param E: 弹性模量 (Pa)
    :param A: 截面积 (m^2)
    :param de: 节点位移列阵 [u1,v1,w1,u2,v2,w2]
    :return: epsilon(应变), sigma(应力,Pa), N(轴力,N)
    """
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]
    L = np.sqrt(dx**2 + dy**2 + dz**2)

    if np.isclose(L, 0.0):
        raise ValueError("错误：两个节点坐标重合，单元退化，无法计算！")

    cx = dx / L
    cy = dy / L
    cz = dz / L

    # 应变位移矩阵 B
    B = np.array([-cx, -cy, -cz, cx, cy, cz]) / L
    epsilon = B @ np.array(de)

    # 应力、轴力
    sigma = E * epsilon
    N = sigma * A

    return epsilon, sigma, N


def main():
    np.set_printoptions(precision=6, suppress=True)
    print("========== 三维空间桁架单元计算程序 ==========\n")

    # -------------------- 算例1：沿X轴一维杆单元 --------------------
    print("---------- 算例1：沿X轴一维杆单元 ----------")
    x1_1 = [0, 0, 0]
    x2_1 = [2, 0, 0]
    E1 = 200e9    # 200 GPa
    A1 = 1.0e-4   # m^2
    de1 = [0, 0, 0, 1.0e-3, 0, 0]

    try:
        L1, dir1, Ke1 = truss3d_element_stiffness(x1_1, x2_1, E1, A1)
        eps1, sig1, N1 = truss3d_element_stress(x1_1, x2_1, E1, A1, de1)

        print(f"单元长度 L = {L1:.4f} m")
        print(f"方向余弦 (cx, cy, cz) = ({dir1[0]:.4f}, {dir1[1]:.4f}, {dir1[2]:.4f})")
        print("全局刚度矩阵 Ke：")
        print(Ke1)
        print(f"轴向应变 ε = {eps1:.6e}")
        print(f"轴向应力 σ = {sig1/1e6:.4f} MPa")
        print(f"单元轴力 N = {N1:.4e} N\n")
    except Exception as e:
        print(e)

    # -------------------- 算例2：空间任意方向杆单元 --------------------
    print("---------- 算例2：空间任意方向杆单元 ----------")
    x1_2 = [0, 0, 0]
    x2_2 = [1, 2, 2]
    E2 = 210e9    # 210 GPa
    A2 = 2.0e-4   # m^2
    de2 = [0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3]

    try:
        L2, dir2, Ke2 = truss3d_element_stiffness(x1_2, x2_2, E2, A2)
        eps2, sig2, N2 = truss3d_element_stress(x1_2, x2_2, E2, A2, de2)

        print(f"单元长度 L = {L2:.4f} m")
        print(f"方向余弦 (cx, cy, cz) = ({dir2[0]:.4f}, {dir2[1]:.4f}, {dir2[2]:.4f})")
        print("全局刚度矩阵 Ke：")
        print(Ke2)

        # 验证刚度矩阵对称性
        is_symmetric = np.allclose(Ke2, Ke2.T)
        print(f"刚度矩阵是否对称：{is_symmetric}")

        # 验证刚体平移：整体平移位移，内力应为0
        de_rigid = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
        eps_r, sig_r, N_r = truss3d_element_stress(x1_2, x2_2, E2, A2, de_rigid)
        print(f"刚体平移下应变 ε = {eps_r:.2e} (趋近于0)")
        print(f"刚体平移下轴力 N = {N_r:.2e} (趋近于0)")

        # 计算特征值，验证半正定
        eig_vals = np.linalg.eigvals(Ke2)
        print("刚度矩阵特征值：")
        print(np.round(eig_vals, 4))
        print(f"轴向应变 ε = {eps2:.6e}")
        print(f"轴向应力 σ = {sig2/1e6:.4f} MPa")
        print(f"单元轴力 N = {N2:.4e} N\n")
    except Exception as e:
        print(e)

    # -------------------- 任务4：刚度矩阵物理意义验证 --------------------
    print("---------- 任务4：刚度矩阵物理意义验证 ----------")
    # 选取算例2单元，令第1个自由度位移=1，其余为0
    de_test = np.zeros(6)
    de_test[0] = 1.0
    _, _, Ke_test = truss3d_element_stiffness(x1_2, x2_2, E2, A2)
    Fe = Ke_test @ de_test
    print(f"位移向量 de = {de_test}")
    print(f"节点力向量 Fe = Ke * de = ")
    print(Fe)
    print("结论：Fe 等于刚度矩阵 Ke 的第 1 列向量。")


if __name__ == "__main__":
    main()