import numpy as np

def invert_transform(T):
    """
    对 4x4 齐次变换矩阵求逆（刚体变换）。
    T = [R t
         0 1]
    T_inv = [R^T  -R^T t
             0      1 ]
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"期望 (4,4)，实际是 {T.shape}")

    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ✅ 正确写法：外面再包一层 []
T = [
    [-0.62675009,  0.52427434, -0.57647267,  2.47151737],
    [ 0.58412967, -0.17353109, -0.79289312, -0.69130498],
    [-0.51572945, -0.83368062, -0.19748357,  4.04871686],
    [ 0.0,         0.0,         0.0,         1.0]
]

T_inv = invert_transform(T)
print(T_inv)
