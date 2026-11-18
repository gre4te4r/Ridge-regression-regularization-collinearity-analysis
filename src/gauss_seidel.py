"""
Module cai dat phuong phap lap Gauss-Seidel cho he phuong trinh tuyen tinh Aw = b.
Phan ra ma tran A = D + L + U.
Cong thuc cap nhat thanh phan:
    w_i^(k+1) = (b_i - sum_{j < i} a_ij * w_j^(k+1) - sum_{j > i} a_ij * w_j^(k)) / a_ii
Su dung ngay gia tri moi cap nhat cho cac chi so j < i.
"""

from typing import List, Optional, Tuple
import numpy as np


def gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 20000,
    div_tol: float = 1e12,
) -> Tuple[np.ndarray, int, float, float, str, List[float]]:
    """
    Giai he phuong trinh tuyen tinh Aw = b bang phuong phap lap Gauss-Seidel.

    Tham so:
        A: Ma tran he so kich thuoc (n, n).
        b: Vector ve phai kich thuoc (n,).
        x0: Vector nghiem khoi tao kich thuoc (n,). Mac dinh la vector 0.
        tol: Dung sai hoi tu cua phan du tuong doi (mac dinh 1e-10).
        max_iter: So vong lap toi da (mac dinh 20000).
        div_tol: Nguong canh bao phan ky khi chuan nghiem hoac phan du vuot qua (mac dinh 1e12).

    Tra ve:
        x: Vector nghiem gan dung o vong lap cuoi.
        iterations: So vong lap da thuc hien.
        residual: Phan du tuong doi o vong lap cuoi.
        rho: Ban kinh pho rho(M_GS) cua ma tran lap Gauss-Seidel.
        status: Trang thai ket thuc: 'converged', 'diverged', hoac 'not_converged'.
        history: Danh sach phan du tuong doi qua cac vong lap.
    """
    A_arr = np.asarray(A, dtype=float)
    b_arr = np.asarray(b, dtype=float).ravel()

    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError(f"A phai la ma tran vuong, nhan duoc shape={A_arr.shape}.")

    n = A_arr.shape[0]
    if b_arr.shape[0] != n:
        raise ValueError(f"Kich thuoc b ({b_arr.shape[0]}) khong khop voi so hang cua A ({n}).")

    diag_A = np.diag(A_arr)
    if np.any(np.isclose(diag_A, 0.0, atol=1e-15)):
        raise ValueError("Ma tran A co phan tu tren duong cheo bang 0 (hoac qua gan 0), khong the chia cho 0.")

    # Khoi tao nghiem
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy().ravel()
        if x.shape[0] != n:
            raise ValueError(f"Kich thuoc x0 ({x.shape[0]}) khong khop voi he phuong trinh ({n}).")

    # Phan ra A = D + L + U de tinh ma tran lap M_GS = -(D + L)^(-1) U
    D = np.diag(diag_A)
    L = np.tril(A_arr, k=-1)
    U = np.triu(A_arr, k=1)

    M_GS = -np.linalg.solve(D + L, U)
    eigvals = np.linalg.eigvals(M_GS)
    rho = float(np.max(np.abs(eigvals)))

    norm_b = float(np.linalg.norm(b_arr, ord=2))
    denom_res = max(norm_b, 1e-12)

    history: List[float] = []

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        # Cap nhat toa do theo thu tu i = 0, ..., n - 1
        for i in range(n):
            sum_lower = float(np.dot(A_arr[i, :i], x[:i]))
            sum_upper = float(np.dot(A_arr[i, i + 1:], x_old[i + 1:]))
            x[i] = (b_arr[i] - sum_lower - sum_upper) / A_arr[i, i]

        # Kiem tra gia tri bat thuong (NaN, Inf)
        if not np.all(np.isfinite(x)):
            history.append(float("inf"))
            return x, k, float("inf"), rho, "diverged", history

        # Tinh phan du tuong doi eta_r = ||A @ x - b||_2 / max(||b||_2, 1e-12)
        residual = float(np.linalg.norm(A_arr @ x - b_arr, ord=2) / denom_res)
        history.append(residual)

        # Kiem tra phan ky theo nguong div_tol
        norm_x = float(np.linalg.norm(x, ord=2))
        if norm_x >= div_tol or residual >= div_tol or not np.isfinite(residual):
            return x, k, residual, rho, "diverged", history

        # Kiem tra dieu kien dung hoi tu
        if residual <= tol:
            return x, k, residual, rho, "converged", history

    return x, max_iter, history[-1], rho, "not_converged", history


if __name__ == "__main__":
    A_test = np.array([
        [10.0, 1.0, 1.0],
        [1.0, 10.0, 1.0],
        [1.0, 1.0, 10.0]
    ])
    b_test = np.array([12.0, 12.0, 12.0])

    sol, iters, res, rho, status, hist = gauss_seidel(A_test, b_test)
    print("Gauss-Seidel Test:")
    print("  Solution:", sol)
    print("  Iterations:", iters)
    print("  Residual:", res)
    print("  Spectral radius:", rho)
    print("  Status:", status)
