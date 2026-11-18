"""
Module cai dat phuong phap lap Jacobi cho he phuong trinh tuyen tinh Aw = b.
Phan ra ma tran A = D + L + U, trong do D la duong cheo, L la tam giac duoi nghiem ngat,
U la tam giac tren nghiem ngat.
Cong thuc lap:
    w^(k+1) = D^(-1) (b - (L + U) w^(k))
"""

from typing import List, Optional, Tuple
import numpy as np


def jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 20000,
    div_tol: float = 1e12,
) -> Tuple[np.ndarray, int, float, float, str, List[float]]:
    """
    Giai he phuong trinh tuyen tinh Aw = b bang phuong phap lap Jacobi.

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
        rho: Ban kinh pho rho(M_J) cua ma tran lap Jacobi.
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

    # Phan ra A = D + R voi R = L + U (ngoai duong cheo)
    R = A_arr.copy()
    np.fill_diagonal(R, 0.0)

    # Ma tran lap Jacobi: M_J = -D^(-1) R
    # Vi D la ma tran duong cheo, M_J[i, :] = -R[i, :] / A[i, i]
    M_J = -R / diag_A[:, np.newaxis]
    eigvals = np.linalg.eigvals(M_J)
    rho = float(np.max(np.abs(eigvals)))

    norm_b = float(np.linalg.norm(b_arr, ord=2))
    denom_res = max(norm_b, 1e-12)

    history: List[float] = []

    for k in range(1, max_iter + 1):
        # Cong thuc Jacobi: x_new = (b - R @ x) / diag_A
        x_new = (b_arr - R @ x) / diag_A

        # Kiem tra gia tri bat thuong (NaN, Inf)
        if not np.all(np.isfinite(x_new)):
            history.append(float("inf"))
            return x_new, k, float("inf"), rho, "diverged", history

        # Tinh phan du tuong doi eta_r = ||A @ x_new - b||_2 / max(||b||_2, 1e-12)
        residual = float(np.linalg.norm(A_arr @ x_new - b_arr, ord=2) / denom_res)
        history.append(residual)

        # Kiem tra phan ky theo nguong div_tol
        norm_x = float(np.linalg.norm(x_new, ord=2))
        if norm_x >= div_tol or residual >= div_tol or not np.isfinite(residual):
            return x_new, k, residual, rho, "diverged", history

        # Kiem tra dieu kien dung hoi tu
        if residual <= tol:
            return x_new, k, residual, rho, "converged", history

        x = x_new

    return x, max_iter, history[-1], rho, "not_converged", history


if __name__ == "__main__":
    A_test = np.array([
        [10.0, 1.0, 1.0],
        [1.0, 10.0, 1.0],
        [1.0, 1.0, 10.0]
    ])
    b_test = np.array([12.0, 12.0, 12.0])

    sol, iters, res, rho, status, hist = jacobi(A_test, b_test)
    print("Jacobi Test:")
    print("  Solution:", sol)
    print("  Iterations:", iters)
    print("  Residual:", res)
    print("  Spectral radius:", rho)
    print("  Status:", status)
