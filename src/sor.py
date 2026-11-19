"""Giai he Aw = b bang phuong phap lap SOR."""

from typing import List, Optional, Tuple
import numpy as np


def sor(
    A: np.ndarray,
    b: np.ndarray,
    omega: float = 1.0,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 20000,
    div_tol: float = 1e12,
) -> Tuple[np.ndarray, int, float, float, str, List[float]]:
    
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

    D = np.diag(diag_A)
    L = np.tril(A_arr, k=-1)
    U = np.triu(A_arr, k=1)

    M_sor = np.linalg.solve(
        D + omega * L,
        (1.0 - omega) * D - omega * U,
    )
    rho = float(np.max(np.abs(np.linalg.eigvals(M_sor))))
    norm_b = float(np.linalg.norm(b_arr, ord=2))
    denom_res = max(norm_b, 1e-12)

    history: List[float] = []

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        # Cap nhat toa do theo thu tu i = 0, ..., n - 1
        for i in range(n):
            sum_lower = float(np.dot(A_arr[i, :i], x[:i]))
            sum_upper = float(np.dot(A_arr[i, i + 1:], x_old[i + 1:]))
            gs_value = (b_arr[i] - sum_lower - sum_upper) / A_arr[i, i]
            x[i] = (1.0 - omega) * x_old[i] + omega * gs_value

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
