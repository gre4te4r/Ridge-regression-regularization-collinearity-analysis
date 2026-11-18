"""
Module xay dung va kiem tra he phuong trinh tuyen tinh trong hoi quy Ridge.
A = X.T @ X + lambda * I
b = X.T @ y
Aw = b
"""

from typing import Tuple
import numpy as np


def is_symmetric(A: np.ndarray, tol: float = 1e-12) -> bool:
    """Kiem tra ma tran A co doi xung hay khong."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    return bool(np.allclose(A, A.T, atol=tol))


def is_positive_definite(A: np.ndarray) -> bool:
    """Kiem tra ma tran A co xac dinh duong hay khong bang phan ra Cholesky."""
    if not is_symmetric(A):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def build_ridge_system(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    check_properties: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Xay dung he phuong trinh tuyen tinh cho bai toan hoi quy Ridge:
        min_w ||Xw - y||_2^2 + lam * ||w||_2^2,  lam > 0

    Lay gradient theo w va triet tieu ve 0 ta duoc:
        (X^T X + lam * I) w = X^T y
        <=> Aw = b

    Tham so:
        X: Ma tran dac trung kich thuoc (n, p).
        y: Vector bien muc tieu kich thuoc (n,) hoac (n, 1).
        lam: Tham so dieu chuan Ridge (lambda > 0).
        check_properties: Neu True, kiem tra tinh doi xung va xac dinh duong (SPD) cua A.

    Tra ve:
        A: Ma tran he so kich thuoc (p, p), la ma tran SPD khi lam > 0.
        b: Vector ve phai kich thuoc (p,).
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()

    if X_arr.ndim != 2:
        raise ValueError(f"X phai la mang 2 chieu (n, p), nhan duoc ndim={X_arr.ndim}.")
    n, p = X_arr.shape

    if y_arr.shape[0] != n:
        raise ValueError(f"Kich thuoc y ({y_arr.shape[0]}) khong khop voi so hang cua X ({n}).")

    if lam <= 0:
        raise ValueError(f"Tham so dieu chuan lambda phai la so duong (lam > 0), nhan duoc lam={lam}.")

    A = X_arr.T @ X_arr + lam * np.eye(p, dtype=float)
    b = X_arr.T @ y_arr

    if check_properties:
        if not is_symmetric(A):
            raise ValueError("Ma tran he so A khong doi xung.")
        if not is_positive_definite(A):
            raise ValueError("Ma tran he so A khong xac dinh duong (SPD).")

    return A, b


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.generate_data import generate_data

    X, y = generate_data(r=0.97, seed=4201)
    for test_lam in [0.01, 1.0, 100.0]:
        A, b = build_ridge_system(X, y, lam=test_lam)
        print(f"\nLambda = {test_lam}:")
        print(f"  Shape A: {A.shape}, Shape b: {b.shape}")
        print(f"  Is symmetric: {is_symmetric(A)}")
        print(f"  Is SPD: {is_positive_definite(A)}")
        print(f"  Condition number kappa_2(A): {np.linalg.cond(A):.4f}")
        print(f"  A diag: {np.diag(A).round(4)}")
