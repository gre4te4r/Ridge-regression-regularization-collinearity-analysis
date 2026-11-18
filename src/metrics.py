"""
Module tinh toan cac chi so danh gia he phuong trinh va thuat toan lap:
- Phan du tuong doi (relative residual)
- Sai so so voi nghiem tham chieu (solution error)
- Ban kinh pho (spectral radius)
- So dieu kien chuan 2 (condition number)
- Nghiem tham chieu giai truc tiep (reference solution)
"""

from typing import Union
import numpy as np
from scipy.linalg import solve


def compute_relative_residual(A: np.ndarray, w: np.ndarray, b: np.ndarray) -> float:
    """
    Tinh phan du tuong doi:
        eta_r = ||A w - b||_2 / max(||b||_2, 1e-12)

    Muc do thoa man he phuong trinh tuyen tinh.
    """
    res_norm = float(np.linalg.norm(A @ w - b, ord=2))
    b_norm = float(np.linalg.norm(b, ord=2))
    denom = max(b_norm, 1e-12)
    return res_norm / denom


def compute_spectral_radius(M: np.ndarray) -> float:
    """
    Tinh ban kinh pho cua ma tran lap M:
        rho(M) = max_i |mu_i(M)|
    Dieu kien can va du de phuong phap lap hoi tu voi moi diem dau la rho(M) < 1.
    """
    eigvals = np.linalg.eigvals(M)
    return float(np.max(np.abs(eigvals)))


def compute_condition_number(A: np.ndarray) -> float:
    """
    Tinh so dieu kien theo chuan 2 cua ma tran A:
        kappa_2(A) = sigma_max(A) / sigma_min(A)
    """
    return float(np.linalg.cond(A, p=2))


def compute_solution_error(w: np.ndarray, w_ref: np.ndarray) -> float:
    """
    Tinh sai so so voi nghiem tham chieu:
        e_w = ||w - w_ref||_2
    """
    return float(np.linalg.norm(w - w_ref, ord=2))


def solve_reference(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Giai he phuong trinh Aw = b truc tiep bang phan ra Cholesky cho he SPD
    thong qua scipy.linalg.solve(A, b, assume_a="pos").
    Dung lam nghiem tham chieu doi chieu cho cac phuong phap lap.
    """
    return solve(A, b, assume_a="pos")
