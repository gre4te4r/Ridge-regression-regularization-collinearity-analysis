from pathlib import Path

import pandas as pd

from src.sor import sor
from src.metrics import compute_condition_number
from src.ridge import build_ridge_system

thu_muc_du_lieu = Path(__file__).parent / "data" / "raw"
du_lieu = pd.read_csv(thu_muc_du_lieu / "TN3_r0.97_seed6201.csv")

X = du_lieu[["x1", "x2", "x3"]].to_numpy()
y = du_lieu["y"].to_numpy()

A, b = build_ridge_system(X, y, lam=1.0)

print("Kích thước X và y:", X.shape, y.shape)
print("Kích thước A và b:", A.shape, b.shape)
print("Số điều kiện của A:", round(compute_condition_number(A), 4))
for omega in (1.0, 1.3, 2.05):
    w, iterations, residual, rho, status, history = sor(A, b, omega=omega)
    print(f"SOR omega={omega}: {status}, k={iterations}, rho={rho:.4f}, residual={residual:.2e}")
