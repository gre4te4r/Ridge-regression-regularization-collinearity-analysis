"""
Chuong trinh chinh (main.py) - TV2:
Khao sat he phuong trinh tuyen tinh trong hoi quy Ridge voi Jacobi va Gauss-Seidel.
Thuc hien:
- Sinh/load 5 bo du lieu thuc nghiem.
- Xay dung 9 he phuong trinh tuyen tinh Aw = b.
- Chay 18 luot thuc nghiem (Jacobi va Gauss-Seidel cho tung he).
- Tinh toan toan bo cac chi so: spectral_radius, iterations, residual, solution_error, runtime.
- Xuat ket qua ra file CSV tai thu muc results/.
"""

import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from src.generate_data import generate_data, save_dataset_to_csv
from src.ridge import build_ridge_system, is_positive_definite
from src.jacobi import jacobi
from src.gauss_seidel import gauss_seidel
from src.metrics import (
    compute_condition_number,
    compute_relative_residual,
    compute_solution_error,
    solve_reference,
)


def run_tv2_benchmarks(data_dir: str = "data/raw", results_dir: str = "results") -> pd.DataFrame:
    """
    Chay toan bo 9 he phuong trinh thuoc 3 thi nghiem (18 luot chay):
    - TN1: r = 0.97, seed = 4201, lambda in [0.01, 0.1, 1.0, 10.0, 100.0] (5 he)
    - TN2: r in [0.10, 0.60, 0.97], seed tuong ung [5201, 5202, 5203], lambda = 1.0 (3 he)
    - TN3: r = 0.97, seed = 6201, lambda = 1.0 (1 he baseline cho TV3)
    """
    raw_path = Path(data_dir)
    res_path = Path(results_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    res_path.mkdir(parents=True, exist_ok=True)

    # 1. Dinh nghia danh sach cac thi nghiem
    # Dung chung mot bo du lieu cho TN1 (tat ca cac lambda)
    X_tn1, y_tn1 = generate_data(r=0.97, seed=4201)
    save_dataset_to_csv(X_tn1, y_tn1, raw_path / "TN1_r0.97_seed4201.csv")

    # Cac bo du lieu cho TN2
    X_tn2_low, y_tn2_low = generate_data(r=0.10, seed=5201)
    save_dataset_to_csv(X_tn2_low, y_tn2_low, raw_path / "TN2_r0.10_seed5201.csv")

    X_tn2_mid, y_tn2_mid = generate_data(r=0.60, seed=5202)
    save_dataset_to_csv(X_tn2_mid, y_tn2_mid, raw_path / "TN2_r0.60_seed5202.csv")

    X_tn2_high, y_tn2_high = generate_data(r=0.97, seed=5203)
    save_dataset_to_csv(X_tn2_high, y_tn2_high, raw_path / "TN2_r0.97_seed5203.csv")

    # Bo du lieu cho TN3
    X_tn3, y_tn3 = generate_data(r=0.97, seed=6201)
    save_dataset_to_csv(X_tn3, y_tn3, raw_path / "TN3_r0.97_seed6201.csv")

    experiment_systems = [
        # TN1: thay doi lambda tren bo du lieu co tuong quan cao
        {"exp": "TN1", "dataset": "TN1_r0.97_seed4201", "r": 0.97, "lam": 0.01, "X": X_tn1, "y": y_tn1},
        {"exp": "TN1", "dataset": "TN1_r0.97_seed4201", "r": 0.97, "lam": 0.10, "X": X_tn1, "y": y_tn1},
        {"exp": "TN1", "dataset": "TN1_r0.97_seed4201", "r": 0.97, "lam": 1.00, "X": X_tn1, "y": y_tn1},
        {"exp": "TN1", "dataset": "TN1_r0.97_seed4201", "r": 0.97, "lam": 10.00, "X": X_tn1, "y": y_tn1},
        {"exp": "TN1", "dataset": "TN1_r0.97_seed4201", "r": 0.97, "lam": 100.00, "X": X_tn1, "y": y_tn1},

        # TN2: thay doi muc tuong quan r, giu lambda = 1
        {"exp": "TN2", "dataset": "TN2_r0.10_seed5201", "r": 0.10, "lam": 1.00, "X": X_tn2_low, "y": y_tn2_low},
        {"exp": "TN2", "dataset": "TN2_r0.60_seed5202", "r": 0.60, "lam": 1.00, "X": X_tn2_mid, "y": y_tn2_mid},
        {"exp": "TN2", "dataset": "TN2_r0.97_seed5203", "r": 0.97, "lam": 1.00, "X": X_tn2_high, "y": y_tn2_high},

        # TN3: baseline tren bo du lieu seed 6201, lambda = 1
        {"exp": "TN3", "dataset": "TN3_r0.97_seed6201", "r": 0.97, "lam": 1.00, "X": X_tn3, "y": y_tn3},
    ]

    records: List[Dict[str, Any]] = []

    print("=" * 105)
    print(f"{'Exp':<5} | {'Dataset':<20} | {'r':<4} | {'lambda':<6} | {'Alg':<12} | {'kappa2(A)':<9} | {'rho(M)':<8} | {'Iter':<5} | {'Status':<13} | {'Residual':<10} | {'Sol Error':<10} | {'Time (ms)':<9}")
    print("=" * 105)

    for item in experiment_systems:
        exp = item["exp"]
        ds_name = item["dataset"]
        r_val = item["r"]
        lam_val = item["lam"]
        X = item["X"]
        y = item["y"]

        A, b = build_ridge_system(X, y, lam=lam_val)
        cond_A = compute_condition_number(A)
        w_ref = solve_reference(A, b)

        # Chay Jacobi
        t0 = time.perf_counter()
        w_j, it_j, res_j, rho_j, status_j, _ = jacobi(A, b, tol=1e-10, max_iter=20000)
        t_j = (time.perf_counter() - t0) * 1000.0
        err_j = compute_solution_error(w_j, w_ref)

        records.append({
            "experiment": exp,
            "dataset": ds_name,
            "r": r_val,
            "lambda": lam_val,
            "algorithm": "Jacobi",
            "cond_A": cond_A,
            "spectral_radius": rho_j,
            "iterations": it_j,
            "residual": res_j,
            "solution_error": err_j,
            "status": status_j,
            "runtime": round(t_j, 3),
        })

        st_j_display = "Phân kỳ" if status_j == "diverged" else ("Hội tụ" if status_j == "converged" else "Chưa hội tụ")
        print(f"{exp:<5} | {ds_name:<20} | {r_val:<4.2f} | {lam_val:<6.2f} | {'Jacobi':<12} | {cond_A:<9.2f} | {rho_j:<8.4f} | {it_j:<5d} | {st_j_display:<13} | {res_j:<10.2e} | {err_j:<10.2e} | {t_j:<9.3f}")

        # Chay Gauss-Seidel
        t0 = time.perf_counter()
        w_g, it_g, res_g, rho_g, status_g, _ = gauss_seidel(A, b, tol=1e-10, max_iter=20000)
        t_g = (time.perf_counter() - t0) * 1000.0
        err_g = compute_solution_error(w_g, w_ref)

        records.append({
            "experiment": exp,
            "dataset": ds_name,
            "r": r_val,
            "lambda": lam_val,
            "algorithm": "Gauss-Seidel",
            "cond_A": cond_A,
            "spectral_radius": rho_g,
            "iterations": it_g,
            "residual": res_g,
            "solution_error": err_g,
            "status": status_g,
            "runtime": round(t_g, 3),
        })

        st_g_display = "Phân kỳ" if status_g == "diverged" else ("Hội tụ" if status_g == "converged" else "Chưa hội tụ")
        print(f"{exp:<5} | {ds_name:<20} | {r_val:<4.2f} | {lam_val:<6.2f} | {'Gauss-Seidel':<12} | {cond_A:<9.2f} | {rho_g:<8.4f} | {it_g:<5d} | {st_g_display:<13} | {res_g:<10.2e} | {err_g:<10.2e} | {t_g:<9.3f}")

    print("=" * 105)

    df_results = pd.DataFrame(records)

    # Chuan hoa cac cot xuat ra dung theo yeu cau Section 10
    output_columns = [
        "experiment",
        "dataset",
        "r",
        "lambda",
        "algorithm",
        "spectral_radius",
        "iterations",
        "residual",
        "solution_error",
        "status",
        "runtime",
    ]
    df_export = df_results[output_columns]

    out_csv = res_path / "tv2_experiment_results.csv"
    df_export.to_csv(out_csv, index=False)
    print(f"\nDa xuat toan bo ket qua 18 luot chay vao: {out_csv}")

    # Xuat them ban co them cot cond_A de de theo doi
    df_results.to_csv(res_path / "tv2_full_metrics.csv", index=False)

    return df_results


if __name__ == "__main__":
    df = run_tv2_benchmarks()
