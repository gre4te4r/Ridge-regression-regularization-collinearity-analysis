"""
Module sinh du lieu thuc nghiem cho bai toan Ridge Regression.
Dap ung yeu cau TV2: Kiem soat tuong quan da bien, seed, va chuan hoa du lieu.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def generate_data(
    n: int = 80,
    p: int = 3,
    r: float = 0.97,
    seed: int = 4201,
    beta: Optional[np.ndarray] = None,
    noise_std: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sinh bo du lieu tong hop (X, y) theo dung cong thuc va tham so trong bao cao.

    Mo hinh sinh du lieu:
        z ~ N(0, 1) (nhan to an chung kich thuoc n x 1)
        epsilon_j ~ N(0, 1) (nhiễu rieng tung dac trung kich thuoc n x p)
        X_j = sqrt(r) * z + sqrt(1 - r) * epsilon_j
        Chuan hoa tung cot cua X ve mean = 0, std = 1 (do lech chuan tong the ddof=0).
        y = X @ beta + epsilon_y, voi epsilon_y ~ N(0, noise_std^2).

    Tham so:
        n: So luong quan sat (mac dinh 80).
        p: So luong dac trung (mac dinh 3).
        r: He so tuong quan muc tieu (0 <= r < 1).
        seed: Hat giong ngau nhien cho numpy.random.default_rng.
        beta: Vector trong so thuc te (mac dinh [0.5, 1.0, 1.5]).
        noise_std: Do lech chuan cua nhiễu epsilon_y (mac dinh 0.2).

    Tra ve:
        X: Ma tran dac trung kich thuoc (n, p) da chuan hoa.
        y: Vector bien muc tieu kich thuoc (n,).
    """
    if n <= 0 or p <= 0:
        raise ValueError("n va p phai la so nguyen duong.")
    if not (0.0 <= r < 1.0):
        raise ValueError("r phai nam trong khoang [0, 1).")
    if noise_std < 0:
        raise ValueError("noise_std phai khong am.")

    if beta is None:
        beta = np.array([0.5, 1.0, 1.5], dtype=float)
    else:
        beta = np.asarray(beta, dtype=float)
        if beta.shape != (p,):
            raise ValueError(f"beta phai co kich thuoc ({p},), nhan duoc {beta.shape}.")

    rng = np.random.default_rng(seed)

    # 1. Sinh nhan to chung
    z = rng.normal(0.0, 1.0, size=(n, 1))

    # 2. Sinh nhieu rieng cho tung dac trung
    epsilon_x = rng.normal(0.0, 1.0, size=(n, p))

    # 3. Tao ma tran X
    X = np.sqrt(r) * z + np.sqrt(1.0 - r) * epsilon_x

    # 4. Chuan hoa tung cot ve mean = 0 va std = 1 (ddof=0 theo bao cao)
    mean_X = X.mean(axis=0)
    std_X = X.std(axis=0, ddof=0)
    if np.any(std_X == 0):
        raise ValueError("Do lech chuan cua mot hoac nhieu cot bang 0.")
    X = (X - mean_X) / std_X

    # 5. Sinh bien muc tieu y
    epsilon_y = rng.normal(0.0, noise_std, size=n)
    y = X @ beta + epsilon_y

    return X, y


def save_dataset_to_csv(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path | str,
) -> None:
    """Luu bo du lieu X, y vao file CSV."""
    p = X.shape[1]
    col_names = [f"x{i+1}" for i in range(p)]
    df = pd.DataFrame(X, columns=col_names)
    df["y"] = y
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def generate_all_experiment_datasets(output_dir: Path | str = "data/raw") -> dict:
    """
    Sinh toan bo 5 bo du lieu thuc nghiem theo cau hinh bao cao:
    - TN1: r = 0.97, seed = 4201 (dung chung cho moi lambda)
    - TN2: r = 0.10, seed = 5201
    - TN2: r = 0.60, seed = 5202
    - TN2: r = 0.97, seed = 5203
    - TN3: r = 0.97, seed = 6201

    Luu tung tap vao thu muc output_dir va tra ve dictionary chua du lieu.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {"exp": "TN1", "dataset": "TN1_r0.97_seed4201", "r": 0.97, "seed": 4201},
        {"exp": "TN2", "dataset": "TN2_r0.10_seed5201", "r": 0.10, "seed": 5201},
        {"exp": "TN2", "dataset": "TN2_r0.60_seed5202", "r": 0.60, "seed": 5202},
        {"exp": "TN2", "dataset": "TN2_r0.97_seed5203", "r": 0.97, "seed": 5203},
        {"exp": "TN3", "dataset": "TN3_r0.97_seed6201", "r": 0.97, "seed": 6201},
    ]

    datasets = {}
    for cfg in configs:
        X, y = generate_data(r=cfg["r"], seed=cfg["seed"])
        file_path = out_dir / f"{cfg['dataset']}.csv"
        save_dataset_to_csv(X, y, file_path)
        datasets[cfg["dataset"]] = {
            "exp": cfg["exp"],
            "r": cfg["r"],
            "seed": cfg["seed"],
            "X": X,
            "y": y,
            "file": file_path,
        }

    return datasets


if __name__ == "__main__":
    datasets = generate_all_experiment_datasets("data/raw")
    print(f"Da sinh thanh cong {len(datasets)} bo du lieu:")
    for name, item in datasets.items():
        X, y = item["X"], item["y"]
        corr = np.corrcoef(X.T)
        print(f"\n[{name}]")
        print(f"  Shape X: {X.shape}, Shape y: {y.shape}")
        print(f"  Means: {X.mean(axis=0).round(4)}")
        print(f"  Stds:  {X.std(axis=0, ddof=0).round(4)}")
        print(f"  Mean off-diagonal correlation: {((np.sum(corr) - X.shape[1]) / (X.shape[1]*(X.shape[1]-1))):.4f}")
        print(f"  Saved to: {item['file']}")
