from pathlib import Path
from time import perf_counter

import pandas as pd

from src.metrics import compute_condition_number, compute_solution_error, solve_reference
from src.ridge import build_ridge_system
from src.sor import sor


ROOT = Path(__file__).resolve().parent
TN3_OMEGAS = (0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 1.95, 2.05)


def run_experiments():
    results_dir = ROOT / "results"
    tv2 = pd.read_csv(results_dir / "tv2_full_metrics.csv").rename(
        columns={"runtime": "runtime_ms"}
    )
    configs = tv2[["experiment", "dataset", "r", "lambda"]].drop_duplicates()
    rows = []
    for experiment, dataset, r, lam in configs.itertuples(index=False, name=None):
        data = pd.read_csv(ROOT / "data" / "raw" / f"{dataset}.csv")
        A, b = build_ridge_system(data[["x1", "x2", "x3"]], data["y"], lam)
        reference = solve_reference(A, b)
        condition = compute_condition_number(A)
        omegas = TN3_OMEGAS if experiment == "TN3" else (1.2,)
        for omega in omegas:
            start = perf_counter()
            w, iterations, residual, rho, status, _ = sor(A, b, omega=omega)
            runtime_ms = (perf_counter() - start) * 1000
            rows.append({
                "experiment": experiment, "dataset": dataset, "r": r, "lambda": lam,
                "algorithm": "SOR", "omega": omega, "cond_A": condition,
                "spectral_radius": rho, "iterations": iterations, "residual": residual,
                "solution_error": compute_solution_error(w, reference),
                "status": status, "runtime_ms": runtime_ms,
            })

    tv3 = pd.DataFrame(rows)
    combined = pd.concat([tv2, tv3], ignore_index=True)[tv3.columns]
    combined = combined.sort_values(["experiment", "r", "lambda"], kind="stable")
    tv3.to_csv(results_dir / "tv3_sor_results.csv", index=False)
    combined.to_csv(results_dir / "tv4_combined_results.csv", index=False)


if __name__ == "__main__":
    run_experiments()
