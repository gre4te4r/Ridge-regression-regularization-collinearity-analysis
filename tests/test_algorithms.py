"""
Unit tests cho toan bo cac module cua TV2:
- generate_data
- build_ridge_system
- jacobi
- gauss_seidel
- metrics
"""

import sys
from pathlib import Path

# Them thu muc goc vao sys.path de import cac module tu src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np
from scipy.linalg import solve

from src.generate_data import generate_data
from src.ridge import build_ridge_system, is_positive_definite, is_symmetric
from src.jacobi import jacobi
from src.gauss_seidel import gauss_seidel
from src.metrics import (
    compute_condition_number,
    compute_relative_residual,
    compute_solution_error,
    compute_spectral_radius,
    solve_reference,
)


class TestGenerateData(unittest.TestCase):
    """Kiem thu module sinh du lieu."""

    def test_output_shapes_and_types(self):
        X, y = generate_data(n=80, p=3, r=0.97, seed=4201)
        self.assertEqual(X.shape, (80, 3))
        self.assertEqual(y.shape, (80,))
        self.assertTrue(np.issubdtype(X.dtype, np.floating))
        self.assertTrue(np.issubdtype(y.dtype, np.floating))

    def test_normalization_statistics(self):
        X, _ = generate_data(n=80, p=3, r=0.60, seed=5202)
        np.testing.assert_allclose(X.mean(axis=0), np.zeros(3), atol=1e-14)
        np.testing.assert_allclose(X.std(axis=0, ddof=0), np.ones(3), atol=1e-14)

    def test_reproducibility(self):
        X1, y1 = generate_data(n=80, p=3, r=0.97, seed=4201)
        X2, y2 = generate_data(n=80, p=3, r=0.97, seed=4201)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            generate_data(n=-10)
        with self.assertRaises(ValueError):
            generate_data(r=1.5)
        with self.assertRaises(ValueError):
            generate_data(noise_std=-0.1)


class TestRidgeSystem(unittest.TestCase):
    """Kiem thu he phuong trinh Ridge."""

    def setUp(self):
        self.X, self.y = generate_data(n=80, p=3, r=0.97, seed=4201)

    def test_dimensions(self):
        A, b = build_ridge_system(self.X, self.y, lam=1.0)
        self.assertEqual(A.shape, (3, 3))
        self.assertEqual(b.shape, (3,))

    def test_symmetry_and_spd(self):
        for lam in [0.01, 1.0, 100.0]:
            A, b = build_ridge_system(self.X, self.y, lam=lam)
            self.assertTrue(is_symmetric(A))
            self.assertTrue(is_positive_definite(A))
            eigvals = np.linalg.eigvalsh(A)
            self.assertTrue(np.all(eigvals > 0))

    def test_invalid_lambda(self):
        with self.assertRaises(ValueError):
            build_ridge_system(self.X, self.y, lam=0.0)
        with self.assertRaises(ValueError):
            build_ridge_system(self.X, self.y, lam=-1.0)

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            build_ridge_system(self.X, self.y[:50], lam=1.0)


class TestAlgorithms(unittest.TestCase):
    """Kiem thu thuat toan Jacobi va Gauss-Seidel theo cac kich ban yeu cau."""

    def test_diagonal_matrix(self):
        """Ma tran duong cheo hoi tu ngay sau 1 vong lap."""
        A = np.diag([4.0, 5.0, 6.0])
        b = np.array([8.0, 15.0, 24.0])
        expected = np.array([2.0, 3.0, 4.0])

        w_j, it_j, res_j, rho_j, st_j, _ = jacobi(A, b)
        self.assertEqual(st_j, "converged")
        self.assertEqual(it_j, 1)
        np.testing.assert_allclose(w_j, expected, atol=1e-12)
        self.assertEqual(rho_j, 0.0)

        w_g, it_g, res_g, rho_g, st_g, _ = gauss_seidel(A, b)
        self.assertEqual(st_g, "converged")
        self.assertEqual(it_g, 1)
        np.testing.assert_allclose(w_g, expected, atol=1e-12)
        self.assertEqual(rho_g, 0.0)

    def test_strictly_diagonally_dominant(self):
        """Ma tran cheo troi vi du: A=[[10,1,1],[1,10,1],[1,1,10]], b=[12,12,12]."""
        A = np.array([
            [10.0, 1.0, 1.0],
            [1.0, 10.0, 1.0],
            [1.0, 1.0, 10.0]
        ])
        b = np.array([12.0, 12.0, 12.0])
        expected = np.array([1.0, 1.0, 1.0])

        w_j, it_j, res_j, rho_j, st_j, _ = jacobi(A, b, tol=1e-10)
        self.assertEqual(st_j, "converged")
        np.testing.assert_allclose(w_j, expected, atol=1e-9)
        self.assertLess(res_j, 1e-10)
        self.assertLess(rho_j, 1.0)

        w_g, it_g, res_g, rho_g, st_g, _ = gauss_seidel(A, b, tol=1e-10)
        self.assertEqual(st_g, "converged")
        np.testing.assert_allclose(w_g, expected, atol=1e-9)
        self.assertLess(res_g, 1e-10)
        self.assertLess(rho_g, 1.0)

    def test_spd_jacobi_diverges_gauss_seidel_converges(self):
        """
        Kiem tra truong hop ma tran SPD ma Jacobi phan ky (rho_J > 1)
        nhung Gauss-Seidel van hoi tu (rho_GS < 1).
        Ma tran: A = [[1, 0.9, 0.9], [0.9, 1, 0.9], [0.9, 0.9, 1]].
        Trị riêng của A: 2.8, 0.1, 0.1 > 0 -> SPD.
        rho(M_J) = 1.8 > 1 -> Jacobi phan ky.
        Gauss-Seidel tren ma tran SPD luon hoi tu theo dinh ly.
        """
        A = np.array([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.9],
            [0.9, 0.9, 1.0]
        ])
        b = np.array([2.8, 2.8, 2.8])
        expected = np.array([1.0, 1.0, 1.0])

        # Kiem tra A la SPD
        self.assertTrue(is_positive_definite(A))

        # Jacobi phan ky
        w_j, it_j, res_j, rho_j, st_j, _ = jacobi(A, b)
        self.assertGreater(rho_j, 1.0)
        self.assertEqual(st_j, "diverged")

        # Gauss-Seidel hoi tu
        w_g, it_g, res_g, rho_g, st_g, _ = gauss_seidel(A, b, tol=1e-10)
        self.assertLess(rho_g, 1.0)
        self.assertEqual(st_g, "converged")
        np.testing.assert_allclose(w_g, expected, atol=1e-8)
        self.assertLess(res_g, 1e-10)

    def test_compare_with_scipy_solve(self):
        """So sanh nghiem cua Jacobi va Gauss-Seidel voi scipy.linalg.solve tren he hoi tu."""
        X, y = generate_data(n=80, p=3, r=0.97, seed=4201)
        A, b = build_ridge_system(X, y, lam=100.0)
        w_ref = solve_reference(A, b)

        # O lambda=100, ca Jacobi va GS deu hoi tu
        w_j, _, res_j, rho_j, st_j, _ = jacobi(A, b, tol=1e-10)
        self.assertEqual(st_j, "converged")
        err_j = compute_solution_error(w_j, w_ref)
        self.assertLess(err_j, 1e-8)
        self.assertLess(res_j, 1e-10)

        w_g, _, res_g, rho_g, st_g, _ = gauss_seidel(A, b, tol=1e-10)
        self.assertEqual(st_g, "converged")
        err_g = compute_solution_error(w_g, w_ref)
        self.assertLess(err_g, 1e-8)
        self.assertLess(res_g, 1e-10)

    def test_zero_on_diagonal_raises_error(self):
        """Ma tran co phan tu tren duong cheo bang 0 phai bao loi ValueError phu hop."""
        A = np.array([
            [0.0, 1.0],
            [1.0, 2.0]
        ])
        b = np.array([1.0, 2.0])

        with self.assertRaises(ValueError) as ctx_j:
            jacobi(A, b)
        self.assertIn("duong cheo", str(ctx_j.exception).lower())

        with self.assertRaises(ValueError) as ctx_g:
            gauss_seidel(A, b)
        self.assertIn("duong cheo", str(ctx_g.exception).lower())

    def test_relative_residual_definition(self):
        """Kiem tra cong thuc tinh phan du tuong doi eta_r."""
        A = np.eye(3) * 2.0
        b = np.array([4.0, 6.0, 8.0])
        w = np.array([2.0, 3.0, 4.0])
        res = compute_relative_residual(A, w, b)
        self.assertAlmostEqual(res, 0.0, places=14)

        w_approx = np.array([2.01, 3.0, 4.0])
        expected_res = np.linalg.norm(A @ w_approx - b) / np.linalg.norm(b)
        res_calc = compute_relative_residual(A, w_approx, b)
        self.assertAlmostEqual(res_calc, expected_res, places=12)


if __name__ == "__main__":
    unittest.main()
