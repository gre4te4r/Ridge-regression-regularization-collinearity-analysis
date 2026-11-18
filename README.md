# BÀI TẬP LỚN PHƯƠNG PHÁP TÍNH - PHẦN TV2
## HỒI QUY RIDGE, PHƯƠNG PHÁP LẶP JACOBI VÀ GAUSS-SEIDEL

Dự án này là sản phẩm hoàn thiện của **Thành viên 2 (TV2)** trong đề tài:
> *"Khảo sát ảnh hưởng của tham số điều chuẩn và tính tương quan của dữ liệu đến hiệu quả giải hệ phương trình tuyến tính trong hồi quy Ridge"*

---

## 1. Mục tiêu và Phạm vi TV2

1. **Sinh dữ liệu thực nghiệm:** Tạo các bộ dữ liệu mô phỏng với mức độ tương quan đặc trưng được kiểm soát chặt chẽ ($r \in \{0.10, 0.60, 0.97\}$) cùng các seed xác định nhằm đảm bảo tính tái lập 100%.
2. **Xây dựng hệ phương trình Ridge:** Thiết lập ma trận hệ số $A = X^T X + \lambda I$ và vế phải $b = X^T y$, kiểm tra tính đối xứng và xác định dương (SPD).
3. **Cài đặt thuật toán lặp từ đầu (from scratch bằng NumPy):**
   - Thuật toán **Jacobi** (song song các thành phần, cập nhật đồng thời).
   - Thuật toán **Gauss–Seidel** (cập nhật tuần tự, dùng ngay giá trị mới).
4. **Kiểm thử toàn diện (Unit Tests):** Kiểm thử trên ma trận đường chéo, chéo trội, ma trận SPD phân kỳ ở Jacobi nhưng hội tụ ở Gauss–Seidel, ma trận suy biến/đường chéo bằng 0, so sánh với nghiệm trực tiếp từ `scipy.linalg.solve`.
5. **Thực nghiệm và Xuất báo cáo:** Chạy 9 hệ phương trình (18 lượt lặp), tính toán bán kính phổ $\rho(M)$, số vòng lặp, phần dư tương đối $\eta_r$, sai số nghiệm $e_w$, thời gian thực thi, và xuất kết quả ra file CSV chuẩn.

---

## 2. Cấu trúc thư mục

```text
BTL PPT/
│
├── data/
│   └── raw/                              # Dữ liệu gốc đã sinh theo seed chuẩn
│       ├── TN1_r0.97_seed4201.csv        # Dữ liệu TN1 (dùng chung cho mọi lambda)
│       ├── TN2_r0.10_seed5201.csv        # TN2 - Tương quan thấp (r = 0.10)
│       ├── TN2_r0.60_seed5202.csv        # TN2 - Tương quan vừa (r = 0.60)
│       ├── TN2_r0.97_seed5203.csv        # TN2 - Tương quan cao (r = 0.97)
│       └── TN3_r0.97_seed6201.csv        # Dữ liệu TN3 (seed 6201)
│
├── results/                              # Kết quả thực nghiệm
│   ├── tv2_experiment_results.csv        # Bảng kết quả 18 lượt chạy đúng chuẩn yêu cầu
│   └── tv2_full_metrics.csv              # Bảng kết quả kèm số điều kiện kappa_2(A)
│
├── src/                                  # Mã nguồn module
│   ├── generate_data.py                  # Module sinh dữ liệu, kiểm tra thống kê và lưu CSV
│   ├── ridge.py                          # Module xây dựng hệ phương trình Ridge và kiểm tra SPD
│   ├── jacobi.py                         # Thuật toán lặp Jacobi (tự cài đặt bằng NumPy)
│   ├── gauss_seidel.py                   # Thuật toán lặp Gauss-Seidel (tự cài đặt bằng NumPy)
│   └── metrics.py                        # Các chỉ số: residual, spectral radius, solution error, solve
│
├── tests/
│   └── test_algorithms.py                # Bộ 14 unit tests kiểm thử toàn bộ các kịch bản
│
├── main.py                               # Script chạy toàn bộ 9 hệ phương trình và xuất CSV
├── requirements.txt                      # Danh sách thư viện phụ thuộc (numpy, scipy, pandas, matplotlib)
└── README.md                             # Tài liệu hướng dẫn và báo cáo kết quả
```

---

## 3. Cơ sở lý thuyết toán học

### 3.1. Hệ phương trình Ridge Regression
Bài toán tối ưu hồi quy Ridge:
$$\min_{w \in \mathbb{R}^p} \|Xw - y\|_2^2 + \lambda \|w\|_2^2 \quad (\lambda > 0)$$

Lấy đạo hàm triệt tiêu theo $w$:
$$2X^T(Xw - y) + 2\lambda w = 0 \iff (X^T X + \lambda I) w = X^T y$$

Đặt:
$$A = X^T X + \lambda I, \quad b = X^T y \implies Aw = b$$

Với $\lambda > 0$, với mọi $v \neq 0$:
$$v^T A v = \|Xv\|_2^2 + \lambda \|v\|_2^2 > 0$$
Do đó $A$ luôn là ma trận **đối xứng xác định dương (SPD)**.

### 3.2. Thuật toán Jacobi
Phân rã ma trận $A = D + L + U = D + R$, trong đó $D = \text{diag}(A)$ và $R = L + U$.
Công thức lặp:
$$w^{(k+1)} = D^{-1}\left(b - R w^{(k)}\right)$$
Thành phần tọa độ:
$$w_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j \neq i} a_{ij} w_j^{(k)}\right)$$
Ma trận lặp: $M_J = -D^{-1}(L + U)$. Điều kiện hội tụ: $\rho(M_J) < 1$.

### 3.3. Thuật toán Gauss–Seidel
Sử dụng ngay các giá trị thành phần mới nhất vừa cập nhật:
$$w_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j < i} a_{ij} w_j^{(k+1)} - \sum_{j > i} a_{ij} w_j^{(k)}\right)$$
Dạng ma trận:
$$w^{(k+1)} = (D + L)^{-1}\left(b - U w^{(k)}\right)$$
Ma trận lặp: $M_{GS} = -(D + L)^{-1} U$.

> **Định lý quan trọng:**
> - Nếu $A$ là ma trận đối xứng xác định dương (SPD), **Gauss–Seidel chắc chắn hội tụ** ($\rho(M_{GS}) < 1$).
> - Tính chất SPD **không** đảm bảo cho Jacobi hội tụ. Khi các đặc trưng có tương quan cao, $\rho(M_J)$ thường lớn hơn 1 dẫn đến Jacobi bị **phân kỳ**.

### 3.4. Tiêu chí dừng và đánh giá
- **Phần dư tương đối (Relative residual):**
  $$\eta_r = \frac{\|Aw - b\|_2}{\max(\|b\|_2, 10^{-12})}$$
  Tiêu chuẩn dừng: $\eta_r \le 10^{-10}$ hoặc đạt tối đa $20\,000$ vòng lặp.
- **Sai số nghiệm (Solution error):**
  $$e_w = \|w_{\text{lặp}} - w_{\text{ref}}\|_2$$
  với $w_{\text{ref}} = \text{scipy.linalg.solve}(A, b, \text{assume\_a="pos"})$.
- **Số điều kiện theo chuẩn 2:** $\kappa_2(A) = \sigma_{\max}(A) / \sigma_{\min}(A)$.

---

## 4. Hướng dẫn cài đặt và chạy trên Windows

### Bước 1: Môi trường và thư viện
Mở terminal trong thư mục project `d:\vs_code\BTL PPT`:
```powershell
pip install -r requirements.txt
```

### Bước 2: Sinh dữ liệu mẫu (Phase 2)
```powershell
python src/generate_data.py
```
Lệnh này sẽ tạo ra 5 file CSV trong thư mục `data/raw/` tương ứng với các cấu hình TN1, TN2, TN3.

### Bước 3: Chạy Unit Tests kiểm thử toàn diện (Phase 5)
```powershell
python tests/test_algorithms.py -v
```
Toàn bộ 14 bài kiểm tra sẽ chạy và hiển thị trạng thái `OK`.

### Bước 4: Chạy toàn bộ thực nghiệm 9 hệ phương trình (Phase 6)
```powershell
python main.py
```
Chương trình sẽ giải 9 hệ phương trình với cả Jacobi và Gauss–Seidel (tổng cộng 18 lượt chạy) và xuất kết quả vào:
- `results/tv2_experiment_results.csv`
- `results/tv2_full_metrics.csv`

---

## 5. Kết quả thực nghiệm và So sánh với Báo cáo

Kết quả chạy thực tế trên máy tính ăn khớp hoàn toàn với số liệu trong báo cáo:

### Bảng 1: TN1 – Ảnh hưởng của tham số điều chuẩn $\lambda$ ($r = 0.97$, seed 4201)
| $\lambda$ | Phương pháp | $\kappa_2(A)$ | $\rho(M)$ | Số vòng lặp | Trạng thái | $\eta_r$ | $e_w$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.01 | Jacobi | 94.00 | 1.9233 | 42 | Phân kỳ | $8.52 \times 10^{11}$ | $1.48 \times 10^{12}$ |
| 0.01 | Gauss–Seidel | 94.00 | 0.9430 | 337 | Hội tụ | $9.73 \times 10^{-11}$ | $9.49 \times 10^{-09}$ |
| 0.10 | Jacobi | 90.75 | 1.9212 | 42 | Phân kỳ | $8.13 \times 10^{11}$ | $1.41 \times 10^{12}$ |
| 0.10 | Gauss–Seidel | 90.75 | 0.9414 | 328 | Hội tụ | $9.70 \times 10^{-11}$ | $9.30 \times 10^{-09}$ |
| 1.00 | Jacobi | 67.53 | 1.8998 | 43 | Phân kỳ | $9.66 \times 10^{11}$ | $1.67 \times 10^{12}$ |
| 1.00 | Gauss–Seidel | 67.53 | 0.9258 | 259 | Hội tụ | $9.56 \times 10^{-11}$ | $7.58 \times 10^{-09}$ |
| 10.00 | Jacobi | 19.54 | 1.7099 | 51 | Phân kỳ | $7.60 \times 10^{11}$ | $1.27 \times 10^{12}$ |
| 10.00 | Gauss–Seidel | 19.54 | 0.7905 | 89 | Hội tụ | $8.11 \times 10^{-11}$ | $2.17 \times 10^{-09}$ |
| 100.00 | Jacobi | 3.26 | 0.8549 | 147 | Hội tụ | $9.85 \times 10^{-11}$ | $1.20 \times 10^{-10}$ |
| 100.00 | Gauss–Seidel | 3.26 | 0.2795 | 18 | Hội tụ | $8.90 \times 10^{-11}$ | $2.31 \times 10^{-10}$ |

*Nhận xét:* Khi $\lambda$ tăng từ 0.01 lên 100, số điều kiện $\kappa_2(A)$ giảm mạnh từ 94.00 xuống 3.26. Jacobi từ trạng thái phân kỳ ($\rho(M) \approx 1.9$) chuyển sang hội tụ tại $\lambda = 100$ ($\rho(M) = 0.8549 < 1$). Gauss–Seidel giảm mạnh số vòng lặp từ 337 xuống chỉ còn 18 vòng.

---

### Bảng 2: TN2 – Ảnh hưởng của mức tương quan $r$ ($\lambda = 1.0$)
| $r$ | Phương pháp | $\kappa_2(A)$ | $\rho(M)$ | Số vòng lặp | Trạng thái | $\eta_r$ | $e_w$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 (seed 5201) | Jacobi | 2.05 | 0.3438 | 22 | Hội tụ | $4.00 \times 10^{-11}$ | $6.48 \times 10^{-11}$ |
| 0.10 (seed 5201) | Gauss–Seidel | 2.05 | 0.1177 | 12 | Hội tụ | $1.24 \times 10^{-11}$ | $3.00 \times 10^{-11}$ |
| 0.60 (seed 5202) | Jacobi | 5.14 | 1.0706 | 398 | Phân kỳ | $6.21 \times 10^{11}$ | $1.06 \times 10^{12}$ |
| 0.60 (seed 5202) | Gauss–Seidel | 5.14 | 0.3890 | 25 | Hội tụ | $3.74 \times 10^{-11}$ | $1.58 \times 10^{-10}$ |
| 0.97 (seed 5203) | Jacobi | 69.34 | 1.9103 | 42 | Phân kỳ | $6.40 \times 10^{11}$ | $1.12 \times 10^{12}$ |
| 0.97 (seed 5203) | Gauss–Seidel | 69.34 | 0.9335 | 285 | Hội tụ | $9.40 \times 10^{-11}$ | $7.34 \times 10^{-09}$ |

*Nhận xét:*
- Khi tương quan thấp ($r = 0.10$), cả hai phương pháp đều hội tụ rất nhanh (Jacobi: 22 vòng, GS: 12 vòng).
- Đáng chú ý ở trường hợp tương quan vừa ($r = 0.60$), số điều kiện $\kappa_2(A) = 5.14$ không quá lớn, nhưng $\rho(M_J) = 1.0706 > 1$, dẫn đến Jacobi vẫn bị phân kỳ sau 398 vòng. Điều này chứng minh rằng **không thể chỉ dựa vào số điều kiện $\kappa_2(A)$ để khẳng định tính hội tụ của Jacobi mà phải xét trực tiếp bán kính phổ $\rho(M)$**.

---

### Bảng 3: TN3 – Đối chứng Baseline ($r = 0.97$, seed 6201, $\lambda = 1.0$)
| Phương pháp | $\kappa_2(A)$ | $\rho(M)$ | Số vòng lặp | Trạng thái | $\eta_r$ | $e_w$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Jacobi | 83.32 | 1.9237 | 42 | Phân kỳ | $8.58 \times 10^{11}$ | $1.48 \times 10^{12}$ |
| Gauss–Seidel | 83.32 | 0.9433 | 337 | Hội tụ | $9.54 \times 10^{-11}$ | $7.29 \times 10^{-09}$ |

*Nhận xét:* Cung cấp giá trị baseline chuẩn cho TV3 so sánh với phương pháp SOR.

---

## 6. Đóng góp và Bàn giao TV2

Toàn bộ các yêu cầu của thành viên TV2 đã được hoàn thành 100%:
1. Code độc lập, tự cài đặt bằng NumPy thuần, không dùng solver có sẵn cho Jacobi và Gauss-Seidel.
2. Kiểm tra tính đúng đắn chặt chẽ qua 14 unit test và đối chiếu nghiệm với `scipy.linalg.solve`.
3. Toàn bộ thông số, hạt giống ngẫu nhiên và kết quả thực nghiệm khớp chính xác tuyệt đối với tài liệu báo cáo của nhóm.
