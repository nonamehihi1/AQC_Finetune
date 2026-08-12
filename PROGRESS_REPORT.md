# Báo Cáo Tiến Độ (Progress Report) — AQC: Adaptive Q-Chunking

**Cập nhật lần cuối**: 2026-08-12

---

## Tóm Tắt

Chúng tôi đang mở rộng Q-Chunking (QC) với **Adaptive Q-Chunking (AQC)** — một phương pháp chọn kích thước chunk tự động dựa trên trạng thái, sử dụng **Learned Variance Normalization (LVN) (Chuẩn hóa phương sai học được)** để thay thế cho việc tính toán Z-score dựa trên mẫu thường bị nhiễu.

---

## Công Việc Đã Hoàn Thành

### 1. Khảo Sát Tài Liệu & Nhận Diện Vấn Đề
- [x] Đọc và phân tích bài báo Q-Chunking (Li et al., NeurIPS 2025)
- [x] Nhận diện hạn chế của kích thước chunk cố định là điểm chính cần cải thiện
- [x] Ghi chép 4 hướng cải tiến tiềm năng (xem `AQC_proposed_improvements.md`)
- [x] Thực hiện phân tích tính khả thi của cả 4 đề xuất → chọn **LVN** là ưu tiên cao nhất

### 2. Triển Khai Agent AQC (`agents/aqc.py`)
- [x] Triển khai kiến trúc multi-scale critic: tách biệt các mạng $Q^k$, $V^k$, $M^k$ cho mỗi $k \in \{1, 3, 5\}$
- [x] Triển khai hàm loss cho critic bao gồm:
  - $Q^h$ loss (FQL chuẩn trên toàn bộ chân trời thời gian)
  - $V^h$ loss (hồi quy expectile)
  - Các loss $Q^k$, $V^k$, $M^k$ trên từng quy mô với bootstrapping hợp lý thông qua $V^h(s_{t+k})$
- [x] Triển khai `sample_actions_adaptive()`: sinh các chuỗi hành động ứng viên, đánh giá advantage chuẩn hóa $\tilde{z}^k = A^k / \sqrt{M^k}$ trên tất cả các kích thước chunk, cùng lúc chọn tốt nhất cả $(k^*, a^*)$
- [x] Sửa lỗi sai kích thước (dimension mismatch) trong `sample_actions_adaptive()` khi broadcasting tensor V^k và M^k

### 3. Hỗ Trợ Đánh Giá (`evaluation.py`)
- [x] Sửa đổi vòng lặp đánh giá để tự động nhận diện agent AQC (qua `hasattr(agent, 'sample_actions_adaptive')`)
- [x] Quá trình đánh giá đã tự động cắt các chunk hành động đúng bằng số bước $k^*$ được chọn

### 4. Kiểm Tra Nhanh (Smoke Test)
- [x] Đã xác minh quá trình huấn luyện AQC chạy không bị lỗi crash (10 bước offline + 15 bước online)
- [x] Đã xác nhận các giá trị loss (Q, V, M losses) đều hợp lý và giảm dần

---

## Đang Thực Hiện

### 5. Thực Nghiệm Toàn Diện (Kaggle GPU T4x2)

**Kế hoạch thực nghiệm:**

| # | Phương Pháp | Môi Trường | Số Bước | Loại Actor | Trạng Thái |
|---|--------|-------------|-------|-----------|--------|
| 1 | ACFQL baseline (h=5) | `cube-triple-play-singletask-task4-v0` | 1M offline + 1M online | `distill-ddpg` | ⏳ Đang đợi |
| 2 | AQC-LVN (k∈{1,3,5}) | `cube-triple-play-singletask-task4-v0` | 1M offline + 1M online | `distill-ddpg` | ⏳ Đang đợi |

**Thời gian ước tính**: ~8-11 giờ mỗi thực nghiệm trên Kaggle GPU T4.

**Các số liệu cần so sánh**: 
- Tỷ lệ thành công (quan trọng nhất)
- Đồ thị huấn luyện (tỷ lệ thành công vs số bước)
- Phân phối các giá trị $k^*$ được chọn ở nhiều trạng thái
- Tốc độ huấn luyện (số vòng lặp/giây)

---

## Các Bước Tiếp Theo

### Ngắn hạn (Tuần này)
- [ ] Chạy Thực nghiệm 1 (ACFQL baseline) trên Kaggle
- [ ] Chạy Thực nghiệm 2 (AQC-LVN) trên Kaggle
- [ ] So sánh kết quả và vẽ đồ thị
- [ ] Nếu AQC-LVN hoạt động kém: thực hiện ablation study để chẩn đoán (quá trình học M^k, chuẩn độ z-score, v.v.)

### Trung hạn (Nếu LVN hiệu quả)
- [ ] Chạy thêm trên các môi trường khác (task2, task3) và nhiều seed (3-5 seed để đảm bảo ý nghĩa thống kê)
- [ ] Ablation: so sánh LVN vs Z-score gốc vs Chuẩn hóa Running EMA
- [ ] Ablation: thay đổi $N$ (số mẫu ứng viên) — dự kiến LVN ít nhạy cảm với $N$ hơn Z-score
- [ ] Phân tích phân phối k*: AQC có thực sự chọn các k khác nhau cho các trạng thái khác nhau không?

### Khám Phá (Nếu có thời gian)
- [ ] Triển khai biến thể LVN tối giản sử dụng Running EMA (không cần thêm mạng)
- [ ] Thử nghiệm với Unified Multi-Scale Critic (Đề xuất 2) — phương pháp nhúng theo chân trời thời gian
- [ ] Nghiên cứu về re-planning dựa trên yếu tố bất ngờ (surprise-based) trong khi thực thi chunk

---

## Rủi Ro & Biện Pháp Giảm Thiểu

| Rủi Ro | Khả Năng | Biện Pháp Giảm Thiểu |
|------|-----------|------------|
| AQC-LVN hoạt động kém hơn baseline | Trung bình | LVN có nhiều mạng hơn để học → có thể cần train lâu hơn hoặc tinh chỉnh siêu tham số. Bắt đầu với cùng siêu tham số như ACFQL. |
| Không đủ thời gian GPU | Thấp | Việc dùng `distill-ddpg` actor giúp giảm thời gian train khoảng 4 lần. Mỗi thực nghiệm nằm trong giới hạn 12h của Kaggle. |
| Mạng M^k khó hội tụ | Trung bình | Theo dõi M^k loss khi huấn luyện. Nếu bất ổn, hãy thử phương án Running EMA. |
| Lỗi kích thước/hình dạng trong AQC | Thấp | Đã sửa xong một lỗi lớn. Kiểm tra nhanh (smoke test) cũng đã qua. |

---

## Các File Đã Thay Đổi (So với Codebase QC Gốc)

| File | Loại Thay Đổi | Mô tả |
|------|-----------|-------------|
| `agents/aqc.py` | **MỚI** | Agent AQC: multi-scale critics, LVN, chọn chunk thích ứng |
| `agents/__init__.py` | ĐÃ SỬA | Đã đăng ký agent AQC trong agent registry |
| `evaluation.py` | ĐÃ SỬA | Thêm hỗ trợ chunk thích ứng vào vòng lặp đánh giá |
| `main.py` | ĐÃ SỬA | Thêm cờ `--chunk_sizes` |
| `AQC_proposed_improvements.md` | **MỚI** | Tài liệu đề xuất chi tiết (4 ý tưởng cải tiến) |
| `PROGRESS_REPORT.md` | **MỚI** | Chính là file này |
| `README.md` | ĐÃ SỬA | Thêm Ý Tưởng Tổng Quan và hướng dẫn chạy |
