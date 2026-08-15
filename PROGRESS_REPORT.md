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

### 5. Sửa Lỗi Nghiêm Trọng (Hotfix 2026-08-16)
- [x] Phát hiện lỗi 0% Success Rate do mạng $M^k$ tính sai toán học (dùng Expectile thay vì Mean). **Tạm gỡ LVN, lùi về dùng Sample Z-Score** để đảm bảo công thức chuẩn hóa đúng.
- [x] Sửa lỗi "Dead Branch" trong `sample_actions_adaptive`: ép dùng mạng `actor_onestep_flow` khi mode là `distill-ddpg`, giúp tăng tốc độ đánh giá lên hàng chục lần.

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
