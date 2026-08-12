<div align="center">

# AQC: Adaptive Q-Chunking with Learned Variance Normalization
### Mở rộng từ [Q-Chunking (NeurIPS 2025)](https://arxiv.org/abs/2507.07969) với khả năng lựa chọn chunk thích ứng

</div>

---

## Ý Tưởng Tổng Quan (General Idea)

### Bối cảnh: Q-Chunking (QC)

Q-Chunking (Li et al., NeurIPS 2025) đề xuất **action chunking cho RL**: thay vì chọn một hành động ở mỗi bước thời gian, tác tử cam kết thực hiện một chuỗi $h$ hành động (một "chunk"), cho phép khám phá có tính nhất quán về mặt thời gian và tận dụng dữ liệu mẫu offline hiệu quả hơn. Kết hợp với Flow Q-Learning (FQL), QC đạt được kết quả SOTA trên các tác vụ thao tác khéo léo trong OGBench.

**Hạn chế chính**: QC sử dụng **kích thước chunk cố định** $h$ cho tất cả các trạng thái. Trong thực tế, một số trạng thái sẽ hưởng lợi từ chunk dài (ví dụ: các chuyển động tiếp cận trong không gian tự do), trong khi các trạng thái khác lại cần chunk ngắn (ví dụ: khi thao tác tiếp xúc chính xác với vật thể). Một $h$ cố định là không tối ưu — quá dài gây ra phản ứng chậm chạp gần các điểm tiếp xúc, quá ngắn làm lãng phí lợi ích của abstraction theo thời gian.

### Adaptive Q-Chunking (AQC)

Mở rộng QC với **khả năng lựa chọn chunk thích ứng phụ thuộc vào trạng thái**. Ý tưởng cốt lõi:

1. **Multi-scale critics (Các mạng critic đa quy mô)**: Huấn luyện các mạng $Q^k(s, a_{1:k})$ riêng biệt cho mỗi kích thước chunk ứng viên $k \in K = \{1, 3, 5\}$, cùng với mạng giá trị $V^k(s)$ và mạng moment bậc hai $M^k(s)$.

2. **Learned Variance Normalization (LVN) (Chuẩn hóa phương sai học được)**: Thay vì tính Z-scores từ các số liệu thống kê mẫu (thường bị nhiễu khi $N$ nhỏ), chúng tôi học phương sai của advantage trực tiếp:

$$\tilde{z}^k(s, a) = \frac{Q^k(s, a) - V^k(s)}{\sqrt{M^k(s)} + \epsilon}$$

trong đó $M^k(s) \approx \mathbb{E}_{a \sim \pi}[(A^k(s,a))^2]$ là moment bậc hai học được.

3. **Lựa chọn thích ứng**: Tại bước suy luận (inference), tác tử lấy mẫu $N$ chuỗi hành động ứng viên, đánh giá advantage đã chuẩn hóa $\tilde{z}^k$ trên tất cả các kích thước chunk, và chọn đồng thời cả kích thước chunk tốt nhất $k^*$ và chuỗi hành động tốt nhất.

### Tại Sao Lại Quan Trọng

| Khía cạnh | QC (Gốc) | AQC (Của chúng tôi) |
|--------|:---:|:---:|
| Kích thước chunk | Cố định $h$ cho mọi trạng thái | Thích ứng $k^*$ cho từng trạng thái |
| Chuẩn hóa Z-score | Thống kê mẫu (nhiễu khi $N$ nhỏ) | Phương sai học được (ổn định) |
| Tác vụ nhiều va chạm | Phải dùng $h$ nhỏ cho toàn bộ | Có thể dùng $k$ lớn ở không gian trống, $k$ nhỏ khi gần va chạm |
| Tinh chỉnh online | Z-score bị trôi (drift) khi policy đổi | Phương sai học được thích ứng mượt mà |

### Sơ Đồ Kiến Trúc

```text
                    State s_t
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    Q^1(s,a_{1:1})  Q^3(s,a_{1:3})  Q^5(s,a_{1:5})
    V^1(s)          V^3(s)          V^5(s)
    M^1(s)          M^3(s)          M^5(s)
        │              │              │
        ▼              ▼              ▼
    z̃^1 = A^1/√M^1  z̃^3 = A^3/√M^3  z̃^5 = A^5/√M^5
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              k* = argmax z̃^k
              (chọn chunk size + action tốt nhất)
```

---

## Cấu Trúc Repository

```text
.
├── agents/
│   ├── acfql.py          # ACFQL — Q-Chunking baseline (chunk cố định)
│   ├── aqc.py            # AQC — Adaptive Q-Chunking với LVN (phương pháp của chúng tôi)
│   ├── acrlpd.py         # Agent dựa trên RLPD
│   └── model.py          # Kiến trúc mạng dùng chung
├── envs/                 # Các tiện ích môi trường (OGBench, Robomimic)
├── utils/                # Dataset, tiện ích Flax, định nghĩa mạng
├── evaluation.py         # Vòng lặp đánh giá (hỗ trợ chọn chunk thích ứng)
├── main.py               # Script huấn luyện chính (offline + online RL)
├── main_online.py        # Script huấn luyện chỉ dùng online
├── AQC_proposed_improvements.md   # Tài liệu chi tiết các đề xuất
├── PROGRESS_REPORT.md    # Tiến độ hiện tại và các bước tiếp theo
└── requirements.txt      # Các thư viện phụ thuộc
```

### Các File Quan Trọng

| File | Mô tả |
|------|-------------|
| `agents/aqc.py` | **Đóng góp chính** — Agent AQC với multi-scale critics ($Q^k$, $V^k$, $M^k$) và chọn chunk thích ứng qua LVN |
| `agents/acfql.py` | Baseline — Q-Chunking gốc với chunk size cố định |
| `evaluation.py` | Đã sửa đổi để hỗ trợ `sample_actions_adaptive()` — tự động phát hiện AQC và dùng chọn chunk thích ứng khi đánh giá |
| `main.py` | Vòng lặp huấn luyện với cờ `--chunk_sizes` để chỉ định các kích thước chunk ứng viên |

---

## Hướng Dẫn Chạy

### Cài Đặt
```bash
git clone https://github.com/nonamehihi1/AQC_Finetune.git
cd AQC_Finetune
pip install -r requirements.txt
```

### Chạy AQC (Phương Pháp Của Chúng Tôi)
```bash
MUJOCO_GL=egl python main.py \
    --agent=agents/aqc.py \
    --agent.actor_type=distill-ddpg \
    --run_group=AQC_LVN \
    --env_name=cube-triple-play-singletask-task4-v0 \
    --offline_steps=1000000 \
    --online_steps=1000000 \
    --horizon_length=5 \
    --chunk_sizes=1,3,5 \
    --seed=1
```

### Chạy Baseline (ACFQL — Chunk Cố Định)
```bash
MUJOCO_GL=egl python main.py \
    --agent=agents/acfql.py \
    --agent.actor_type=distill-ddpg \
    --run_group=Baseline_ACFQL \
    --env_name=cube-triple-play-singletask-task4-v0 \
    --offline_steps=1000000 \
    --online_steps=1000000 \
    --horizon_length=5 \
    --seed=1
```

---

## Tham Khảo (References)

- **Q-Chunking**: Li et al., "Reinforcement Learning with Action Chunking", NeurIPS 2025. [[paper](https://arxiv.org/abs/2507.07969)] [[website](https://colinqiyangli.github.io/qc/)]
- **FQL**: Park et al., "Flow Q-Learning". [[code](https://github.com/seohongpark/fql)]
- **OGBench**: Park et al., "OGBench: Benchmarking Offline Goal-Conditioned RL". [[code](https://github.com/seohongpark/ogbench)]

## Lời Cảm Ơn (Acknowledgments)

Codebase này được xây dựng dựa trên [Q-Chunking](https://github.com/ColinQiyangLi/qc) và [FQL](https://github.com/seohongpark/fql). Hai thư mục `rlpd_*` được lấy trực tiếp từ [RLPD](https://github.com/ikostrikov/rlpd).
