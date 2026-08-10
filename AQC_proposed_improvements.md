# Các đề xuất cải tiến AQC

## 1. Vượt qua chuẩn hóa Z-Score bằng Cấu trúc Học phương sai (Learned Variance Normalization)

### Vấn đề hiện tại

Ở bước **Inference**, AQC tính Z-score nội bộ dựa trên giá trị trung bình và độ lệch chuẩn của đúng `N` mẫu chuỗi hành động vừa được sinh ra.

Điều này có thể gây bất ổn trong hai trường hợp:

- Khi số lượng mẫu `N` nhỏ, các thống kê thực nghiệm (empirical statistics) như mean và standard deviation trở nên nhiễu.
- Khi phân phối của chính sách thay đổi trong quá trình **online fine-tuning**, các thống kê được tính từ batch hiện tại có thể không còn phản ánh tốt phân phối lợi thế thực tế.

Do đó, Z-score có thể trở nên **brittle**, đặc biệt khi `N` nhỏ hoặc xảy ra **distribution shift**.

### Đề xuất

Thay vì tính toán trực tiếp mean và variance trên `N` mẫu tại thời điểm inference, ta học một mạng dự đoán phương sai của advantage cho từng thang đo `k`.

Cụ thể, huấn luyện thêm một mạng để xấp xỉ moment bậc hai:

$$
M^k(s_t) \approx
\mathbb{E}_{a \sim \pi_\beta}
\left[
\left(A^k(s_t,a)\right)^2
\right]
$$

Sau đó tính phương sai:

$$
\operatorname{Var}^k(s_t)
=
M^k(s_t)
-
\left(V^k_{\text{advantage}}(s_t)\right)^2
$$

Trong đó, $V^k_{\text{advantage}}(s_t)$ là giá trị xấp xỉ kỳ vọng của advantage. Với cách định nghĩa advantage trong AQC, giá trị kỳ vọng này thường nằm xung quanh 0.

### Công thức chọn lọc mới

Khi inference, không cần tính thống kê từ batch `N` mẫu nữa. Score có thể được chuẩn hóa trực tiếp bằng phương sai đã học:

$$
\tilde{z}^k(s_t,a)
=
\frac{
A^k(s_t,a)
}{
\sqrt{\operatorname{Var}^k(s_t)}+\epsilon
}
$$

Trong đó:

- $A^k(s_t,a)$: advantage của hành động/chuỗi hành động với horizon `k`.
- $\operatorname{Var}^k(s_t)$: phương sai advantage được mạng học trước.
- $\epsilon$: hằng số nhỏ nhằm tránh chia cho 0.

### Lợi ích

- **Ổn định hơn khi `N` nhỏ:** không còn phụ thuộc vào phương sai mẫu được tính từ một batch nhỏ.
- **Độc lập với số lượng mẫu `N`:** không cần tính mean/std lại cho từng batch ở inference.
- **Phù hợp với online fine-tuning:** thống kê được học và có thể được cập nhật dần theo phân phối dữ liệu mới.
- **Inference nhanh hơn:** giảm các phép tính thống kê trên tập `N` mẫu.

---

## 2. Gộp chung các Mạng Phê bình (Unified Multi-Scale Critic với Horizon Embedding)

### Vấn đề hiện tại

AQC yêu cầu huấn luyện một mạng critic riêng cho mỗi kích thước chunk/horizon:

$$
k \in K
$$

Ví dụ, nếu:

$$
K = \{1,5,10,20\}
$$

thì cần huấn luyện bốn mạng critic riêng biệt.

Điều này dẫn đến:

- Chi phí huấn luyện tăng gần tuyến tính theo $|K|$.
- Chi phí bộ nhớ tăng theo số lượng critic.
- Các mạng $Q^{k_1}$ và $Q^{k_2}$ không chia sẻ tham số, mặc dù chúng đang đánh giá những hành vi có liên quan đến nhau.

### Đề xuất kiến trúc

Thay vì sử dụng một mạng critic độc lập cho từng `k`, sử dụng **một Unified Multi-Scale Critic** duy nhất:

$$
Q_\psi
\left(
s_t,
a_{t:t+k},
k
\right)
$$

Trong đó, `k` được đưa vào mạng thông qua một **horizon embedding**.

Có thể hình dung kiến trúc như sau:

```text
                    k
                    │
            Horizon Embedding
                    │
                    ▼
s_t ──────┐
          ├──► Shared Feature Extractor ──► Qψ
a_t:t+k ──┘
```

Các tầng trích xuất đặc trưng từ trạng thái/hình ảnh và hành động được chia sẻ giữa tất cả các horizon.

### Hàm mất mát

Thay vì tối ưu một loss riêng cho từng $k$, ta lấy kỳ vọng trên toàn bộ không gian horizon:

$$
\mathcal{L}_{\text{unified}}(\psi)
=
\mathbb{E}_{k\sim K,\mathcal{D}}
\left[
\left(
Q_\psi(s_t,a_{t:t+k},k)
-
y_t^k
\right)^2
\right]
$$

với target:

$$
y_t^k
=
\sum_{j=0}^{k-1}
\gamma^j r_{t+j}
+
\gamma^k V_{\xi_h}(s_{t+k})
$$

### Lợi ích

- **Giảm số lượng mạng cần huấn luyện:** chỉ còn một critic thay vì một critic cho mỗi `k`.
- **Chia sẻ đặc trưng:** CNN/MLP dùng để trích xuất đặc trưng được dùng chung.
- **Giảm bộ nhớ và chi phí huấn luyện.**
- **Học quan hệ giữa các horizon:** mạng có thể học rằng giá trị của các horizon khác nhau có mối liên hệ với nhau.
- **Có khả năng nội suy:** horizon embedding có thể giúp mạng ước lượng giá trị cho những `k` chưa xuất hiện trực tiếp trong tập horizon huấn luyện.

> Lưu ý: khả năng nội suy tới các `k` chưa được huấn luyện là một giả thuyết cần được kiểm chứng thực nghiệm, không phải đảm bảo kiến trúc sẽ luôn đạt được.

---

## 3. Tự động hóa Không gian `k` (Continuous / State-Conditioned Chunk Selection)

### Vấn đề hiện tại

Không gian kích thước chunk:

$$
K = \{k_1,k_2,\ldots,h\}
$$

hiện là một tập rời rạc được thiết kế thủ công dựa trên kiến thức của con người.

Ví dụ:

$$
K = \{1,5,10\}
$$

Trong trường hợp một trạng thái thực tế cần horizon gần `7`, hệ thống chỉ có thể lựa chọn một trong các giá trị có sẵn như `5` hoặc `10`.

Điều này có thể gây lãng phí hoặc làm giảm khả năng thích nghi với từng trạng thái.

### Đề xuất

Tích hợp cơ chế tương tự **Options Framework**, trong đó việc lựa chọn `k` được quyết định động dựa trên trạng thái.

Thay vì quyết định `k` cố định ngay từ đầu, policy có thể sinh ra một chuỗi hành động tối đa dài `h`:

$$
a_t,a_{t+1},\ldots,a_{t+h-1}
$$

Sau đó, tại mỗi bước con $\tau$, một mạng termination condition:

$$
\beta(s_\tau)
$$

sẽ quyết định liệu chunk có nên kết thúc tại trạng thái hiện tại hay không.

Quy trình:

```text
Sinh chunk tối đa h bước
        │
        ▼
   s_t → β(s_t)
        │
        ├── Không dừng → tiếp tục
        │
        ▼
   s_t+1 → β(s_t+1)
        │
        ├── Không dừng → tiếp tục
        │
        ▼
       ...
        │
        └── β(sτ) > threshold
                  │
                  ▼
            Cắt chunk tại τ
                  │
                  ▼
             Re-query policy
```

Khi:

$$
\beta(s_\tau) > \text{threshold}
$$

hệ thống kết thúc chunk tại $\tau$, loại bỏ các hành động phía sau và re-query policy từ trạng thái mới.

### Ý nghĩa

Khi đó, độ dài chunk thực tế:

$$
k^* = \tau - t
$$

được quyết định bởi trạng thái thay vì được cố định trước.

Ví dụ:

- Không gian mở → có thể chọn `k` lớn.
- Gần vật cản → termination sớm → `k` nhỏ.
- Tình huống cần thao tác chính xác → chunk có thể chỉ dài 1–2 bước.
- Chuyển động ổn định → chunk có thể dài hơn.

### Lợi ích

- **State-conditioned:** `k` thay đổi theo trạng thái.
- **Giảm phụ thuộc vào việc thiết kế thủ công tập `K`.**
- **Thích nghi với các tình huống có mức độ khó khác nhau.**
- Có thể tạo ra nhiều giá trị `k` hơn so với một tập horizon rời rạc cố định.

> Về mặt kỹ thuật, cách tiếp cận này gần với **adaptive termination** hơn là làm cho `k` thực sự liên tục. Nếu action chunk vẫn được thực thi theo từng bước rời rạc, `k` vẫn là một số nguyên; điểm mới nằm ở việc hệ thống tự quyết định thời điểm kết thúc chunk.

---

## 4. Chuyển từ Open-loop sang Closed-loop Low-level Executor

### Vấn đề hiện tại

Sau khi AQC lựa chọn một chuỗi hành động:

$$
a_{t:t+k}
$$

chuỗi này được thực thi theo kiểu **open-loop**.

Điều đó có nghĩa là hệ thống sinh ra một chuỗi hành động rồi thực thi nó mà không liên tục điều chỉnh dựa trên trạng thái thực tế trong quá trình thực hiện.

Điều này đặc biệt bất lợi trong các tình huống có tương tác vật lý.

Ví dụ:

- robot tiến gần vật thể;
- vật thể có vị trí hơi khác dự kiến;
- robot bị sai lệch quỹ đạo;
- môi trường có nhiễu hoặc chuyển động ngoài dự đoán.

Trong những trường hợp này, việc thực thi một chunk dài có thể khiến robot phản ứng chậm.

Ngược lại, nếu giảm xuống `k = 1`, AQC phải gọi policy liên tục. Điều này lại trở nên tốn kém nếu policy $\pi_\beta$ là một mô hình Flow Matching nặng, vốn được thiết kế để sinh ra cả chuỗi hành động dài `h` bước.

### Đề xuất: Hierarchical Execution

Tách hệ thống thành hai tầng:

```text
                 HIGH-LEVEL
                    AQC
                     │
                     ▼
          Kế hoạch / trajectory mẫu
                     │
                     ▼
              a*_{t:t+k}
                     │
                     ▼
              LOW-LEVEL
          Closed-loop Controller
                     │
                     ▼
               Robot / Env
                     ▲
                     │
              State feedback
```

### Tầng 1 — AQC High-level

AQC vẫn chịu trách nhiệm:

- đánh giá các action chunk;
- lựa chọn horizon phù hợp;
- lập kế hoạch quỹ đạo;
- sinh ra một chuỗi mục tiêu/trajectory mẫu.

Ví dụ:

$$
\tau^*
=
\left[
s_t^*,s_{t+1}^*,\ldots,s_{t+k}^*
\right]
$$

hoặc một chuỗi action:

$$
a_{t:t+k}^*
$$

Đây được xem như **reference trajectory** cho tầng điều khiển phía dưới.

### Tầng 2 — Closed-loop Low-level Controller

Thay vì gửi trực tiếp các action đã sinh tới động cơ theo kiểu open-loop, một controller tốc độ cao sẽ liên tục quan sát trạng thái thực tế:

$$
s_{t+i}
$$

và điều chỉnh action để robot bám theo trajectory mục tiêu.

Có thể sử dụng:

- PD Controller;
- PID Controller;
- Learned PD Controller;
- MLP nhỏ;
- hoặc một policy lightweight được tối ưu cho inference tốc độ cao.

Ví dụ, controller có thể hoạt động ở tần số:

$$
500\text{ Hz}
$$

Trong khi AQC chỉ cần cập nhật trajectory ở tần số thấp hơn.

Một dạng điều khiển đơn giản có thể là:

$$
a_t
=
K_p(s_t^*-s_t)
+
K_d(\dot{s}_t^*-\dot{s}_t)
$$

Trong đó:

- $s_t^*$: trạng thái mục tiêu từ trajectory của AQC.
- $s_t$: trạng thái thực tế.
- $\dot{s}_t^*$: vận tốc mục tiêu.
- $\dot{s}_t$: vận tốc thực tế.
- $K_p,K_d$: hệ số điều khiển.

### Lợi ích

- **Tách planning và control:** AQC tập trung vào quyết định dài hạn, low-level controller tập trung vào phản ứng tức thời.
- **Closed-loop:** controller có thể sửa sai dựa trên trạng thái thực tế.
- **Phản ứng nhanh:** low-level controller có thể chạy ở tần số rất cao.
- **Giảm tải cho AQC:** không cần re-query policy nặng ở mọi timestep.
- **Cho phép sử dụng chunk dài hơn:** ngay cả khi AQC chọn `k` lớn, low-level controller vẫn có thể điều chỉnh liên tục khi robot tiến gần vật thể hoặc gặp nhiễu.

### Điểm cần lưu ý

Ý tưởng này không có nghĩa là có thể tăng `k` tùy ý. Nếu AQC tạo ra một trajectory quá sai hoặc không khả thi, low-level controller không thể luôn luôn "cứu" được hệ thống.

Do đó, cần đánh giá thực nghiệm sự đánh đổi giữa:

$$
\text{Planning Horizon}
\quad\leftrightarrow\quad
\text{Tracking Error}
\quad\leftrightarrow\quad
\text{Reactivity}
$$

Một kiến trúc hợp lý là để AQC quyết định **mục tiêu/trajectory ở high-level**, còn low-level controller đảm nhiệm việc **tracking và phản ứng với nhiễu ở tần số cao**.
