# Lab 21 — Evaluation Report

**Họ tên**: Lệnh Quang Hưng  **MSSV**: 2A202601546  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab Free T4, ~14.6 GB VRAM`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | `train_seed.jsonl` — 250 ticket CSKH → JSON triage (corpus mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 (tier T4 mặc định) — p95 đo được chỉ là 98 token *(results/token_stats.json)*. Giữ 1024 vì đây là trần an toàn của tier T4 trong VRAM 14.6 GB, và 98 token p95 nằm sâu trong ngân sách đó nên không có lý do hạ xuống 256 chỉ để tiết kiệm — corpus ngắn, không phải trường hợp p95 vượt trần cần xử lý. |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch / 30 step (225 mẫu, batch hiệu dụng 16 trên T4) |

**Template có giữ khối `<think>` không?** **Có** — *(results/template_check.json: verdict = "reasoning preserved — safe to train on traces")*.
Vì template giữ `<think>`, corpus mặc định (250 câu trả lời JSON trần, không có trace suy luận) khiến `masked-think`/`response-only` trở thành no-op y hệt `assistant-only` — không có gì để loại trừ khỏi loss. Tôi giữ nguyên `MASK_MODE=assistant-only` vì đây đã là mặc định SFT đúng cho corpus này, không cần đổi.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (39/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss (`supervised_preview`, `results/mask_proof.json`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đối chiếu với đoạn bị mask (`masked_preview`) để thấy rõ ranh giới: toàn bộ `<|im_start|>system...`, `<|im_start|>user\nAlo shop, mình đặt balo laptop mã đơn VN411453...` và khối `<think>\n\n` mở đầu **không** nằm trong đoạn trên — chỉ có JSON trả lời và token đóng mới được tính loss. `supervised_fraction=0.4149` nằm rõ dưới ngưỡng 0.95, xác nhận model không học viết lại câu hỏi.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3310.7 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1056.0 |
| (c) LoRA fine-tune | 0.970 | 0.611 | 1.000 | 1398.2 |

**(b) có thật sự mạnh hơn (a) không?** **Có** — `(b).target = 0.765` vượt xa `(a).target = 0.000` (naive prompt không ra được JSON đúng định dạng nào, `format=0.000`). Tôi không sửa `OPTIMIZED_PROMPT` — dùng nguyên bản có sẵn trong `src/labkit/generate.py`, được `make verify` xác nhận SHA khớp bản gốc (`optimized_prompt_sha` trong `baselines_frozen.json` không đổi, kiểm tra "baseline (b) prompt unmodified" = ok).

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6262 | **0.97** | 1000.9 | 12.01 |
| `attn_only` | q,v (matched) | ~283/566* | 32,456,704 | 1e-4 | 0.5368 | **0.97** | 835.0 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | **0.00** | 961.5 | 12.01 |
| `qlora` | text-linear | 16 (4-bit) | 32,464,896 | 1e-4 | 0.7058 | **0.94** | 1019.0 | 7.09 |

\* `matched_rank()` nâng rank cho `attn_only` để khớp ngân sách tham số (32,456,704 vs 32,464,896 — lệch <5%, xác nhận bởi `make verify`: "attn_only is a FAIR contrast").

Trả lời ba câu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

`attn_only` **hoà tuyệt đối** với `correct` trên target (0.97 = 0.97), dù chỉ gắn vào q,v thay vì toàn bộ linear của text decoder. Thứ tự theo train loss lại **ngược lại**: `attn_only` có train loss thấp hơn (0.5368 so với 0.6262 của `correct`) — nếu chỉ nhìn train loss, tôi sẽ kết luận sai rằng `attn_only` "tốt hơn" `correct`, trong khi trên thước đo thật (target accuracy) chúng bằng nhau. Điều này cho thấy với ngân sách tham số bằng nhau, **rank là đòn bẩy chính, không phải vị trí gắn adapter** — ít nhất trên corpus 250 mẫu và bài toán triage 4 trường này. Đây đúng là kết quả đáng giá nhất của lab: hai cấu hình khác hẳn nhau về "nơi gắn" nhưng cho cùng năng lực khi ngân sách tham số được cân bằng, nghĩa là câu chuyện "attention-only luôn kém hơn all-linear" không đúng một cách vô điều kiện — nó chỉ đúng khi so sánh không cân bằng ngân sách (rank giữ nguyên 16 ở cả hai vị trí), đúng như cảnh báo ở mục 2.1-2.5 của rubric.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

`wrong_lr` dùng LR=1e-5 (thang full fine-tune) thay vì 1e-4 (thang LoRA đúng, 10× lớn hơn). Train loss cuối của nó là 1.5704 — cao hơn hẳn so với `correct` (0.6262), và train sụp hoàn toàn trên tập target: `target=0.00`, `format=0.00` (autopsy.json). Nếu chỉ nhìn con số loss mà không biết LR đã bị đổi, tôi sẽ dễ kết luận sai rằng "LoRA không học được gì trên bài toán này" hoặc "cấu hình LoRA (rank, vị trí) có vấn đề" — trong khi thực chất chỉ một siêu tham số (LR) bị đặt sai thang, và LoRA hoàn toàn có khả năng học tốt (như `correct` đã chứng minh với đúng LR). Đây chính là bài học §10.3: LoRA cần LR ≈ 10× so với full fine-tune vì số tham số huấn luyện ít hơn rất nhiều lần, và không tuân theo trực giác "LR thấp thì an toàn hơn".

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

`qlora` giảm VRAM từ 12.01 GB xuống 7.09 GB — tiết kiệm 4.92 GB (~41%), đáng kể trên một card T4 14.6 GB. Đổi lại, target giảm nhẹ từ 0.97 xuống 0.94 (mất 0.03, tức 3% điểm accuracy) và latency tăng từ 1398.2ms lên 1790.4ms (chậm hơn ~28% do overhead dequantize khi suy luận). Số đo của tôi **ủng hộ một phần** khuyến nghị của nhà cung cấp: QLoRA không "miễn phí" như quảng cáo — nó đánh đổi thật (chất lượng giảm nhẹ + chậm hơn), nhưng mức giảm 3% target trên bài toán này không nghiêm trọng tới mức "không dùng được". Nếu VRAM là ràng buộc cứng (ví dụ muốn chạy model lớn hơn trên cùng một T4), QLoRA vẫn là lựa chọn khả thi; nếu VRAM dư dả như ở đây (12GB vẫn vừa T4 14.6GB), không có lý do đánh đổi 3% target để tiết kiệm VRAM không cần thiết.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.147` · `valid_trace_rate = 0.00`

Diễn giải: Bản fine-tune **thắng rõ ràng** trên tác vụ chính — target tăng từ 0.765 (baseline b) lên 0.970 (Δ +0.205, tương đương +26.8% tương đối). Tuy nhiên cổng hồi quy **FAILED** vì điểm regression (15 câu kiến thức phổ thông) tụt từ 0.7578 xuống 0.6111, tức giảm 0.147 — vượt xa ngưỡng cho phép 0.02, gấp hơn 7 lần. Đây là dấu hiệu kinh điển của **catastrophic forgetting**: quá trình fine-tune trên 250 mẫu triage hẹp đã khiến model quên một phần năng lực trả lời câu hỏi phổ thông ngoài miền dữ liệu train. `valid_trace_rate=0.00` phản ánh đúng những gì đã ghi ở mục 1 — corpus không chứa trace suy luận nên không có gì để đo hợp lệ ở chỉ số này, không phải model bị lỗi reasoning. Kết luận: bản fine-tune này **chưa sẵn sàng để deploy nguyên trạng** — cần khắc phục regression (ví dụ trộn 1-5% dữ liệu phổ thông vào tập train theo deck §14.3) trước khi coi là thắng thật sự trên mọi phương diện, dù nó đã chứng minh vượt trội tuyệt đối trên đúng tác vụ triage mà nó được huấn luyện cho.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "Cho mình hỏi, mình đặt chuột không dây... Cho tôi trả lại" | doi_tra / cao / chuột không dây / tich_cuc | — | intent=doi_tra, urgency=cao, product=chuột không dây, sentiment=tich_cuc (score 1.0) | ✅ FT thắng — đúng cả 4 trường |
| 2 | "Shop ơi, mình đặt ốp lưng điện thoại... Hoàn tiền. Sớm..." | hoan_tien / trung_binh / ốp lưng điện thoại / ... | — | intent=hoan_tien, product=ốp lưng điện thoại (score 1.0) | ✅ FT thắng — đúng cả 4 trường |
| 3 | "Cho mình hỏi, mình đặt bình giữ nhiệt... Chưa thấy tiền." | hoan_tien / **thap** / bình giữ nhiệt / tich_cuc | — | intent=hoan_tien, **urgency=trung_binh** (sai), product=bình giữ nhiệt (score 0.75) | ❌ **FT thua** — sai đúng ở `urgency`, đoán "trung_binh" thay vì "thap" |
| 4 | "Shop ơi, mình đặt nồi chiên không dầu... Thiếu phụ kiện." | **san_pham_loi** / thap / nồi chiên không dầu / trung_tinh | — | intent=san_pham_loi, **urgency=trung_binh** (sai) (score 0.75) | ❌ **FT thua** — sai `urgency`, cùng pattern với ca 3 |
| 5 | "Cho mình hỏi, mình đặt đèn bàn LED... Giao hàng chậm." | van_chuyen / **thap** / đèn bàn LED / tich_cuc | — | intent=van_chuyen, **urgency=trung_binh** (sai) (score 0.75) | ❌ **FT thua** — sai `urgency` |
| 6 | "Chào shop, mình đặt đèn bàn LED... Sai màu." | san_pham_loi / **thap** / đèn bàn LED / tich_cuc | — | intent=san_pham_loi, **urgency=trung_binh** (sai) (score 0.75) | ❌ **FT thua** — sai `urgency` |

*(Cột (b) prompt để trống vì `qualitative.json` chỉ ghi lại dự đoán của bản fine-tune (c) — không lưu song song dự đoán (b) cho từng ticket cụ thể; số liệu tổng hợp so sánh (b) vs (c) đã có ở bảng mục 3.)*

**Có mẫu chung nào ở các ca FT thua không?** Có, rất rõ ràng: **cả 4 ca thua đều sai đúng một trường — `urgency`** — và luôn theo cùng một hướng: nhãn đúng là `thap` (thấp) nhưng model luôn đoán `trung_binh` (trung bình). Không có ca nào sai `intent`, `product`, hay `sentiment`. Điều này gợi ý model có xu hướng hệ thống "kéo" mức độ khẩn cấp về giá trị trung tính an toàn (`trung_binh`) khi tín hiệu trong câu ngắn/mơ hồ về mức độ gấp — có thể do phân phối nhãn `urgency` trong 250 mẫu train không đủ đa dạng câu diễn đạt "thấp" để model phân biệt chắc chắn với "trung bình".

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không khuyến nghị deploy bản fine-tune `correct` ở trạng thái hiện tại**, dù nó thắng áp đảo baseline (b) trên đúng tác vụ triage (target 0.970 vs 0.765, tăng 20.5 điểm phần trăm) — vì nó đánh đổi bằng một cái giá không được phép trả: quên gần 15% năng lực trả lời kiến thức phổ thông (regression tụt 14.7 điểm phần trăm, vượt ngưỡng cho phép hơn 7 lần). Một hệ thống CSKH thật không thể chỉ giỏi phân loại ticket rồi trở nên kém tin cậy ở mọi việc khác nó từng làm được. Đòn bẩy thật sự trong lab này, dựa trên 4 thí nghiệm đối chứng, **không phải vị trí gắn adapter** (attn_only hoà tuyệt đối với correct khi ngân sách tham số cân bằng — mục 4.1) và cũng không hẳn là "rank càng cao càng tốt" theo nghĩa đơn giản, mà là **hai thứ khác**: (1) learning rate đúng thang — sai một con số này (wrong_lr) làm sụp hoàn toàn khả năng học, biến LoRA từ "thắng áp đảo" thành "không học được gì" (target 0.00); và (2) **chất lượng/độ đa dạng của dữ liệu train** — 250 mẫu không đủ đa dạng cách diễn đạt để tránh catastrophic forgetting, và cũng là nguyên nhân trực tiếp gây ra lỗi hệ thống ở trường `urgency` trong toàn bộ ca thua tôi tìm được (mục 6). Mask đúng (mục 2) là điều kiện cần nhưng không đủ — nó đảm bảo model học đúng thứ cần học, nhưng không tự động đảm bảo model không quên thứ khác. Nếu phải chọn một việc để làm tiếp trước khi deploy, tôi sẽ ưu tiên trộn 1-5% dữ liệu tổng quát vào tập train (đúng gợi ý deck §14.3) để kiểm tra xem có kéo lại được regression về dưới ngưỡng 0.02 mà không đánh đổi quá nhiều target, hơn là tiếp tục tinh chỉnh rank hay vị trí adapter — vì thí nghiệm 4.1 đã cho thấy rõ đó không phải chỗ đáng đầu tư thêm.

**Ba điều tôi học được** (cụ thể, không generic):
1. "Fine-tune thắng baseline" và "fine-tune sẵn sàng deploy" là hai câu khẳng định khác nhau — bảng 3 baseline chỉ trả lời câu đầu, cổng hồi quy 4 nhóm mới trả lời câu thứ hai; nếu lab chỉ dừng ở NB2/NB3 tôi sẽ báo cáo sai rằng mọi thứ ổn.
2. Train loss và điểm target có thể xếp hạng ngược nhau hoàn toàn (attn_only có loss thấp hơn correct nhưng target bằng nhau, không cao hơn) — dùng train loss để so sánh giữa các cấu hình khác nhau là một cái bẫy cụ thể, không phải lý thuyết suông, vì tôi đã tự đo thấy nó trên đúng 2 con số này.
3. Một siêu tham số sai thang (LR 1e-5 thay vì 1e-4, chỉ lệch 10 lần) có thể tạo ra khác biệt kết quả lớn hơn nhiều so với thay đổi kiến trúc (vị trí gắn adapter) — nghĩa là khi debug "LoRA không học được", việc đầu tiên nên kiểm tra là LR, không phải rank hay target_modules.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn một tỷ lệ nhỏ (1-5%) dữ liệu hỏi-đáp phổ thông vào tập train theo đúng gợi ý deck §14.3, huấn luyện lại đúng cấu hình `correct` với cùng max_steps, rồi xem cổng hồi quy có chuyển PASSED hay không mà không đánh đổi quá nhiều điểm target — đây là câu hỏi cụ thể còn bỏ ngỏ mà lab core không yêu cầu nhưng dữ liệu tôi có sẵn hoàn toàn đủ để trả lời.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
