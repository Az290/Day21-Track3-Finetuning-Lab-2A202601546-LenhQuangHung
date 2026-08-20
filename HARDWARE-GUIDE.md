# Hardware Guide — chọn tier

> **Lab 21 ≠ Lab 22.** SFT giữ **một** model trong bộ nhớ. DPO (Ngày 22) giữ policy *và*
> reference → gấp ~2× VRAM. Vì thế Lab 21 chạy được model lớn hơn Lab 22 trên cùng GPU.

---

## 1. VRAM cho LoRA SFT

Con số dưới đây là mức tối thiểu **bf16 LoRA** do chính nhà cung cấp công bố cho họ
Qwen3.5. Đổi model thì phải đổi số — và đo lại bằng `torch.cuda.max_memory_allocated()`.

| Base | bf16 LoRA | Vừa với |
|---|---:|---|
| Qwen3.5-0.8B | ~3 GB | mọi GPU; CPU chỉ để làm NB1 |
| Qwen3.5-2B | ~5 GB | laptop 8–12 GB |
| **Qwen3.5-4B** | **~10 GB** | **Colab Free T4 (16 GB)** ✓ |
| Qwen3.5-9B | ~22 GB | L4 22.5 GB, A100 40 GB, RTX 3090/4090 |
| Qwen3.5-27B | ~56 GB | A100 80 GB, H100 |

Activation + KV cache tăng theo `max_length × batch`. Tier T4 dùng `max_length=1024`,
`batch=1`, `grad_accum=16` → batch hiệu dụng 16 (dưới trần 32 của deck §10.4).

### Vì sao mặc định là bf16 LoRA chứ không phải QLoRA?

Lab Day 21 cũ mặc định QLoRA 4-bit. Với **dòng model 2026 này, nhà cung cấp khuyến nghị
KHÔNG dùng QLoRA** — sai số lượng tử hoá cao hơn bình thường — và đề xuất bf16 LoRA,
nhất là với biến thể MoE (deck §12). Ở 4B, bf16 LoRA vẫn vừa T4, nên bạn không phải đánh đổi.

4-bit vẫn có mặt trong lab, nhưng **như một phép đo** (run `qlora` ở NB4), không phải mặc định.

---

## 2. Chọn tier

| Bạn có | Tier | Ghi chú |
|---|---|---|
| Không GPU | `CPU` | NB1 + toàn bộ test. Huấn luyện thì dùng Colab. |
| Colab Free T4 16 GB | **`T4`** | mặc định của lab |
| Kaggle T4 ×2 | `T4` | dùng 1 GPU; lab không cần multi-GPU |
| Laptop RTX 3060/4060 12 GB | `LAPTOP` | Qwen3.5-2B cho chắc; 4B có thể OOM ở NB4 |
| RTX 3090/4090, L4, A100 | `BIGGPU` | Qwen3.5-9B |
| macOS (M-series) | `CPU` | MPS chạy được NB1; **không** huấn luyện `bitsandbytes` |

```
Có GPU NVIDIA dùng được không?
├─ Không ─→ COMPUTE_TIER=CPU  → làm NB1 + test tại chỗ, train trên Colab Free T4
├─ Có, ≥22 GB ─→ COMPUTE_TIER=BIGGPU
├─ Có, 12–22 GB ─→ COMPUTE_TIER=T4
└─ Có, 8–12 GB ─→ COMPUTE_TIER=LAPTOP
```

---

## 3. Ngân sách thời gian (tier T4, corpus mặc định 250 mẫu)

| Notebook | T4 | A100 |
|---|---:|---:|
| NB1 data + mask | 2 ph | 2 ph |
| NB2 baselines | ~10 ph | ~4 ph |
| NB3 train | ~25 ph | ~8 ph |
| NB4 ba đối chứng | ~35 ph | ~12 ph |
| NB5 eval + verdict | ~10 ph | ~4 ph |
| **Core (NB1–NB5)** | **~80 ph** | **~30 ph** |
| NB6 merge (tuỳ chọn) | ~10 ph | ~4 ph |

Hết giờ? Thứ tự ưu tiên: **NB1 → NB2 → NB3 → NB5**, rồi NB4 nếu còn thời gian. NB1 và NB5
là hai notebook mang nhiều điểm nhất.

---

## 4. Lỗi phần cứng thường gặp

| Lỗi | Nguyên nhân | Sửa |
|---|---|---|
| OOM ngay ở NB3 | tier quá lớn so với GPU | hạ một bậc tier |
| OOM ở run thứ 2 của NB4 | model cũ chưa được giải phóng | `generate.free_memory()` (đã có sẵn giữa các run) |
| `bitsandbytes` không cài được | macOS / không CUDA | bỏ qua run `qlora`; hai đối chứng còn lại vẫn chạy |
| Colab ngắt kết nối giữa NB4 | runtime free bị giới hạn | chạy từng `make nb4` một; adapter đã lưu sau mỗi run |
| Rất chậm dù có GPU | torch bản CPU | `python -c "import torch; print(torch.cuda.is_available())"` |
