# %% [markdown]
# # NB4 — Giải phẫu cấu hình sai (phần quan trọng nhất của lab)
#
# Lab Day 21 **phiên bản cũ** lấy "quét rank r=8/16/64" làm thí nghiệm trung tâm, gắn
# LoRA vào `q_proj, v_proj`, và chấm bằng perplexity. Deck hiện tại gọi đúng ba thứ đó
# là **Lỗi #1, #2, #3** (§10.2–§10.4).
#
# Notebook này không xoá thí nghiệm cũ — nó **chạy lại thí nghiệm cũ như một đối chứng**,
# để bạn tự tay thấy danh tiếng *"LoRA học kém hơn full fine-tune"* xuất hiện rồi biến mất.
#
# Ba run đối chứng, **cùng số step**, chỉ đổi một biến mỗi lần:
#
# | Run | Đổi gì | Kỳ vọng |
# |---|---|---|
# | `attn_only` | chỉ q,v — **rank nâng lên cho bằng số tham số** | thua `correct` |
# | `wrong_lr` | LR thang full-FT (÷10) | loss gần như phẳng |
# | `qlora` | 4-bit thay bf16 | nhẹ hơn, chất lượng ? |
#
# > **Vì sao phải "bằng số tham số".** So `q,v @ r=16` với `all-linear @ r=16` là so
# > *ngân sách*, không phải so *vị trí* — và không chứng minh được gì. `matched_rank()`
# > giải ra rank đưa attention-only về đúng ngân sách của `correct`.

# %%
import json, os, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

from labkit import data, generate, modeling, report, train
from labkit.config import CONTRAST_KEYS, CONTRAST_MAX_STEPS, SPECS, get_tier

ROOT = pathlib.Path.cwd() if (pathlib.Path.cwd() / "data").exists() else pathlib.Path.cwd().parent
TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))

from datasets import Dataset

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

train_rows = load_jsonl(ROOT / "data" / "split" / "train.jsonl")
train_ds = Dataset.from_list([{"messages": data.to_messages(r)} for r in train_rows])

# %% [markdown]
# ## 1. Bảng vị trí × rank × số tham số
#
# Đọc bảng này **trước** khi chạy. Nó cho thấy vì sao so cùng-rank là không công bằng.

# %%
model, tok = generate.load_base(TIER)
placement = modeling.describe_placement(model, SPECS["correct"].r)
print(report.markdown_table(placement))
del model
generate.free_memory()

# %% [markdown]
# ## 2. Ba run đối chứng — cùng ngân sách step

# %%
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def run_contrast(key: str) -> dict:
    spec = SPECS[key]
    model, tok = generate.load_base(TIER, load_in_4bit=spec.load_in_4bit)
    targets = modeling.resolve_target_modules(model, spec.target)

    if spec.r is None:                       # attn_only: solve for the matched rank
        base_targets = modeling.resolve_target_modules(model, "text-linear")
        r = modeling.matched_rank(model, base_targets, SPECS["correct"].r, targets)
        spec = spec.resolved(r)
        print(f"  matched rank for {key}: r={r} (alpha={spec.alpha})")

    trainable = modeling.count_lora_params(model, targets, spec.r)
    want = train.sft_config_kwargs(TIER, spec, str(ROOT / "adapters" / key),
                                   max_steps=CONTRAST_MAX_STEPS)
    sft_kwargs, _ = train.filter_kwargs(SFTConfig, want, label=f"SFTConfig[{key}]")
    lora_kwargs, _ = train.filter_kwargs(
        LoraConfig, train.lora_config_kwargs(spec, targets), label=f"LoraConfig[{key}]")

    trainer = SFTTrainer(model=model, args=SFTConfig(**sft_kwargs),
                         train_dataset=train_ds, processing_class=tok,
                         peft_config=LoraConfig(**lora_kwargs))
    t0 = time.perf_counter()
    res = trainer.train()
    elapsed = time.perf_counter() - t0

    out = ROOT / "adapters" / key
    trainer.model.save_pretrained(out)

    row = train.summarize_run(spec, TIER, targets, trainable, elapsed, generate.peak_vram_gb())
    row["final_loss"] = round(res.training_loss, 4)
    row["max_steps"] = CONTRAST_MAX_STEPS
    row["teaches"] = spec.teaches
    report.append_row(row, results_dir=ROOT / "results")

    del trainer, model
    generate.free_memory()
    return row


rows = []
for key in CONTRAST_KEYS:
    print("=" * 70)
    print(f"RUN {key}: {SPECS[key].label}")
    print(f"     {SPECS[key].teaches}")
    rows.append(run_contrast(key))

# %% [markdown]
# ## 3. Bảng đối chứng
#
# `correct` từ NB3 chạy nhiều step hơn — **đừng so loss trực tiếp với nó**. Ba run ở đây
# so được với nhau vì cùng `max_steps`. Muốn so đầy đủ với `correct`, hãy chạy lại
# `correct` với `max_steps=CONTRAST_MAX_STEPS` (một dòng, ~10 phút) và ghi rõ trong REPORT.

# %%
cols = ["run", "label", "r", "trainable_params", "learning_rate", "final_loss",
        "train_seconds", "peak_vram_gb"]
print(report.markdown_table(rows, cols))

# %% [markdown]
# ## 4. Câu hỏi phải trả lời trong REPORT.md
#
# 1. `attn_only` có **cùng số tham số huấn luyện** với `correct`. Nó thắng hay thua? Điều
#    đó nói gì về *rank* so với *vị trí gắn adapter*?
# 2. `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau bao nhiêu? Nếu chỉ nhìn
#    loss mà không biết LR, bạn sẽ kết luận gì — và kết luận đó có đúng không?
# 3. `qlora` tiết kiệm bao nhiêu VRAM, và **trả giá bằng gì**? Nhà cung cấp khuyến nghị
#    *không* dùng QLoRA cho dòng model này (deck §12) — số đo của bạn có ủng hộ điều đó không?
