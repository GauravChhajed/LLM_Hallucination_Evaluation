# LLM Response Generation Architecture & Design

## 1. Overview

The `src/generation/` package implements a modular, backend-agnostic response-generation layer that feeds identical prompts and contexts to two or more configured Large Language Models (LLMs), records deterministic generation metadata and latency, and persists outputs for downstream evaluation.

```text
Dataset Row (question_id, question, context)
        │
        ▼
[prompt_builder.py] ──► PromptPayload (system + user prompt / chat messages)
        │
        ▼
[generator.py] (create_generator factory via models/model_config.py)
   ├── HuggingFaceLocalGenerator ──► [model_loader.py] (AutoTokenizer + Causal/Seq2Seq LM)
   └── APIBasedGenerator         ──► Free-Tier HTTP APIs (Groq, HF Router, Gemini)
        │
        ▼
GenerationResult (answer, timing, status, token counts, metadata)
        │
        ▼
[response_generator.py] ──► Paired Model A/B runs + JSON/CSV export (results/model_responses/)
```

---

## 2. Module Responsibilities

- **`src/generation/prompt_builder.py`**:
  - Builds a standardized `PromptPayload` via `build_qa_prompt(question, context, system_prompt, template)` using `STANDARD_EVAL_TEMPLATE` (`standard_neutral_v1`).
  - Uses a single neutral instruction for all models (*"You are answering questions in an evaluation setting. Answer the question using the information available to you. If context is provided, consider it when formulating your answer. Provide a concise answer."*) without category-specific coaching on refusals, false premises, or uncertainty.
  - Formats context-bearing prompts (`Context:\n...\n\nQuestion:\n...\n\nAnswer:`) and closed-book prompts (`Question:\n...\n\nAnswer:`), while preserving `PromptTemplate` extensibility for future prompt-ablation experiments.
  - Uses `format_prompt_for_tokenizer(...)` to apply native `tokenizer.apply_chat_template(...)` for instruction-tuned causal models and plain-text prompts for Seq2Seq models (e.g., FLAN-T5).

- **`src/generation/model_loader.py`**:
  - Resolves runtime compute device (`cpu` or `cuda`) via `src/utils/config.py`.
  - Selects `torch.float32` automatically on CPU (preventing unsupported FP16 CPU kernel failures) while honoring `float16`/`bfloat16` when CUDA is available.
  - Loads Hugging Face tokenizers and `AutoModelForCausalLM` / `AutoModelForSeq2SeqLM` models into `D:\hf_cache`.
  - Validates API credentials via `.env` and raises `ModelLoadError` / `MissingAPIKeyError` explicitly without ever silently substituting a different model.

- **`src/generation/generator.py`**:
  - Defines `BaseLLMGenerator` and structured `GenerationResult`.
  - Implements `HuggingFaceLocalGenerator` (with reproducible seeding, `torch.no_grad()`, causal new-token slicing vs. Seq2Seq decoding, timing, and graceful empty/error handling).
  - Implements `APIBasedGenerator` for OpenAI-compatible endpoints (Groq, Hugging Face Inference Router, OpenRouter) and Google Gemini REST API.
  - Exposes `create_generator(model_key_or_id, ...)` factory bound to `models/model_config.py`.

- **`src/generation/response_generator.py`**:
  - Runs single-model or paired (`generate_paired_responses`) dataset inference ensuring Model A and Model B receive identical ordered inputs and `GenerationConfig`.
  - Exports structured runs as JSON and CSV to `results/model_responses/`.

---

## 3. Generation Parameters (`GenerationConfig`)

Default settings prioritize fair, reproducible comparison:

| Parameter | Default | Purpose |
| :--- | :--- | :--- |
| `max_new_tokens` | `256` | Caps output length consistently across models and keeps CPU latency bounded |
| `do_sample` | `False` | Enables greedy deterministic decoding for reproducible answers |
| `temperature` | `0.0` | Passed only when `do_sample=True` |
| `top_p` | `1.0` | Passed only when `do_sample=True` |
| `repetition_penalty` | `1.05` | Mild repetition suppression without distorting factual phrasing |
| `seed` | `42` | Seeds Python, NumPy, PyTorch, and Transformers prior to each inference |

---

## 4. Hardware Constraints (Measured System Profile)

- **PyTorch Runtime:** `torch==2.15.0+cpu` (`torch.cuda.is_available() == False`; CPU fallback active)
- **CPU:** 12 logical Intel x64 cores
- **RAM:** `15.65 GB` total (`~4.85 GB` free under typical desktop load; closing background apps increases headroom)
- **Cache Storage (`HF_HOME`):** `D:\hf_cache` (`~283 GB` free)

> Because CPU inference runs in FP32 (`~4 bytes/parameter` plus KV-cache/activation overhead), a 0.5B model uses roughly `~2.2–2.8 GB` RAM, whereas a 1.5B model requires `~6.2–7.0 GB` RAM.

---

## 5. Candidate Model Comparison (Pre-Download Technical Assessment)

| Candidate Model | Params / Arch | Est. FP32 CPU RAM | Disk Footprint | License | CPU Feasibility | QA & Hallucination Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Qwen/Qwen2.5-0.5B-Instruct`** | 0.5B Causal LM | ~2.2–2.8 GB | ~1.0 GB | Apache-2.0 | **High** (fast on 12-core CPU) | Strong small instruction-following; good lightweight baseline |
| **`google/flan-t5-base`** | 250M Seq2Seq | ~1.1–1.5 GB | ~1.0 GB | Apache-2.0 | **Very High** (very fast) | Concise extractive/context QA; weak on nuanced refusal/open-ended traps |
| **`google/flan-t5-large`** | 780M Seq2Seq | ~3.2–3.8 GB | ~3.1 GB | Apache-2.0 | **High** (moderate speed) | Strong context-grounded QA contrast vs. causal decoder models; 512-token window |
| **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`** | 1.1B Causal LM | ~4.5–5.2 GB | ~2.2 GB | Apache-2.0 | **Medium-High** | Useful contrast model—often exhibits higher hallucination/overconfidence than Qwen2.5 |
| **`HuggingFaceTB/SmolLM2-360M-Instruct`** | 360M Causal LM | ~1.6–2.0 GB | ~0.75 GB | Apache-2.0 | **Very High** | Very fast ultra-small chat model, though lower factual depth |
| **`Qwen/Qwen2.5-1.5B-Instruct`** | 1.5B Causal LM | ~6.2–7.0 GB | ~3.1 GB | Apache-2.0 | **Medium** (requires freeing RAM; slower on CPU) | Higher reasoning & instruction adherence; ideal once CUDA PyTorch is installed or on subset runs |
| **Free-Tier APIs** (e.g., Groq `llama-3.1-8b-instant`, Google `gemini-2.5-flash`) | 8B+ / Hosted | ~0 GB local RAM | 0 GB | Free-tier API | **Instant** (network-bound) | Optional high-capability baseline or judge if a free API key is added to `.env` |

### Recommended Local Comparison Pair for Current CPU Setup
- **Model A:** `Qwen/Qwen2.5-0.5B-Instruct` (`qwen2.5-0.5b`) — modern chat-templated causal instruction model (~2.2–2.8 GB RAM).
- **Model B:** `google/flan-t5-base` (`flan-t5-base`, ~1.1–1.5 GB RAM) or `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (`tinyllama-1.1b`, ~4.5 GB RAM) — provides a meaningful architectural and hallucination-rate contrast while staying safely inside CPU RAM limits.

---

## 6. How to Run Generation (After Model Approval)

```python
from models.model_config import GenerationConfig
from src.generation import create_generator, generate_paired_responses, save_generation_results
from src.utils.config import get_config
from src.utils.data_loader import load_dataset

cfg = get_config(ensure_dirs=True)
df, _ = load_dataset(cfg.paths.default_dataset_path)

gen_cfg = GenerationConfig(max_new_tokens=256, do_sample=False, seed=42)
gen_a = create_generator("qwen2.5-0.5b", generation_config=gen_cfg, app_config=cfg)
gen_b = create_generator("flan-t5-base", generation_config=gen_cfg, app_config=cfg)

# Use max_questions=10 for quick pilot runs, or None for all 100 questions
res_a, res_b = generate_paired_responses(df, gen_a, gen_b, max_questions=10)
save_generation_results(res_a, cfg.paths.model_responses_dir, "model_a_qwen25_05b")
save_generation_results(res_b, cfg.paths.model_responses_dir, "model_b_flan_t5_base")
```
