# Phase 7: LLM-as-a-Judge Evaluation Framework

## 1. Overview & Architecture

The `src/judge/` package adds an independent, rubric-driven **LLM-as-a-Judge** evaluation layer alongside Phase 5 (lexical/semantic correctness) and Phase 6 (claim-level faithfulness & hallucination detection), without overwriting earlier metrics.

```text
Phase 4 GenerationResult + Dataset Ground Truth (+ optional Phase 5 & 6 metadata)
        │
        ▼
[judge_prompt.py]  ──► Blind, versioned rubric prompt (model identity hidden)
        │
        ▼
[judge_client.py]  ──► Hosted API (Groq / Gemini / HF Router) OR Local HF Judge
        │                  (explicit error statuses; never silently swaps models)
        ▼
[judge_parser.py]  ──► Strict JSON extraction, 1-5 range & NOT_APPLICABLE validation
        │
        ▼
[judge_metrics.py] ──► Documented overall weighting + per-model & consistency stats
        │
        ▼
[judge_evaluator.py] ► JudgeEvaluationRecord + raw audit response + JSON/CSV export
```

---

## 2. Evaluation Rubric (1–5 Scale)

All models are scored independently under identical instructions (`prompt_version = "judge_rubric_v1.0"`):

| Dimension | Definition | 1 (Lowest) | 3 (Mid) | 5 (Highest) | Default Weight |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Relevance** | Directly addresses the question | Completely off-topic | Partially addresses with drift | Directly and wholly addresses | `0.20` |
| **Factual Correctness** | Consistent with reference answer & facts | Fabricates or accepts false premise | Mixed accuracy / omissions | Fully consistent (incl. valid paraphrases & honest refusals) | `0.25` |
| **Completeness** | Covers essential information needed | Missing nearly all key points | Covers some points, omits key caveats | Complete key-point coverage | `0.20` |
| **Contextual Faithfulness** | Supported strictly by supplied context | Directly contradicts context | Mixes supported & unsupported claims | Every claim strictly supported (`NOT_APPLICABLE` if no context) | `0.25` |
| **Clarity** | Understandable, coherent, well-structured | Incoherent or garbled | Understandable but awkward | Clear, concise, well-structured | `0.10` |

### Score & Confidence Interpretation
- **No-Context Rule:** When no context is supplied, `contextual_faithfulness_score` MUST be `"NOT_APPLICABLE"`, and `overall_score` re-normalizes the remaining four weights (`0.20 / 0.25 / 0.20 / 0.10` summing to `0.75`) so closed-book questions remain strictly on `[1.0, 5.0]`.
- **`judge_confidence` (`0.0–1.0`):** Self-reported confidence emitted by the judge model regarding its own assessment. It is **not** a calibrated probability, **not** an objective measure of factual correctness, and is strictly separated from the 1–5 answer quality scores.

---

## 3. Prompt Design, Bias Mitigation & Reliability

- **Blind Single-Answer Judging:** `build_judge_prompt(...)` never accepts or includes generator `model_id`, `model_key`, model rank, or peer scores.
- **Appropriate Refusal & Counterfactual Rules:** Prompts explicitly instruct the judge not to penalize valid paraphrases or honest refusals on unanswerable/false-premise questions, and to judge counterfactual-context questions strictly against the supplied context.
- **Why Blind Prompting Is Not Bias-Free:** Hiding model names removes overt brand bias, **but does not eliminate LLM judge biases**, including:
  - *Verbosity bias* (preferring longer answers over concise ones),
  - *Self-preference / family bias* (favoring outputs phrased similarly to the judge model's own style),
  - *Stochasticity / prompt sensitivity* (score drift across runs or prompt revisions).
- **Reliability Support:** `judge_metrics.py` includes `compute_repeated_judging_consistency(...)` to measure standard deviation and score range across repeated trials, and every record logs `prompt_version` and `raw_judge_response` for full auditing.

---

## 4. Judge Model Comparison (Hardware & Free-Tier Trade-offs)

| Candidate Judge Model | Execution Mode | Local RAM / VRAM | Latency | Cost / Limits | JSON Structured Reliability | Recommendation |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **`llama-3.3-70b-versatile` (Groq)** | Hosted Free API (`GROQ_API_KEY`) | 0 GB local | Very Fast (~1–2s) | Free tier RPM/TPD limits | **High** (strong instruction/JSON adherence) | **Recommended API Judge** when key configured |
| **`gemini-2.5-flash` (Google)** | Hosted Free API (`GEMINI_API_KEY`) | 0 GB local | Fast (~1–2s) | Free tier RPM/RPD limits | **High** (native `application/json` mode) | **Recommended API Judge** with native JSON output |
| **`meta-llama/Llama-3.1-8B-Instruct` (HF Router)** | Hosted Free API (`HF_TOKEN`) | 0 GB local | Moderate | Free-tier HF rate limits | **Medium-High** | Good open-weights hosted baseline |
| **`Qwen/Qwen2.5-1.5B-Instruct`** | Local HF (`CPU` FP32) | ~6.2–7.0 GB RAM | Slow on CPU | Free / unlimited offline | **Medium** (occasional JSON formatting errors) | **Best offline CPU fallback** when RAM is freed |
| **`Qwen/Qwen2.5-0.5B-Instruct`** | Local HF (`CPU` FP32) | ~2.2–2.8 GB RAM | Moderate on CPU | Free / unlimited offline | **Low-Medium** (weak complex rubric reasoning) | Suitable only for pipeline smoke tests, not strong grading |

*(Note: Paid APIs are never required, and no judge weights were downloaded during Phase 7.)*

---

## 5. Examples of Valid vs. Invalid Judge Outputs

### Valid JSON Output (Illustrative Fixture Consistent with Phase 6 `Q066` Hallucination Example)
*Context:* *"The Meridian Earth-observation constellation was launched in three phases. Phase 2 added 18 synthetic-aperture radar satellites in 2023."*
*Reference Answer:* *"According to the passage, 18 synthetic-aperture radar satellites were deployed in Phase 2."*
*Candidate Answer (Hallucinated):* *"Phase 2 added 24 synthetic-aperture radar satellites in 2023. The program cost 450 million dollars."*

*(Note: Illustrative schema example aligned with Phase 6's `Q066` claim-level example; not generated by a live judge API call.)*

```json
{
  "relevance_score": 4,
  "factual_correctness_score": 1,
  "completeness_score": 2,
  "contextual_faithfulness_score": 1,
  "clarity_score": 5,
  "dimension_explanations": {
    "relevance": "Directly addresses the Phase 2 satellite count, with one extraneous cost claim.",
    "factual_correctness": "States 24 satellites instead of the reference value of 18 and invents an unverified $450 million cost.",
    "completeness": "Mentions Phase 2 radar satellites in 2023 but gets the core quantity wrong.",
    "contextual_faithfulness": "Contradicts the context (24 vs. 18 satellites) and introduces an unsupported $450 million program cost.",
    "clarity": "Grammatically clear and concise despite containing factual hallucinations."
  },
  "judge_confidence": 0.91,
  "overall_summary": "Fluent response that contradicts the context satellite count and fabricates an unsupported cost figure."
}
```

### Rejected Invalid Outputs (Caught by `judge_parser.py`)
1. **Malformed / Non-JSON Output:** `"I give this answer 5 out of 5."` → `evaluation_status = "invalid_judge_output"`, raw text preserved in `raw_judge_response`.
2. **Out-of-Range Score:** `"relevance_score": 6` or `"judge_confidence": 1.5` → rejected with validation error.
3. **Invalid `NOT_APPLICABLE` on Context Question:** `"contextual_faithfulness_score": "NOT_APPLICABLE"` when `has_context = True` → rejected with validation error.

---

## 6. Limitations

1. **Approximate Evaluator:** Even strong 70B/cloud judges can misgrade subtle math, edge-case logic, or specialized facts; LLM-as-a-Judge complements rather than replaces Phase 5/6 deterministic checks.
2. **Small Local Judges:** On a CPU-only 16 GB machine, small (<2B) local judges struggle with multi-criterion JSON rubrics compared to free-tier 8B–70B hosted APIs.
3. **Single-Answer Rubric Only:** Pairwise preference comparison is intentionally deferred as a future experimental extension.
