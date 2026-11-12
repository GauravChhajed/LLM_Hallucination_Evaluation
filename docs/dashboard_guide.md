# Phase 9: Streamlit Research Dashboard Guide

## 1. Launching the Dashboard

Activate the virtual environment inside `LLM_Hallucination_Evaluation` and launch Streamlit:

```powershell
cd "c:\Users\hp\OneDrive\Desktop\llm project\LLM_Hallucination_Evaluation"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

*(Alternatively, `streamlit run dashboard/app.py` delegates cleanly to `app.main()`.)*

---

## 2. Dashboard Architecture

```text
app.py (Main Streamlit Entrypoint)
  └── src/dashboard/
        ├── navigation.py            ──► Sidebar page selector, run picker, global filters, cache refresh
        ├── data_loader.py           ──► Dynamic run discovery, checkpoint reconstruction, safe JSON/CSV loading
        ├── dashboard_utils.py       ──► Formatting with `n`, env-key masking, Plotly chart builders
        ├── overview.py              ──► Page 1: Run KPIs, core summary charts, filtered CSV/JSON export
        ├── model_comparison.py      ──► Page 2: Side-by-side table, radar chart, category breakdown, parity warnings
        ├── hallucination_analysis.py► Page 3: Context-only claim breakdown, faithfulness rates & claim inspector
        ├── question_analysis.py     ──► Page 4: Interactive multi-filter question & model explorer
        ├── judge_analysis.py        ──► Page 5: Rubric dimensions, raw judge response audit, confidence caveats
        └── experiment_management.py ► Page 6: Browse runs, upload custom CSV, configure & launch pipeline
```

---

## 3. Page-by-Page Guide

1. **Overview:** Summarizes total saved runs, evaluated questions, successful/failed/empty generations, and core metrics with exact `n` sample counts. Notes explicitly that semantic similarity does not prove factual accuracy.
2. **Model Comparison:** Compares models across semantic similarity, Token F1, reference-based correctness distribution, faithfulness ratio, contradiction rate, unsupported claim rate, context hallucination rate, latency, and optional judge scores. Missing metrics remain `N/A` (never replaced with zero), and cross-model sample-size mismatches trigger explicit parity warnings.
3. **Hallucination Analysis:** Separates `SUPPORTED`, `CONTRADICTED`, and `NOT_ENOUGH_INFORMATION` claims strictly across context-bearing questions. Context-free items are excluded from faithfulness statistics, and unsupported claims are explicitly distinguished from confirmed falsehoods.
4. **Question-Level Analysis:** Filter by `Question ID`, `Category`, `Difficulty`, `Model`, `Hallucination test type`, `Generation status`, and `Correctness label`. Displays context, reference answer, model outputs, individual metrics, claim-level evidence/explanations, and judge scores in clean expandable sections.
5. **LLM-as-a-Judge:** Displays the 5 rubric scores (`Relevance`, `Factual Correctness`, `Completeness`, `Contextual Faithfulness`, `Clarity`), overall weighted score, `judge_confidence` (explicitly marked as self-reported and uncalibrated), and raw judge JSON audit logs. Handles judge-disabled runs gracefully.
6. **Experiment Management:**
   - Browse and open saved runs or export CSV/JSON bundles.
   - Configure new or resumed runs using the default benchmark or a validated custom CSV upload.
   - Displays local CPU/RAM warnings and masked API key availability (`Configured` vs. `Not set`) without exposing secrets. Requires a deliberate click on **Run Evaluation** (never auto-launches or auto-downloads on page load).
