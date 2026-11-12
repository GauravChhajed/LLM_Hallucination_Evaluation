"""
Page 5: LLM-as-a-Judge — Rubric dimension analysis, raw audit inspection, and confidence notes.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.dashboard_utils import build_judge_dimensions_chart, format_metric, format_metric_with_n
from src.dashboard.navigation import DashboardContext


def render_judge_analysis_page(ctx: DashboardContext) -> None:
    """Render Page 5: LLM-as-a-Judge."""
    st.header("LLM-as-a-Judge Evaluation")

    if ctx.active_run is None or not ctx.filtered_records:
        st.info("Select an evaluation run to inspect LLM-as-a-Judge scores.")
        return

    records = ctx.filtered_records
    judge_records = [
        r for r in records
        if r.get("judge_evaluation_status") not in {None, "disabled", "pending"}
    ]
    evaluated_records = [r for r in judge_records if r.get("judge_evaluation_status") == "evaluated"]

    if not judge_records:
        st.info(
            "LLM-as-a-Judge was **disabled** or has no recorded outputs for this run. "
            "Deterministic Phase 5 and Phase 6 metrics remain available across the other pages."
        )
        return

    st.caption(
        "**Research Note:** LLM-as-a-Judge provides an approximate rubric evaluation alongside deterministic metrics. "
        "`judge_confidence` is self-reported by the judge model and must **not** be interpreted as a calibrated probability."
    )

    summary_models = ctx.filtered_summary.get("models") or {}
    judge_rows = []
    for model_id, m_data in summary_models.items():
        j = m_data.get("llm_judge") or {}
        judge_rows.append(
            {
                "Model ID": model_id,
                "Evaluated (n)": j.get("n_evaluated", 0),
                "Failed / Skipped": j.get("n_failed_or_skipped", 0),
                "Relevance (1-5)": format_metric_with_n(j.get("relevance_score"), 2),
                "Factual Correctness (1-5)": format_metric_with_n(j.get("factual_correctness_score"), 2),
                "Completeness (1-5)": format_metric_with_n(j.get("completeness_score"), 2),
                "Contextual Faithfulness (1-5)": format_metric_with_n(j.get("contextual_faithfulness_score"), 2),
                "Clarity (1-5)": format_metric_with_n(j.get("clarity_score"), 2),
                "Overall Weighted (1-5)": format_metric_with_n(j.get("overall_score"), 2),
                "Self-Reported Confidence": format_metric_with_n(j.get("judge_confidence"), 2),
            }
        )

    st.dataframe(pd.DataFrame(judge_rows), width="stretch", hide_index=True)

    if evaluated_records:
        st.plotly_chart(build_judge_dimensions_chart(summary_models), width="stretch")

    st.subheader("Question-Level Judge Audit")
    audit_df = pd.DataFrame(
        [
            {
                "Question ID": r.get("question_id"),
                "Model ID": r.get("model_id"),
                "Judge Model": r.get("judge_model_id"),
                "Status": r.get("judge_evaluation_status"),
                "Relevance": r.get("judge_relevance_score"),
                "Factual": r.get("judge_factual_correctness_score"),
                "Completeness": r.get("judge_completeness_score"),
                "Faithfulness": r.get("judge_contextual_faithfulness_score"),
                "Clarity": r.get("judge_clarity_score"),
                "Overall": r.get("judge_overall_score"),
                "Judge Confidence": r.get("judge_confidence"),
                "Error": r.get("judge_error_message"),
            }
            for r in judge_records
        ]
    )
    st.dataframe(audit_df, width="stretch", hide_index=True)

    options = [f"{r.get('question_id')} | {r.get('model_id')} [{r.get('judge_evaluation_status')}]" for r in judge_records]
    chosen_idx = options.index(st.selectbox("Inspect Judge Record & Raw Output", options=options))
    chosen = judge_records[chosen_idx]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Dimension Explanations:**")
        st.json(chosen.get("judge_dimension_explanations") or {})
    with c2:
        st.markdown("**Judge Status & Confidence:**")
        st.write(
            f"- **Status:** `{chosen.get('judge_evaluation_status')}`\n"
            f"- **Overall Weighted Score:** `{format_metric(chosen.get('judge_overall_score'), 2)}`\n"
            f"- **Self-Reported Judge Confidence:** `{format_metric(chosen.get('judge_confidence'), 2)}` *(uncalibrated)*\n"
            f"- **Prompt Version:** `{chosen.get('judge_prompt_version')}`"
        )
        if chosen.get("judge_error_message"):
            st.error(chosen.get("judge_error_message"))

    with st.expander("Raw Judge Response (Audit Log)", expanded=False):
        st.code(chosen.get("raw_judge_response") or "[No raw response recorded]", language="json")
