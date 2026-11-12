"""
Page 3: Hallucination Analysis — Claim-level verification, faithfulness rates,
and context-vs-answer auditing.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.dashboard_utils import (
    build_category_metric_chart,
    build_claim_distribution_chart,
    format_metric_with_n,
)
from src.dashboard.navigation import DashboardContext


def render_hallucination_analysis_page(ctx: DashboardContext) -> None:
    """Render Page 3: Hallucination Analysis."""
    st.header("Hallucination & Contextual Faithfulness Analysis")

    if ctx.active_run is None or not ctx.filtered_records:
        st.info("Select an evaluation run with records to inspect claim-level hallucination analysis.")
        return

    records = ctx.filtered_records
    ctx_records = [r for r in records if r.get("has_context") is True]
    no_ctx_records = [r for r in records if r.get("has_context") is False]
    summary_models = ctx.filtered_summary.get("models") or {}

    total_claims = sum(int(r.get("total_factual_claims") or 0) for r in ctx_records)
    supported_claims = sum(int(r.get("supported_claims") or 0) for r in ctx_records)
    contradicted_claims = sum(int(r.get("contradicted_claims") or 0) for r in ctx_records)
    unsupported_claims = sum(int(r.get("unsupported_claims") or 0) for r in ctx_records)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Context Records", len(ctx_records))
    k2.metric("Total Claims (Context)", total_claims)
    k3.metric("Supported Claims", supported_claims)
    k4.metric("Contradicted Claims", contradicted_claims)
    k5.metric("NOT_ENOUGH_INFO Claims", unsupported_claims)

    st.info(
        f"Excluded **{len(no_ctx_records)}** no-context question-model record(s) from context-faithfulness "
        "rates. **Important:** `NOT_ENOUGH_INFORMATION` (unsupported) claims indicate lack of evidence "
        "in the supplied passage—they are **not** automatically classified as factually false. "
        "`context_hallucination_rate` is a broad unsupported-by-context measure combining contradicted and unsupported claims."
    )

    if not ctx_records:
        st.warning("No context-bearing questions match the current filters.")
        return

    # Model-level faithfulness metrics table
    faith_rows = []
    for model_id, m_data in summary_models.items():
        faith = m_data.get("faithfulness_and_hallucination") or {}
        faith_rows.append(
            {
                "Model ID": model_id,
                "Computed Context Items (n)": faith.get("n_context_computed", 0),
                "Appropriate Refusals": faith.get("n_context_appropriate_refusal", 0),
                "No-Context Skipped": faith.get("n_no_context_skipped", 0),
                "Faithfulness Ratio": format_metric_with_n(faith.get("faithfulness_ratio")),
                "Contradiction Rate": format_metric_with_n(faith.get("contradiction_rate")),
                "Unsupported Claim Rate": format_metric_with_n(faith.get("unsupported_claim_rate")),
                "Context Hallucination Rate": format_metric_with_n(faith.get("context_hallucination_rate")),
            }
        )
    st.dataframe(pd.DataFrame(faith_rows), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_claim_distribution_chart(ctx_records), width="stretch")
    with c2:
        st.plotly_chart(
            build_category_metric_chart(
                summary_models=summary_models,
                section_key="faithfulness_and_hallucination",
                metric_key="context_hallucination_rate",
                title="Context Hallucination Rate by Category",
                yaxis_title="Context Hallucination Rate",
            ),
            width="stretch",
        )

    st.subheader("Context-Bearing Question Table")
    table_df = pd.DataFrame(
        [
            {
                "Question ID": r.get("question_id"),
                "Model ID": r.get("model_id"),
                "Test Type": r.get("hallucination_test_type"),
                "Counterfactual": r.get("is_counterfactual_context"),
                "Status": r.get("hallucination_detection_status"),
                "Total Claims": r.get("total_factual_claims"),
                "Supported": r.get("supported_claims"),
                "Contradicted": r.get("contradicted_claims"),
                "NOT_ENOUGH_INFO": r.get("unsupported_claims"),
                "Faithfulness": r.get("faithfulness_ratio"),
                "Context Halluc. Rate": r.get("context_hallucination_rate"),
                "Flags": "; ".join(r.get("hallucination_types") or []),
            }
            for r in ctx_records
        ]
    )
    st.dataframe(table_df, width="stretch", hide_index=True)

    st.subheader("Context vs. Generated Answer & Claim Inspector")
    options = [
        f"{r.get('question_id')} | {r.get('model_id')}" for r in ctx_records
    ]
    selected_label = st.selectbox("Select Context Record to Inspect", options=options)
    chosen = ctx_records[options.index(selected_label)]

    col_ctx, col_ans = st.columns(2)
    with col_ctx:
        st.markdown("**Supplied Context**")
        st.write(chosen.get("context") or "")
        if chosen.get("is_counterfactual_context"):
            st.caption("Counterfactual context test: faithfulness is evaluated strictly against this passage.")
    with col_ans:
        st.markdown("**Generated Answer**")
        st.write(chosen.get("generated_answer") or "*[Empty]*")
        st.caption(chosen.get("hallucination_explanation") or "")

    claims = chosen.get("claims") or []
    if not claims:
        st.info(f"No factual claims extracted (status: `{chosen.get('hallucination_detection_status')}`).")
    else:
        for c in claims:
            label = c.get("verification_label", "UNKNOWN")
            with st.expander(f"[{label}] {c.get('claim_id')}: {c.get('claim_text')}", expanded=(label != "SUPPORTED")):
                st.write(
                    f"**Claim Type:** `{c.get('claim_type')}` | "
                    f"**Classification Confidence:** `{c.get('confidence')}` | "
                    f"**Evidence Strength:** `{c.get('evidence_strength')}` | "
                    f"**Verifier:** `{c.get('verifier_backend')}`"
                )
                st.write(f"**Explanation:** {c.get('explanation')}")
                ev_list = c.get("retrieved_evidence") or []
                if ev_list:
                    st.markdown("**Retrieved Context Evidence:**")
                    for ev_sent, ev_score in zip(ev_list, c.get("evidence_scores") or []):
                        st.markdown(f"- *(sim={ev_score:.3f})* {ev_sent}")
                else:
                    st.caption("No context sentence exceeded the retrieval threshold.")
