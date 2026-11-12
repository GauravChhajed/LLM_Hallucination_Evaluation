"""
Page 2: Model Comparison — Multi-metric side-by-side comparison with explicit sample counts.
"""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from src.dashboard.dashboard_utils import (
    build_category_metric_chart,
    build_correctness_distribution_chart,
    build_model_comparison_bar_chart,
    build_model_radar_chart,
    format_metric_with_n,
)
from src.dashboard.navigation import DashboardContext


def render_model_comparison_page(ctx: DashboardContext) -> None:
    """Render Page 2: Model Comparison."""
    st.header("Model Comparison")

    if ctx.active_run is None or not ctx.filtered_records:
        st.info("Select an evaluation run with matching records to compare models.")
        return

    summary_models = ctx.filtered_summary.get("models") or {}
    parity_notes = ctx.filtered_summary.get("sample_population_parity_notes") or []

    for note in parity_notes:
        st.warning(note)

    # Comparison table showing explicit sample count `n` for every metric
    table_rows = []
    for model_id, m_data in summary_models.items():
        gen = m_data.get("generation") or {}
        ref = m_data.get("reference_evaluation") or {}
        faith = m_data.get("faithfulness_and_hallucination") or {}
        judge = m_data.get("llm_judge") or {}

        table_rows.append(
            {
                "Model ID": model_id,
                "Gen Success / Total": f"{gen.get('successful_count', 0)} / {m_data.get('total_questions', 0)}",
                "Semantic Similarity": format_metric_with_n(ref.get("semantic_similarity")),
                "Token F1": format_metric_with_n(ref.get("token_f1")),
                "Ref. Correctness": format_metric_with_n(ref.get("correctness_score")),
                "Faithfulness Ratio (Context)": format_metric_with_n(faith.get("faithfulness_ratio")),
                "Contradiction Rate": format_metric_with_n(faith.get("contradiction_rate")),
                "Unsupported Claim Rate": format_metric_with_n(faith.get("unsupported_claim_rate")),
                "Context Hallucination Rate": format_metric_with_n(faith.get("context_hallucination_rate")),
                "Avg Response Time": format_metric_with_n(gen.get("response_time_seconds"), decimals=2, suffix="s"),
                "Judge Overall (1-5)": format_metric_with_n(judge.get("overall_score"), decimals=2),
            }
        )

    st.subheader("Side-by-Side Model Summary (with Valid Sample Counts `n`)")
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
    st.caption(
        "Scoring Note: `Ref. Correctness` reports the mean continuous score in [0, 1] (`correct` >= 0.72 or normalized exact match, "
        "`partially_correct` in [0.42, 0.72), `incorrect` < 0.42, `contradictory_numeric` capped <= 0.30). "
        "`Avg Time` includes executed non-error attempts (`success` + `empty_response`) with recorded latency and excludes failed (`error`) runs."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_rates = build_model_comparison_bar_chart(
            summary_models=summary_models,
            metric_specs=[
                ("reference_evaluation", "semantic_similarity", "Semantic Sim."),
                ("reference_evaluation", "token_f1", "Token F1"),
                ("reference_evaluation", "correctness_score", "Ref. Correctness"),
                ("faithfulness_and_hallucination", "faithfulness_ratio", "Faithfulness"),
                ("faithfulness_and_hallucination", "contradiction_rate", "Contradiction Rate"),
                ("faithfulness_and_hallucination", "unsupported_claim_rate", "Unsupported Rate"),
                ("faithfulness_and_hallucination", "context_hallucination_rate", "Context Halluc. Rate"),
            ],
            title="Reference, Faithfulness & Hallucination Rates by Model",
            yaxis_title="Score / Rate [0, 1]",
            y_range=[0, 1.08],
        )
        st.plotly_chart(fig_rates, width="stretch")

    with c2:
        fig_radar = build_model_radar_chart(summary_models)
        st.plotly_chart(fig_radar, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        fig_time = build_model_comparison_bar_chart(
            summary_models=summary_models,
            metric_specs=[("generation", "response_time_seconds", "Avg Generation Latency")],
            title="Response Generation Latency (Seconds)",
            yaxis_title="Seconds",
        )
        st.plotly_chart(fig_time, width="stretch")

    with c4:
        fig_corr = build_correctness_distribution_chart(summary_models)
        st.plotly_chart(fig_corr, width="stretch")

    st.subheader("Category-Wise Performance")
    metric_choice = st.selectbox(
        "Category Breakdown Metric",
        options=[
            ("reference_evaluation", "correctness_score", "Reference-Based Correctness"),
            ("reference_evaluation", "semantic_similarity", "Semantic Similarity"),
            ("reference_evaluation", "token_f1", "Token F1"),
            ("faithfulness_and_hallucination", "faithfulness_ratio", "Faithfulness Ratio (Context Only)"),
            ("faithfulness_and_hallucination", "context_hallucination_rate", "Context Hallucination Rate"),
            ("llm_judge", "overall_score", "LLM Judge Overall (1-5)"),
        ],
        format_func=lambda item: item[2],
    )
    fig_cat = build_category_metric_chart(
        summary_models=summary_models,
        section_key=metric_choice[0],
        metric_key=metric_choice[1],
        title=f"{metric_choice[2]} Across Categories",
        yaxis_title=metric_choice[2],
    )
    st.plotly_chart(fig_cat, width="stretch")

    st.download_button(
        label="Download Comparison Summary (JSON)",
        data=json.dumps(ctx.filtered_summary, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name=f"{ctx.active_run.descriptor.run_id}_comparison_summary.json",
        mime="application/json",
    )
