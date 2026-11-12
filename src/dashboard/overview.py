"""
Page 1: Overview — High-level experiment KPIs, methodology notes, and quick exports.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard.dashboard_utils import (
    build_correctness_distribution_chart,
    build_model_comparison_bar_chart,
    format_metric,
    format_metric_with_n,
)
from src.dashboard.data_loader import records_to_export_csv_bytes, run_payload_to_export_json_bytes
from src.dashboard.navigation import DashboardContext
from src.utils.config import get_config
from src.utils.data_loader import get_dataset_summary, load_dataset


def _weighted_stat_over_models(
    summary_models: dict,
    section: str,
    metric: str,
) -> tuple[float | None, int]:
    """Compute sample-weighted average and total valid `n` across selected models."""
    total_n = 0
    weighted_sum = 0.0
    for m_data in summary_models.values():
        stat = (m_data.get(section) or {}).get(metric) or {}
        mean_v = stat.get("mean")
        n_v = int(stat.get("n") or 0)
        if isinstance(mean_v, (int, float)) and n_v > 0:
            weighted_sum += float(mean_v) * n_v
            total_n += n_v
    if total_n == 0:
        return None, 0
    return round(weighted_sum / total_n, 4), total_n


def render_overview_page(ctx: DashboardContext) -> None:
    """Render Page 1: Overview."""
    st.header("Evaluation Overview")

    if ctx.active_run is None:
        st.info(
            "No evaluation runs were found under `results/runs/` yet. "
            "Use **Experiment Management** in the sidebar to configure and launch a run, "
            "or place completed run folders in `results/runs/`."
        )
        cfg = get_config()
        if cfg.paths.default_dataset_path.exists():
            try:
                df_ds, _ = load_dataset(cfg.paths.default_dataset_path, validate=False)
                ds_sum = get_dataset_summary(df_ds)
                st.subheader("Default Benchmark Dataset Ready")
                c1, c2, c3 = st.columns(3)
                c1.metric("Benchmark Questions", ds_sum["total_questions"])
                c2.metric("Context-Based Questions", ds_sum["questions_with_context"])
                c3.metric("Closed-Book / Adversarial", ds_sum["questions_without_context"])
                st.json(ds_sum["categories"])
            except Exception as exc:
                st.warning(f"Could not preview default dataset: {exc}")
        return

    records = ctx.filtered_records
    summary_models = ctx.filtered_summary.get("models") or {}

    unique_questions = len({r.get("question_id") for r in records})
    num_models = len(summary_models)
    total_success = sum(
        int((m.get("generation") or {}).get("successful_count") or 0)
        for m in summary_models.values()
    )
    total_failed = sum(
        int((m.get("generation") or {}).get("failed_count") or 0)
        for m in summary_models.values()
    )
    total_empty = sum(
        int((m.get("generation") or {}).get("empty_response_count") or 0)
        for m in summary_models.values()
    )

    sem_mean, sem_n = _weighted_stat_over_models(summary_models, "reference_evaluation", "semantic_similarity")
    corr_mean, corr_n = _weighted_stat_over_models(summary_models, "reference_evaluation", "correctness_score")
    faith_mean, faith_n = _weighted_stat_over_models(summary_models, "faithfulness_and_hallucination", "faithfulness_ratio")
    time_mean, time_n = _weighted_stat_over_models(summary_models, "generation", "response_time_seconds")
    judge_mean, judge_n = _weighted_stat_over_models(summary_models, "llm_judge", "overall_score")

    row1 = st.columns(5)
    row1[0].metric("Total Saved Runs", len(ctx.runs))
    row1[1].metric("Questions Evaluated", unique_questions)
    row1[2].metric("Models Selected", num_models)
    row1[3].metric("Successful Generations", total_success)
    row1[4].metric("Failed / Empty", f"{total_failed} / {total_empty}")

    row2 = st.columns(5)
    row2[0].metric("Avg Semantic Similarity", f"{format_metric(sem_mean)} (n={sem_n})")
    row2[1].metric("Avg Ref. Correctness", f"{format_metric(corr_mean)} (n={corr_n})")
    row2[2].metric("Avg Faithfulness (Context Only)", f"{format_metric(faith_mean)} (n={faith_n})")
    row2[3].metric("Avg Response Time", f"{format_metric(time_mean, decimals=2, suffix='s')} (n={time_n})")
    row2[4].metric(
        "Avg LLM Judge Overall",
        f"{format_metric(judge_mean, decimals=2)}/5 (n={judge_n})" if judge_n > 0 else "Disabled / N/A",
    )

    st.caption(
        "Research Note: Semantic similarity measures embedding-space closeness in phrasing and meaning, "
        "**not** factual correctness. Context faithfulness is computed strictly for questions that include a supplied context. "
        "Reference correctness averages continuous `correctness_score` values in [0, 1] (`correct` >= 0.72 or exact match; "
        "`partially_correct` in [0.42, 0.72); `incorrect` < 0.42; `contradictory_numeric` capped <= 0.30). "
        "Avg Response Time covers executed non-error attempts (`success` + `empty_response`) with valid recorded latency, "
        "excluding failed (`error`) or missing generations."
    )

    if not records:
        st.warning("No records match the current sidebar filters.")
        return

    col_left, col_right = st.columns(2)
    with col_left:
        fig_core = build_model_comparison_bar_chart(
            summary_models=summary_models,
            metric_specs=[
                ("reference_evaluation", "semantic_similarity", "Semantic Similarity"),
                ("reference_evaluation", "token_f1", "Token F1"),
                ("reference_evaluation", "correctness_score", "Ref. Correctness"),
                ("faithfulness_and_hallucination", "faithfulness_ratio", "Faithfulness Ratio"),
            ],
            title="Core Evaluation Metrics by Model",
            yaxis_title="Score [0, 1]",
            y_range=[0, 1.08],
        )
        st.plotly_chart(fig_core, width="stretch")

    with col_right:
        fig_corr = build_correctness_distribution_chart(summary_models)
        st.plotly_chart(fig_corr, width="stretch")

    with st.expander("Run Metadata & Export", expanded=False):
        meta = ctx.active_run.metadata
        st.json(
            {
                "run_id": ctx.active_run.descriptor.run_id,
                "status": ctx.active_run.descriptor.status,
                "started_at_utc": ctx.active_run.descriptor.started_at_utc,
                "finished_at_utc": ctx.active_run.descriptor.finished_at_utc,
                "dataset_hash": ctx.active_run.descriptor.dataset_hash,
                "execution_mode": meta.get("execution_mode"),
                "resolved_device": meta.get("resolved_device"),
                "enabled_stages": meta.get("enabled_stages"),
                "omitted_stage_files": meta.get("omitted_stage_files"),
            }
        )
        exp_c1, exp_c2 = st.columns(2)
        exp_c1.download_button(
            label="Download Filtered Results (CSV)",
            data=records_to_export_csv_bytes(records),
            file_name=f"{ctx.active_run.descriptor.run_id}_filtered.csv",
            mime="text/csv",
        )
        exp_c2.download_button(
            label="Download Run Report (JSON)",
            data=run_payload_to_export_json_bytes(ctx.active_run, records, ctx.filtered_summary),
            file_name=f"{ctx.active_run.descriptor.run_id}_report.json",
            mime="application/json",
        )
