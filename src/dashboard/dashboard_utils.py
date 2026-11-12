"""
Formatting, security-safe environment checks, and Plotly chart builders for the dashboard.

All chart builders preserve missing values (never converting None to 0.0) and
surface exact sample counts (`n`) alongside every aggregated metric.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go


def format_metric(value: Optional[float | int], decimals: int = 3, suffix: str = "") -> str:
    """Format a numeric metric cleanly, returning 'N/A' when missing."""
    if value is None or isinstance(value, bool):
        return "N/A"
    try:
        num = float(value)
        if pd.isna(num):
            return "N/A"
        return f"{num:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def format_metric_with_n(stat_block: Optional[Mapping[str, Any]], decimals: int = 3, suffix: str = "") -> str:
    """Format an aggregate stat block `{'mean': ..., 'n': ...}` with explicit `n`."""
    if not stat_block or stat_block.get("mean") is None:
        n = stat_block.get("n", 0) if stat_block else 0
        return f"N/A (n={n})"
    return f"{format_metric(stat_block.get('mean'), decimals=decimals, suffix=suffix)} (n={stat_block.get('n', 0)})"


def is_env_key_configured(env_var_name: Optional[str]) -> bool:
    """Check whether an optional API key is set in the environment without exposing its value."""
    if not env_var_name:
        return True
    return bool(os.environ.get(env_var_name, "").strip())


def build_model_comparison_bar_chart(
    summary_models: Mapping[str, Any],
    metric_specs: Sequence[tuple[str, str, str]],  # (section_key, metric_key, display_label)
    title: str,
    yaxis_title: str = "Score",
    y_range: Optional[List[float]] = None,
) -> go.Figure:
    """
    Build a grouped bar chart across models omitting missing (`None`) metrics rather
    than plotting zero, and showing `n` sample counts in hover text.
    """
    fig = go.Figure()
    model_ids = list(summary_models.keys())

    for model_id in model_ids:
        model_data = summary_models[model_id]
        x_labels: List[str] = []
        y_vals: List[Optional[float]] = []
        hover_texts: List[str] = []

        for section_key, metric_key, display_label in metric_specs:
            stat = (model_data.get(section_key) or {}).get(metric_key) or {}
            mean_val = stat.get("mean")
            n_val = stat.get("n", 0)
            x_labels.append(display_label)
            y_vals.append(mean_val if isinstance(mean_val, (int, float)) else None)
            hover_texts.append(
                f"Model: {model_id}<br>{display_label}: {format_metric(mean_val)}<br>n = {n_val}"
            )

        fig.add_trace(
            go.Bar(
                name=model_id,
                x=x_labels,
                y=y_vals,
                text=[format_metric(v, decimals=2) if v is not None else "N/A" for v in y_vals],
                textposition="auto",
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=title,
        barmode="group",
        yaxis_title=yaxis_title,
        yaxis_range=y_range,
        legend_title="Model",
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig


def build_correctness_distribution_chart(summary_models: Mapping[str, Any]) -> go.Figure:
    """Build a stacked/grouped bar chart of reference-based correctness label counts."""
    label_order = [
        "correct",
        "partially_correct",
        "incorrect",
        "contradictory_numeric",
        "unassessed",
    ]
    fig = go.Figure()
    model_ids = list(summary_models.keys())

    for label in label_order:
        y_counts: List[int] = []
        for model_id in model_ids:
            dist = (summary_models[model_id].get("reference_evaluation") or {}).get(
                "correctness_label_distribution"
            ) or {}
            counts = dist.get("counts") or {}
            y_counts.append(int(counts.get(label, 0)))

        if sum(y_counts) > 0:
            fig.add_trace(go.Bar(name=label, x=model_ids, y=y_counts, text=y_counts, textposition="auto"))

    fig.update_layout(
        title="Reference-Based Correctness Label Distribution",
        barmode="stack",
        xaxis_title="Model",
        yaxis_title="Question Count",
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig


def build_model_radar_chart(summary_models: Mapping[str, Any]) -> go.Figure:
    """
    Build a normalized [0, 1] multi-dimensional radar chart across models.

    Only includes axes where at least one model has a valid non-None observation;
    Judge Overall (1-5) is explicitly normalized to [0, 1] by dividing by 5.0.
    """
    axes_defs = [
        ("Semantic Similarity", lambda m: (m.get("reference_evaluation") or {}).get("semantic_similarity"), 1.0),
        ("Token F1", lambda m: (m.get("reference_evaluation") or {}).get("token_f1"), 1.0),
        ("Ref. Correctness", lambda m: (m.get("reference_evaluation") or {}).get("correctness_score"), 1.0),
        ("Faithfulness Ratio", lambda m: (m.get("faithfulness_and_hallucination") or {}).get("faithfulness_ratio"), 1.0),
        ("Judge Overall (norm /5)", lambda m: (m.get("llm_judge") or {}).get("overall_score"), 5.0),
    ]

    active_axes = []
    for axis_name, getter, scale in axes_defs:
        has_any = any(
            isinstance((getter(m_data) or {}).get("mean"), (int, float))
            for m_data in summary_models.values()
        )
        if has_any:
            active_axes.append((axis_name, getter, scale))

    fig = go.Figure()
    if len(active_axes) < 3:
        fig.update_layout(title="Radar Comparison (requires >= 3 populated dimensions)")
        return fig

    for model_id, m_data in summary_models.items():
        r_vals: List[Optional[float]] = []
        theta_vals: List[str] = []
        hover_texts: List[str] = []
        for axis_name, getter, scale in active_axes:
            stat = getter(m_data) or {}
            raw_mean = stat.get("mean")
            n_val = stat.get("n", 0)
            norm_val = round(float(raw_mean) / scale, 4) if isinstance(raw_mean, (int, float)) else None
            theta_vals.append(axis_name)
            r_vals.append(norm_val)
            hover_texts.append(
                f"{model_id}<br>{axis_name}: {format_metric(norm_val)} (raw={format_metric(raw_mean)}, n={n_val})"
            )

        # Close polygon cleanly only if all axes on this model are populated
        if all(v is not None for v in r_vals):
            theta_vals.append(theta_vals[0])
            r_vals.append(r_vals[0])
            hover_texts.append(hover_texts[0])

        fig.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill="toself",
                name=model_id,
                hovertext=hover_texts,
                hoverinfo="text",
                connectgaps=False,
            )
        )

    fig.update_layout(
        title="Normalized Multi-Dimension Model Profile ([0, 1] scale)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(t=60, b=40, l=40, r=40),
    )
    return fig


def build_category_metric_chart(
    summary_models: Mapping[str, Any],
    section_key: str,
    metric_key: str,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    """Plot a metric broken down by question category per model with `n` in hover text."""
    fig = go.Figure()
    all_categories = sorted(
        {
            cat
            for m_data in summary_models.values()
            for cat in (m_data.get("by_category") or {}).keys()
        }
    )

    for model_id, m_data in summary_models.items():
        by_cat = m_data.get("by_category") or {}
        y_vals: List[Optional[float]] = []
        hover_texts: List[str] = []
        for cat in all_categories:
            stat = ((by_cat.get(cat) or {}).get(section_key) or {}).get(metric_key) or {}
            val = stat.get("mean")
            n_val = stat.get("n", 0)
            y_vals.append(val if isinstance(val, (int, float)) else None)
            hover_texts.append(f"{model_id}<br>{cat}: {format_metric(val)} (n={n_val})")

        fig.add_trace(
            go.Bar(
                name=model_id,
                x=all_categories,
                y=y_vals,
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Category",
        yaxis_title=yaxis_title,
        margin=dict(t=50, b=80, l=40, r=20),
    )
    return fig


def build_claim_distribution_chart(records: Sequence[Mapping[str, Any]]) -> go.Figure:
    """
    Visualize Supported, Contradicted, and NOT_ENOUGH_INFORMATION claim counts per model
    strictly across context-bearing questions.
    """
    ctx_records = [r for r in records if r.get("has_context") is True]
    by_model: Dict[str, Dict[str, int]] = {}
    for rec in ctx_records:
        mid = str(rec.get("model_id", ""))
        bucket = by_model.setdefault(
            mid, {"SUPPORTED": 0, "CONTRADICTED": 0, "NOT_ENOUGH_INFORMATION": 0}
        )
        bucket["SUPPORTED"] += int(rec.get("supported_claims") or 0)
        bucket["CONTRADICTED"] += int(rec.get("contradicted_claims") or 0)
        bucket["NOT_ENOUGH_INFORMATION"] += int(rec.get("unsupported_claims") or 0)

    fig = go.Figure()
    model_ids = list(by_model.keys())
    colors = {
        "SUPPORTED": "#2ca02c",
        "CONTRADICTED": "#d62728",
        "NOT_ENOUGH_INFORMATION": "#ff7f0e",
    }
    for label in ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFORMATION"):
        vals = [by_model[m][label] for m in model_ids]
        fig.add_trace(
            go.Bar(
                name=label,
                x=model_ids,
                y=vals,
                text=vals,
                textposition="auto",
                marker_color=colors[label],
            )
        )

    fig.update_layout(
        title="Context-Bearing Claim Verification Breakdown by Model",
        barmode="stack",
        xaxis_title="Model",
        yaxis_title="Claim Count",
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig


def build_judge_dimensions_chart(summary_models: Mapping[str, Any]) -> go.Figure:
    """Compare all 5 LLM-as-a-Judge rubric dimensions (1-5) across models."""
    dims = [
        ("llm_judge", "relevance_score", "Relevance"),
        ("llm_judge", "factual_correctness_score", "Factual Correctness"),
        ("llm_judge", "completeness_score", "Completeness"),
        ("llm_judge", "contextual_faithfulness_score", "Contextual Faithfulness"),
        ("llm_judge", "clarity_score", "Clarity"),
        ("llm_judge", "overall_score", "Overall Weighted"),
    ]
    return build_model_comparison_bar_chart(
        summary_models=summary_models,
        metric_specs=dims,
        title="LLM-as-a-Judge Scores by Rubric Dimension (1–5 Scale)",
        yaxis_title="Score (1–5)",
        y_range=[0, 5.4],
    )
