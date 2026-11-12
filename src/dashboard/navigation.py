"""
Sidebar navigation, run selector, and global filter controls for the Streamlit dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import streamlit as st

from src.dashboard.data_loader import LoadedRunData, RunDescriptor, discover_runs, filter_records, load_run_data
from src.pipeline.aggregation import aggregate_pipeline_results

PAGES: tuple[str, ...] = (
    "Overview",
    "Model Comparison",
    "Hallucination Analysis",
    "Question-Level Analysis",
    "LLM-as-a-Judge",
    "Experiment Management",
)


@dataclass
class DashboardContext:
    """Global sidebar state passed to the active page renderer."""

    selected_page: str
    runs: List[RunDescriptor]
    active_run: Optional[LoadedRunData]
    filtered_records: List[Dict[str, Any]]
    filtered_summary: Dict[str, Any]
    selected_models: List[str]
    selected_categories: List[str]
    selected_difficulties: List[str]


@st.cache_data(show_spinner=False, ttl=30)
def _cached_load_run(run_dir_str: str) -> LoadedRunData:
    """Safe read-only cache for loading run JSON/CSV artifacts."""
    return load_run_data(Path(run_dir_str))


def clear_run_caches() -> None:
    """Invalidate cached run reads when a new run finishes or user clicks Refresh."""
    _cached_load_run.clear()


def render_sidebar_and_context(runs_dir: Path) -> DashboardContext:
    """Render sidebar navigation, run selector, and global filters."""
    st.sidebar.title("LLM Hallucination Eval")
    st.sidebar.caption("Comparative Evaluation Framework")

    if "pending_nav_page" in st.session_state:
        pending = st.session_state.pop("pending_nav_page")
        if pending in PAGES:
            st.session_state["nav_page"] = pending
    elif "nav_page" not in st.session_state or st.session_state["nav_page"] not in PAGES:
        st.session_state["nav_page"] = PAGES[0]

    selected_page = st.sidebar.radio(
        "Navigation",
        options=list(PAGES),
        key="nav_page",
    )

    st.sidebar.divider()
    col_head, col_btn = st.sidebar.columns([3, 1])
    col_head.subheader("Evaluation Run")
    if col_btn.button("↻", help="Refresh discovered runs"):
        clear_run_caches()
        st.rerun()

    runs = discover_runs(runs_dir)
    if not runs:
        st.sidebar.info("No runs found yet. Launch or inspect experiments in Experiment Management.")
        return DashboardContext(
            selected_page=selected_page,
            runs=[],
            active_run=None,
            filtered_records=[],
            filtered_summary={"models": {}, "sample_population_parity_notes": []},
            selected_models=[],
            selected_categories=[],
            selected_difficulties=[],
        )

    run_labels = {
        r.run_id: f"{r.run_id} [{r.status}]"
        for r in runs
    }
    run_ids = [r.run_id for r in runs]
    default_idx = 0
    if "active_run_id" in st.session_state and st.session_state["active_run_id"] in run_ids:
        default_idx = run_ids.index(st.session_state["active_run_id"])

    selected_run_id = st.sidebar.selectbox(
        "Select Run",
        options=run_ids,
        index=default_idx,
        format_func=lambda rid: run_labels[rid],
    )
    st.session_state["active_run_id"] = selected_run_id

    chosen_desc = next(r for r in runs if r.run_id == selected_run_id)
    active_run = _cached_load_run(str(chosen_desc.run_dir))

    for warn in active_run.warnings:
        st.sidebar.warning(warn)

    records = active_run.combined_results
    available_models = sorted({str(r.get("model_id", "")) for r in records if r.get("model_id")})
    available_categories = sorted({str(r.get("category", "")) for r in records if r.get("category")})
    available_difficulties = sorted({str(r.get("difficulty", "")) for r in records if r.get("difficulty")})

    st.sidebar.divider()
    st.sidebar.subheader("Global Filters")
    selected_models = st.sidebar.multiselect(
        "Models",
        options=available_models,
        default=available_models,
    )
    selected_categories = st.sidebar.multiselect(
        "Categories",
        options=available_categories,
        default=available_categories,
    )
    selected_difficulties = st.sidebar.multiselect(
        "Difficulties",
        options=available_difficulties,
        default=available_difficulties,
    )

    filtered = filter_records(
        records,
        models=selected_models if available_models else None,
        categories=selected_categories if available_categories else None,
        difficulties=selected_difficulties if available_difficulties else None,
    )
    filtered_summary = aggregate_pipeline_results(filtered)

    return DashboardContext(
        selected_page=selected_page,
        runs=runs,
        active_run=active_run,
        filtered_records=filtered,
        filtered_summary=filtered_summary,
        selected_models=selected_models,
        selected_categories=selected_categories,
        selected_difficulties=selected_difficulties,
    )
