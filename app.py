"""
Streamlit Dashboard Entrypoint for LLM Hallucination Detection and Evaluation.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.dashboard.experiment_management import render_experiment_management_page
from src.dashboard.hallucination_analysis import render_hallucination_analysis_page
from src.dashboard.judge_analysis import render_judge_analysis_page
from src.dashboard.model_comparison import render_model_comparison_page
from src.dashboard.navigation import render_sidebar_and_context
from src.dashboard.overview import render_overview_page
from src.dashboard.question_analysis import render_question_analysis_page
from src.utils.config import get_config


def main() -> None:
    """Configure Streamlit layout and dispatch to the selected research page."""
    st.set_page_config(
        page_title="LLM Hallucination Detection & Evaluation",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cfg = get_config(ensure_dirs=True)
    runs_dir = cfg.paths.results_dir / "runs"
    ctx = render_sidebar_and_context(runs_dir=runs_dir)

    page_renderers = {
        "Overview": render_overview_page,
        "Model Comparison": render_model_comparison_page,
        "Hallucination Analysis": render_hallucination_analysis_page,
        "Question-Level Analysis": render_question_analysis_page,
        "LLM-as-a-Judge": render_judge_analysis_page,
        "Experiment Management": render_experiment_management_page,
    }

    renderer = page_renderers.get(ctx.selected_page, render_overview_page)
    renderer(ctx)


if __name__ == "__main__":
    main()
