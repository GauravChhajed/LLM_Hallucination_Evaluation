"""
Page 6: Experiment Management — Browse runs, configure parameters, validate custom
datasets, export results, and deliberately launch checkpointed pipeline runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from models.model_config import GenerationConfig, get_model_spec, list_available_models
from src.dashboard.dashboard_utils import is_env_key_configured
from src.dashboard.data_loader import records_to_export_csv_bytes, run_payload_to_export_json_bytes
from src.dashboard.navigation import DashboardContext, clear_run_caches
from src.judge.judge_config import JudgeConfig
from src.pipeline import EvaluationPipelineRunner, PipelineConfig
from src.utils.config import get_config
from src.utils.data_loader import load_dataset, validate_dataset


def render_experiment_management_page(ctx: DashboardContext) -> None:
    """Render Page 6: Experiment Management."""
    st.header("Experiment Management")
    cfg = get_config(ensure_dirs=True)

    # ------------------------------------------------------------------
    # Section A: Browse Previous Runs & Export
    # ------------------------------------------------------------------
    st.subheader("A. Browse Previous Evaluation Runs")
    if not ctx.runs:
        st.info("No previous runs exist under `results/runs/` yet.")
    else:
        runs_df = pd.DataFrame(
            [
                {
                    "Run ID": r.run_id,
                    "Status": r.status,
                    "Started (UTC)": r.started_at_utc,
                    "Models": ", ".join(r.models),
                    "Pairs": r.total_pairs,
                    "Judge": "Yes" if r.judge_enabled else "No",
                    "Dataset Hash": r.dataset_hash[:12] + "..." if r.dataset_hash else "",
                }
                for r in ctx.runs
            ]
        )
        st.dataframe(runs_df, width="stretch", hide_index=True)

        col_open, col_exp_csv, col_exp_json = st.columns(3)
        target_run_id = col_open.selectbox(
            "Select Run to Open or Export",
            options=[r.run_id for r in ctx.runs],
        )
        if col_open.button("Open Selected Run in Overview"):
            st.session_state["active_run_id"] = target_run_id
            st.session_state["pending_nav_page"] = "Overview"
            st.rerun()

        if ctx.active_run is not None and ctx.active_run.descriptor.run_id == target_run_id:
            col_exp_csv.download_button(
                "Export Active Run (CSV)",
                data=records_to_export_csv_bytes(ctx.active_run.combined_results),
                file_name=f"{target_run_id}_results.csv",
                mime="text/csv",
            )
            col_exp_json.download_button(
                "Export Active Run (JSON)",
                data=run_payload_to_export_json_bytes(
                    ctx.active_run,
                    ctx.active_run.combined_results,
                    ctx.active_run.summary,
                ),
                file_name=f"{target_run_id}_results.json",
                mime="application/json",
            )

    st.divider()

    # ------------------------------------------------------------------
    # Section B: Configure New Evaluation Run
    # ------------------------------------------------------------------
    st.subheader("B. Configure New or Resumed Evaluation Run")

    dataset_mode = st.radio(
        "Dataset Source",
        options=["Default Benchmark Dataset", "Upload Custom CSV Dataset"],
        horizontal=True,
    )

    active_dataset_path: Path = cfg.paths.default_dataset_path
    dataset_valid = True
    available_qids: List[str] = []

    if dataset_mode == "Upload Custom CSV Dataset":
        uploaded = st.file_uploader("Upload evaluation CSV", type=["csv"])
        if uploaded is not None:
            try:
                up_df = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
                val_res = validate_dataset(up_df)
                if not val_res.is_valid:
                    dataset_valid = False
                    st.error("Uploaded CSV failed validation:\n- " + "\n- ".join(val_res.errors))
                else:
                    safe_name = Path(uploaded.name).name
                    target_path = cfg.paths.processed_data_dir / f"uploaded_{safe_name}"
                    up_df.to_csv(target_path, index=False, encoding="utf-8")
                    active_dataset_path = target_path
                    available_qids = up_df["question_id"].tolist()
                    st.success(f"Validated uploaded dataset ({len(up_df)} questions) saved to `{target_path.name}`.")
            except Exception as exc:
                dataset_valid = False
                st.error(f"Could not read uploaded CSV: {exc}")
        else:
            dataset_valid = False
            st.info("Upload a valid CSV file to configure a custom-dataset run.")
    else:
        if active_dataset_path.exists():
            ds_df, _ = load_dataset(active_dataset_path, validate=False)
            available_qids = ds_df["question_id"].tolist()
            st.caption(f"Using `{active_dataset_path}` ({len(ds_df)} questions).")
        else:
            dataset_valid = False
            st.error(f"Default dataset not found at `{active_dataset_path}`.")

    all_models = list_available_models(include_api=True)
    model_options = {m.key: f"{m.display_name} [{m.backend.value}]" for m in all_models}

    col_m1, col_m2 = st.columns(2)
    selected_models = col_m1.multiselect(
        "Models to Evaluate",
        options=list(model_options.keys()),
        default=["qwen2.5-0.5b", "flan-t5-base"],
        format_func=lambda k: model_options[k],
    )

    subset_mode = col_m2.radio(
        "Question Subset",
        options=["First N Questions (Pilot)", "Specific Question IDs", "All Questions"],
        horizontal=False,
    )
    max_questions = None
    selected_qids: List[str] = []
    if subset_mode == "First N Questions (Pilot)":
        max_questions = int(col_m2.number_input("Max Questions", min_value=1, max_value=max(1, len(available_qids) or 100), value=5))
    elif subset_mode == "Specific Question IDs" and available_qids:
        selected_qids = col_m2.multiselect("Select Question IDs", options=available_qids, default=available_qids[:3])

    c_gen1, c_gen2, c_gen3, c_gen4 = st.columns(4)
    max_new_tokens = int(c_gen1.number_input("Max New Tokens", min_value=16, max_value=1024, value=256, step=32))
    do_sample = c_gen2.checkbox("Stochastic Sampling", value=False)
    temperature = float(c_gen3.number_input("Temperature", min_value=0.0, max_value=1.5, value=0.0, step=0.1, disabled=not do_sample))
    seed = int(c_gen4.number_input("Random Seed", min_value=0, max_value=999999, value=42))

    c_opt1, c_opt2, c_opt3, c_opt4 = st.columns(4)
    sequential_exec = c_opt1.checkbox("Sequential Model Execution", value=True, help="Unloads each local model before loading the next.")
    enable_ref_eval = c_opt2.checkbox("Reference Evaluation (Phase 5)", value=True)
    enable_hallucination = c_opt3.checkbox("Hallucination Detection (Phase 6)", value=True)
    enable_judge = c_opt4.checkbox("Enable LLM-as-a-Judge (Phase 7)", value=False)

    judge_model_key = "qwen2.5-1.5b"
    if enable_judge:
        judge_candidates = [m.key for m in all_models if "judge" in m.tags or m.is_local]
        judge_model_key = st.selectbox(
            "Select Judge Model",
            options=judge_candidates,
            format_func=lambda k: model_options.get(k, k),
        )

    c_res1, c_res2 = st.columns(2)
    resume_run = c_res1.checkbox("Resume Existing Run", value=False)
    custom_run_id = ""
    if resume_run and ctx.runs:
        custom_run_id = c_res2.selectbox("Run ID to Resume", options=[r.run_id for r in ctx.runs])
    else:
        custom_run_id = c_res2.text_input("Optional Run ID (leave blank for auto timestamp)", value="").strip()

    # ------------------------------------------------------------------
    # Section C: Pre-Execution Safety Preview & Deliberate Launch
    # ------------------------------------------------------------------
    st.subheader("C. Review Configuration & Launch")

    if not selected_models:
        st.warning("Select at least one model before running an evaluation.")
        return

    # Inspect credentials safely without exposing key values
    missing_keys: List[str] = []
    has_local_model = False
    for m_key in selected_models + ([judge_model_key] if enable_judge else []):
        spec = get_model_spec(m_key)
        if spec.is_local:
            has_local_model = True
        elif spec.api_key_env and not is_env_key_configured(spec.api_key_env):
            missing_keys.append(f"{spec.display_name} requires `{spec.api_key_env}` (currently not set in environment).")

    if has_local_model:
        st.warning(
            "**Resource Warning:** Local Hugging Face inference on CPU/16 GB RAM may download model weights "
            "to `D:\\hf_cache` (if not cached) and consume substantial RAM and CPU time. "
            "Use a small `Max Questions` pilot subset first."
        )
    if missing_keys:
        st.error("**Missing API Credentials:**\n- " + "\n- ".join(missing_keys))

    pipeline_cfg = PipelineConfig(
        models=selected_models,
        dataset_path=active_dataset_path,
        output_dir=cfg.paths.results_dir / "runs",
        summaries_dir=cfg.paths.results_dir / "summaries",
        run_id=custom_run_id or None,
        max_questions=max_questions,
        question_ids=selected_qids or None,
        generation_config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            do_sample=do_sample,
            seed=seed,
        ),
        sequential_model_execution=sequential_exec,
        enable_reference_evaluation=enable_ref_eval,
        enable_hallucination_detection=enable_hallucination,
        enable_judge=enable_judge,
        judge_config=JudgeConfig(judge_model_key_or_id=judge_model_key) if enable_judge else None,
        resume=resume_run,
    )

    with st.expander("Preview Selected PipelineConfig", expanded=False):
        st.json(pipeline_cfg.to_dict())

    can_run = dataset_valid and len(selected_models) > 0 and not missing_keys
    if st.button("Run Evaluation", type="primary", disabled=not can_run):
        progress_bar = st.progress(0.0, text="Starting evaluation pipeline...")
        status_text = st.empty()

        def on_progress(stage: str, step: int, total: int) -> None:
            frac = min(1.0, max(0.0, step / max(1, total)))
            progress_bar.progress(frac, text=f"Stage `{stage}`: {step}/{total}")
            status_text.caption(f"Executing `{stage}` ({step}/{total})...")

        try:
            runner = EvaluationPipelineRunner(config=pipeline_cfg, app_config=cfg)
            run_result = runner.run(progress_callback=on_progress)
            progress_bar.progress(1.0, text="Evaluation completed!")
            clear_run_caches()
            st.session_state["active_run_id"] = run_result.run_id
            st.success(f"Run `{run_result.run_id}` finished and saved to `{run_result.run_dir}`.")
        except Exception as exc:
            st.error(
                f"Evaluation stopped with an error: {exc}. "
                "Any incremental checkpoints saved prior to the error remain preserved under `results/runs/`."
            )
