"""
Page 4: Question-Level Analysis — Interactive multi-filter question & model explorer.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard.dashboard_utils import format_metric
from src.dashboard.data_loader import filter_records
from src.dashboard.navigation import DashboardContext


def render_question_analysis_page(ctx: DashboardContext) -> None:
    """Render Page 4: Question-Level Analysis."""
    st.header("Question-Level Analysis")

    if ctx.active_run is None or not ctx.filtered_records:
        st.info("Select an evaluation run with matching records to explore individual questions.")
        return

    base_records = ctx.filtered_records

    # Page-specific filters (in addition to global sidebar filters)
    all_h_types = sorted({str(r.get("hallucination_test_type") or "") for r in base_records if r.get("hallucination_test_type")})
    all_gen_statuses = sorted({str(r.get("generation_status") or "") for r in base_records if r.get("generation_status")})
    all_corr_labels = sorted({str(r.get("correctness_label") or "") for r in base_records if r.get("correctness_label")})

    f1, f2, f3, f4 = st.columns(4)
    sel_h_types = f1.multiselect("Hallucination Test Type", options=all_h_types, default=all_h_types)
    sel_gen_status = f2.multiselect("Generation Status", options=all_gen_statuses, default=all_gen_statuses)
    sel_corr_labels = f3.multiselect("Correctness Label", options=all_corr_labels, default=all_corr_labels)
    search_qid = f4.text_input("Search Question ID / Text", value="").strip().lower()

    scoped = filter_records(
        base_records,
        hallucination_test_types=sel_h_types if all_h_types else None,
        generation_statuses=sel_gen_status if all_gen_statuses else None,
        correctness_labels=sel_corr_labels if all_corr_labels else None,
    )
    if search_qid:
        scoped = [
            r for r in scoped
            if search_qid in str(r.get("question_id", "")).lower()
            or search_qid in str(r.get("question", "")).lower()
        ]

    if not scoped:
        st.warning("No records match the selected question filters.")
        return

    qids = []
    seen = set()
    for r in scoped:
        qid = str(r.get("question_id", ""))
        if qid not in seen:
            seen.add(qid)
            qids.append(qid)

    qid_labels = {}
    for qid in qids:
        first = next(r for r in scoped if r.get("question_id") == qid)
        q_preview = str(first.get("question", ""))[:80]
        qid_labels[qid] = f"{qid} — {q_preview}"

    selected_qid = st.selectbox(
        "Select Question ID",
        options=qids,
        format_func=lambda q: qid_labels[q],
    )

    q_records = [r for r in scoped if r.get("question_id") == selected_qid]
    first_rec = q_records[0]

    st.subheader(f"Question `{selected_qid}`")
    meta_cols = st.columns(4)
    meta_cols[0].markdown(f"**Category:** {first_rec.get('category') or 'N/A'}")
    meta_cols[1].markdown(f"**Difficulty:** {first_rec.get('difficulty') or 'N/A'}")
    meta_cols[2].markdown(f"**Test Type:** `{first_rec.get('hallucination_test_type') or 'N/A'}`")
    meta_cols[3].markdown(f"**Expected Behavior:** `{first_rec.get('expected_behavior') or 'N/A'}`")

    st.markdown("**Question:**")
    st.info(first_rec.get("question") or "")

    col_ctx, col_ref = st.columns(2)
    with col_ctx:
        st.markdown("**Supplied Context:**")
        ctx_text = str(first_rec.get("context") or "").strip()
        if ctx_text:
            st.write(ctx_text)
        else:
            st.caption("[No context supplied — closed-book / adversarial question]")
    with col_ref:
        st.markdown("**Reference Answer:**")
        st.success(first_rec.get("reference_answer") or "")

    st.divider()
    st.subheader("Model Responses & Evaluations")

    model_cols = st.columns(max(1, min(len(q_records), 3)))
    for idx, rec in enumerate(q_records):
        col = model_cols[idx % len(model_cols)]
        with col:
            st.markdown(f"### `{rec.get('model_id')}`")
            st.caption(
                f"Status: `{rec.get('generation_status')}` | "
                f"Latency: `{format_metric(rec.get('generation_time_seconds'), decimals=2, suffix='s')}`"
            )
            if rec.get("generation_error"):
                st.error(f"Generation Error: {rec.get('generation_error')}")
            st.markdown("**Generated Response:**")
            st.write(rec.get("generated_answer") or "*[Empty Response]*")

            m1, m2 = st.columns(2)
            m1.metric("Exact / Norm EM", f"{format_metric(rec.get('exact_match'), 1)} / {format_metric(rec.get('normalized_exact_match'), 1)}")
            m2.metric("Token F1", format_metric(rec.get("token_f1")))
            m1.metric("Semantic Similarity", format_metric(rec.get("semantic_similarity")))
            m2.metric(
                "Ref. Correctness",
                f"{format_metric(rec.get('correctness_score'))} ({rec.get('correctness_label') or 'N/A'})",
            )
            if rec.get("has_numeric_contradiction"):
                st.warning("Numeric contradiction detected against reference answer.")

            with st.expander("Claim-Level Faithfulness & Verification", expanded=False):
                if not rec.get("has_context"):
                    st.caption("No context supplied; context faithfulness not computed.")
                else:
                    st.write(
                        f"**Status:** `{rec.get('hallucination_detection_status')}` | "
                        f"**Faithfulness Ratio:** `{format_metric(rec.get('faithfulness_ratio'))}` | "
                        f"**Context Hallucination Rate:** `{format_metric(rec.get('context_hallucination_rate'))}`"
                    )
                st.write(f"**Summary:** {rec.get('hallucination_explanation') or 'N/A'}")
                for claim in rec.get("claims") or []:
                    st.markdown(
                        f"- **[{claim.get('verification_label')}]** `{claim.get('claim_id')}`: "
                        f"{claim.get('claim_text')}  \n"
                        f"  *Confidence:* `{claim.get('confidence')}` | "
                        f"*Evidence Strength:* `{claim.get('evidence_strength')}` — {claim.get('explanation')}"
                    )

            with st.expander("LLM-as-a-Judge Scores", expanded=False):
                j_status = rec.get("judge_evaluation_status")
                if j_status in {None, "disabled", "pending"}:
                    st.caption(f"Judge evaluation not available (`{j_status}`).")
                elif j_status != "evaluated":
                    st.error(f"Judge status `{j_status}`: {rec.get('judge_error_message')}")
                else:
                    st.write(
                        f"**Judge:** `{rec.get('judge_model_id')}` | "
                        f"**Overall:** `{format_metric(rec.get('judge_overall_score'), 2)}/5` | "
                        f"**Self-Reported Confidence:** `{format_metric(rec.get('judge_confidence'), 2)}`"
                    )
                    st.write(
                        f"Relevance: `{rec.get('judge_relevance_score')}` | "
                        f"Factual: `{rec.get('judge_factual_correctness_score')}` | "
                        f"Completeness: `{rec.get('judge_completeness_score')}` | "
                        f"Faithfulness: `{rec.get('judge_contextual_faithfulness_score')}` | "
                        f"Clarity: `{rec.get('judge_clarity_score')}`"
                    )
                    st.json(rec.get("judge_dimension_explanations") or {})
