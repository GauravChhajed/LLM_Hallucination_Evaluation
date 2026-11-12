# Phase 5 Evaluation Metrics & Reference Correctness Documentation

## 1. Overview

The `src/evaluation/` package evaluates LLM-generated responses against verified reference answers without relying on a single collapsed score. It preserves **lexical**, **semantic**, and **reference-correctness diagnostic** signals independently so researchers and users can inspect *why* an answer succeeded or failed.

---

## 2. Normalization & Tokenization Procedure (`lexical_metrics.py`)

Before computing Normalized Exact Match and token-level overlap:

1. Trim leading/trailing whitespace and convert text to lowercase.
2. Remove thousands-separator commas inside numbers (e.g., `1,000` → `1000`) without altering decimals (`7.5`), fractions (`1/6`), negative signs (`-6`), or hyphenated identifiers/dates (`2024-03`, `syn-ack`).
3. Extract alphanumeric/numeric tokens via regex (`-?\d+(?:[./]\d+)*|[a-z0-9]+(?:[-'/][a-z0-9]+)*`), stripping external punctuation (`?`, `!`, `,`, `;`, trailing `.`) while **never deleting decimals, units, names, or dates**.
4. Optionally drop standalone English articles (`a`, `an`, `the`) when `remove_articles=True`.

---

## 3. Mathematical Definitions of Implemented Metrics

### A. Exact Match (EM) & Normalized Exact Match (NEM)

$$\text{EM}(g, r) = \mathbb{I}[\text{strip}(g) = \text{strip}(r)] \quad (\text{where } g, r \neq \emptyset)$$

$$\text{NEM}(g, r) = \mathbb{I}[\text{norm}(g) = \text{norm}(r)] \quad (\text{where } \text{norm}(g), \text{norm}(r) \neq \emptyset)$$

### B. Token-Level Precision, Recall, and F1 (Multiset / Bag-of-Tokens)

Let $T_g$ be the sequence of normalized tokens in the generated answer and $T_r$ be the sequence of normalized tokens in the reference answer. Using multiset (`Counter`) intersection prevents repeated tokens from inflating precision:

$$\text{common} = \sum_{w \in V} \min\big(\text{count}_{T_g}(w),\; \text{count}_{T_r}(w)\big)$$

$$\text{Precision} = \frac{\text{common}}{|T_g|}, \qquad \text{Recall} = \frac{\text{common}}{|T_r|}$$

$$\text{Token F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \quad (\text{or } 0 \text{ if } \text{common} = 0)$$

### C. Semantic Similarity (Cosine Similarity)

Using dense embeddings $\mathbf{A} = E(g)$ and $\mathbf{B} = E(r)$:

$$\text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \, \|\mathbf{B}\|_2} = \frac{\sum_{i=1}^{d} A_i B_i}{\sqrt{\sum_{i=1}^{d} A_i^2} \, \sqrt{\sum_{i=1}^{d} B_i^2}} \in [-1, 1]$$

If either embedding has zero norm ($\|\mathbf{A}\|_2 = 0$ or $\|\mathbf{B}\|_2 = 0$), cosine similarity returns `0.0` safely.

---

## 4. Embedding Model Choice

- **Default Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional sentence embeddings, ~90 MB).
- **Rationale:** Lightweight, fast on 12-core CPU fallback, well-suited for short-to-medium sentence similarity, and lazily loaded only when semantic evaluation runs (with `D:\hf_cache` storage and mock provider support for offline tests).

---

## 5. Semantic Similarity vs. Reference-Based Correctness

**Crucial Research Principle:** A high cosine similarity score is **not** proof of factual correctness.

1. **Numerical Contradictions:** *"17 multiplied by 24 equals 409"* and *"17 multiplied by 24 equals 408"* share almost identical phrasing and embeddings ($\text{cos} > 0.95$), yet the generated answer is mathematically wrong.
2. **Negation & Refusal Flip:** *"Venus has 2 moons"* vs. *"Venus has zero moons"* often receive high embedding similarity despite opposite truth values.
3. **Adversarial Traps:** On false-premise or unanswerable questions, a model that fabricates an answer using words from the question/reference can score moderate lexical overlap without acknowledging insufficiency or rejecting the false premise.

### How `correctness.py` Mitigates This

`ReferenceCorrectnessEvaluator` computes transparent auxiliary signals alongside lexical and semantic scores:
- **Key Information Coverage:** Fraction of non-stopword reference content terms present in the generated answer.
- **Numeric Consistency Check:** Extracts normalized numbers (`extract_numeric_tokens`) and flags `has_numeric_contradiction = True` if conflicting or missing numbers contradict the reference.
- **Refusal / Premise-Rejection Alignment:** Checks whether questions expecting refusal, uncertainty, clarification, or false-premise rejection actually trigger explicit hedging/rejection cues.
- **Optional Composite Score:** When enabled, combines `0.40 * clamp(semantic_similarity, 0, 1) + 0.40 * key_info_coverage + 0.20 * token_f1` (re-normalized if semantic similarity is unavailable) and **caps** scores at `0.30` on numeric contradiction or `0.35` on refusal mismatch, assigning an interpretable label (`correct`, `partially_correct`, `incorrect`, `contradictory_numeric`, `unassessed`).

---

## 6. Interpreting `EvaluationRecord` Output

Each evaluated question-model pair produces an `EvaluationRecord` containing:
- Identity & text: `question_id`, `model_id`, `generated_answer`, `reference_answer`
- Individual uncollapsed metrics: `exact_match`, `normalized_exact_match`, `token_precision`, `token_recall`, `token_f1`, `semantic_similarity`
- Reliability status: `evaluation_status` (`evaluated`, `empty_generated_answer`, `empty_reference_answer`, `generation_error`, `evaluated_without_semantic`)
- Diagnostics: `evaluation_metadata` (lexical token counts, embedding model/status, `correctness` sub-report with `matched_key_terms`, `missing_key_terms`, and explanatory `notes`).

---

## 7. Known Limitations (Phase 5 Scope)

1. **No Faithfulness / Claim-Level Hallucination Detection Yet:** Context-grounded claim verification and NLI contradiction checks belong to Phase 6.
2. **No LLM-as-a-Judge Yet:** Semantic/reasoning grading by an evaluator LLM belongs to Phase 7.
3. **Heuristic Numeric & Cue Matching:** Rule-based numeric extraction and refusal-phrase matching improve precision over raw cosine similarity, but paraphrased numerical forms (e.g., *"forty-one"* vs. *"41"*) or subtle paraphrases of refusal are not yet normalized to canonical values.
