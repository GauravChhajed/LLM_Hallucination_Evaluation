# Evaluation Dataset Design & Documentation

## 1. Purpose and Design Philosophy

This benchmark (`100` curated items) is designed to evaluate Large Language Models (LLMs) along two complementary dimensions:

1. **Closed-book Factual & Reasoning Accuracy:** Testing whether a model produces concise, accurate answers across General Knowledge, Science & Technology, Mathematics, History, and Logical Reasoning without inventing spurious facts.
2. **Faithfulness & Hallucination Resistance:** Testing whether a model remains strictly grounded in supplied context, obeys counterfactual context overrides, refuses unanswerable queries, rejects false premises, identifies ambiguity, and avoids fabricating citations or numbers.

Every record is stored consistently in:
- `data/raw/original_dataset.json` (canonical raw JSON list)
- `data/processed/validated_dataset.json` (validated JSON with schema metadata and summary statistics)
- `data/evaluation_dataset.csv` (tabular benchmark consumed by the evaluation pipeline and Streamlit dashboard)

---

## 2. Schema Definition

Each dataset item contains 8 required fields and 1 optional traceability field:

| Field | Required | Description |
| :--- | :---: | :--- |
| `question_id` | Yes | Unique identifier (`Q001`–`Q100`). |
| `question` | Yes | Exact prompt question presented identically to both compared LLMs. |
| `context` | Yes | Supporting passage for context-based QA (empty string `""` for closed-book items). |
| `reference_answer` | Yes | Verified ground-truth answer or explicit refusal/clarification criterion. |
| `category` | Yes | One of the 7 benchmark categories. |
| `difficulty` | Yes | `easy`, `medium`, or `hard`. |
| `expected_behavior` | Yes | Target behavior label (e.g., `factual_answer`, `reject_false_premise`, `acknowledge_insufficient_info`). |
| `hallucination_test_type` | Yes | Specific hallucination stress-test tag. |
| `reference_source` | Optional | Authoritative provenance for factual/context answers (left empty for synthetic unanswerable prompts). |

---

## 3. Category Distribution (`N = 100`)

| Category | Count | Share | Focus |
| :--- | :---: | :---: | :--- |
| General Knowledge | 15 | 15% | Stable, verifiable geography, culture, and world facts |
| Science and Technology | 15 | 15% | Physics, chemistry, biology, and computer science precision |
| Mathematics | 15 | 15% | Arithmetic, algebra, calculus, combinatorics, geometry, probability (programmatically verified) |
| History | 10 | 10% | Unambiguous historical dates, treaties, and figures |
| Logical Reasoning | 10 | 10% | Deductive syllogisms, constraint ordering, formal fallacies |
| Context-based Question Answering | 20 | 20% | Grounded extraction, unanswerable context, counterfactual rules, distractors |
| Adversarial and Hallucination Testing | 15 | 15% | False premises, missing info, ambiguity, uncertainty, fake citations |
| **Total** | **100** | **100%** | |

---

## 4. Hallucination Test Scenarios Covered

| `hallucination_test_type` | Count | What It Tests |
| :--- | :---: | :--- |
| `factual_precision` | 65 | Closed-book accuracy across GK, Science, Math, History, and Logic |
| `context_supported_precision` | 8 | Exact extraction and reasoning grounded in a supplied passage |
| `unanswerable_from_context` | 5 | Refusing to guess facts missing from a related passage |
| `counterfactual_context` | 4 | Following supplied hypothetical rules instead of overriding them with pretraining priors |
| `misleading_irrelevant_context` | 3 | Ignoring distractor numbers or unrelated passages |
| `false_premise` | 4 | Identifying and correcting invalid presuppositions in the question |
| `insufficient_information` | 3 | Recognizing underspecified word problems or missing documents |
| `ambiguous_question` | 3 | Requesting disambiguation instead of guessing a single entity |
| `uncertainty_acknowledgment` | 2 | Admitting unknowable future or unmeasured quantities |
| `fabricated_citation_or_number` | 3 | Refusing to invent DOIs, ISBNs, page numbers, or non-existent reports |

---

## 5. Known Limitations

1. **Dataset Size (`100` items):** While broad across 7 categories and 9 hallucination failure modes, 100 questions provide a controlled comparative benchmark rather than an exhaustive domain stress test.
2. **English-Only Prompts:** All prompts and reference answers are in English; multilingual hallucination behavior is not measured.
3. **Single-Turn Evaluation:** Evaluates single-turn QA rather than multi-turn sycophancy or conversational drift.
4. **Reference Answer Phrasing:** Concise reference answers represent one valid phrasing; therefore, downstream evaluation combines semantic similarity, key-information coverage, claim-level faithfulness, and LLM-as-a-Judge rather than relying on exact string match alone.
