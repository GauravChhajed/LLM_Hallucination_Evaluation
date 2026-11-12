# Phase 6: Faithfulness & Hallucination Detection Design

## 1. Research Principles & Distinctions

The `src/hallucination/` package separates three distinct concepts that are frequently conflated:

1. **Faithfulness (Context-Grounded):** Whether claims in the generated response are supported by the *supplied context passage*—even when that context describes hypothetical or counterfactual rules. Context faithfulness is **never** calculated for closed-book questions that have no supplied context.
2. **Factual Correctness (Reference-Based):** Whether the response matches real-world ground truth (evaluated separately in Phase 5 `src/evaluation/`).
3. **Hallucination Detection:** Identifying fabricated, contradicted, or unverified claims relative to supplied context, as well as adversarial traps (false premises, unanswerable prompts, fabricated citations) when context is absent—without automatically marking every unsupported statement as false.

---

## 2. Modular Architecture

```text
Generated Answer + Optional Context + Dataset Metadata
        │
        ▼
[claim_extractor.py] ──► Atomic typed claims (numeric, temporal, named_entity, factual)
        │                  + filters conversational filler & appropriate refusals
        ▼
[evidence_retriever.py] ► Context sentence segmentation + top-k semantic & lexical retrieval
        │
        ▼
[claim_verifier.py]  ──► SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFORMATION
        │                  (Numeric/negation consistency guards + optional NLI + semantic check)
        ▼
[faithfulness.py]    ──► Supported/Contradicted/Unsupported counts & explicit ratios
        │
        ▼
[hallucination_detector.py] ► HallucinationDetectionRecord + JSON/CSV persistence
```

---

## 3. Claim Extraction (`claim_extractor.py`)

- **Sentence Segmentation (`split_sentences`):** Protects decimals (`7.5`, `99.95`) and abbreviations (`Dr.`, `Prof.`, `et al.`, `St.`) while splitting sentences.
- **Compound Clause Decomposition:** Splits semicolons and clause-level conjunctions (`and`, `but`, `while`, `whereas`) only when both halves form independent verb-containing clauses.
- **Filtering Non-Factual Segments:** Conversational boilerplate (*"Sure!"*, *"Hope this helps."*) and explicit refusals/uncertainty statements (*"The provided context does not state..."*) are separated from factual claims so appropriate refusals are not penalized as hallucinations.
- **Claim Typing:** Assigns `numeric_claim`, `temporal_claim`, `named_entity_claim`, or `factual_assertion`.

---

## 4. Evidence Retrieval (`evidence_retriever.py`)

- Segments the supplied context into sentences and encodes both claims and context sentences via `BaseEmbeddingProvider` (`sentence-transformers/all-MiniLM-L6-v2` by default, or mocked providers).
- Computes cosine similarity and secondary token Jaccard overlap for every context sentence.
- Retrieves up to `top_k=2` sentences meeting `min_similarity_threshold = 0.30` (or lexical overlap `>= 0.25`) while retaining `best_evidence` for auditability.
- **Crucial Safeguard:** Low similarity triggers `NOT_ENOUGH_INFORMATION`, **never** `CONTRADICTED`.

---

## 5. Claim Verification & Threshold Rationale (`claim_verifier.py`)

Each factual claim receives one of three labels:
- `SUPPORTED`
- `CONTRADICTED`
- `NOT_ENOUGH_INFORMATION`

### Confidence vs. Evidence Strength Interpretation

`ClaimVerificationResult` separates two distinct fields to prevent ambiguity:
1. **`confidence`**: Represents **classification confidence in the assigned `verification_label`** (`[0.0, 1.0]`).
   - For `NOT_ENOUGH_INFORMATION`, `confidence` reflects certainty that *the supplied context lacks sufficient evidence* (bounded in `[0.50, 0.85]` in heuristic mode or derived from NLI `neutral` probability—never set to `1.0` solely because similarity is `0.0`). It **never** implies that the claim is factually false.
   - For `no_context` questions, `confidence` is `0.0` (`not_applicable_no_context`).
2. **`evidence_strength`**: Represents the **observed support/retrieval signal** in `[0.0, 1.0]` (`0.0` when no context or no relevant evidence exists; similarity/entailment strength otherwise).

Verification pipeline:
1. **Numeric/Temporal Conflict Guard:** If a claim and closely related evidence sentence (`topic_overlap >= 0.35`) contain conflicting numbers/dates (e.g., `24` vs. `18` satellites), the claim is labeled `CONTRADICTED`.
2. **Negation Polarity Flip Guard:** If a claim and closely overlapping evidence (`overlap >= 0.45`) have opposite negation polarity (`no/not/never/zero`), the claim is labeled `CONTRADICTED`.
3. **Optional NLI Cross-Encoder (`BaseNLIScorer`):**
   - `p_contradiction >= 0.60` → `CONTRADICTED` (conservative threshold to avoid false-positive contradiction flags on paraphrased text).
   - `p_entailment >= 0.55` → `SUPPORTED`.
   - Otherwise → `NOT_ENOUGH_INFORMATION`.
4. **Hybrid Semantic + Key-Term Fallback (when no NLI model is loaded):**
   - Requires `best_similarity >= 0.55`, `key_term_coverage >= 0.50`, and all claim numbers grounded in evidence (`numbers_grounded = True`) to label `SUPPORTED`; otherwise labels `NOT_ENOUGH_INFORMATION`.

### Candidate NLI Models Compared for Available Hardware (CPU-only, 15.65 GB RAM)

| Model | Params | Est. FP32 CPU RAM | Disk Size | CPU Suitability | Notes |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **`cross-encoder/nli-deberta-v3-small`** | 141M | ~560 MB | ~540 MB | **Recommended** | Fast on 12-core CPU; strong MNLI/SNLI balance |
| **`cross-encoder/nli-deberta-v3-xsmall`** | 71M | ~285 MB | ~280 MB | **Very High** | Fastest NLI option with slightly lower accuracy |
| **`facebook/bart-large-mnli`** | 407M | ~1.6 GB | ~1.6 GB | **Medium** | Accurate zero-shot NLI, ~3× slower on CPU |
| **Hybrid Embedding + Rule Verifier** | ~22M (MiniLM) | ~90 MB (shared) | ~90 MB | **Very High** | Reuses Phase 5 embedder; zero extra model load |

---

## 6. Faithfulness & Hallucination Metric Formulas (`faithfulness.py`)

For context-bearing questions (`has_context = True`) with $N = \text{total\_factual\_claims} > 0$, four separate metrics are preserved:

$$\text{Faithfulness Ratio} = \frac{N_{\text{SUPPORTED}}}{N}, \qquad \text{Contradiction Rate} = \frac{N_{\text{CONTRADICTED}}}{N}, \qquad \text{Unsupported Claim Rate} = \frac{N_{\text{NOT\_ENOUGH\_INFO}}}{N}$$

$$\text{Context Hallucination Rate} = \frac{N_{\text{CONTRADICTED}} + N_{\text{NOT\_ENOUGH\_INFO}}}{N} = \text{Contradiction Rate} + \text{Unsupported Claim Rate}$$

> **Important Interpretation Note:** `context_hallucination_rate` combines contradicted and unsupported claims as a broad **unsupported-by-context** measure. It measures departure from the supplied passage and **does not establish that every unsupported claim is factually false** (a claim may be true in the real world while still absent from the context).

- If `has_context = False` → `status = "no_context"`, all four ratios (`faithfulness_ratio`, `contradiction_rate`, `unsupported_claim_rate`, `context_hallucination_rate`) are `None`.
- If `N = 0` due to refusal → `status = "appropriate_refusal"`, all ratios are `None` (no division by zero).
- If `N = 0` due to empty output → `status = "empty_response"`, all ratios are `None`.

---

## 7. Examples of Verification Outcomes

1. **SUPPORTED:** Context: *"Phase 2 added 18 synthetic-aperture radar satellites in 2023."* | Claim: *"Phase 2 added 18 synthetic-aperture radar satellites in 2023"* → Grounded numbers (`18`, `2023`), high similarity/key-term match → `SUPPORTED`.
2. **CONTRADICTED:** Context: *"water boils at 72 degrees Celsius... capital of France is Lyon."* | Claim: *"the capital of France is Paris"* → NLI contradiction / entity conflict against counterfactual context → `CONTRADICTED`.
3. **NOT_ENOUGH_INFORMATION:** Context: *"The Kestrel Suspension Bridge spans 1.4 kilometers..."* | Claim: *"The bridge cost \$120 million to build"* → No cost sentence passes retrieval/entailment → `NOT_ENOUGH_INFORMATION` (flagged as unsupported, not assumed false in the abstract).

---

## 8. Limitations

1. **Heuristic Clause Decomposition:** Very complex multi-clause sentences with nested relative pronouns may stay as one compound claim rather than splitting into finer atoms.
2. **Pronoun Coreference:** Claims using unresolved pronouns (*"It cost \$5M"*) without repeating the entity name reduce retrieval and NLI precision when evaluated out of context.
3. **Lightweight CPU Verifiers:** Without an NLI cross-encoder loaded, subtle semantic contradictions that do not involve differing numbers or explicit negation keywords default conservatively to `NOT_ENOUGH_INFORMATION` rather than `CONTRADICTED`.
4. **No LLM-as-a-Judge Yet:** Deep pragmatic grading of nuance is deferred to Phase 7.
