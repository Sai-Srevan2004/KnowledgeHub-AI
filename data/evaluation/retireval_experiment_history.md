# KnowledgeHub AI — Retrieval Experiment History

## 1. Objective

Evaluate and improve the retrieval component of KnowledgeHub AI using a manually curated ground-truth dataset of 28 answerable questions.

The goal is to improve retrieval relevance and ranking while preventing regressions across the complete evaluation dataset.

## 2. Evaluation Dataset

- Total evaluation questions: 30
- Answerable questions: 28
- Unanswerable questions: 2
- Evaluation method: Manual ground-truth relevance labels
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: LangChain `InMemoryVectorStore`
- Similarity search: Cosine similarity
- Initial retrieval configuration: Top-K = 3

## 3. Experiment 1 — Initial Baseline

**Configuration**

| Parameter | Value |
|---|---|
| Chunk size | 500 characters |
| Chunk overlap | 50 characters |
| Top-K | 3 |
| Total chunks | 51 |

**Results**

| Metric | Score |
|---|---:|
| Precision@3 | 0.3333 |
| Recall@3 | 0.9643 |
| MRR | 0.8869 |
| nDCG@3 | 0.9069 |

**Observation**

- Q010 failed to retrieve its relevant chunk.
- The original chunk combined multiple policy topics.
- The system retrieved relevant information for 27 of 28 answerable questions.

**Decision**

Test smaller chunks to improve retrieval precision and coverage.

## 4. Experiment 2 — Smaller Chunk Size

**Configuration**

| Parameter | Value |
|---|---|
| Chunk size | 300 characters |
| Chunk overlap | 50 characters |
| Top-K | 3 |
| Total chunks | 91 |

**Results before relevance-label correction**

| Metric | Score |
|---|---:|
| Precision@3 | 0.3929 |
| Recall@3 | 0.9821 |
| MRR | 0.8810 |
| nDCG@3 | 0.8942 |

**Observation**

- Q010 was successfully retrieved.
- Precision and recall improved compared with the initial baseline.
- MRR and nDCG decreased slightly.

**Ground-truth review**

Three questions had overly broad relevance labels:

- Q009: Updated relevant chunks to `[24]`.
- Q016: Updated relevant chunks to `[55]`.
- Q023: Updated relevant chunks to `[47]`.

The labels were corrected to represent chunks directly needed to answer each question.

## 5. Experiment 3 — Corrected 300-Character Baseline

**Configuration**

| Parameter | Value |
|---|---|
| Chunk size | 300 characters |
| Chunk overlap | 50 characters |
| Top-K | 3 |
| Total chunks | 91 |
| Answerable questions | 28 |

**Results**

| Metric | Score |
|---|---:|
| Precision@3 | 0.3571 |
| Recall@3 | 0.9821 |
| MRR | 0.8810 |
| nDCG@3 | 0.8942 |

**Observation**

- 27 of 28 questions retrieved at least one relevant chunk.
- Q020 and Q022 retrieved their relevant chunks at rank 3.
- Q011 retrieved only one of its two labeled relevant chunks.

**Decision**

Preserve this experiment as the corrected baseline for subsequent comparisons.

## 6. Experiment 4 — Increasing Top-K

**Configuration**

| Parameter | Value |
|---|---|
| Chunk size | 300 characters |
| Chunk overlap | 50 characters |
| Top-K | 5 |
| Total chunks | 91 |

**Results**

| Metric | Top-K = 3 | Top-K = 5 |
|---|---:|---:|
| Precision | 0.3571 | 0.2143 |
| Recall | 0.9821 | 0.9821 |
| MRR | 0.8810 | 0.8810 |
| nDCG | 0.8942 | 0.8942 |

**Observation**

- Precision decreased significantly.
- Recall did not improve.
- MRR and nDCG remained unchanged.
- Additional retrieved chunks did not recover new relevant information.

**Decision**

Retain Top-K = 3 as the preferred configuration for now.

## 7. Key Retrieval Failures Identified

| Question | Issue | Finding |
|---|---|---|
| Q011 | Retrieval coverage | Only one of two labeled relevant chunks appeared in the top three


## 8 Experiment: LLM-Based Query Rewriting

**Objective:** Evaluate whether rewriting user questions using an LLM improves retrieval quality.

**Approach:**
- Used Groq LLM to rewrite user questions into policy-oriented search queries.
- Generated rewritten queries for all 28 answerable evaluation questions.
- Evaluated retrieval using the same dataset, embedding model, chunk configuration (300 characters, 50 overlap), and `top_k = 3`.
- Compared results against the original retrieval baseline.

**Results:**

| Metric | Original Retrieval | Query Rewriting |
|---|---:|---:|
| Precision@3 | 0.3571 | 0.3333 |
| Recall@3 | 0.9821 | 0.9107 |
| MRR | 0.8810 | 0.8333 |
| nDCG@3 | 0.8942 | 0.8348 |

**Key Findings:**

Query rewriting reduced retrieval performance across all four evaluation metrics.

Two questions experienced a complete loss of relevant chunks from the top-3 results:

1. **Q005 – Remote Work Approval:** The rewritten query emphasized the approval process but failed to retrieve the relevant chunk containing the original remote-work request information.
2. **Q022 – Software Procurement:** The rewritten query used broader terminology and failed to retrieve the chunk containing the specific software acquisition instructions.

**Root Cause Analysis:**

The experiments indicate that LLM-based query rewriting can introduce *query drift*. During rewriting, important terms or parts of the original information need may be lost, negatively affecting semantic retrieval and ranking.

**Decision:**

- Did not integrate query rewriting into the main RAG pipeline.
- Preserved the original retrieval implementation as the baseline.
- Saved rewritten queries and experiment results for reproducibility.
- Identified query expansion (preserving the original query while adding alternative terminology) as a candidate for a separate experiment.

**Artifacts:**
- `src/knowledgehub_ai/retrieval/query_rewriter.py`
- `data/evaluation/rewritten_queries.json`
- `data/evaluation/retrieval_query_rewriting.json`
- `scripts/evaluate_query_rewriting.py`
- `scripts/analyze_query_rewriting.py`

**Conclusion:** Query rewriting was not beneficial for the current dataset and retrieval configuration. Optimization decisions should be based on measured retrieval quality rather than assuming that LLM-generated queries will improve search performance.