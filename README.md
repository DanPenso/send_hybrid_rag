# SEND-RAG: Hybrid Retrieval-Augmented Generation for Statutory SEND Support

This repository contains the complete, end-to-end codebase for the fine-tuned Hybrid-RAG dissertation project. The system implements a local,  Retrieval-Augmented Generation (RAG) architecture combined with Supervised Fine-Tuning (QLoRA). The purpose is to adapt a baseline Large Language Model (Meta-Llama-3.1-8B-Instruct) into a factually grounded, pedagogically aligned tool to support teachers working with SEND children.

---

## Key Results

This project compared a **zero-shot Llama 3.1 baseline** against a **hybrid RAG + QLoRA fine-tuned model** on 250 stratified UK SEND advisory prompts. Evaluation used four streams: RAGAS (faithfulness/relevancy), LLM-as-judge (tone/grounding), PCA/K-Means semantic drift, and cross-stream statistical testing.

### RAGAS evaluation (Stream A)

Hybrid RAG improved mean faithfulness from **0.82** to **0.88** (+0.054, p = 0.034). Answer relevancy decreased from **0.72** to **0.55**, suggesting a trade-off between grounding in retrieved context and conversational fluency.

![RAGAS overall metric means](data/06_evaluation/figures/5a_ragas/v14/5a_overall_metric_means.png)

### Pedagogical evaluation (Stream B)

The LLM judge showed a significant uplift in technical grounding (**3.87 → 4.03**, +0.164, p < 0.001). Tone alignment and accuracy showed small but statistically significant decreases, indicating the hybrid system prioritised source fidelity over stylistic warmth in some cases.

![LLM judge overall metric means](data/06_evaluation/figures/5b_llm_judge/v14_llm_judge_notebook/5b_overall_metric_means.png)

### Semantic drift & clustering (Stream C)

Responses were embedded and projected into 2D PCA space after K-Means clustering (k=6). The plot below colours points by SEND category and distinguishes baseline from hybrid outputs by shape, making it easier to see how semantic drift varies across topic areas. Hybrid responses showed higher mean context similarity (**0.63 → 0.71**, +0.085, p < 0.001), indicating stronger alignment with retrieved statutory material.

![PCA by category — baseline vs hybrid](data/06_evaluation/stream_c_r/v14/stream_c_pca_by_category.png)

### Conclusion

- **Hybrid RAG improved factual faithfulness** over the zero-shot baseline (mean delta +0.054, p = 0.034).
- **Technical grounding improved significantly** under LLM-as-judge evaluation (+0.164, p < 0.001).
- **Retrieved-context alignment strengthened**, with hybrid mean context similarity rising from 0.63 to 0.71.
- **Semantic drift analysis** showed category-level separation in PCA space, with baseline and hybrid responses shifting differently across SEND topic areas.
- **Trade-offs were observed**: answer relevancy and tone alignment decreased in some streams, so the system leans toward statutory grounding over conversational fluency — an important consideration for teacher-facing deployment.

### Limitations

- The ChromaDB index is not included in this repository due to size; Phase 4a must be re-run to rebuild it.
- Evaluation used synthetic training data and API-based judges; results should be validated in real school settings before deployment.
- This is a research prototype, not a clinical or legal advice tool.

### Links

- **Dataset:** [send-senco-instruction-pairs on Hugging Face](https://huggingface.co/datasets/DanPenso/send-senco-instruction-pairs)
- **Dissertation grade:** Distinction (80%)

---

## Setup

**Python:** 3.10 or above is required.

**1. Create and activate a virtual environment**
```bash
python -m venv venv311
# Windows
venv311\Scripts\activate
# macOS / Linux
source venv311/bin/activate
```

**2. Install Python dependencies**
```bash
# GPU training (Phase 3a) — install PyTorch with CUDA 12.4 first:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Then install everything else:
pip install -r requirements.txt
```

**3. Configure API keys** — copy `.env` and fill in your credentials:
```
ANTHROPIC_API_KEY=...   # Phase 2
AWS_ACCESS_KEY_ID=...   # Phase 1 (Textract)
AWS_SECRET_ACCESS_KEY=...
COHERE_API_KEY=...      # Phase 4
OPENAI_API_KEY=...      # Phase 5a / 5b
HF_TOKEN=...            # Phase 3a (Hugging Face model download)
```

**4. External tools**
* **Ollama** — required for Phase 3c (model registration) and Phase 4b (local inference). Install from [ollama.com](https://ollama.com) and ensure `ollama serve` is running.
* **llama.cpp** — required for Phase 3c GGUF conversion. The script generates the shell commands; `llama.cpp` must be built and on your `PATH`.

**5. R packages (Phase 5c only)**
```r
install.packages(c("jsonlite", "dplyr", "tidyr", "readr", "stringr",
                   "purrr", "ggplot2", "cluster", "reticulate"))
```

---

## Tech Stack & Architecture
* **Base Model:** `Meta-Llama-3.1-8B-Instruct`
* **Fine-Tuning:** `Hugging Face PEFT`, `TRL`, `BitsAndBytes` (4-bit QLoRA)
* **Local Inference:** `Ollama` (GGUF format for deterministic CPU/GPU execution)
* **Vector Database:** `ChromaDB` with `Cohere` (`embed-english-v3.0`)
* **Evaluation Frameworks:** `RAGAS` (Faithfulness/Relevancy), `LLM-as-a-Judge` (Tone/Grounding), `PCA/K-Means` (Semantic Latent Space), `Cross-Stream Analysis`

---

## Execution Pipeline (Phases 1-5)

The project is structured into 5 chronological phases. Constants and evaluation targets (e.g., the `v14` inference tag and 250 evaluation prompt count) are centrally managed in `scripts/pipeline_constants.py`.

### Phase 1: Data Ingestion (`phase1_aws_pdf_ingest.py`)
Ingests raw documents downloaded from the Department for Education and NHS. Utilises **AWS Textract** via `boto3` to extract machine-readable text from multi-page PDFs.

### Phase 2: Synthetic Data Generation (`phase2_synthetic_data_generation.py`)
Generates the 1,000-pair instruction dataset required to teach the model its supportive "SENCo" persona. Utilizes `Anthropic Claude 4.6 Sonnet` constrained by Pydantic validation to output strict JSONL format.

### Phase 3: Supervised Fine-Tuning & Deployment
* **`phase3a_lora_sft.py`**: Executes the QLoRA fine-tuning. Freezes the base model and targets attention projections (`q_proj`, `v_proj`) to mitigate catastrophic forgetting.
* **`phase3b_lora_hf_merge.py`**: Safely merges the PEFT adapter weights back into the base Llama 3.1 model. **Note:** Explicitly executed on GPU RAM.
* **`phase3c_gguf_ollama.py`**: Generates shell commands to compile the merged checkpoint into a 16-bit GGUF format via `llama.cpp` and registers it to the local Ollama engine with deterministic safety parameters (`Temperature=0.0`).

### Phase 4: Retrieval & Inference
* **`phase4a_chroma_index_ingest.py`**: Chunks the Phase 1 text and embeds it using Cohere, populating the local ChromaDB vector store.
* **`phase4b_local_ollama_rag.py`**: Runs the 250 stratified evaluation prompts against both the Zero-Shot Baseline and the Hybrid-RAG model, outputting raw responses to JSONL.

### Phase 5: Multi-Stream Evaluation
* **Stream A (`phase5a_ragas_eval.ipynb`)**: API calls to OpenAI GPT-4o-mini to calculate strict RAGAS Faithfulness and Answer Relevancy metrics.
* **Stream B (`phase5b_llm_judge_eval.ipynb`)**: API calls to GPT-4o-mini to calculate pedagogical Tone Alignment and Technical Grounding.
* **Stream C (`phase5c_pca_kmeans.R`)**: Unsupervised clustering (K-Means, k=6) of the response embedding vectors to calculate geometric semantic drift .
* **Stream D (`phase5d_cross_stream_eval.ipynb`)**: Joins all data streams to generate final statistical hypothesis tests, logistic regressions, and system uplift visualizations.

---

## Data folder


```text
data/
├── 01_raw_pdfs/                    # 47 statutory SEND PDFs (DfE, NHS)
├── 02_extracted_text/              # Textract OCR outputs (one .txt per PDF)
├── 03_synthetic_eval/
│   └── bedrock_ready_llama.jsonl   # 1,000 synthetic instruction pairs (Phase 2)
├── 04_chroma_db_cohere/            # Not included (size). Rebuild by running phase4a (requires COHERE_API_KEY).
├── 05_results/                     # v14 inference outputs (250 questions)
│   ├── questions_250_v10.jsonl
│   ├── baseline_results_250_local_ollama_v14.jsonl
│   └── hybrid_rag_results_250_local_ollama_v14.jsonl
└── 06_evaluation/
    ├── ragas_baseline_scores_v14.csv
    ├── ragas_hybrid_scores_v14.csv
    ├── llm_judge_per_row_v14_llm_judge_notebook.csv
    ├── llm_judge_summary_v14_llm_judge_notebook.csv
    ├── llm_judge_hypothesis_tests_v14_llm_judge_notebook.csv
    ├── llm_judge_tone_distribution_table.csv
    ├── figures/
    │   ├── 5a_ragas/v14/
    │   │   ├── 5a_overall_metric_means.png
    │   │   ├── 5a_category_metric_bars.png
    │   │   └── 5a_category_delta_heatmap.png
    │   └── 5b_llm_judge/v14_llm_judge_notebook/
    │       ├── 5b_overall_metric_means.png
    │       ├── 5b_tone_distribution.png
    │       ├── 5b_category_metric_bars.png
    │       └── 5b_category_delta_heatmap.png
    ├── stream_c_r/v14/             # Stream C PCA/K-Means outputs (Phase 5c)
    │   ├── stream_c_per_row_scores.csv
    │   ├── stream_c_paired_stats.csv
    │   ├── stream_c_overall_summary.csv
    │   ├── stream_c_cluster_mix.csv
    │   ├── stream_c_category_drift_delta.csv
    │   ├── stream_c_k_selection.csv
    │   ├── stream_c_pca_explained_variance.csv
    │   ├── stream_c_discussion_view.csv
    │   └── stream_c_pca_by_category.png
    └── stream_d/v14_cross_stream_synthesis/
        ├── joined_per_question.csv
        └── significance_test.csv   # Stream D figures are generated when phase5d is run
```


| Artifact | Location |
|----------|----------|
| **Evaluation Prompts** | `data/05_results/questions_250_v10.jsonl` |
| **Inference Tag** | `v14` (Set in `scripts/pipeline_constants.py`) |
| **Model Outputs** | `data/05_results/*_v14.jsonl` |
| **Evaluation Metrics** | `data/06_evaluation/` (Contains v14 tags for RAGAS, Judge, and PCA) |
| **Thesis Figures** | `data/06_evaluation/figures/` |




