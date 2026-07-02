finrag-arch-eval/
├── README.md
├── pyproject.toml / requirements.txt
├── .env.example
├── configs/
│   ├── base.yaml              # shared/controlled variables (LLM, embed model, chunking, prompt)
│   ├── p0_closedbook.yaml
│   ├── p1_dense.yaml
│   ├── p2_hybrid.yaml
│   ├── p3_hybrid_rerank.yaml
│   └── p4_table_aware.yaml
├── data/
│   ├── raw/                   # 10-K filings (or download script, not raw files if licensing matters)
│   ├── processed/              # chunked/parsed text + tables
│   └── qa/
│       ├── questions.jsonl     # gold QA pairs with category + evidence labels
│       └── README.md           # how the dataset was built/sourced
├── src/
│   ├── ingestion/
│   │   ├── parse_10k.py
│   │   ├── table_parser.py
│   │   └── chunker.py
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── hybrid_fusion.py    # RRF etc.
│   │   ├── reranker.py
│   │   └── table_retriever.py
│   ├── generation/
│   │   ├── prompts.py
│   │   └── generator.py
│   ├── pipelines/
│   │   ├── p0_closedbook.py
│   │   ├── p1_dense.py
│   │   ├── p2_hybrid.py
│   │   ├── p3_hybrid_rerank.py
│   │   └── p4_table_aware.py
│   └── eval/
│       ├── retrieval_metrics.py    # Recall@k, MRR, nDCG
│       ├── answer_metrics.py       # faithfulness, hallucination, numeric accuracy
│       ├── latency.py
│       └── stats_tests.py          # bootstrap / Wilcoxon
├── scripts/
│   ├── run_pipeline.py         # CLI: run one architecture end-to-end
│   ├── run_all.py              # run all 5 pipelines on full eval set
│   └── build_dataset.py
├── results/
│   ├── raw_runs/                # per-pipeline raw outputs (jsonl)
│   ├── tables/                  # generated results tables (csv/md)
│   └── figures/                 # faithfulness-vs-latency plot etc.
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_results_analysis.ipynb
│   └── 03_failure_analysis.ipynb
├── paper/
│   └── mini_paper.md / .tex     # the write-up itself, versioned alongside code
└── tests/
    └── ...