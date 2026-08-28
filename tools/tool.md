
# Tools and Libraries

Software tools and libraries relevant to detecting, evaluating, and mitigating citation fabrication and hallucination in LLM and RAG-based systems.

| Tool | Purpose | Paper | Repository |
|---|---|---|---|
| **SelfCheckGPT** | Zero-resource, black-box hallucination detection via sampling-based consistency checking — no external database required. | Manakul, Liusie, & Gales (2023), EMNLP · [10.18653/v1/2023.emnlp-main.557](https://doi.org/10.18653/v1/2023.emnlp-main.557) | [github.com/potsawee/selfcheckgpt](https://github.com/potsawee/selfcheckgpt) |
| **FActScore** | Breaks long-form generations into atomic facts and scores the percentage supported by a reliable knowledge source. | Min et al. (2023), EMNLP · [10.18653/v1/2023.emnlp-main.741](https://doi.org/10.18653/v1/2023.emnlp-main.741) | [github.com/shmsw25/FActScore](https://github.com/shmsw25/FActScore) |
| **FacTool** | Tool-augmented factuality detection framework covering knowledge-based QA, code, math, and scientific-literature review (including hallucinated-citation detection). | Chern et al. (2023) · arXiv: [2307.13528](https://arxiv.org/abs/2307.13528) | [github.com/GAIR-NLP/factool](https://github.com/GAIR-NLP/factool) |
| **RAGAs** | Reference-free evaluation framework for RAG pipelines (faithfulness, answer relevance, context precision/recall). | Es et al. (2024), EACL · [10.18653/v1/2024.eacl-demo.16](https://doi.org/10.18653/v1/2024.eacl-demo.16) | [github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas) |
| **CiteVerifier** | Open-source citation-verification framework combining reference parsing with cascaded multi-source retrieval (CrossRef, OpenAlex, Semantic Scholar, web search) to classify citations as valid/invalid at scale. | Xu et al. (2026), GhostCite · arXiv: [2602.06718](https://arxiv.org/abs/2602.06718) | See paper for release status |

## General-Purpose Bibliographic / Citation-Matching APIs

These are not citation-fabrication-specific tools but are the authoritative databases against which fabricated citations are typically checked in the reviewed literature:

- **CrossRef API** — https://www.crossref.org/documentation/retrieve-metadata/rest-api/
- **OpenAlex API** — https://docs.openalex.org/
- **Semantic Scholar API** — https://api.semanticscholar.org/
- **PubMed / NCBI E-utilities** — https://www.ncbi.nlm.nih.gov/books/NBK25501/
- **Internet Archive Wayback Machine API** (used for URL-citation existence checks) — https://archive.org/help/wayback_api.php

## Retrieval-Augmented Generation (RAG) Frameworks

General frameworks used to build the RAG pipelines discussed in the paper's mitigation section (Section 4.1):

- **LangChain** — https://github.com/langchain-ai/langchain
- **LlamaIndex** — https://github.com/run-llama/llama_index
