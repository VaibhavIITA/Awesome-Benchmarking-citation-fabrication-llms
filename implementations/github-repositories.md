
# GitHub Implementations

Existing, high-quality open-source implementations relevant to citation fabrication detection, hallucination evaluation, and RAG grounding — referenced from the research paper and datasets/tools reviewed in this repository.

| Repository | Description | Associated Paper |
|---|---|---|
| [potsawee/selfcheckgpt](https://github.com/potsawee/selfcheckgpt) | Official implementation of SelfCheckGPT's zero-resource, sampling-based hallucination detection. | Manakul, Liusie, & Gales (2023) |
| [shmsw25/FActScore](https://github.com/shmsw25/FActScore) | Official implementation of the FActScore atomic-fact factual-precision evaluator. | Min et al. (2023) |
| [GAIR-NLP/factool](https://github.com/GAIR-NLP/factool) | Official implementation of FacTool, including a "scientific literature review" mode for detecting hallucinated citations. | Chern et al. (2023) |
| [explodinggradients/ragas](https://github.com/explodinggradients/ragas) | Official implementation of the RAGAs reference-free RAG evaluation framework. | Es et al. (2024) |
| [chaitanyamalaviya/ExpertQA](https://github.com/chaitanyamalaviya/ExpertQA) | Data and code release for the ExpertQA dataset used to evaluate citation attribution across 32 expert domains. | Malaviya et al. (2023) |
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | General-purpose RAG/agent orchestration framework used to build many of the retrieval-grounded systems discussed in Section 4.1 of the paper. | — |
| [run-llama/llama_index](https://github.com/run-llama/llama_index) | General-purpose data-indexing/RAG framework, an alternative to LangChain for building citation-grounded pipelines. | — |

## Notes

- Repositories are listed as of the time of writing; check each repository's own license before reuse.
- Some tools referenced in the paper (e.g., **CiteVerifier** from Xu et al., 2026) had not published a public code release at the time this repository was compiled — consult the originating paper for current availability.
- This list intentionally favors **official, author-maintained** implementations over third-party reimplementations to preserve fidelity to the published methodology.
