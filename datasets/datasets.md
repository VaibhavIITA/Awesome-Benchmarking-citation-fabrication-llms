
# Datasets

Datasets referenced or used in the empirical studies reviewed by this repository's research paper. These are datasets **created by the original study authors**, not new datasets produced for this repository.

| Dataset | Description | Used By | Source / Link |
|---|---|---|---|
| **ExpertQA** | 2,177 expert-curated questions across 32 academic and professional fields, each paired with citation-grounded answers; used to test citation URL validity in domain-expert settings. | Rao, Wong, & Callison-Burch (2026) | Malaviya, C., Lee, S., Chen, S., Sieber, E., Yatskar, M., & Roth, D. (2023). arXiv: [2309.07852](https://arxiv.org/abs/2309.07852) |
| **DRBench** | Benchmark of "deep research" style queries designed to evaluate multi-step, citation-producing agentic search behavior at scale. | Rao, Wong, & Callison-Burch (2026) | Referenced in arXiv:2604.03173 |
| **GhostCite Citation Corpus** | 2.2 million citations extracted from 56,381 papers published at top-tier AI/ML and security venues (2020–2025), used to measure fabricated-citation prevalence in the *published* scholarly record. | Xu et al. (2026) | Referenced in arXiv:2602.06718 |
| **Legal Query Hallucination Set** | Custom-built set of specific, verifiable legal questions (case holdings, existence, citation) used to probe LLM legal hallucination rates. | Dahl, Magesh, Suzgun, & Ho (2024) | Referenced in arXiv:2401.01301 |
| **AuthorityBench Prompt Set** | 220,564 prompts in a fully balanced 2×2 factorial design crossing claim veracity (true/false) with citation veracity (real/fabricated) across general knowledge, science, law, and medicine. | Khurana, Ramana RN, & Kumar (2026) | Referenced in arXiv:2606.13104 |
| **FActScore Biography Set** | Long-form biography generations from InstructGPT, ChatGPT, and PerplexityAI, manually decomposed into atomic facts for factual-precision scoring. | Min et al. (2023) | [arXiv:2305.14251](https://arxiv.org/abs/2305.14251) |
| **HaluEval** | 35,000 generated and human-annotated hallucinated/non-hallucinated samples spanning QA, knowledge-grounded dialogue, and summarization, used broadly in hallucination-detection benchmarking. | Referenced across the hallucination-detection literature | Li, J., Cheng, X., Zhao, W. X., Nie, J.-Y., & Wen, J.-R. (2023). *EMNLP 2023*. arXiv: [2305.11747](https://arxiv.org/abs/2305.11747) |

## Notes on Access

- Most datasets above are distributed alongside their originating paper's official code repository (see [`implementations/github-repositories.md`](../implementations/github-repositories.md)).
- The GhostCite citation corpus and the Legal Query Hallucination Set are not fully public at the time of writing; consult the originating papers for data-availability statements.
- No dataset in this repository contains personally identifiable information; all are derived from published academic or legal text.
