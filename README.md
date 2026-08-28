# Awesome Citation Fabrication Benchmarking

> A curated, verified research collection on citation fabrication and hallucination benchmarking in large language models and agentic search tools.

## Short Description

This repository curates verified scholarly research, datasets, tools, and implementations related to **citation fabrication in large language models (LLMs) and agentic search/deep-research systems**. It accompanies an original AI-assisted research paper that synthesizes empirical benchmarking studies published between 2023 and 2026, comparing fabrication rates across general-purpose chatbots, retrieval-augmented generation (RAG) pipelines, commercial AI legal research tools, and agentic web-browsing search agents. All 21 curated papers were independently verified for authenticity — see the Citation Integrity Audit below.

## Table of Contents

- [Topic Overview](#topic-overview)
- [AI-Assisted Research Paper](#ai-assisted-research-paper)
- [Citation Integrity Audit](#citation-integrity-audit)
- [Curated Research Papers](#curated-research-papers)
- [Datasets](#datasets)
- [Tools and Libraries](#tools-and-libraries)
- [GitHub Implementations](#github-implementations)
- [Tutorials and Learning Resources](#tutorials-and-learning-resources)
- [License](#license)

## Topic Overview

Citation fabrication — the generation of bibliographic references, legal citations, or source attributions that do not correspond to real, verifiable works — is one of the most measurable and consequential forms of hallucination in large language models. Because a claimed citation either resolves to a real, matching source or it does not, this failure mode is unusually tractable to benchmark compared to other kinds of hallucination, and it has become a major focus of empirical AI-reliability research since ChatGPT's public release in late 2022.

This repository focuses on the full landscape of that research: foundational work defining and taxonomizing hallucination; empirical studies quantifying fabrication rates for general-purpose chatbots (GPT-3.5, GPT-4, GPT-4o) across scholarly, medical, and legal writing; evaluations of commercial retrieval-augmented generation (RAG) products explicitly marketed as "hallucination-free"; and the newest and fastest-growing area — agentic AI search engines and "deep research" tools that autonomously browse the web and generate cited reports. Reported fabrication rates across this literature range from roughly 14% to 95%, depending on model generation, domain, and system architecture, underscoring that this remains an unresolved, actively evolving problem with direct implications for scholarly integrity, legal practice, journalism, and public trust in AI-mediated information.

## AI-Assisted Research Paper

**Title:** *Benchmarking Citation Fabrication Rates Across Large Language Models and Agentic Search Tools*

**Abstract (short):** This paper synthesizes empirical benchmarking work published between 2023 and 2026 that quantifies citation fabrication rates across general-purpose chatbots, RAG pipelines, commercial AI legal research tools, and agentic deep-research systems. Reported fabrication rates range from roughly 14% to 95% depending on model, domain, and prompting condition. The paper organizes this literature around a taxonomy of fabrication types, reviews current detection and mitigation approaches (retrieval grounding, sampling-based consistency checking, and post-hoc verification pipelines), and identifies open methodological challenges and a future research agenda.

📄 **Full paper:** [`Benchmarking_Citation_Fabrication_LLMs.docx`](./Benchmarking_Citation_Fabrication_LLMs.docx)

## Citation Integrity Audit

All 21 curated research-paper references in this repository were independently checked against publisher DOI records, arXiv abstract pages, PubMed/PMC records, or official code repositories before inclusion — no citation was included from model memory without independent verification.

📄 **Full audit report:** [`Citation_Integrity_Audit.pdf`](./Citation_Integrity_Audit.pdf)

## Curated Research Papers

21 verified scholarly papers organized into 7 categories: Foundational Hallucination Research; Citation Fabrication Benchmarks (General-Purpose LLMs); Domain-Specific Studies (Medicine & Mental Health); Domain-Specific Studies (Law); Agentic Search & Deep Research Tools; Detection, Verification & Mitigation Methods; and Citation Bias & Trust Effects.

📄 **Full list with DOIs / arXiv IDs / PMIDs:** [`references/references.md`](./refrences/refrences.md)

## Datasets

Datasets used across the reviewed literature to benchmark citation and hallucination reliability, including ExpertQA, DRBench, the GhostCite citation corpus, HaluEval, and others.

📄 **Full list with descriptions and links:** [`datasets/datasets.md`](./datasets/datasets.md)

## Tools and Libraries

Open-source tools for hallucination and citation-fabrication detection (SelfCheckGPT, FActScore, FacTool, RAGAs, CiteVerifier), plus authoritative bibliographic APIs (CrossRef, OpenAlex, Semantic Scholar, PubMed, Wayback Machine) and general RAG frameworks.

📄 **Full list with descriptions and links:** [`tools/tools.md`](./tools/tool.md)

## GitHub Implementations

Official, author-maintained code repositories for the tools and datasets referenced above.

📄 **Full list:** [`implementations/github-repositories.md`](./implementations/github-repositories.md)

## Tutorials and Learning Resources

Authoritative resources for learning more about LLM hallucination and RAG evaluation:

- **ACM Computing Surveys — "Survey of Hallucination in Natural Language Generation"** (Ji et al., 2023): foundational taxonomy and terminology. https://doi.org/10.1145/3571730
- **ACM Transactions on Information Systems — "A Survey on Hallucination in Large Language Models"** (Huang et al., 2023/2025): comprehensive, regularly cited overview of detection and mitigation methods. https://doi.org/10.1145/3703155
- **RAGAs documentation** (official docs for the RAG evaluation framework): https://docs.ragas.io/
- **LangChain documentation** on retrieval-augmented generation: https://python.langchain.com/docs/concepts/rag/
- **Columbia Journalism Review / Tow Center — "AI Search Has a Citation Problem"** (Jaźwińska & Chandrasekar, 2025): accessible, non-technical introduction to agentic search citation failures. https://www.cjr.org/tow_center/we-compared-eight-ai-search-engines-theyre-all-bad-at-citing-news.php

## Repository Structure

```
.
|-- README.md
|-- Citation_Integrity_Audit.pdf
|-- Benchmarking_Citation_Fabrication_LLMs.docx
|-- references/
|   `-- references.md
|-- datasets/
|   `-- datasets.md
|-- tools/
|   `-- tools.md
|-- implementations/
|   `-- github-repositories.md
`-- LICENSE
```

## License

This repository's original content (README, audit documentation, and the accompanying research paper) is released under the [MIT License](./license/license.md). Linked third-party datasets, tools, and papers remain under their own original licenses.

## Contributing

Issues and pull requests are welcome — particularly corrections, additional verified benchmark papers, or updated fabrication-rate figures as newer studies are published. Please verify any new reference against a publisher/arXiv/PubMed record before submitting, consistent with this repository's citation integrity standard.
