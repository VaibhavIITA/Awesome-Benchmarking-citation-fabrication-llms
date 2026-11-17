# Baseline validation (summary of Prompt 10)

Full report: Prompt 10 PDF. Validation is against the method description (Naser 2026, my protocol), the real pilot output and 57 reference labels; no numerical replication is claimed (data and code of Naser 2026 not public, D08).

## Verified
- Parsing of all 12 real pilot answers after the fixes (debugging log 6, 7).
- Models, 4-bit loading, chat templates, decoding settings; revisions pinned on Kaggle before the main run.
- Seeds fixed and recorded; pilot and main topics kept apart; B1 folds grouped by topic.
- DOI resolution through CrossRef and DataCite (D20); accent and multi-copy fixes in B2 checks (D22).
- Blind labelling with a written rubric; B1 uses string features only.
- B2 threshold 0.85 and B0 min_flags 1 chosen on the pilot (D16, D21b).

## Uncertain (reported as limitations)
- Reference labels are AI-made and share sources with B2 (D24, D27): B2's agreement is an upper bound.
- Real papers missing from CrossRef/OpenAlex would be called fabricated by B2.
- Topic prominence uses keyword search counts.

## Discrepancies with the planned protocol
- RQ1 reproduction dropped (D08); main run reduced to 6 topics (D25); 1,024-token limit kept (D25).
