# AI assistance log (Brief Sec. 10)

Vaibhav Sharma, MSE2026010. Tool: Claude (Anthropic), chat interface.

| Prompt | Date | What the AI produced | What I checked or changed | Status |
|---|---|---|---|---|
| 01 to 05 | 2026-11-12 to 14 | Scoping, literature table, gap, design | Reviewed and marked verified | Verified by me |
| 06 to 07 | 2026-11-14 | Resource audit, frozen protocol draft | Reviewed; checklist left open by design | Verified by me |
| 08 | 2026-11-17 | env_test.py, notebook, seed.py | I ran it on Colab; real outputs in results/env_report.json, requirements.txt | Verified by run |
| D08 | 2026-11-17 | Change notice text | Decision mine, after Naser data returned 401/404 | Verified by me |
| 09 | 2026-11-17 | b0_rules.py, b1_detector.py, analyse_a.py, export_labels.py, tests, pilot notebook; edits to generate.py and verify.py | I ran the pilot on Colab (17 Nov 2026). Real output exposed 2 parser bugs and a DataCite gap; fixed (33 tests) and Cells 9 to 12 rerun. Decisions D17 to D20 mine | Verified by run |
| 10 | 2026-11-17 | Audit report; 2 further B2 fixes (accents, top candidates); 57 pilot reference labels with evidence (D24); threshold sweep and B0 check run on my pilot files | I reviewed the audit and all 57 labels (no changes) and confirmed D16, D18, D19, D21b, D24 | Verified by me; Colab checks carried forward |
| 11, 12 | 2026-11-17 | Debugging write-up; pathway comparison and test plan | Pending my review; D26 pending | Pending |
| 13 | 2026-11-17 | Kaggle notebook (parallel GPUs, revision pinning, freezing); 108 reference labels (D27); analysis additions (Wilson CIs, specificity, trivial reference, B1 ablation, token-limit count); results, tables, figures | I ran the main run on Kaggle; labels and analysis by AI on my files | Pending my review |
| 14, 15 | 2026-11-17 | Interpretation outline; paper skeleton, then the full draft text of the paper (6 pages) from the logged evidence | I review and revise the draft and record my changes in the paper's disclosure | Pending my revision |
