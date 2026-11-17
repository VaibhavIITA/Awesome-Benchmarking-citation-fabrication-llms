# Labelling rubric (gold labels, T21 Version A)

Vaibhav Sharma, MSE2026010. Fixed before any pilot label was written (17 Nov 2026).

For each row: search the title in Google Scholar inside quotes; if a DOI is given, open https://doi.org/<DOI>. Spend at most about 3 minutes per row.

| Label | Rule |
|---|---|
| exists | A real publication matches the title (small wording or case differences allowed), and first author, year (within 1) and DOI (if given) are correct |
| metadata_error | The publication is real, but at least one of: different first author, year off by more than 1, DOI fails or opens a different work, clearly shortened or altered title |
| fabricated | No real publication with this title or a close version exists, even if the authors or venue are real |
| unresolvable | Cannot decide in about 3 minutes (too vague, cut off, inaccessible) |

Notes column: write "truncated", "title altered", "real authors, fake title", "duplicate" or anything unclear.
Rules: the labeller does not see the model name or the B2 result; keys are not edited; rows are not added or removed.
Round 2 (for Cohen's kappa): a random 20% of the main gold set is labelled again at least 3 days later without looking at round 1.
