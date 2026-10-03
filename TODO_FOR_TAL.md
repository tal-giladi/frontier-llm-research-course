# TODO for Tal

Plan only so far (`plan.md`, 2026-10-03). Nothing built, no git repo yet.

Decisions needed before building (details in `plan.md` section 13):
1. DONE (2026-10-03): name "Frontier LLM Research Engineering", repo `frontier-llm-research-course`, slug `frontier-llm-research`.
2. DONE (2026-10-03): free.
3. DONE (2026-10-03): every lab has free CPU and free Colab GPU options; the main path runs on rented GPUs (production grade).
4. DONE (2026-10-03): one course.
5. DONE (2026-10-03): yes, `PUBLISHING_WARNING.md` added.

Plan revision 2 (2026-10-03, after the external review) — new decisions needed (`plan.md` section 15):
6. DONE (2026-10-03): accepted the per-learner main-path budget (~650–1,150 H100-hours, ~USD 1,300–3,500).
7. DONE (2026-10-03): no funding; pilots run as scaled pilots on the paid Colab account, projected to the main path.
7b. DONE (2026-10-03): host pilot traces and checkpoints on a free Hugging Face account.
8. DONE (2026-10-03): multimodal elective (Module 21) comes after the first release.

## To run the pilots (2026-10-04)

1. Create the GitHub repo `tal-giladi/frontier-llm-research-course` (public is fine) so I can push, or tell me to create it with `gh`. Colab clones the code from there.
2. Open `curriculum/pilots/pilot_phase0.ipynb` in Colab (A100 runtime), run it top to bottom. P1 first; the rest in order while compute units last. Outputs land in Google Drive `frontier-llm-pilots/`.
3. Share the `frontier-llm-pilots/summary/` folder and the `p*/` text outputs (or paste them) so I can fill `curriculum/pilots/RESULTS.md` and replace PROJECTED figures.
