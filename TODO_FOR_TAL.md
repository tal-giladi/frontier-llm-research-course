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

1. DONE (2026-10-04): public repo https://github.com/tal-giladi/frontier-llm-research-course created and pushed.
2. (Changed 2026-10-08, after the single notebook ran out of compute units.) The pilots are now 21 short notebooks in `curriculum/pilots/notebooks/`, numbered in priority order, one per Colab session. Open the next unfinished one (runtime type in its first heading: A100, except 12 and 13 on T4 and 18 on L4) and run all cells. Rerunning a notebook, or running it again after a disconnect or in a new month, skips every command that already finished, including what your first run left in Drive (the `p1/` runs resume where they stopped). Start with `01_P1.ipynb` (about 4.5 A100-hours in total; it can be spread over several sessions).
   `00_status.ipynb` runs on a free CPU runtime: it shows which pilots are complete and writes `frontier-llm-pilots-results.zip` to Drive.
3. After each session (or once a month), run `00_status.ipynb`, download `frontier-llm-pilots-results.zip` and put it in `curriculum/pilots/incoming/`, so I can fill `curriculum/pilots/RESULTS.md` and replace PROJECTED figures.

## Decision (2026-10-07)

9. DONE (2026-10-07): Stage D base model for Modules 12–16: proposed Qwen3-1.7B-Base (revision `ea980cb`, Apache-2.0, same layout as Baseline-0); alternative OLMo-2-0425-1B (fully open data). Rejected Qwen3.5-2B-Base (hybrid attention + vision). Tal confirmed Qwen3 main, OLMo-2 documented alternative for contamination-sensitive lessons.
10. DONE (2026-10-07, main session, change if you disagree): Module 16 reward-hacking material describes the published harness exploits (AlwaysEqual, sys.exit(0), conftest report rewriting, editing/reading tests) in prose only; runnable code covers only toy loopholes in the course's own verifiers (visible-test hardcoding, format/length rewards). A safety classifier stopped the first draft that shipped working harness bypasses, and fired again on the resume; Module 16 was relaunched with a narrower brief: no exploit code at all, published hacks as reading only, hands-on work on deliberately misspecified toy rewards (format, length, visible-pairs-only verifier).

## Later (2026-10-09)

11. TODO: write lesson 02.5 · FlashAttention-2, 3 and 4 (`lessons/module-02/lesson-05.md` is an empty placeholder in the sidebar, no quiz yet, so the dry-run import will flag it until it is written). Scope in `plan.md` Module 2.
