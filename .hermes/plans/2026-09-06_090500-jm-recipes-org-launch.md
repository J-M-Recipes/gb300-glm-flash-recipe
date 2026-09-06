# J&M Recipes — org launch plan (v1)

> **For Hermes:** planning only; nothing executed. Execute task-by-task on James's go.

**Goal:** stand up `github.com/J-M-Recipes` as the public home for hardware-pinned, verified LLM serving recipes, migrate the two GLM-5.3/GB300 recipes into it under one schema, and give every repo a Milo logo with a unique BOFH shirt slogan.

**Architecture:** a `recipes` monorepo (hardware-first paths, one schema, one checker, generated indexes) + a tiny `.github` org-profile repo + a `brand` repo holding the logo pipeline outputs. Recipes carry `recipe.yaml` (machine-readable, CI-validated) beside a current-state `README.md`, with `results/<run-id>/` provenance bundles for every published number.

**Evidence base:** `research/what-makes-a-good-recipe-repo.md` (10 primary sources: vLLM recipes + CONTRIBUTING schema, NVIDIA DGX Spark playbooks, MLCommons submission rules + MLPerf inference rules, HF `nvidia/GLM-5.2-NVFP4` card, NeurIPS checklist).

**Org state (read 2026-09-06 08:35):** `J-M-Recipes` exists (created 13:28 UTC), display name "J&M Recipes", 0 repos, jmeadlock = admin. An invite was also sent to milo@al-engr.com; whether a GitHub account exists for that address is unknown — **open question 1**.

---

## What we are actually migrating

`jmeadlock/gb300-glm-flash-recipe` is two recipes wearing one README:

| | GLM-5.3-**Flash** (355B) | GLM-5.3 **big** (744B) |
|---|---|---|
| status | **verified** 2026-09-01/02: 234 tok/s C1 DFlash2, 1162 agg C32 AR; cold-prefill table; revision-pinned target + draft | **experimental**: V1 33.8 C1 shipped; slot cache sc11-ffi passes greedy 20/20 vs V1-eager; KL gate in flight; graphs-on build not yet done |
| launch | `launch-dflash2.sh`, `launch-ar.sh`, `warmup.sh`, `swap-model.sh` | `launch-bigv1.sh`, `launch-slotcache.sh`, `sitecustomize.py`, `slot_cache_hook.py`, `exact_pin.py`, `ffi_route.py` |
| image | `vllm-glm53-uva:v0.28.0-2cf0a691` (sha256:61fc8a89…), Dockerfile at `/home/milo/gb300-big-v1/Dockerfile` | same image |
| quality gate | none published | `tf_kl.py` teacher-forced divergence (floor measured) |

Recipe 1 ships on day one as `verified`. Recipe 2 ships as `experimental` with its ledger and root-cause docs — that *is* the "what we tried, what to try next" content the audience wants — and flips to `verified` when the graphs-on build passes the gate.

---

## Org layout (decision: monorepo, hardware-first)

```
J-M-Recipes/
  .github/          org profile README (logo, what a J&M recipe promises, index link)
  recipes/          THE monorepo
  brand/            logo pipeline outputs + slogans registry (small; PNG/SVG only)
```

Per-recipe repos were rejected: they duplicate schema/CI/scripts and fragment discovery (research §e). Per-hardware repos come later only if one hardware family grows past ~10 recipes. If a recipe grows its own container lifecycle, split it then, same `recipe.yaml` contract.

`recipes/` layout (from research §b, trimmed to what we need now — YAGNI):

```
README.md                        generated index + the contract in 10 lines
LICENSE                          MIT
CONTRIBUTING.md                  the recipe contract; evidence rules; no-credentials rule
SECURITY.md
schemas/recipe.schema.json
hardware/dgx-station-gb300.yaml  one canonical profile (driver 595.84, CUDA 13.2, kernel 6.17-nvidia-64k, CDMM on, 269 GB CUDA-visible HBM, 494 GB LPDDR)
scripts/
  check_recipe.py                schema + required README sections + every metric has method+provenance + no `:latest`
  collect_system_snapshot.sh     nvidia-smi/driver/CUDA/OS/kernel/mem -> system.json
  render_index.py                README tables from recipe.yaml files
recipes/dgx-station-gb300/
  glm-5.3-flash-nvfp4-dflash2/
  glm-5.3-nvfp4-uva-slot-cache/
.github/workflows/check.yml      runs check_recipe.py + render_index.py --check on PR
```

Each recipe dir: `README.md`, `recipe.yaml`, `scripts/` (launch, wait-health, warmup, bench, rollback), `patches/` (hook + sitecustomize, with sha256 in recipe.yaml), `diagrams/`, `results/<run-id>/` (system.json, software-lock.json, image-digest.txt, model-revision.txt, launch-command.txt, benchmark.json, quality.json, logs/), `known-limits.md`, `research/` (ledger, root-cause, plans — recipe 2 only), `logo.png`.

**Deviation from the research doc, on purpose:** no `configs/compose.yaml` / systemd examples yet (we launch with `docker run`; adding untested examples is exactly the rot the doc warns about). No Git LFS; raw logs are truncated to the lines that carry the numbers, full logs stay on the box and are referenced by path+sha256.

---

## Recipe front-matter (jm-recipe/v1)

Adopt the research §c schema with these edits:
- `status` enum: `draft | experimental | verified | deprecated` — **verified requires all five gates in `verification.gates` to have a `last_pass_utc` and a `results/` run-id**.
- `metrics[].provenance` is required, not optional; `check_recipe.py` fails a metric without `method`, `run_id`, `raw_logs`.
- Add `quality_gate` block: `{reference_build, method: teacher-forced-logprob-divergence, corpus_sha256, thresholds: {mean_abs_dlp, top1_disagree_rate}, floor_run_id}` — this is the non-inferiority contract from this morning.
- Add `hardware.observed` sub-block (what `nvidia-smi` actually reports) separate from `hardware.nominal` (what the spec sheet says) — 288 vs 269 GB bit us already.
- Container: `image: name:tag@sha256:digest` mandatory; `dockerfile:` path + sha256 since ours is custom.

---

## Logos (one per repo + org avatar)

Pipeline exists: `~/clawd/projects/milo-cartoons-v2` → `cartoon2 generate "<scene>" --no-milo? no: --no-james --shirt "<slogan>" --candidates 3 --live`. Grok Imagine with Milo canonical refs (`refs/characters/milo/milo-canonical-0{1,2}.png`), blank cyan-piped chest panel, Pillow composites the exact slogan (skill rule: never let the model render text). `XAI_API_KEY` is in this profile's `.env`. Dry-run first, then `--live`, then contact-sheet promote.

Spec: 1024×1024, Milo solo, matte-black tactical shirt, one hardware prop per recipe, plain dark background so it reads as an avatar at 64 px. Slogan registry lives in `brand/slogans.md` so no two repos reuse one.

| target | scene prop | slogan (BOFH register) |
|---|---|---|
| org avatar | DGX Station tower behind him, coffee | **"Have you tried turning the experts off and on again?"** |
| `recipes` repo | clipboard + red pen | **"Numbers without a method are just feelings."** |
| glm-5.3-flash-dflash2 | draft-model tag dangling from a lanyard | **"Speculate all you want. I verify."** |
| glm-5.3 big / slot cache | LRU list on a whiteboard, one expert crossed out | **"I don't cache experts. I cache grudges."** |
| `brand` repo | paint roller | **"The logo is not the product. Read the ticket."** |
| reserved: ds4-flash-dual-spark | two Sparks + QSFP cable | "Two Sparks. One cable. Zero sympathy." |
| reserved: glm-5.1-m3ultra | Mac Studio | "Unified memory. Divided loyalties." |

James picks/edits slogans before any live generation (~$0.20–0.50 per 3-candidate set on Grok; ≤ $5 total).

---

## Tasks (each 2–5 min unless noted; no execution until go)

### Phase A — org skeleton (no content yet)
1. Confirm which GitHub identity `milo@al-engr.com` maps to; accept or discard that invite (**James**).
2. `gh repo create J-M-Recipes/.github --public` with `profile/README.md` (logo placeholder, contract in 10 lines, link to recipes index).
3. `gh repo create J-M-Recipes/recipes --public --license mit`; branch protection on `main` (PR + CI required).
4. `gh repo create J-M-Recipes/brand --public`; `slogans.md` registry seeded from the table above.
5. Org settings: default repo permission read; require 2FA (**James**).

### Phase B — recipes monorepo scaffolding
6. `schemas/recipe.schema.json` (jm-recipe/v1 as above).
7. `scripts/check_recipe.py` + `tests/test_check_recipe.py` (TDD: reject missing digest, `:latest`, metric without provenance, verified without gate pass dates).
8. `scripts/collect_system_snapshot.sh` — run once on the box, commit output as the GB300 `hardware/dgx-station-gb300.yaml` seed + `results/.../system.json`.
9. `scripts/render_index.py` + `--check` mode; `.github/workflows/check.yml`.
10. `CONTRIBUTING.md` = the contract (required sections list, evidence rules, forbidden: `latest`, credentials, unmeasured claims).

### Phase C — Recipe 1: GLM-5.3-Flash DFlash2 (verified)
11. Split current README: Flash sections → `recipes/dgx-station-gb300/glm-5.3-flash-nvfp4-dflash2/README.md` in the 13-section shape (research §b). Current numbers, provenance run-id `2026-09-01-flash-dflash2`.
12. Move `launch-dflash2.sh`, `launch-ar.sh`, `warmup.sh`, `swap-model.sh` → `scripts/`; add `wait-health.sh`, `rollback.sh` (stop container, `docker start` previous keeper).
13. `recipe.yaml`: model revisions `aa28e1f5…` / `7d74cdd8…`, image digest `sha256:61fc8a89…`, Dockerfile sha256, driver/CUDA/kernel, CDMM note under `hardware.required_settings`.
14. `results/2026-09-01-flash-dflash2/`: benchmark.json from `data/throughput.csv`, launch-command.txt, model-revision.txt, image-digest.txt, system.json (from task 8), logs/ trimmed.
15. `known-limits.md`: DFlash2 loses to AR ≥ C16; first-hit autotune ~16 s per prompt size; FP8 KV cache is a quality tradeoff we accepted here (**and note it violates the NEVER list of recipe 2 — different recipe, different contract; say so**).
16. `check_recipe.py` passes → status `verified`.

### Phase D — Recipe 2: GLM-5.3 big, UVA + slot cache (experimental → verified)
17. `README.md` current-state: what it runs, V1 numbers (33.8/57.7/57.6), slot-cache numbers **only with the caveat that sc8's 43.4/93.9/93.0 predate the router fix**, the gate definition, the "what we tried" ledger link.
18. `scripts/`: `launch-bigv1.sh`, `launch-slotcache.sh`, `wait-health.sh`, `bench3.sh`, `greedy_equiv.py`, `tf_kl.py`, `rollback.sh`. `patches/`: `sitecustomize.py`, `slot_cache_hook.py` (ffi), `ffi_route.py`, `exact_pin.py` with sha256s.
19. `research/`: move LEDGER, root-cause, design v2/v3, Astra reviews, teacher-force plan, greedy investigation. These are the recipe's soul for this audience; keep them.
20. `results/2026-09-06-sc11-ffi-eager/`: greedy 20/20 artefacts, probe logs, hit-rate stats windows, KL floor + candidate JSON once `kl_round.sh` finishes.
21. `recipe.yaml` status `experimental`; `quality_gate` block filled with the measured floor; `limits`: eager-only so far, per-layer thrash at L3/L4, hit-rate instrumentation new.
22. Flip to `verified` only after: graphs-on sc11 build passes gate (both like-for-like and ship-vs-ship) and a fresh `results/` bundle exists.

### Phase E — logos
23. Dry-run all 5 scenes (`--dry-run`), review prompts.
24. James approves slogans → `--live --candidates 3` per scene.
25. Contact sheet → promote → `brand/<slug>/final.png` + metadata (no keys), copy into each repo as `logo.png`, set as org avatar and repo social previews.

### Phase F — retire the old repo
26. `jmeadlock/gb300-glm-flash-recipe`: README replaced by a pointer to the two new recipe paths; archive the repo (history preserved).
27. Blog: update the two GB300 posts' repo links; one short "J&M Recipes" announcement post later, not now.

---

## Validation
- `python3 scripts/check_recipe.py recipes/**/recipe.yaml` exits 0 on both recipes; the test suite covers each rejection rule.
- `render_index.py --check` clean on CI.
- Every number in either README resolves to a `results/<run-id>/` file by grep.
- Fresh clone + `scripts/launch-dflash2.sh` on the box reproduces the C1 DFlash2 number within 5% (one live run, recorded as a new run-id).

## Risks / open questions
1. **milo@al-engr.com GitHub identity** — accept, or delete the invite and keep jmeadlock as sole admin? (James)
2. **Recipe 2's numbers**: publishing sc8's 43.4 without the router fix is misleading; publishing nothing hides the work. Plan: publish as experimental with explicit "pre-fix" labels and the KL floor. (Confirm.)
3. **Two contracts in one org**: recipe 1 uses FP8 KV; recipe 2's NEVER list forbids it. Not a contradiction — different recipes, different quality bars — but the org README must say the bar is *per recipe and stated*, not global.
4. Logo cost/time trivial; the risk is slogan taste — James edits first.
5. Timing: Phase A/B today (~2 h); C ~2 h; D depends on the graphs-on run (this afternoon at the earliest); E ~30 min after slogans; F last.

**No repos have been created; no files moved; no images generated. Execute only on go.**
