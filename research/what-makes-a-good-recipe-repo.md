# What Makes a Good Large-Model Serving Recipe Repository

Audience: J-M-Recipes maintainers and “wannabe inference engineers.”

Thesis: a good recipe is not a blog post and not a benchmark claim; it is a reproducible, hardware-pinned, version-pinned, checkable serving configuration with honest limits, measured numbers, and a rollback path.

## Executive recommendations

1. Make the recipe unit machine-readable first, human-readable second: keep a `recipe.yaml` / front-matter record beside the README, validate it in CI, and render an index from it. vLLM’s recipe repo is moving from legacy Markdown guides to structured YAML under `models/<org>/<model>.yaml`, exported as JSON for tools and rendered as an interactive command builder.[1][8]
2. Treat hardware verification as a scarce label. vLLM says only `verified` hardware is meaningful and instructs contributors not to fill hardware support with speculative guesses.[8]
3. Require exact runtime provenance: model revision, container image digest, runtime version, CUDA/driver stack, launch command, environment variables, and local patches. MLPerf’s system description metadata requires hardware and software fields such as accelerator memory, framework, other software stack, operating system, and notes.[5]
4. Require performance numbers to carry their method: benchmark command, input/output shape, request rate/concurrency, run count, warmup, timestamp, raw logs, and pass/fail threshold. vLLM’s DeepSeek guide is useful because it publishes both the `vllm bench serve` command and its output metrics, while MLPerf requires result logs such as `mlperf_log_summary.txt`, `mlperf_log_detail.txt`, accuracy logs, and `measurements.json`.[2][5][9]
5. Put verification and rollback in every recipe. NVIDIA’s DGX Spark playbook pattern includes prerequisites, time/risk, validation, cleanup/rollback, and troubleshooting; that pattern is directly reusable for model-serving recipes.[4]
6. Start J-M-Recipes as a monorepo with hardware profiles and per-recipe directories, not one repo per recipe. A monorepo gives one schema, one CI gate, shared scripts, and a generated index; split only if one recipe becomes a standalone product.

## Fetch log

Fetch budget used: 10 page fetches total.

| # | Source | Why fetched |
|---:|---|---|
| 1 | vLLM recipes README | Catalog structure, recipe framing, YAML migration. |
| 2 | vLLM DeepSeek-V3.2 recipe | Per-model recipe structure, launch flags, benchmark/accuracy examples. |
| 3 | NVIDIA DGX Spark playbooks README | Playbook collection structure. |
| 4 | NVIDIA `connect-two-sparks` playbook | Hardware pinning, risk, verification, rollback pattern. |
| 5 | MLCommons general submission rules | Required submission content, system metadata, logs. |
| 6 | MLPerf Inference rules | Replicability, SUT definition, run/result semantics. |
| 7 | Hugging Face `nvidia/GLM-5.2-NVFP4` model card | Serving model card with hardware/runtime/version/eval/limits. |
| 8 | vLLM recipes CONTRIBUTING | Authoritative vLLM recipe YAML schema and validation workflow. |
| 9 | MLCommons inference submission CLI docs | Concrete LoadGen/endpoint result tree and checker flow. |
| 10 | NeurIPS paper checklist | Reproducibility/limitations/statistical-significance rubric. |

## (a) Comparison table: source structures

| Source | Structure observed | Does well | Does badly / gaps | Transferable pattern |
|---|---|---|---|---|
| vLLM recipes README | The repo frames recipes around “How do I run model X on hardware Y for task Z?”, lists model-provider sections, and says new recipes live as structured YAML while legacy Markdown guides remain as references.[1] | Good catalog/navigation pattern; model/provider grouping makes discovery easy for users who know the model name.[1] | README is mostly an index, not an operational artifact; it delegates schema and validation details to `CONTRIBUTING.md`.[1][8] | Keep a human catalog, but make it generated from validated recipe metadata rather than hand-maintained prose. |
| vLLM per-model Markdown recipe: DeepSeek-V3.2 | Sections include introduction, dependency install, vLLM install, launch command, performance tuning, accuracy benchmarking, serving benchmark output, mode advice, tool-calling examples, and troubleshooting.[2] | Shows exact `vllm serve` flags, exact `lm_eval` commands, exact `vllm bench serve` command, and actual output metrics including throughput, TTFT, TPOT, ITL, successful/failed requests, and configured request rate.[2] | `uv pip install vllm --extra-index-url .../nightly` is not release-pinned, no container digest or OS/driver lock is given, “Some users reported” performance advice lacks run provenance, and rollback is absent.[2] | Copy the command/result style, but add lockfiles, hardware snapshots, raw logs, confidence/range, and rollback. |
| vLLM recipe YAML schema / CONTRIBUTING | One recipe is one YAML file at `models/<org>/<model>.yaml`; it has `meta`, `hardware`, `model`, install controls, architecture, variants, base args/env, dependencies, features, hardware overrides, strategy overrides, and Markdown guide content.[8] | Strong machine-readable schema; includes `min_vllm_version`, optional `docker_image`, hardware verification labels, VRAM formula, validation command, preview workflow, and generated JSON/API output.[8] | Docker examples pin image tags but not digests; the schema allows missing Docker brand keys to fall back to `:latest`, which is convenient but weak for reproducible recipes.[8] | Use schema + renderer + CI, but make image digests and verified measurement artifacts mandatory for J-M-Recipes. |
| NVIDIA DGX Spark playbooks README | The collection describes step-by-step playbooks for DGX Spark AI/ML workloads and says each playbook includes prerequisites, instructions, troubleshooting, and example code.[3] | Strong “playbook” promise: a reader expects operational steps, not just background.[3] | Collection README is only an index; it does not itself define a schema or quality bar for playbooks.[3] | Add a root README that promises a standard contract and links to a schema/checker. |
| NVIDIA `connect-two-sparks` playbook | The playbook has overview, what you accomplish, what to know, prerequisites, ancillary files, time/risk, numbered steps, expected outputs/notes, verification, cleanup/rollback, and troubleshooting.[4] | Excellent hardware pinning: it requires two DGX Spark systems, one QSFP cable, SSH/sudo access, same username, and interface validation with `ibdev2netdev`; it also names duration, risk level, rollback, and last update.[4] | Verification is mostly manual and illustrative; no machine-readable front-matter or CI-checkable pass/fail artifact is present.[4] | Every serving recipe should have `Prerequisites`, `Time & risk`, `Verify`, `Cleanup/Rollback`, and `Troubleshooting` as required sections. |
| MLCommons / MLPerf Inference rules and submission docs | MLPerf defines a system under test as a set of hardware and software resources, requires benchmark implementations to be shared, restricts non-determinism, says non-replicable results are invalid, and requires structured submission content with systems metadata, source, scripts, and result logs.[5][6] | Best-in-class artifact discipline: system description JSON, software/hardware metadata, result logs, accuracy logs, compliance outputs, measurement files, and submission checker flow.[5][9] | Full MLPerf compliance is too heavyweight for a hobbyist recipe repo; copying it wholesale would bury beginners under benchmark-suite bureaucracy.[5][9] | Borrow the artifact contract: system snapshot, software lock, measured results, raw logs, checker output, and repeatable runner scripts. |
| Hugging Face `nvidia/GLM-5.2-NVFP4` model card | Front-matter includes pipeline tag, base model, license, library, tags; body includes model overview, terms, use case, architecture, input/output, software integration, model version, evaluation datasets, inference hardware, SGLang/vLLM usage, evaluation table, and limitations.[7] | Good model identity layer: base model, MIT license, NVFP4 version, ModelOpt version, supported runtimes, Blackwell compatibility, Linux preference, B200/B300 test hardware, exact vLLM flags, evaluation settings, and limitations.[7] | Serving section uses `lmsysorg/sglang:latest` for SGLang and a vLLM tag without digest; test hardware is listed by GPU family, not full system description; evaluation numbers do not include raw harness logs.[7] | Model cards are good upstream references, but a serving recipe must add system-level reproducibility and raw measurement artifacts. |
| NeurIPS Paper Checklist | The checklist is required in NeurIPS submissions, asks yes/no/n/a with justifications and section references, and covers claims, limitations, experimental reproducibility, datasets, code/model access, experiment details, statistical significance, and compute resources.[10] | Strong honesty rubric: claims must match scope, limitations should be explicit, reproducibility can be code/data/model/checkpoint/detailed instructions, and error bars/statistical-significance methods should be explained.[10] | It is paper-oriented and does not tell an operator how to launch or roll back a server.[10] | Add a short “reproducibility checklist” to every recipe PR: claims, limits, exact replication path, statistical/method notes, compute resources. |

## (b) Recommended layout

### Recommended recipe repository layout

Use this for a J-M-Recipes monorepo:

```text
README.md
LICENSE                         # MIT unless a recipe-specific dependency forbids it
CONTRIBUTING.md                 # recipe contract, evidence rules, no-credentials rule
SECURITY.md                     # where to report leaked tokens / unsafe instructions
schemas/
  recipe.schema.json            # validates recipe.yaml/front-matter
  result.schema.json            # validates measured results
hardware/
  dgx-station-gb300.yaml        # canonical hardware profile
  dgx-spark-gb10.yaml
scripts/
  check-recipe.py               # schema + required sections + citation/link checks
  collect-system-snapshot.sh    # nvidia-smi, driver, CUDA, OS, CPU, RAM, disk
  render-index.py               # generates model/hardware/runtime indexes
recipes/
  dgx-station-gb300/
    glm-5.3-flash-nvfp4-vllm-uva-slot-cache/
      README.md                 # current-state human recipe
      recipe.yaml               # machine-readable front-matter/body source of truth
      diagrams/
      configs/
        vllm.env.example
        compose.yaml
        systemd.service.example
      scripts/
        launch.sh
        wait-health.sh
        verify-quality.py
        bench-vllm.sh
        rollback.sh
      patches/
        slot-cache-hook.patch
      results/
        2026-09-06-gb300-vllm028-slot-cache/
          system.json
          software-lock.json
          image-digest.txt
          model-revision.txt
          launch-command.txt
          quality.json
          benchmark.json
          logs/
            server.log
            bench-1stream.json
            bench-8stream.json
            quality.log
      known-limits.md
indexes/
  by-hardware.md                # generated
  by-model.md                   # generated
  by-runtime.md                 # generated
```

Why this shape:

- vLLM’s move to structured YAML plus generated JSON/API output is the right direction for agent-readable recipes.[8]
- NVIDIA’s playbook shape proves beginners need prerequisites, time/risk, validation, rollback, and troubleshooting in the same artifact.[4]
- MLPerf’s submission discipline shows why a recipe should retain system metadata, software metadata, measured results, and logs rather than only publishing a README number.[5][9]

### Recommended individual recipe layout

Each recipe README should be current-state, not an ops diary:

1. **What this runs**: model, quantization, runtime, hardware, target workload, status, and who should not use it.
2. **Verified hardware**: exact machine class, GPU/accelerator, HBM, host RAM, interconnect/UVA assumption, storage, OS, driver, CUDA, and any BIOS/kernel settings.
3. **Exact software lock**: container image name, tag, digest, runtime version, Python package versions, local patch commit, model revision/SHA, and config file hashes.
4. **Prerequisites and risk**: required accounts/download rights, disk/RAM space, expected setup time, destructive actions, and rollback summary.
5. **Launch**: exact command plus environment file; no “roughly like this” command.
6. **Health verification**: readiness command, expected response, timeout, and failure triage.
7. **Quality gate**: teacher-forced logprob divergence, benchmark/eval prompt suite, thresholds, reference build, and fail-open/fail-closed behavior.
8. **Performance method**: benchmark command, dataset/request shape, stream/concurrency levels, warmup, run count, aggregation method, raw result path, and expected range.
9. **Expected numbers**: table of measured metrics with date, run id, hardware, runtime, and method link.
10. **Known limits**: unsupported contexts, memory pressure, concurrency cliff, quality tradeoffs, unsafe flags, and unverified claims.
11. **Troubleshooting**: symptoms, probable causes, cheapest next checks, and known upstream issues.
12. **Rollback / cleanup**: command to stop service, restore previous image/config/patch, and remove optional large caches.
13. **Changelog**: only material current-state changes, not play-by-play.

### Required fields

Minimum hard requirements for a J-M-Recipes recipe:

- **Hardware pin**: exact product name, accelerator count, accelerator memory, host memory, interconnect, storage, OS, driver/CUDA, and hardware feature assumptions.
- **Container image digest**: `name:tag@sha256:...`; if no container is used, record wheel hashes and package lock instead.
- **Exact launch command**: command, environment variables, config files, ports, working directory, and expected model download path.
- **Expected numbers with method/provenance**: every number needs command, workload shape, timestamp, system snapshot, raw logs, and aggregation method.
- **Verification gates**: schema check, model revision check, container digest check, server health, quality gate, performance floor, and logs saved.
- **Known limits**: tested/not-tested boundaries, hardware-specific caveats, accuracy/performance tradeoffs, and security constraints.
- **Rollback**: exact stop/restore commands and previous-known-good image/config references.

## (c) YAML recipe front-matter schema proposal

This is a YAML data shape, not a full JSON Schema. CI should validate required keys, enum values, date formats, command existence, local file references, and that any metric with `value` has `method` and `provenance`.

```yaml
schema_version: jm-recipe/v1
recipe:
  id: gb300-glm-5-3-flash-nvfp4-vllm-uva-slot-cache
  title: GLM-5.3 Flash NVFP4 on one NVIDIA DGX Station GB300 with vLLM UVA slot cache
  status: verified            # draft | experimental | verified | deprecated
  difficulty: advanced        # beginner | intermediate | advanced
  audience: wannabe inference engineers
  license: MIT
  maintainers:
    - github: jmeadlock
  created_utc: 2026-09-06
  last_verified_utc: 2026-09-06
  summary: Reproducible vLLM launch and verification path for GLM-5.3 Flash NVFP4 on a single DGX Station GB300.

model:
  model_id: zai-org/GLM-5.3-Flash-NVFP4
  model_revision: REPLACE_WITH_HF_COMMIT_SHA
  base_model_id: zai-org/GLM-5.3
  architecture: moe
  precision: nvfp4
  total_parameters: 744B
  active_parameters: REPLACE_WITH_ACTIVE_PARAMS
  context_length_tokens: REPLACE_WITH_TESTED_CONTEXT
  license: MIT
  trust_remote_code: true

hardware:
  profile_id: dgx-station-gb300
  system_name: NVIDIA DGX Station GB300
  node_count: 1
  accelerators_per_node: 1
  accelerator_model: NVIDIA GB300 Grace Blackwell Ultra Superchip
  accelerator_memory_gb: 288
  host_memory_gb: 496
  memory_topology: Grace LPDDR5X plus GPU HBM over NVLink-C2C/UVA
  required_features:
    - blackwell
    - uva
    - nvlink-c2c
  verification_commands:
    - nvidia-smi
    - python scripts/check-uva.py

software:
  runtime: vllm
  runtime_version: 0.28.0
  runtime_commit: REPLACE_WITH_COMMIT_SHA_IF_PATCHED
  container:
    image: vllm/vllm-openai
    tag: v0.28.0
    digest: sha256:REPLACE_WITH_DIGEST
  cuda_version: REPLACE_WITH_CUDA_VERSION
  driver_version: REPLACE_WITH_DRIVER_VERSION
  os: REPLACE_WITH_OS_IMAGE
  python_packages_lock: results/latest/software-lock.json
  local_patches:
    - path: patches/slot-cache-hook.patch
      sha256: REPLACE_WITH_PATCH_SHA256

launch:
  working_directory: RECIPE_ROOT
  env_file: configs/vllm.env.example
  command: >-
    vllm serve zai-org/GLM-5.3-Flash-NVFP4
    --host 0.0.0.0
    --port 8000
    --trust-remote-code
    REPLACE_WITH_FULL_FLAGS
  ports:
    - 8000
  artifacts:
    model_cache: REPLACE_WITH_PATH
    server_log: results/latest/logs/server.log

verification:
  gates:
    - id: schema
      command: python scripts/check-recipe.py recipes/dgx-station-gb300/glm-5.3-flash-nvfp4-vllm-uva-slot-cache
      pass: exits_zero
    - id: container_digest
      command: docker image inspect vllm/vllm-openai:v0.28.0 --format '{{.RepoDigests}}'
      pass: matches software.container.digest
    - id: health
      command: scripts/wait-health.sh http://localhost:8000/health 900
      pass: HTTP 200 before timeout
    - id: quality
      command: python scripts/verify-quality.py --reference REPLACE --target http://localhost:8000/v1 --threshold REPLACE
      pass: teacher_forced_logprob_divergence_below_threshold
    - id: performance
      command: scripts/bench-vllm.sh
      pass: expected_metrics floors satisfied and raw logs written

metrics:
  - name: decode_single_stream_tok_s
    value: 43.0
    unit: tok/s
    direction: higher_is_better
    method: fixed prompt/output benchmark; see results/latest/benchmark.json
    provenance:
      run_id: 2026-09-06-gb300-vllm028-slot-cache
      command_file: results/latest/launch-command.txt
      raw_logs:
        - results/latest/logs/bench-1stream.json
      system_snapshot: results/latest/system.json
      software_lock: results/latest/software-lock.json
  - name: decode_4_to_8_stream_tok_s
    value: 94.0
    unit: tok/s
    direction: higher_is_better
    method: same benchmark, concurrency 4-8; see results/latest/benchmark.json
    provenance:
      run_id: 2026-09-06-gb300-vllm028-slot-cache
      raw_logs:
        - results/latest/logs/bench-8stream.json

limits:
  - Slot cache is a custom hook and must be disabled when comparing against unmodified upstream vLLM.
  - Metrics are valid only for the pinned GB300 hardware/software/model tuple.
  - Long-context and high-concurrency behavior require separate measurement before claims.

rollback:
  command: scripts/rollback.sh
  previous_good:
    container_digest: sha256:REPLACE_WITH_PREVIOUS_DIGEST
    recipe_commit: REPLACE_WITH_PREVIOUS_RECIPE_COMMIT
  cleanup_notes: Do not delete model cache unless disk recovery is required.

security:
  credentials_required: false
  forbidden_in_repo:
    - Hugging Face tokens
    - SSH keys
    - cloud credentials
```

## (d) Rot pitfalls with evidence

| Pitfall | Evidence from primary sources | Rule for J-M-Recipes |
|---|---|---|
| Unpinned runtime channels rot quickly. | vLLM’s DeepSeek recipe installs vLLM from the nightly wheel index without pinning an exact nightly build, and the HF GLM card uses `lmsysorg/sglang:latest` for one serving path.[2][7] | Ban bare `latest` and unpinned nightly commands in verified recipes; require image digest or wheel hash. |
| Tags are better than `latest` but still not enough. | vLLM’s schema includes `min_vllm_version` and `docker_image`, but examples are tag-based and missing Docker keys can fall back to `:latest`.[8] | Store `image:tag@sha256:digest`; validate digest in CI and in the recipe’s local verification gate. |
| Hardware support claims become folklore when not backed by end-to-end tests. | vLLM’s CONTRIBUTING says only `verified` is meaningful and tells authors not to fill hardware lists with guesses.[8] | A hardware profile can be “supported” only after health, quality, and performance gates pass on that exact profile. |
| Numbers without method are marketing, not recipes. | vLLM’s DeepSeek guide is more useful because it publishes benchmark/eval commands and outputs; MLPerf submission docs require raw logs/summaries and measurement files, and NeurIPS asks for experimental reproducibility and error-bar/statistical-significance details.[2][9][10] | Every metric row must include method, run id, raw logs, system snapshot, and aggregation rule. |
| Missing software/hardware metadata makes a result unreproducible. | MLPerf defines the SUT as hardware plus software resources and requires system metadata including host, accelerator, memory, framework, software stack, and OS fields.[5][6] | Capture `system.json` and `software-lock.json` for every measured run. |
| No verification gate leaves readers with a command, not confidence. | NVIDIA’s playbook includes explicit verification steps, and MLCommons docs run a submission generator with `--run_checker=yes` before upload.[4][9] | Every recipe must include `scripts/wait-health.sh`, `scripts/verify-quality.py`, `scripts/bench-vllm.sh`, and CI schema checks. |
| No rollback path turns a recipe into a trap. | NVIDIA’s playbook states rollback in its time/risk block and gives cleanup commands to remove netplan configs or IP assignments.[4] | Every recipe must include a safe stop/restore path and previous-known-good references. |
| Evaluation claims drift when limitations are hidden. | The HF GLM card includes a model-limitation section, and the NeurIPS checklist explicitly asks authors to state assumptions, scope, limitations, and factors that influence performance.[7][10] | Require `known-limits.md` and a `limits` list in front matter. |
| Generated artifacts rot if hand-edited. | vLLM’s CONTRIBUTING says recipes render to static JSON and tells contributors not to stage generated `public/` output.[8] | Keep generated indexes/results summaries separate; CI regenerates and checks them. |

## (e) J-M-Recipes org-layout options

### Option 1: one repository per recipe

Example: `J-M-Recipes/gb300-glm-5-3-flash-nvfp4-vllm-slot-cache`.

Pros:

- Clean project boundary, issue tracker, releases, and archive policy per recipe.
- Easy for a user to clone only the exact recipe they need.
- Strong if a recipe includes large patches, custom containers, or its own lifecycle.

Cons:

- Duplicates schema, scripts, templates, CI, and docs across repos.
- Discovery fragments quickly once there are many models/hardware targets.
- Harder to enforce a single evidence bar or generate cross-recipe indexes.

Best fit: a flagship recipe that becomes a maintained product or needs independent release cadence.

### Option 2: monorepo with per-recipe directories

Example: `J-M-Recipes/recipes`, with `recipes/dgx-station-gb300/glm-5.3-flash-nvfp4-vllm-uva-slot-cache/`.

Pros:

- One schema, one checker, one template, one CI contract, one generated index.
- Shared hardware profiles prevent copy/paste drift for GB300, GB10, H200, B200, etc.
- Easy to compare recipes by hardware, model, runtime, and verification status.
- Best match for vLLM’s structured recipe catalog and MLPerf-style artifact discipline.[8][9]

Cons:

- Per-recipe release tags and issues need labels/conventions.
- A bad large artifact policy can bloat the repo; raw logs may need Git LFS or external releases.
- Contributors must learn the repo schema before adding a tiny recipe.

Best fit: first 10-50 recipes, shared maintainership, and agent-readable indexes.

### Option 3: one repository per hardware family

Examples: `J-M-Recipes/dgx-station-gb300`, `J-M-Recipes/dgx-spark-gb10`, `J-M-Recipes/h200-cluster`.

Pros:

- Matches how many readers search: “what can I run on this box?”
- Shared hardware bring-up, system snapshots, power/network caveats, and rollback patterns stay close to recipes.
- Good for hardware with unusual memory topology such as GB300 Grace/HBM/UVA.

Cons:

- Model-specific fixes duplicate across hardware repos.
- Cross-hardware comparison and model index generation get harder.
- Hardware repos can become mixed piles of model recipes, driver notes, and local ops unless schema is strict.

Best fit: when the org has many recipes per hardware target and the hardware itself is the product surface.

### Recommendation

Use **Option 2: a monorepo**, but make hardware the first path segment inside `recipes/`:

```text
J-M-Recipes/recipes/
  recipes/dgx-station-gb300/glm-5.3-flash-nvfp4-vllm-uva-slot-cache/
  hardware/dgx-station-gb300.yaml
  indexes/by-hardware.md
  indexes/by-model.md
```

Reason: J-M-Recipes is young, the first recipe has a lot of reusable machinery, and the audience needs consistency more than repository purity. A monorepo lets Milo and contributors enforce schema, citations, digest pinning, verification gates, and result provenance once. If the GLM-5.3 GB300 slot-cache recipe becomes a standalone artifact with its own custom runtime/container lifecycle, split it later from the monorepo using the same `recipe.yaml` contract.

## Acceptance checklist for the first GLM-5.3 GB300 recipe

Before marking it `verified`, require:

- `recipe.yaml` validates against `schemas/recipe.schema.json`.
- Container image digest is present and `docker image inspect` confirms it.
- Model revision/SHA is present.
- `scripts/launch.sh` contains the same command as the README.
- `scripts/wait-health.sh` reaches the vLLM health endpoint.
- Teacher-forced logprob divergence quality gate passes against the named reference build.
- Benchmark result files include single-stream and 4-8 stream runs, raw logs, and method metadata.
- Expected numbers table cites run ids and raw logs.
- `known-limits.md` states that the hot-expert slot cache is custom and that claims apply only to the pinned GB300 system.
- `scripts/rollback.sh` restores the previous known-good container/config/patch state.

## Sources

[1] https://raw.githubusercontent.com/vllm-project/recipes/main/README.md
[2] https://raw.githubusercontent.com/vllm-project/recipes/main/DeepSeek/DeepSeek-V3_2.md
[3] https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/main/README.md
[4] https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/main/nvidia/connect-two-sparks/README.md
[5] https://raw.githubusercontent.com/mlcommons/policies/master/submission_rules.adoc
[6] https://raw.githubusercontent.com/mlperf/inference_policies/master/inference_rules.adoc
[7] https://huggingface.co/nvidia/GLM-5.2-NVFP4/raw/main/README.md
[8] https://raw.githubusercontent.com/vllm-project/recipes/main/CONTRIBUTING.md
[9] https://docs.mlcommons.org/inference/submission/submission-cli
[10] https://neurips.cc/public/guides/PaperChecklist
