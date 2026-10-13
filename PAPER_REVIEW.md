# Paper Review and GPU Handover Audit

Reviewed: 2026-09-05. UniRL source: `f5d710406b215bb7a0b387fdd37e4d4778b92338`.
This is a substantive source-and-protocol review, not evidence of new training.
No GPU experiments were run during this review. Historical CPU artifacts were
kept separate; source-derived counterexamples below were not newly executed.

## 1. Verdict and logical chain

The defensible paper is about **retaining trajectory semantics across model
families and execution choices**, not an algorithm catalogue or another claim
that asynchronous/multimodal RL exists. Two ideas carry the argument: a typed,
lineage-preserving trajectory boundary and explicit execution contracts around it.

The evidence chain is:

1. AR tokens, sparse diffusion replay and cross-stage ancestry require different
   state, even when the outer feedback loop looks identical.
2. A sequence of typed frontiers records those distinctions and the parent keys
   needed for grouping; whole-root movement preserves an established ordering.
3. Logical roles consume that boundary while placement, transport and publication
   select a physical execution, subject to explicit supported-path constraints.
4. E1/E2 establish that the exercised paths learn; a mathematically aligned,
   full-budget reference is additionally necessary for reproduction.
5. E4a checks lineage/grouping against an ID-based oracle; E4b studies a genuine
   cross-stage objective. A quality difference is not an IR-overhead measurement.
6. E5-R prices the representation on identical payloads; E5-T measures two
   deployment regimes. E3 supplies an external native-recipe systems comparison,
   with algorithm differences disclosed rather than attributed to framework cost.
7. E6 is a P1 operating-region study. It does not carry an independent novelty
   claim, and its absence cannot be hidden behind a central async contribution.

The revised prose now follows this chain. A publishable empirical conclusion
still depends on the GPU evidence; this review does not close that obligation.

## 2. Material findings and disposition

Paths below are relative to the audited UniRL checkout. **Documentation fixed**
does not mean the underlying missing validation or instrumentation was implemented.

| Finding | Source / reasoning | Revision and remaining GPU gate |
|---|---|---|
| `Sample` is not a list of individual tree nodes | `unirl/types/sample.py`, `Part`, `Sample` | Describe a forest encoded by batched frontiers; align abstract, Section 3 and diagrams |
| Parent adjacency is weaker than the promised structural contract | `Sample.__post_init__`, `Sample.concat`, `propagate_rewards`; `unirl/distributed/tensor/batch.py`, shared fields | State equal-depth, parent-contiguous, schema and shared-value preconditions; require boundary validation and an ID-join oracle in E0 |
| Mixed-version rejection is not generic `Batch.concat` behavior | Shared-field merge takes first value; async `combine_rollout_prompts` performs its own check | Attribute the guarantee to the actual assembler, not every container operation |
| Original-prompt vs rewrite grouping changes the estimator | `unirl/trainer/pe.py`, `Part.compute_advantages` | Split E4 into semantic E4a and objective E4b; compare 32 versus 8 descendants without claiming representation superiority |
| Existing E4 is not held-out integration-effort evidence | Frozen-LLM PE recipe already exists | Replace effort-reduction claim with a reuse inventory; a held-out implementation study is optional |
| SD3.5 launcher names overstate alignment | `sd3_vllmomni.yaml` uses flow/sparse early steps/global reward std; baseline launcher uses CPS/window/actor likelihood | Label E2-R/E3 native recipes; transition distribution, indices, anchors and normalization must pass mathematical and numerical parity before strict reproduction/overhead attribution |
| A 30-step speed run is not a training reference | E3 budget versus E2 training budget | Add 300-step, three-seed E2-R launch template with saves every 50; baseline export/output discovery is explicitly blocked pending its initialized checkout |
| Async logged step time omits outer-loop work | `trainer/async_ar.py`, `async_diffusion.py` emit before `async_rollout.py` publication/eval/save | Require complete driver monotonic clock plus phase timeline; do not sum overlapping role timers |
| D0/A1 changes two controls | Freshness gate and in-flight limit change together | Describe onset of allowed overlap, not a causal estimate of either knob alone |
| Seed CLI fields do not seed all trainable initialization | Data/generation config versus worker/model/LoRA construction | Require audited worker initialization and resume state; E4 AR sampling needs explicit code plumbing, not an invented Hydra flag |
| AR smoke did not actually reduce root count | AR data source reads its own `algorithm.prompts_per_rollout` | Add explicit override to 8 and verify emitted roots; do not rely only on trainer batch size |
| Internal AIME is not untouched test data | AR eval configuration and benchmark registry | Final MATH-500 is primary; fixed-budget AIME is secondary; AIME-selected best checkpoint is validation-selected; require overlap audit |
| Text serving/evaluation can change the task | `benchmarks/core/generate.py`: chat request, no request seed/thinking override, content extraction | Audit rendered token IDs, reasoning/content handling, RNG and resumed requests before scoring |
| `image/geneval2` is not automatically official GenEval2 | `benchmarks/core/registry.py`: synthetic JSONL and custom Qwen3-VL scorer | Rename paper endpoint; retain CLI registry name; official equivalence requires a separate audit |
| Image generation defaults and partial scores weaken comparability | `benchmarks/run.py`, generation resume and summary aggregation | Fix dimensions/steps/guidance/eval seed, use unique tags, archive full settings and require complete per-metric scores |
| E4 held-out outputs were not produced by the stock run | PE `eval_interval=0`; per-track checkpoint writer | Specify cached rewrites and image generation/scoring harness; adapter lives under `checkpoint-<rollout>/diffusion` |
| Diversity metric lacked a defined estimator | LPIPS choice, grouping and image-pair accounting | Prespecify 16 images, 120 pairs per prompt, prompt-level averaging and paired bootstrap; 10% decline is a study trigger, not a universal cutoff |
| Topology is not representation overhead | Separate recipe changes train DP, residency and LoRA publication | Require same-payload/same-transport E5-R; label E5-T as deployment-regime comparison |
| Handover's generic dry-run instruction could launch training | Only single-node wrapper implements `DRY_RUN`; SD3.5 scripts ignore it | Restrict `DRY_RUN` and use Hydra `--cfg job --resolve` for those scripts |
| Single-node wrapper can interrupt an existing Ray runtime | `examples/run_experiment_single_node.sh` calls `ray stop` | Warn against shared/multi-node use; require site launcher and verified allocation |
| Reward endpoint and artifact paths were underspecified | Reward-service CLI/registry; image CLI; E3 log redirection | Add service startup/readiness contract, explicit parquet output, E3 directory creation and correct image/checkpoint paths |
| A third conceptual figure repeated the experiment map | Manuscript layout and paragraph roles | Remove redundant evidence-flow diagram; retain trajectory and execution diagrams; data figures await real artifacts |

### Structural counterexample (static deduction, not a new test result)

Take roots `a,b` and child rows `a/0,b/0,a/1,b/1` with rewards `1,10,3,30`.
Every child's parent exists, but reshaping into two blocks gives parent means
`5.5,16.5`; explicit ID grouping gives `2,20`. The source's propagation routine
uses the former reshape and does not perform the latter join. Likewise,
position-wise concat uses the first input's depth and generic shared fields take
the first value. The archived ten CPU checks exercise valid, parent-major trees;
they do not establish safety for these adversarial inputs. GPU E0 must reject or
correctly canonicalize such inputs before training; a new source patch needs its
own tests and pinned artifact, not retroactive claims about the old commit.

## 3. Paragraph-role audit of the revised manuscript

Within each entry the points follow paragraph order, including named paragraphs.
Equations formalize the adjacent point; tables/figures must add a needed mapping,
mechanism or measured comparison rather than introduce an unsupported claim.

| Location | Ordered paragraph points | Logical handoff |
|---|---|---|
| Abstract | Shared loop but heterogeneous state; trajectory boundary; encoded state; execution contracts; non-interchangeability; audited scope; CPU-only evidence; open results | State thesis and evidence boundary together |
| Introduction | Concrete state mismatch; prior-system overlap; research question; IR answer; two design ideas; optional async realization and evidence order; current empirical limit | Motivate the exact question, not a catalogue |
| Section 2 | Outer-loop equation; trajectory sufficient statistics; rewrite/image running example; role heterogeneity; topology heterogeneity; requirements table; non-goals | Derive what the representation and execution must preserve |
| 3.1 | Batched-frontier definition; ancestry/operations; whole-root grouping; unvalidated preconditions; trajectory schematic | Define structure without claiming arbitrary-tree correctness |
| 3.2 | Three leaf layers; primitives; conditions; replay segments; differing lifetimes | Explain why lineage alone is insufficient for replay |
| 3.3 | Field-composition rules and dispatch reconstruction; benefit and overhead obligation | Turn typed state into a batching contract |
| 3.4 | Request-to-filled trajectory map; adapter/composed/agentic forms | Locate engine specialization behind a common boundary |
| 3.5 | Train-stack versus replay/loss responsibilities; algorithm attribution | Preserve mathematical differences instead of pretending to unify objectives |
| Section 4 opening | Architecture diagram separates program, data boundary and mechanisms | Transition from representation to physical execution |
| 4.1 | Device slabs/slots; role replicas, ranks and dispatch; relation to HybridFlow | Explain allocation without claiming generic orchestration as novelty |
| 4.2 | Tensor-reference spans and hydration; available transports; testable driver-memory consequence | Connect structural handling to movement cost |
| 4.3 | Lifecycle order; engine-specific variants/publication; invalid-combination rejection | Establish the correctness conditions for valid measurements |
| 4.4 | Synchronous ordering and fixed-work comparison | Supply the execution reference |
| 4.5 | Admission/group completeness; version/lag bound and publication; limits and observables | Explain bounded async as an implementation, not a new RL algorithm |
| 5.1 | AR prompt/group/segment/reward/replay path | Show one concrete use of all three leaf layers |
| 5.2 | Sparse diffusion path; incompatible objectives and specialized estimators | Demonstrate reuse without erasing sufficient statistics |
| 5.3 | Composed versus tool paths; frozen E4's actual use of ancestry; ID-based test | Connect mechanism to the central empirical case |
| Section 6 opening | Four RQs/artifact rules; experiment map; priorities; unknown environment fields | Set admission and scope before presenting settings |
| RQ1 opening / E1 | Learning versus reproduction; exact AR setting; reference mismatch; endpoint selection and uncertainty | Separate a run that learns from a reproduced reference |
| RQ1 E2 / image identity | Exact training recipe; long reference and parity limit; fixed judge endpoints; synthetic identity; evaluation protocol; LPIPS estimator and limitation; blank result schema | Prevent benchmark, metric or algorithm substitutions |
| RQ2 | Shape-matched native pair; process repetitions; backend and estimator caveats; optional AR baseline; full resource metrics and quality-budget limit; blank schema | Make speed interpretable within its comparison class |
| RQ3 E4 | Frozen cross-stage setting; ID oracle and normalization populations; root-intent evaluation and cached held-out generation | Test semantics separately from objective quality |
| RQ3 E5 | Controlled payload costs; explicit absent harness; tree/flat matched slice; coupled deployment pair; reuse inventory | Price the abstraction without conflating topology or engineering effort |
| RQ3 CPU evidence | Historical executed checks and small synthetic workload; generated table; interpretation and artifact location | Provide only the evidence that actually exists |
| RQ4 | Allocation/overlap design; fixed grid; stock parity problems; measurement gaps; expected directions and failure ownership; pending result panels | Bound useful async behavior without guaranteed benefit |
| Fairness/statistics | Effective work/resources; seed/process units; final endpoint; pilot target and censoring; complete wall time and complete scoring | Define valid aggregation and prevent post-hoc selection |
| Related work | LLM orchestration; asynchronous systems; multimodal systems; diffusion algorithms | Attribute overlap and keep only the tested distinction |
| Limitations | Architecture-specific integration/structural limits; manual topology; async/failure limits; missing empirical case | Carry real restrictions through to the conclusion |
| Conclusion | Re-state two design contracts; CPU versus open GPU evidence | No stronger claim than the preceding evidence |
| Appendices A-C | Source mapping; artifact schema; executed versus required correctness checks | Make mechanisms and result admission auditable |

## 4. GPU acceptance checklist

The operational source of truth is [GPU_HANDOFF_README.md](GPU_HANDOFF_README.md),
with statistical rules in [EXPERIMENT_RUNBOOK.md](EXPERIMENT_RUNBOOK.md).

- First produce per-experiment E0 readiness evidence; a valid trainer entrypoint
  does not make the full result pipeline ready.
- Capture worker seeds, resolved configs, exact source diffs and all model/data/
  scorer hashes before formal runs; prove checkpoint export and scoring on a smoke.
- Run E1/E2 and retain base/final metrics and every seed. Close at least one
  mathematically aligned full-budget reference before saying "reproduction".
- Run E4a before E4b; keep frozen AR hashes, group manifests and root-conditioned
  held-out evaluations. Ties or worse quality do not invalidate correct semantics.
- Run E3 with full resources/time accounting and native labels until true parity
  exists; run both E5-R and E5-T before claiming abstraction cost is bounded.
- E6 follows primary evidence and requires the matched estimator, complete clock,
  buffer accounting and separately piloted quality target.
- Fix engineering faults under a new recorded patch/run ID. Retain stable negative
  results; changing objective, budget, reward, evaluator or central interpretation
  requires a research decision, not silent tuning to an expected direction.
- Return artifact URIs, checksums, parsers and generated TeX/figures, not handwritten
  scores. Hardware facts, dataset statistics and final scores remain blank now.

## 5. Primary references checked for evaluation identity

- [LPIPS paper](https://arxiv.org/abs/1801.03924): learned perceptual distance; the paper's 10% investigation trigger is our protocol choice.
- [GenEval2 paper](https://arxiv.org/abs/2512.16853) and [official implementation](https://github.com/facebookresearch/GenEval2): official benchmark identity must not be inferred from a local registry key.
- [Pick-a-Pic / PickScore](https://arxiv.org/abs/2305.01569), [HPSv3](https://arxiv.org/abs/2508.03789), [ImageReward](https://arxiv.org/abs/2304.05977): distinguish optimized reward from the frozen external judge views.
- [Qwen3](https://arxiv.org/abs/2505.09388), [DAPO](https://arxiv.org/abs/2503.14476), [SD3.5-Medium model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium): cite actual model/data identities without equating DAPO data with the DAPO objective.

## 6. Validation record (completed 2026-09-06)

- All 29 Bash blocks in the handoff passed `bash -n` after substituting inert
  tokens for documented placeholders; no launch block was executed. This checks
  shell syntax, not Hydra/runtime validity or GPU compatibility.
- Manuscript citation keys, reference labels and local documentation links passed
  static consistency checks. `git diff --check` passed.
- All nine files listed in the historical CPU bundle's `checksums.sha256` passed
  SHA-256 verification; this is artifact integrity verification, not a new CPU run.
- Tectonic compiled the manuscript successfully to 19 pages; all rendered pages
  were visually inspected, including the two mechanism figures and result-schema
  tables. Final references resolved; non-blocking style warnings remain for the
  title's sans-serif small-cap substitution and microtype's equation-number patch.
- The sibling UniRL implementation was read, not modified. New GPU-side harnesses,
  instrumentation and RNG fixes in the readiness table remain unimplemented here.
