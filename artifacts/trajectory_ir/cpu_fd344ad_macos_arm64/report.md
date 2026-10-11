# UniRL trajectory-IR CPU evidence

- Run ID: `cpu_fd344ad_macos_arm64`
- UniRL commit: `fd344ad6036a0ba7be9900a9d0cf78ebf59fa738`
- Checkout clean: `True`
- Python: `3.12.13`
- PyTorch: `2.11.0`
- Platform: `macOS-27.0.1-arm64-arm-64bit`
- Timed device: `CPU` (1 thread)
- Protocol: 10 warm-up + 100 measured repetitions

## Correctness

All 10 executable checks passed. The process aborts on the first failed assertion.

- PASS — `cross_stage_lineage_validation`
- PASS — `split_concat_exact_roundtrip`
- PASS — `whole_tree_select_and_slice`
- PASS — `reward_propagation_and_group_advantages`
- PASS — `invalid_lineage_rejected`
- PASS — `primitive_modality_mismatch_rejected`
- PASS — `primitive_batch_mismatch_rejected`
- PASS — `packed_variable_length_roundtrip`
- PASS — `tensorref_view_selection_and_materialization`
- PASS — `tensorref_bounds_rejected`

## Timing summary

| Root groups | Leaf rows | Operation | Median (us) | p90 (us) |
|---:|---:|---|---:|---:|
| 8 | 64 | `sample.split` | 457.8 | 510.8 |
| 8 | 64 | `sample.split_concat` | 120.8 | 142.6 |
| 8 | 64 | `sample.select_half` | 556.8 | 616.7 |
| 8 | 64 | `sample.propagate_rewards` | 20.6 | 23.5 |
| 8 | 64 | `tensorref.select_contiguous` | 1.1 | 1.2 |
| 8 | 64 | `tensorref.select_strided` | 12.3 | 13.0 |
| 32 | 256 | `sample.split` | 1859.6 | 1968.6 |
| 32 | 256 | `sample.split_concat` | 291.2 | 329.6 |
| 32 | 256 | `sample.select_half` | 2231.5 | 3126.8 |
| 32 | 256 | `sample.propagate_rewards` | 47.4 | 69.2 |
| 32 | 256 | `tensorref.select_contiguous` | 1.1 | 1.2 |
| 32 | 256 | `tensorref.select_strided` | 47.0 | 52.4 |
| 128 | 1024 | `sample.split` | 7525.6 | 7870.8 |
| 128 | 1024 | `sample.split_concat` | 1435.5 | 2231.7 |
| 128 | 1024 | `sample.select_half` | 8152.4 | 8458.7 |
| 128 | 1024 | `sample.propagate_rewards` | 137.3 | 149.8 |
| 128 | 1024 | `tensorref.select_contiguous` | 1.2 | 1.3 |
| 128 | 1024 | `tensorref.select_strided` | 202.2 | 226.4 |

## Scope

These are synthetic, single-process CPU measurements of container and reference operations. They do not measure rollout throughput, learner throughput, network transport, GPU utilization, training quality, or synchronous/asynchronous convergence, and must not be extrapolated to those claims.
