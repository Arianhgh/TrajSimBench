# Reviewable Paper Baselines

These folders contain isolated, reviewable copies of three paper-code
implementations and their smoke tests. They are kept outside the CPU-first
TrajSimBench package because the upstream projects have different dependency
and runtime requirements.

## Verified status

| Baseline | Status | Verified scope |
|---|---|---|
| t2vec | Passed core/model smoke test | Encoder-decoder forward pass and finite toy loss |
| TrajCL | Passed core encoder smoke test | Contrastive forward pass, logits, targets, and finite loss |
| TMN | Modernized core passed | Variable-length forward, loss, backward, finite gradients, optimizer step, retrieval sanity |

These are **not** claims of full paper reproduction. Porto/GeoLife
preprocessing, paper-scale training, checkpoint validation, and paper-table
comparison remain pending. In particular, TMN's original `Traj_Network`
training path still needs additional device, division, checkpoint, and
distance-dependency work.

## Running the checks

From the repository root:

```bash
bash baseline_envs/reviewable_smoke/t2vec/run_smoke.sh
bash baseline_envs/reviewable_smoke/trajcl/run_smoke.sh
bash baseline_envs/reviewable_smoke/tmn/run_smoke.sh
```

Expected results are recorded in each smoke folder and in the main
[smoke-test results](../../reviewable_smoke_tests/SMOKE_TEST_RESULTS.md).

## Provenance

- t2vec: `boathit/t2vec`, upstream commit `942d2faca5c9b79d806f458524bb197e75167514`
- TrajCL: `changyanchuan/TrajCL`, upstream commit `fbc98f58c8b46135bf2507cd83c9a1e96ede0ff3`
- TMN: `PeilunYang/TMN`, upstream commit `0e506b05113b7b28f6e78d2275152cd2474b8dd9`

The upstream repositories did not contain a license file in the checked-out
commits. Keep the attribution and source URLs visible, and confirm the
upstream redistribution terms before publishing these source copies in a
release archive.
