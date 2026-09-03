# Baseline environments

Each optional learned baseline lives in its own environment and communicates
with TrajSimBench through the file protocol in
[`docs/baseline-adapter-protocol.md`](../docs/baseline-adapter-protocol.md).

The CPU MVP includes only `fake`, a deterministic protocol fixture. Named
learned methods are intentionally not installed or trained until their source,
weights, data assumptions, and license are audited.

