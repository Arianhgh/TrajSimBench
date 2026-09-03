# ADR-0002: CPU-first MVP and gated external claims

## Status

Accepted for the initial implementation.

## Decision

The repository ships a complete, deterministic CPU MVP. It implements the
canonical data contract, seven classical measures, task and perturbation
semantics, exact retrieval, evaluation, storage, orchestration, analysis, and
dashboard interfaces. It does not perform large-scale training or require a
GPU. Learned methods are integrated through a strict external adapter protocol
and a deterministic fake adapter used for contract tests.

## Rationale

The available system has no GPU, while training or tuning named learned
baselines would make results irreproducible and would exceed the requested
scope. The benchmark therefore makes compute and missing-data boundaries
explicit in manifests and failure tables. Classical methods and Tiny synthetic
fixtures provide a runnable regression path without making unsupported
scientific claims.

## Consequences

Porto and GeoLife are user-supplied data contracts rather than guessed
downloads. Germany is disabled until its exact source and license are approved.
ANN, cross-domain, application, and named learned-baseline tracks can be added
behind the existing schemas and adapter protocol without changing the MVP
artifacts.
