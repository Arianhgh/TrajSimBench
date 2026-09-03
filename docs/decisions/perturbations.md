# ADR: free-space perturbations and task construction

**Status:** accepted for v1  
**Scope:** perturbation, notion, task, and negative-generation layers  
**Date:** 2026-09-03

## Decision

All free-space perturbations operate on a detached `float64` copy and are driven by a seeded `numpy.random.Generator`. Projected metric columns are authoritative for meter-based changes; optional geographic columns are updated with a deterministic local equirectangular inverse when a foundation projection service is not available. Output arrays are read-only. Every generated or rejected result carries a content-derived ID and complete provenance, including input/output hashes and realized parameters.

The v1 free-space route-overlap definition is symmetric coverage of arc-length-resampled projected paths within a metric tolerance. It is versioned as `resampled_path_coverage_v1` and is explicitly not road-segment overlap. Road-network detours return a typed not-generated result until an approved map-matching source and license exist.

Task generators sort IDs before using their local RNG. This makes task construction independent of candidate container order and method evaluation order. Hard-negative generators stop at an explicit candidate examination bound and preserve rejection/yield reports; they never relax a threshold or substitute a random negative silently.

## Consequences

Short trajectories can produce `not_generated` results when a transform cannot retain a valid sequence. Notions remain separate, so translated copies may be preferred over nearby geometry for shape while the reverse holds for absolute route. The approximate coordinate update is adequate for small free-space fixture perturbations; a later foundation projection service can replace it without changing the perturbation contract.
