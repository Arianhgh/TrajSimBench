# Benchmark V1 Protocol

## Purpose

This is the shared standard-benchmark protocol.  Every method receives the
same underlying trajectories, eligibility rules, splits, query IDs, candidate
database IDs, oracle labels, perturbations, and random seeds.  A method may
use its documented input encoding, padding, architecture, loss, and training
augmentation, but every such transformation is recorded and timed.

This document describes the standard benchmark, not a claim that it
reproduces an author's paper.  Paper reproduction uses the authors' own
dataset, preprocessing, split, and settings in a separately labelled result
category.

## Dataset Register

| Dataset | Role | Status | Main representation | Road representation |
| --- | --- | --- | --- | --- |
| Porto Taxi | Primary urban retrieval and scale dataset | Eligible source recorded | Free-space GPS | Separate, gated matched version |
| GeoLife | Human movement, user holdout, movement-mode analysis | License confirmation required | Free-space GPS | Optional driving-only extension |
| T-Drive | Beijing taxi transfer dataset | License confirmation required | Free-space GPS | Separate, gated matched version |
| AIS | Maritime domain transfer dataset | Region, dates, release, vessel rules, and terms required | Free-space GPS | None |
| Synthetic | Diagnostic control with known transformations | Ready | Free-space and graph-generated diagnostics | Graph diagnostics only |

Germany remains an optional reproduction dataset.  It cannot enter the
standard benchmark until its exact source and preprocessing lineage are
approved.

## Canonical Preparation

1. Preserve original longitude, latitude, timestamps, source trip/session
   IDs, object IDs, and any mode labels.  Every processed trajectory retains a
   link to its raw source record.
2. Reject malformed coordinates and invalid records with an explicit reason
   code.  Remove exact duplicate observations, but preserve repeated positions
   at different times because they can represent a stop.  Record conflicting
   timestamps instead of silently repairing them.
3. Preserve supplied Porto trip boundaries.  For continuous recordings,
   segment by documented gap and stop rules selected on a development subset
   and frozen before evaluation.  The numerical rules may differ by dataset;
   they are not universal taxi, pedestrian, or vessel settings.
4. Preserve WGS84 longitude/latitude.  Calculate distances and meter-based
   perturbations in the declared local metric CRS.  Do not centre, rotate, or
   scale individual trajectories in canonical data.
5. Retain native sampling.  Do not smooth, interpolate, or resample the main
   dataset.  Resampled variants are separately named sampling experiments.
   Porto point times reconstructed from its start time and 15-second sampling
   interval are labelled `reconstructed`.
6. Report length, duration, distance, and sampling-rate strata.  The 20--200
   observed-point cohort is a compatibility experiment, not the definition of
   the benchmark; shorter and longer trajectories remain reportable strata.

Raw checksums, the preparation configuration hash, inspection report, and
canonical-file checksums are required before a dataset is eligible for a run.

## Shared Split and Evaluation Rules

The standard split is 70% train, 10% validation, and 20% test by source
movement.  All fragments and augmented versions inherit the split of their
source movement.  Normalizers, vocabularies, learned encoders, and all other
data-derived values fit on training trajectories only.

Additional named experiments are:

- GeoLife user-held-out split: complete users never overlap across partitions.
- Temporal holdout: later source movements are held out after sorting by UTC
  start time.
- Cross-dataset transfer: train on one named dataset and evaluate on another
  without silently combining their source records.

Each dataset/split combination has fixed query IDs, fixed database IDs, and
fixed self-match exclusions.  Retrieval scale is independent of split size:
the standard tier targets a 10,000-item database and 1,000 queries, but a
smaller tier is explicitly labelled when the available held-out set is too
small.  No replacement sampling is allowed.

## Inputs Shared by Every Method

| Fixed and shared | Method-specific but recorded |
| --- | --- |
| Source movements, eligibility, segmentation, and canonical coordinates | Grid/token construction, patches, padding, masks, and batching |
| Train/validation/test/query/database IDs and self-match exclusions | Architecture, loss, training augmentation, checkpoint, and hyperparameters |
| Oracles, units, labels, perturbations, and random seeds | Published embedding-distance or score conversion |
| Access to raw training data and external information | Required feature extraction and its measured cost |

A method may not silently omit difficult test trajectories.  Failures,
coverage, simplification, resampling, runtime, and peak memory are all part of
the reported pipeline.

## Perturbation and Robustness Rules

Create perturbations only after splitting.  Corrupted variants stay with their
clean source trajectory's partition.  Robustness runs corrupt held-out clean
trajectories and then execute the method's complete evaluated pipeline;
corrupted inputs are not re-cleaned unless that action is itself declared as a
separate experiment.

## Separate Road-Network Track

The main benchmark remains free-space GPS.  Porto and T-Drive may receive a
separate road representation linked to the same canonical trajectory IDs:

`clean GPS -> frozen road graph and matcher -> ordered directed road edges`

Before this track is enabled, freeze and record the road-network snapshot,
graph build options, matcher version, candidate count, search radius, GPS
error parameters, historical GPS/map-date mismatch, and acceptance criteria.
Calibrate these values only on a development set.  Store original GPS,
matched positions, inferred connecting edges, residual errors, and failures.
Publish representative review samples plus acceptance rates by trajectory
length and sampling interval.  A plausible match is not ground truth.

For noisy GPS, run two separately named tests: corrupt GPS then rematch to
measure the full pipeline, or keep the reference road path fixed to evaluate
the representation.  Never combine their results.

## Completion Gate

The standard real-data benchmark is ready only after every selected dataset
has approved provenance, frozen preparation settings, validated canonical
files, fixed split/query/database IDs, and a recorded checksum manifest.  A
model is paper-reproduced only after its original end-to-end protocol and
paper-comparable metrics have been run separately.
