# Model Pipeline Contract

## Purpose
This document defines the interface between the data pipeline, model pipeline, and downstream inference services.

## Inputs from Data-Pipeline
The model pipeline expects the following inputs from `Data-Pipeline/`:

- YOLO-format training images: `Data-Pipeline/data/processed/images/train2017`
- YOLO-format validation images: `Data-Pipeline/data/processed/images/val2017`
- YOLO-format training labels: `Data-Pipeline/data/processed/labels/train2017`
- YOLO-format validation labels: `Data-Pipeline/data/processed/labels/val2017`
- Split files:
  - `Data-Pipeline/data/splits/train.txt`
  - `Data-Pipeline/data/splits/val.txt`
  - `Data-Pipeline/data/splits/test.txt`
- Data quality and schema reports:
  - `Data-Pipeline/data/reports/quality.json`
  - `Data-Pipeline/data/reports/schema.json`
  - `Data-Pipeline/data/reports/stats.json`
  - `Data-Pipeline/data/reports/bias.md`

## Outputs from Model Pipeline
The model pipeline will produce:

- Trained checkpoints for:
  - YOLO Nano
  - YOLO Small
  - YOLO Large
- Evaluation metrics JSON files
- Benchmark result files
- Bias evaluation reports
- Exported inference-ready model artifacts
- Model manifests for downstream services

## Model Selection Policy
At this stage, the pipeline does not choose one global winner.
All three model tiers are retained and compared on:

- Accuracy
- Latency
- Throughput

These outputs will later be used by the routing/orchestration layer.

## Service Handoff
Downstream services should be able to consume:

- exported model path
- model name
- training/evaluation metadata
- class count / label metadata
- inference configuration
- benchmark summary

## Notes
This contract is intentionally versioned in the repository so all team members follow the same assumptions.