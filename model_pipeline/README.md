# Model Pipeline

This module contains the complete **model evaluation, benchmarking, comparison, bias-analysis, and experiment-tracking workflow** for the adaptive ML inference project.

The pipeline uses **pre-trained YOLOv8 variants** (`nano`, `small`, and `large`) and evaluates their tradeoffs across:

- detection quality,
- inference latency,
- throughput,
- workload-slice behavior,
- reproducibility,
- and deployment suitability.

These outputs support **final model selection** and the broader **adaptive routing / orchestration** design used in the project.

---

# 1. Scope and purpose

The goal of this pipeline is to answer:

- Which pre-trained YOLO model gives the best detection quality?
- Which model gives the best latency and throughput?
- How do these tradeoffs change across simple, moderate, and complex scenes?
- What evidence supports choosing one fixed model versus an adaptive routing strategy?

This project does **not** train a brand-new detector architecture from scratch. Instead, it uses **pre-trained YOLOv8 nano, small, and large** models and focuses on:

- validation,
- runtime benchmarking,
- model comparison,
- slice-based analysis,
- bias/disparity reporting,
- experiment tracking with MLflow,
- and CI-based reproducibility.

---

# 2. Models evaluated

The current pipeline evaluates the following pre-trained YOLOv8 variants:

- **YOLO Nano (`yolov8n.pt`)**  
  Lowest-latency / highest-throughput tier.

- **YOLO Small (`yolov8s.pt`)**  
  Balanced middle tier.

- **YOLO Large (`yolov8l.pt`)**  
  Highest-capacity / highest-latency tier.

---

# 3. Repository structure

```text
model_pipeline/
├── artifacts/
│   └── dataset.yaml
├── configs/
│   ├── data/
│   │   └── dataset_config.yaml
│   ├── eval/
│   │   └── eval_config.yaml
│   └── train/
│       └── train_config.yaml
├── reports/
│   ├── benchmarks/
│   ├── bias/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── RL/
│   ├── bias/
│   │   └── generate_bias_report.py
│   └── evaluation/
│       ├── benchmark.py
│       ├── compare_models.py
│       ├── evaluate.py
│       └── generate_slice_comparison.py
└── tests/

---

# 4. Directory Overview

artifacts/

Contains dataset metadata used during evaluation.

configs/

Contains configuration files for:

dataset paths,
evaluation settings,
benchmark settings,
model definitions.
reports/metrics/

Stores per-model evaluation outputs.

reports/benchmarks/

Stores per-model runtime benchmark outputs.

reports/figures/

Stores final comparison and slice-comparison outputs.

reports/bias/

Stores slice-based bias / disparity analysis outputs.

src/evaluation/

Contains the main evaluation, benchmark, comparison, and slice-report scripts.

src/bias/

Contains the script that generates bias/disparity reports from slice comparison outputs.

src/RL/

Contains the adaptive routing / RL logic used by the larger system.
---

# 7. Configuration Files Used
model_pipeline/configs/eval/eval_config.yaml

Controls:

evaluation split,
benchmark device,
output directories.
model_pipeline/configs/train/train_config.yaml

Defines:

model variants,
fallback pretrained YOLO weights,
checkpoint output structure.
model_pipeline/configs/data/dataset_config.yaml

Used by benchmarking logic to locate split files and processed dataset resources.

model_pipeline/artifacts/dataset.yaml

Base dataset metadata used to generate dataset.runtime.yaml

# 8. MLflow Experiment Tracking

This pipeline logs experiments to MLflow.

Start the MLflow UI
###bash
mlflow ui

Then open:

http://127.0.0.1:5000
Current Experiments:

pretrained_yolo_evaluation
pretrained_yolo_benchmark
pretrained_yolo_summary

What Is Logged:

Evaluation Runs
model name
weight source
split
device
mAP50
mAP50_95
precision
recall
metrics JSON artifact
Benchmark Runs
model name
weight source
split name
overall latency
overall throughput
per-bucket latency
per-bucket throughput
benchmark JSON artifact

# 9. CI/CD Reproducibility

The repository includes a GitHub Actions workflow for the model pipeline.

The CI workflow currently does the following:

checks out the repository,
sets up Python,
installs model-pipeline dependencies,
authenticates to GCP using a GitHub secret,
uses DVC to pull only the evaluation data needed for CI,
runs pretrained evaluation,
regenerates comparison and bias reports,
uploads report artifacts.
Current CI Behavior

The workflow pulls only the evaluation data needed for CI instead of restoring the full data pipeline workspace.

This keeps CI focused on model-evaluation reproducibility rather than full pipeline reconstruction.

# 10. Current Model-Selection Conclusion

Based on the current outputs:

YOLO Nano is the fastest model and is the best option when low latency and high throughput matter most.
YOLO Small provides the best balance between detection quality and runtime cost and is the strongest default baseline.
YOLO Large gives the strongest detection quality but is substantially slower and more expensive to run.

These results support an adaptive routing strategy rather than always using one fixed model tier.

# 11. Current Model-Selection Conclusion

Based on the current outputs:

YOLO Nano is the fastest model and is the best option when low latency and high throughput matter most.
YOLO Small provides the best balance between detection quality and runtime cost and is the strongest default baseline.
YOLO Large gives the strongest detection quality but is substantially slower and more expensive to run.

These results support an adaptive routing strategy rather than always using one fixed model tier.

# 12. Summary

This model pipeline provides a complete pre-trained model evaluation workflow for adaptive inference:

evaluates YOLO Nano, Small, and Large,
benchmarks them across workload slices,
compares their tradeoffs,
generates slice-based comparison artifacts,
generates bias/disparity reports,
tracks experiments in MLflow,
and supports reproducible CI evaluation using DVC-backed data access.