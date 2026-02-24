# Data Pipeline - ML Orchestration Platform

## COCO 2017 Data Pipeline

**Course:** IE7374 - MLOps, Spring 2026  
**Team:** Hemanth Sai Madadapu, Sujith Peddireddy, Jan Mollet, Sayee Ashish Aher, Sowmyashree Jayaram

---

## Overview

Reproducible MLOps-style data pipeline for processing COCO 2017 dataset, orchestrated via Apache Airflow and DVC, deployed in Docker containers. Prepares data for our RL-powered multi-model orchestration system.

### What This Pipeline Does

Downloads 118K+ COCO images automatically  
 Converts to YOLO format for object detection  
 Creates stratified train/val/test splits  
 Validates data quality with automated checks  
 Detects and mitigates bias across scene complexity  
 Orchestrates via Airflow DAG (8 stages)  
 Versions data with DVC  
 Deployed in Docker containers

### Resource Requirements

** Important:** Before running, ensure you have:

- **Disk Space:** 20GB minimum
- **First Run Time:** 2-4 hours (mostly download + extraction)
- **Subsequent Runs:** <5 minutes (DVC caching)
- **Network:** Stable connection for 19GB download

---

## Quick Start with Docker

** All commands must be run from repository root, not Data-Pipeline/**

```bash
# 1. Clone repository
git clone https://github.com/hemanth1403/IE7374-MLOps-Adaptive-ML-Inference.git
cd IE7374-MLOps-Adaptive-ML-Inference

# 2. Start Airflow with Docker Compose
docker-compose -f docker-compose.airflow.yml up -d

# 3. Access Airflow UI
open http://localhost:8080

# 4. Login
Username: admin
Password: admin

# 5. Trigger Pipeline
- Find DAG: "dvc_coco_pipeline"
- Toggle ON (switch on left)
- Click ▶ play button
- Watch 8 stages execute

# 6. Stop when done
docker-compose -f docker-compose.airflow.yml down
```

### Alternative: DVC Direct Execution

**For development or if Docker unavailable:**

```bash
# Navigate to Data-Pipeline (important!)
cd Data-Pipeline

# Install dependencies
pip install -r requirements.txt

# Run pipeline
dvc repro

# Run tests
pytest tests/ -v
```

**Note:** All DVC and pytest commands must be run from the `Data-Pipeline/` directory.

---

## Pipeline Architecture

### 8-Stage DVC Pipeline

**Defined in Data-Pipeline/dvc.yaml:**

**Stage 1: download_val_and_ann** - Downloads validation set + annotations (~1GB)

**Stage 2: extract_val_and_ann** - Extracts 5,000 validation images

**Stage 3: download_train** - Downloads training set (~18GB, takes 30-60 min)

**Stage 4: extract_train** - Extracts 118,287 training images

**Stage 5: coco_to_yolo** - Converts COCO JSON to YOLO format + fills missing labels

**Stage 6: preprocess_images_link** - Creates symlinks (disk-efficient, no data duplication)

**Stage 7: splits** - Creates stratified train/val/test (10% test, maintains complexity balance)

**Stage 8: reports** - Generates schema, quality, anomaly, and bias reports

**Visualizations:**

- Airflow Graph: docs/airflow_graph_success.png
- Gantt Chart: docs/airflow_gantt_success.png (shows reports stage takes longest)
- DVC DAG: docs/dvcDAG.png

### Airflow DAG Integration

**File:** Data-Pipeline/dags/dvc_coco_pipeline.py

Each Airflow task executes `dvc repro <stage>`, combining:

- Airflow's orchestration + monitoring
- DVC's caching + reproducibility

**Key Configuration:**

- Sequential execution (max_active_tasks=1) to avoid DVC lock conflicts
- 3 retries per task with 5-minute delays
- Proper dependency management

**Bottleneck Analysis:** Gantt chart (docs/airflow_gantt_success.png) shows reports stage (quality checks + bias slicing) takes the longest. Subsequent runs are cached by DVC, reducing re-run time significantly.

---

## Running the Pipeline

### Method 1: Docker + Airflow (Recommended)

**From repository root:**

```bash
# Start Airflow stack
docker-compose -f docker-compose.airflow.yml up -d

# Access UI
open http://localhost:8080

# Login
Username: admin
Password: admin

# Trigger DAG "dvc_coco_pipeline"
# Monitor execution in Graph View
```

**Why Docker:**

- No dependency conflicts
- PostgreSQL backend included
- Identical environment across machines
- Airflow logs available in UI

**Shutdown:**

```bash
docker-compose -f docker-compose.airflow.yml down
```

### Method 2: DVC Direct Execution

**From Data-Pipeline/ directory:**

```bash
cd Data-Pipeline

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
dvc repro

# View pipeline structure
dvc dag

# Check status
dvc status
```

**DVC Features:**

- Automatic caching (skips completed stages)
- Only reruns changed dependencies
- Faster than full re-execution

---

## Testing

** Run tests from Data-Pipeline/ directory:**

```bash
cd Data-Pipeline

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=scripts --cov-report=html

# View coverage
open htmlcov/index.html

# Specific test
pytest tests/test_splits_integrity.py -v
```

### Test Modules (5 files)

**test_ci_smoke.py** - CI sanity checks (imports, environment)

**test_dvc_yaml_valid.py** - DVC config validation (YAML syntax, stages)

**test_quality_invariants.py** - Data quality (image formats, labels)

**test_splits_integrity.py** - Critical: No data leakage between train/val/test

**test_reports_exist_and_parse.py** - Output validation (reports generated, parseable)

### CI/CD

**GitHub Actions** (.github/workflows/ci.yml):

- Runs on every push
- Automated testing
- Code quality checks

---

## Data Versioning with DVC

** Run DVC commands from Data-Pipeline/ directory:**

### Configuration

**Mode:** Local DVC (no cloud remote)  
**Config:** Data-Pipeline/.dvc/config  
**Pipeline:** Data-Pipeline/dvc.yaml (8 stages)  
**Lock:** Data-Pipeline/dvc.lock (tracks output hashes)

### DVC Commands

```bash
cd Data-Pipeline

# Run pipeline
dvc repro

# Check status
dvc status

# View DAG
dvc dag

# Force re-run specific stage
dvc repro -f reports
```

---

## Quality Assurance & Bias Analysis

### Schema Validation

**Script:** Data-Pipeline/scripts/schema_stats.py

**Outputs:**

- data/reports/schema.json - Expected schema definition
- data/reports/stats.json - Dataset statistics

**Validates:**

- Image formats (JPEG)
- Label formats (YOLO)
- Directory structure
- File counts

### Quality Checks

**Script:** Data-Pipeline/scripts/quality_checks.py

**Output:** data/reports/quality.json

**Example results:**

```json
{
  "total_images": { "train": 106411, "val": 5000, "test": 11876 },
  "missing_label_files": { "train": 0, "val": 0, "test": 0 },
  "empty_label_files": { "train": 932, "val": 48, "test": 89 },
  "invalid_yolo_lines": 0,
  "invalid_bbox_range": 0,
  "objects_per_image_stats": {
    "train": { "mean": 7.18, "max": 90 },
    "val": { "mean": 7.27, "max": 62 }
  }
}
```

**Validates:**

- All images have corresponding labels
- No invalid YOLO format lines
- Bounding boxes within valid range [0, 1]
- Class IDs in range [0-79]

### Anomaly Detection

**Script:** Data-Pipeline/scripts/anomaly_alerts.py

**What it does:**

- Analyzes quality.json results
- Flags suspicious patterns (>1% missing labels, invalid formats)
- Emits alerts to Airflow task logs (searchable with "ALERT:")
- Optional extension: Airflow email/Slack notification hooks can be configured

**How alerts work:**

- Logged to console and Airflow UI
- Critical issues fail the task
- Warnings logged for review

### Bias Detection & Mitigation

**Detection Script:** Data-Pipeline/scripts/bias_slicing.py

**Slicing Strategy:**

We slice data by scene complexity based on object count:

- Simple: ≤2 objects
- Moderate: 3-7 objects
- Complex: ≥8 objects

**Output:** data/reports/bias.md

**Example results:**

```
Train:
- simple: 32,649 (30.68%)
- moderate: 38,612 (36.29%)
- complex: 35,150 (33.03%)

Val:
- simple: 1,566 (31.32%)
- moderate: 1,736 (34.72%)
- complex: 1,698 (33.96%)

Test:
- simple: 3,656 (30.78%)
- moderate: 4,282 (36.06%)
- complex: 3,938 (33.16%)
```

**Analysis:** Distribution is balanced across all splits (approximately 31%/36%/33% across train/val/test).

**Mitigation Implementation:**

**Primary Mitigation Strategy: Stratified Splitting**

Our `create_splits.py` script implements bias mitigation through:

1. **Stratified Sampling**
   - Analyzes scene complexity distribution in source data
   - Creates splits that maintain similar distributions across train/val/test
   - Prevents over-representation of any complexity tier

2. **Validation of Balance**
   - Bias report (bias.md) confirms balanced distribution
   - No single split is skewed toward simple or complex scenes
   - Ensures model will train on representative samples

**This is proactive mitigation** - we prevent bias during split creation rather than detecting and fixing it afterward.

**If Bias Detected (deviation >5% from balanced):**

Reactive mitigation strategies available:

- **Re-sampling:** Oversample underrepresented complexity tiers
- **Weighted Sampling:** Adjust sampling probability by inverse frequency
- **Threshold Tuning:** Adjust complexity classification thresholds
- **Document Trade-offs:** Balance between sample size and fairness

**Current Status:** Analysis shows balanced distribution across splits. Stratified approach successfully prevents complexity bias. No additional re-sampling needed.

**Future Training Mitigations:**

- Weighted loss functions (address class imbalance: person 29%, rare classes <1%)
- Focal loss (focus on hard examples)
- Class-aware data augmentation

---

## Reproducibility

### Setup on Any Machine

**Complete reproduction from scratch:**

```bash
# 1. Clone repository
git clone https://github.com/hemanth1403/IE7374-MLOps-Adaptive-ML-Inference.git
cd IE7374-MLOps-Adaptive-ML-Inference

# 2. Run with Docker (no Python setup needed)
docker-compose -f docker-compose.airflow.yml up -d

# 3. Access Airflow UI
open http://localhost:8080
# Login: admin / admin
# Trigger DAG: dvc_coco_pipeline

# Monitor execution in UI (Graph View or Gantt View)
```

**OR run with DVC directly:**

```bash
# Navigate to Data-Pipeline directory
cd Data-Pipeline

# Install dependencies
pip install -r requirements.txt

# Run pipeline
dvc repro

# Verify
pytest tests/ -v
```

**Expected Runtime:**

- First run: 2-4 hours (download 19GB + processing)
- Subsequent runs: <5 minutes (DVC caches completed stages)

**Expected Outputs:**

```
Data-Pipeline/data/
├── raw_zips/           # Downloaded COCO archives
├── raw/                # Extracted images and annotations
│   ├── train2017/      # 118,287 images
│   ├── val2017/        # 5,000 images
│   └── annotations/    # COCO JSON files
├── processed/
│   ├── images/         # Symlinks to raw images (disk-efficient)
│   ├── labels/         # YOLO format annotations
│   └── coco.names      # Class names (80 classes)
├── splits/
│   ├── train.txt       # 106,411 image paths
│   ├── val.txt         # 5,000 image paths
│   └── test.txt        # 11,876 image paths
└── reports/
    ├── schema.json     # Schema definition
    ├── stats.json      # Dataset statistics
    ├── quality.json    # Quality validation results
    └── bias.md         # Bias analysis report
```

---

## Components

### Pipeline Scripts (10 modules)

** Run from Data-Pipeline/ directory**

**1. download_coco2017.py**

- Downloads COCO 2017 with resume support
- Usage: `python scripts/download_coco2017.py --out data/raw_zips --subset val`

**2. extract_zips.py**

- Extracts downloaded archives
- Creates completion markers

**3. convert_coco_to_yolo.py**

- Converts COCO JSON to YOLO text format
- Bounding box: [class, x_center, y_center, width, height] (normalized)

**4. fill_missing_labels.py**

- Creates empty label files for images without annotations
- Ensures every image has a label file (YOLO requirement)

**5. preprocess_images.py**

- Symlink mode: Links to original images (saves 19GB disk space)
- Alternative copy mode available if resizing needed

**6. create_splits.py**

- Stratified train/val/test splits
- **Implements bias mitigation** through balanced sampling

**7. schema_stats.py**

- Generates schema.json and stats.json
- Validates data structure

**8. quality_checks.py**

- Validates image readability, label format, bbox ranges
- Outputs quality.json

**9. anomaly_alerts.py**

- Analyzes quality results
- Emits alerts to logs (ALERT: prefix in Airflow logs)

**10. bias_slicing.py**

- Slices data by scene complexity
- Outputs bias.md report

---

## 🐳 Docker Deployment

### What's Included

**Docker Compose Stack** (docker-compose.airflow.yml):

**Services:**

- Airflow webserver (UI on port 8080)
- Airflow scheduler (executes tasks)
- PostgreSQL (metadata database)

**Volume Mounts:**

- Data-Pipeline/ → /opt/airflow/repo/Data-Pipeline/
- Shares code and data between host and container

**Login Credentials:**

- Username: admin
- Password: admin
- (Created during container initialization)

### Why Docker for This Project

**Solves Real Problems:**

- Airflow + PyTorch dependency conflicts avoided
- Identical environment for all team members
- Matches how Airflow runs in production

**Benefits:**

- One command to start (`docker-compose up`)
- No Python version issues
- PostgreSQL included (better than SQLite)
- Easy cleanup (`docker-compose down`)

---

## Folder Structure

```
Data-Pipeline/
├── dags/
│   └── dvc_coco_pipeline.py          # Airflow DAG
├── scripts/                           # 10 modular scripts
│   ├── download_coco2017.py
│   ├── extract_zips.py
│   ├── convert_coco_to_yolo.py
│   ├── fill_missing_labels.py
│   ├── preprocess_images.py
│   ├── create_splits.py
│   ├── schema_stats.py
│   ├── quality_checks.py
│   ├── anomaly_alerts.py
│   └── bias_slicing.py
├── tests/                             # 5 test modules
│   ├── test_ci_smoke.py
│   ├── test_dvc_yaml_valid.py
│   ├── test_quality_invariants.py
│   ├── test_splits_integrity.py
│   └── test_reports_exist_and_parse.py
├── .dvc/
│   └── config                         # DVC configuration
├── data/                              # Data directory (DVC tracked)
│   ├── raw_zips/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── reports/
├── dvc.yaml                           # DVC pipeline definition
├── dvc.lock                           # DVC lockfile
├── requirements.txt                   # Dependencies: pytest, requests, tqdm, pillow, pyyaml, dvc
├── pytest.ini                         # Pytest configuration
├── .dvcignore                         # DVC ignore patterns
└── README.md                          # This file
```

**Parent directory contains:**

- Dockerfile.airflow
- docker-compose.airflow.yml
- docs/ (screenshots)
- .github/workflows/ci.yml

---

## Quality Reports (Actual Outputs)

### Schema & Statistics

**Files generated:**

- data/reports/schema.json
- data/reports/stats.json

**Example stats (from your run):**

- Train images: 106,411
- Val images: 5,000
- Test images: 11,876
- Total classes: 80
- Mean objects per image: ~7.2

### Quality Validation

**File:** data/reports/quality.json

**Key metrics:**

- Missing labels: 0 (all images have labels)
- Empty labels: 1,069 total (images with no objects)
- Invalid YOLO lines: 0
- Invalid bounding boxes: 0
- Class IDs: All in valid range [0-79]

** No critical quality issues detected**

### Bias Analysis

**File:** data/reports/bias.md

**Distribution across splits:**

**Train:** 30.68% simple, 36.29% moderate, 33.03% complex  
**Val:** 31.32% simple, 34.72% moderate, 33.96% complex  
**Test:** 30.78% simple, 36.06% moderate, 33.16% complex

**Analysis:** Distribution is consistent across all splits. No significant bias detected.

**Mitigation:** Stratified splitting maintains balanced representation. No additional re-sampling required.

---

## Troubleshooting

### Common Issues

**Issue: "dvc.yaml not found"**

```bash
# Make sure you're in Data-Pipeline/ directory
cd Data-Pipeline
dvc repro
```

**Issue: "Out of disk space"**

```bash
# Check available space
df -h

# Pipeline uses symlinks to save space (preprocess_images_link mode)
```

**Issue: "Docker container fails to start"**

```bash
# Check Docker is running
docker ps

# View logs
docker-compose -f docker-compose.airflow.yml logs

# Restart
docker-compose -f docker-compose.airflow.yml restart
```

**Issue: "Tests fail"**

```bash
# Ensure pipeline completed first
cd Data-Pipeline
dvc status

# Run reports to generate test dependencies
dvc repro reports

# Then run tests
pytest tests/ -v
```

**Issue: "Can't login to Airflow"**

- Username: admin
- Password: admin
- (Set in docker-compose.airflow.yml)

---

## Pipeline Visualizations

**Included in docs/ folder:**

**airflow_graph_success.png** - DAG graph showing all 8 tasks and dependencies

**airflow_gantt_success.png** - Execution timeline showing task durations; reports stage identified as bottleneck

**dvcDAG.png** - DVC pipeline visualization showing data lineage

---

## MLOps Practices Demonstrated

**1. Containerization** - Docker for reproducible deployment

**2. Workflow Orchestration** - Airflow + DVC hybrid approach

**3. Data Versioning** - DVC tracks data via hashes

**4. Testing & CI/CD** - Pytest + GitHub Actions

**5. Monitoring** - Airflow UI for real-time pipeline status

**6. Quality Assurance** - Schema validation, quality checks, anomaly detection

**7. Fairness** - Bias detection via data slicing + mitigation via stratified sampling

**8. Modularity** - Reusable, independently testable scripts

**9. Error Handling** - Retries, logging, graceful failures

**10. Documentation** - Comprehensive README, inline comments

---

## Team

- Hemanth Sai Madadapu - madadapu.h@northeastern.edu
- Sujith Peddireddy - peddireddy.su@northeastern.edu
- Jan Mollet - mollet.j@northeastern.edu
- Sayee Ashish Aher - aher.sa@northeastern.edu
- Sowmyashree Jayaram - jayaram.so@northeastern.edu

**Course:** IE7374 - MLOps, Spring 2026  
**Instructor:** Prof. Ramin Mohammadi

---

## License

MIT License - See LICENSE file

## Acknowledgments

- COCO Dataset: Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014
- Apache Airflow, DVC communities
- Prof. Ramin Mohammadi's course materials

---

**Repository:** https://github.com/hemanth1403/IE7374-MLOps-Adaptive-ML-Inference
