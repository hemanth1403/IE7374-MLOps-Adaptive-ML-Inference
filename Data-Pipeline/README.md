# Data Pipeline - ML Orchestration Platform

## COCO 2017 Data Pipeline with Airflow & DVC

**Course:** IE7374 - MLOps, Spring 2026  
**Team:** Hemanth Sai Madadapu, Sujith Peddireddy, Jan Mollet, Sayee Ashish Aher, Sowmyashree Jayaram

---

## Overview

Production level data pipeline for processing COCO 2017 dataset, orchestrated via **Apache Airflow** and **DVC**, deployed in **Docker containers**. Prepares data for our RL-powered multi-model ML orchestration system.

### What This Pipeline Does

1. Downloads 118K+ COCO images automatically
2. Converts to YOLO format for object detection
3. Creates stratified train/val/test splits
4. Validates data quality with automated checks
5. Detects and mitigates bias across scene complexity
6. Orchestrates via Airflow DAG (8 stages)
7. Versions data with DVC
8. Deployed in Docker (production-ready)

---

## 🐳 Quick Start with Docker

**Why Docker?** Solves all dependency conflicts, production-ready, works immediately.

```bash
# 1. Clone repository
git clone https://github.com/hemanth1403/IE7374-MLOps-Adaptive-ML-Inference.git
cd IE7374-MLOps-Adaptive-ML-Inference

# 2. Start Airflow with Docker Compose (includes PostgreSQL + scheduler + webserver)
docker-compose -f docker-compose.airflow.yml up -d

# 3. Access Airflow UI
open http://localhost:8080

# 4. Login
Username: airflow
Password: airflow

# 5. Trigger Pipeline
- Find DAG: "dvc_coco_pipeline"
- Toggle ON (switch on left)
- Click play button
- Watch 8 stages execute!

# 6. Stop when done
docker-compose -f docker-compose.airflow.yml down
```

**That's it! No Python setup, no venv, no dependency hell.**

---

## Pipeline Architecture

### 8-Stage DVC Pipeline

**Orchestrated by Airflow, defined in dvc.yaml:**

**Stage 1: download_val_and_ann**

- Downloads validation images + annotations (~1GB)

**Stage 2: extract_val_and_ann**

- Extracts 5,000 validation images

**Stage 3: download_train**

- Downloads training images (~18GB)

**Stage 4: extract_train**

- Extracts 118,287 training images

**Stage 5: coco_to_yolo**

- Converts COCO format to YOLO format
- Fills missing label files

**Stage 6: preprocess_images_link**

- Creates symlinks (disk-efficient, no data duplication)

**Stage 7: splits**

- Creates stratified train/val/test (10% test, maintains complexity balance)

**Stage 8: reports**

- Generates schema validation, quality checks, anomaly detection, bias analysis

**Visualizations:**

- Airflow Graph: `docs/airflow_graph_success.png`
- Gantt Chart: `docs/airflow_gantt_success.png`
- DVC DAG: `docs/dvcDAG.png`

### Airflow DAG Design

**File:** `dags/dvc_coco_pipeline.py`

**Smart Integration:** Each Airflow task runs `dvc repro <stage>`, combining:

- Airflow's orchestration + monitoring
- DVC's caching + reproducibility

**Key Features:**

- Sequential execution (avoids DVC lock conflicts)
- 3 retries per task (5-minute delays)
- Proper dependency management
- Clean separation of concerns

---

## All 12 Requirements Met

**1. Proper Documentation**

- README (this file), inline comments, docstrings
- Evidence: Comprehensive documentation throughout

**2. Modular Code**

- 10 independent scripts in scripts/
- Evidence: Each script is self-contained and reusable

**3. Airflow DAG Orchestration**

- dags/dvc_coco_pipeline.py with 8 tasks
- Evidence: Screenshots in docs/ showing successful execution

**4. Tracking & Logging**

- Python logging in all scripts
- Evidence: Check any script for logging.info() statements

**5. DVC Data Versioning**

- dvc.yaml defines pipeline, dvc.lock tracks hashes
- Evidence: Local DVC tracking with hash-based versioning

**6. Pipeline Optimization**

- Airflow Gantt chart identifies bottlenecks
- Evidence: docs/airflow_gantt_success.png shows execution timeline

**7. Schema Generation**

- scripts/schema_stats.py generates data schema
- Evidence: data/reports/schema_stats.yaml output

**8. Anomaly Detection & Alerts**

- scripts/anomaly_alerts.py with automated alerts
- Evidence: Console warnings + detailed logging

**9. Bias Detection & Mitigation**

- scripts/bias_slicing.py + stratified splitting
- Evidence: data/reports/bias.md + balanced splits

**10. Test Modules**

- 5 pytest files in tests/
- Evidence: Comprehensive test suite with CI/CD

**11. Reproducibility**

- Docker + DVC + requirements.txt + docs
- Evidence: Anyone can run docker-compose and get same results

**12. Error Handling**

- Try-catch blocks, 3 retries in Airflow DAG
- Evidence: Graceful failure handling throughout

---

## Running the Pipeline

### Method 1: Docker + Airflow

**Complete setup in 2 minutes:**

```bash
# From repository root
docker-compose -f docker-compose.airflow.yml up -d

# Access Airflow UI
open http://localhost:8080
# Login: airflow / airflow
# Trigger DAG: dvc_coco_pipeline
```

**Features:**

- Isolated environment (no conflicts)
- PostgreSQL backend (better than SQLite)
- Production-ready deployment
- Easy team collaboration

**Monitor execution:**

- Graph View: See task dependencies
- Gantt View: Identify bottlenecks
- Logs: Click any task box

**Cleanup:**

```bash
docker-compose -f docker-compose.airflow.yml down
```

### Method 2: DVC Direct Execution (Development)

**For code development and testing:**

```bash
cd Data-Pipeline

# Install dependencies (first time)
pip install -r requirements.txt

# Run pipeline
dvc repro

# Run specific stage
dvc repro reports
```

**When to use:**

- Developing new scripts
- Testing changes locally
- Quick iterations
- No need for Airflow overhead

### Method 3: Standalone Airflow (Backup)

**If Docker unavailable:**

```bash
cd Data-Pipeline
export AIRFLOW_HOME=$(pwd)/airflow

# One-time setup
pip install apache-airflow==2.11.0
airflow db init
airflow users create --username admin --password admin \
    --firstname Admin --lastname User --role Admin \
    --email admin@example.com

# Run
airflow standalone
# Access http://localhost:8080
```

---

## 🐳 Docker Deployment Details

### What's in docker-compose.airflow.yml

**Services:**

**airflow-webserver**

- Web UI for pipeline monitoring
- Port 8080
- Includes all Airflow providers

**airflow-scheduler**

- Executes DAG tasks
- Monitors dependencies
- Handles retries

**postgres**

- Metadata database for Airflow
- Better performance than SQLite
- Production-recommended

**Volume Mounts:**

- `./Data-Pipeline/dags` -> Airflow DAGs folder
- `./Data-Pipeline/scripts` -> Pipeline scripts
- `./Data-Pipeline/data` -> Data directory

### Docker Advantages

**Why we use Docker:**

**1. Dependency Isolation**

- Airflow in container, ML libraries in container
- No Python version conflicts
- No package dependency hell

**2. Reproducibility**

- Same environment on Mac/Linux/Windows
- Team members get identical setup
- Graders can run with one command

**3. Production Alignment**

- Industry standard for deployment
- Matches how Airflow runs in production
- Demonstrates MLOps best practices

**4. Easy Cleanup**

- `docker-compose down` removes everything
- No leftover files or processes
- Fresh start anytime

---

## Testing

### Test Suite (5 Modules)

**test_ci_smoke.py** - CI sanity checks

- Verifies basic imports
- Environment validation

**test_dvc_yaml_valid.py** - DVC configuration

- Validates dvc.yaml syntax
- Checks stage definitions

**test_quality_invariants.py** - Data quality

- Image format validation
- Label correctness checks

**test_splits_integrity.py** - Critical data leakage test

- Validates train/val/test are disjoint
- No overlap between splits

**test_reports_exist_and_parse.py** - Output validation

- Reports generated successfully
- JSON/YAML files parseable

### Running Tests

```bash
cd Data-Pipeline

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=scripts --cov-report=html
open htmlcov/index.html

# Specific test
pytest tests/test_splits_integrity.py -v
```

### CI/CD Integration

**GitHub Actions** (.github/workflows/ci.yml)

- Runs on every push
- Automated testing
- Coverage reporting

---

## Data Versioning with DVC

### DVC Setup (Local)

**Configuration:** Local DVC (no cloud remote for this checkpoint)

**.dvc/config:**

```ini
[core]
    no_scm = True
```

**Pipeline:** dvc.yaml defines 8 stages  
**Lock:** dvc.lock tracks output hashes for reproducibility

### DVC Commands

```bash
# Run pipeline
dvc repro

# Check status
dvc status

# View DAG
dvc dag

# Force re-run
dvc repro -f <stage_name>
```

### Why DVC?

- **Caching:** Skips unchanged stages (saves hours on re-runs)
- **Versioning:** Tracks data via hashes, not Git
- **Reproducibility:** dvc repro gives same results
- **Lineage:** Complete data provenance

**Future:** Can add cloud remote (GCS) for team data sharing

---

## Quality Assurance

### Schema Validation (scripts/schema_stats.py)

**Validates:**

- Image dimensions correct
- File formats valid (JPEG)
- Color channels correct (RGB, 3 channels)
- Annotation completeness

**Output:** data/reports/schema_stats.yaml

### Quality Checks (scripts/quality_checks.py)

**Checks:**

- All images readable
- Labels exist for all images
- Bounding boxes in range [0, 1]
- Valid YOLO format
- Class IDs valid [0-79]

**Output:** data/reports/quality.json

### Anomaly Detection (scripts/anomaly_alerts.py)

**Detects:**

- Corrupt files
- Missing labels
- Invalid bounding boxes
- Format violations

**Alerts:** Console warnings + detailed logs

### Bias Detection & Mitigation (scripts/bias_slicing.py)

**Detection:**

- Slices data by scene complexity:
  - Simple: ≤2 objects
  - Moderate: 3-7 objects
  - Complex: ≥8 objects
- Analyzes distribution across train/val/test

**Mitigation (Implemented via Stratified Splitting):**

**Primary Strategy:** scripts/create_splits.py

- Maintains 35% simple / 45% moderate / 20% complex distribution
- Prevents over-representation of any scene type
- Proactive bias prevention (better than reactive fixing)

**Findings:** Distribution balanced, no re-sampling needed

**Future Mitigations (if bias detected):**

- Re-sampling underrepresented groups
- Weighted loss functions in training
- Class-aware data augmentation
- Separate evaluation per slice

**Output:** data/reports/bias.md

---

## Reproducibility

### Run on Any Machine

```bash
# 1. Clone
git clone https://github.com/hemanth1403/IE7374-MLOps-Adaptive-ML-Inference.git
cd IE7374-MLOps-Adaptive-ML-Inference

# 2. Run with Docker (RECOMMENDED)
docker-compose -f docker-compose.airflow.yml up -d
# Access http://localhost:8080, trigger DAG

# OR run with DVC locally
cd Data-Pipeline
pip install -r requirements.txt
dvc repro
```

**Runtime:** 2-4 hours first run, under 5 minutes with DVC cache

**Outputs:** 123K processed images, YOLO labels, stratified splits, quality reports

---

## Scripts & Components

**10 Modular Scripts:**

1. download_coco2017.py - Download with resume support
2. extract_zips.py - Extract archives
3. convert_coco_to_yolo.py - Format conversion
4. fill_missing_labels.py - Label completion
5. preprocess_images.py - Symlink creation (disk-efficient)
6. create_splits.py - Stratified splitting
7. schema_stats.py - Schema generation
8. quality_checks.py - Quality validation
9. anomaly_alerts.py - Anomaly detection
10. bias_slicing.py - Bias analysis

**Each script:**

- Independent and reusable
- Command-line arguments
- Comprehensive logging
- Error handling

---

## Project Structure

```
Data-Pipeline/
├── dags/
│   └── dvc_coco_pipeline.py          # Airflow DAG
├── scripts/                           # 10 modular scripts
├── tests/                             # 5 test modules
├── .dvc/                              # DVC configuration
├── data/                              # Data directory (DVC tracked)
│   ├── raw_zips/                     # Downloads
│   ├── raw/                          # Extracted COCO
│   ├── processed/                    # YOLO format
│   ├── splits/                       # Train/val/test
│   └── reports/                      # Quality reports
├── dvc.yaml                           # DVC pipeline
├── dvc.lock                           # DVC lockfile
├── requirements.txt                   # Dependencies
├── pytest.ini                         # Test config
└── README.md                          # This file
```

**Parent directory:**

- Dockerfile.airflow
- docker-compose.airflow.yml
- docs/ (screenshots)
- .github/workflows/ci.yml

---

## Pipeline Visualizations

**Airflow Screenshots in docs/:**

**airflow_graph_success.png**

- DAG graph view
- All 8 tasks with dependencies
- Successful execution (green)

**airflow_gantt_success.png**

- Execution timeline
- Task duration breakdown
- Bottleneck identification

**dvcDAG.png**

- DVC pipeline visualization
- Data lineage graph

---

## Development Workflow

### For Local Development (Optional)

**If you want to modify scripts:**

```bash
cd Data-Pipeline

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test changes
python scripts/download_coco2017.py --help

# Run tests
pytest tests/ -v

# Run DVC pipeline
dvc repro
```

**When to use local setup:**

- Script development
- Quick testing
- Debugging
- No Docker available

**When to use Docker:**

- Running complete pipeline
- Demonstrating to team/professor
- Production deployment
- Avoiding dependency issues

---

## MLOps Best Practices

### What This Project Demonstrates

**1. Containerization**

- Docker for reproducible environments
- Production-ready deployment
- Dependency isolation

**2. Workflow Orchestration**

- Airflow for complex workflows
- DVC for data-centric pipelines
- Hybrid approach (Airflow + DVC)

**3. Data Version Control**

- DVC tracks data via hashes
- Reproducible data transformations
- Git for code, DVC for data

**4. Testing & Quality**

- Pytest for automated testing
- CI/CD with GitHub Actions
- Quality gates prevent bad data

**5. Monitoring & Observability**

- Airflow UI for real-time monitoring
- Schema validation for data quality
- Anomaly detection for data issues

**6. Fairness & Ethics**

- Bias detection via data slicing
- Mitigation through stratified sampling
- Documented trade-offs

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

## Acknowledgments

- COCO Dataset: Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014
- Apache Airflow community
- DVC team
- Course materials from Prof. Mohammadi

---

**Repository:** https://github.com/hemanth1403/IE7374-MLOps-Adaptive-ML-Inference

<!-- _Last Updated: February 24, 2026_ -->
