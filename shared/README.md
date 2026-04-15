# Shared

Cross-service definitions used across the platform. Intended to be the single source of truth for data schemas, API contracts, and constants shared between services.

---

## Directory Structure

```
shared/
├── contracts/
│   └── model_contract.md    # Data flow contracts between pipeline stages
├── schemas/                 # JSON/Pydantic schemas (planned)
└── constants/               # Shared constants (planned)
```

---

## contracts/model_contract.md

Defines the data flow contracts between the pipeline stages:

- What the Data Pipeline produces (image paths, annotation format, split CSVs)
- What the model evaluation pipeline consumes and produces (metrics CSVs, benchmark JSONs)
- What `profile_models.py` produces (`model_performance_profile.csv`) and what the RL environment expects
- What `engine.infer()` returns (detection schema, latency, confidence)
- What the WebSocket protocol exchanges (base64 frame → JSON result)

Read this file before implementing any new service or modifying inter-component data formats.

---

## schemas/ and constants/

Currently empty. Intended future use:

- **schemas/** — Pydantic models or JSON Schema definitions for request/response validation across services
- **constants/** — Shared numeric constants (observation space size, model index mappings, reward weights) to avoid duplication across `environment.py`, `engine.py`, and any future services
