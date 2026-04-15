# Services

Planned microservice decomposition of the inference platform. These directories are scaffolded for a future split into independent, separately deployable services.

**Current state:** The full serving stack is implemented as a monolith in [`model_pipeline/src/RL/serving/`](../model_pipeline/src/RL/serving/). This directory contains empty stubs for the planned decomposition.

---

## Planned Services

| Service | Port | Responsibility |
|---|---|---|
| `orchestrator/` | 8000 | Receives frames, calls RL policy, fans out to selected YOLO service, returns result |
| `rl_policy_service/` | 8001 | Loads PPO agent, accepts observation vector, returns action (0/1/2) |
| `yolo_nano_service/` | 8002 | Serves YOLOv8-nano inference only |
| `yolo_small_service/` | 8003 | Serves YOLOv8-small inference only |
| `yolo_large_service/` | 8004 | Serves YOLOv8-large inference only |

---

## Current Architecture vs Planned

**Today** — `engine.py` loads all three YOLO models in-process alongside the RL agent. One pod, one container, all inference in a single Python process.

**Future** — Each YOLO variant becomes its own pod, scaled independently based on routing frequency. The RL policy service is stateless and can be replicated freely. The orchestrator becomes a lightweight API gateway.

This decomposition would allow, for example, scaling YOLOv8-nano replicas up (high routing frequency) while keeping large at one replica (rarely needed).

---

## Shared Contracts

Service interfaces are defined in [`shared/contracts/model_contract.md`](../shared/contracts/model_contract.md). Any implementation of these services should conform to those schemas.
