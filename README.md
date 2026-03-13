# DPArena - Simulation platform for data poisoning in federated Learning

A full-stack platform for simulating **data poisoning attacks** within a **Federated Learning (FL)** environment. Users define neural network models (TensorFlow or PyTorch), configure attack scenarios, and run multi-GPU simulations — all through a Jupyter-inspired web interface with real-time progress tracking, result analysis, and export capabilities.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Infrastructure Requirements](#infrastructure-requirements)
- [Getting Started](#getting-started)
  - [Machine 1 — Application Server (Docker)](#machine-1--application-server-docker)
  - [Machine 2 — GPU Server (Orchestrator)](#machine-2--gpu-server-orchestrator)
- [Orchestrator Pipeline — Step by Step](#orchestrator-pipeline--step-by-step)
- [Platform Features](#platform-features)
  - [Simulation Parameters](#simulation-parameters)
  - [Data Poisoning Attacks](#data-poisoning-attacks)
  - [Robust Aggregation Defenses](#robust-aggregation-defenses)
  - [Evaluation Metrics](#evaluation-metrics)
- [Frontend Features](#frontend-features)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## Architecture Overview

The platform follows a **two-machine distributed architecture**:

![Platform Architecture](App_Architecture.png)

---

## Infrastructure Requirements

| Component | Machine 1 — App Server | Machine 2 — GPU Server |
|-----------|----------------------|----------------------|
| **Purpose** | Hosts the web UI, API, and database | Runs the actual FL simulations |
| **Docker** | ✅ Required | ❌ Not required |
| **GPU** | ❌ Not required | ✅ Required (NVIDIA, CUDA) |
| **Storage** | ~10 GB (Docker images + DB) | ~50+ GB (datasets, models, conda envs) |
| **RAM** | 4 GB minimum | 16+ GB recommended |
| **Software** | Docker, Docker Compose | Python 3.10+, Anaconda, nvidia-smi |
| **Network** | Must be reachable by users | Must be reachable by Machine 1 |

---

## Getting Started

### Machine 1 — Application Server (Docker)

#### 1. Clone the repository

```bash
git clone https://github.com/Tudorx95/BachelorThesisProject.git
cd BachelorThesisProject
```

#### 2. Configure secrets

Create the `secrets/` directory and populate the secret files:

```bash
mkdir -p secrets
echo "your-db-password" > secrets/db_password.txt
echo "your-jwt-secret-key" > secrets/secret_key.txt
echo "your-orchestrator-password" > secrets/orchestrator_password.txt
```

> **⚠️ Important:** Never commit the `secrets/` directory to version control. It is already excluded via `.gitignore`.

#### 3. Update orchestrator URL

In `docker-compose.yaml`, update the `ORCHESTRATOR_URL` environment variable in the `backend` service to point to your GPU server:

```yaml
- ORCHESTRATOR_URL=http://<GPU_SERVER_IP>:8000
```

#### 4. Start the platform

```bash
docker compose up --build -d
```

This starts three containers:

| Container | Port | Description |
|-----------|------|-------------|
| `fl_postgres` | 5432 | PostgreSQL 16 database |
| `fl_backend` | 8000 | FastAPI REST API + WebSocket |
| `fl_frontend` | 3000 | React web application |

Access the platform at **http://localhost:3000**.

#### 5. Deploying on a Remote Server

When the application server has a different IP (e.g., `10.13.70.3`) and users will access the platform from other machines, you need to update **two files** so the frontend can reach the backend and the backend accepts requests from the correct origin.

##### 5.1. `docker-compose.yaml` — Frontend environment variables

In the `frontend` service, replace `localhost` with the server's IP:

```diff
     environment:
-      - REACT_APP_API_URL=http://localhost:8000
-      - REACT_APP_WS_URL=ws://localhost:8000
+      - REACT_APP_API_URL=http://10.13.70.3:8000
+      - REACT_APP_WS_URL=ws://10.13.70.3:8000
```

> **Why:** The React app runs in the user's browser, so it must call the backend using a routable IP, not `localhost`.

##### 5.2. `backend/main.py` — CORS `allow_origins`

Add the server's IP to the `allow_origins` list in the CORS middleware configuration:

```diff
 app.add_middleware(
     CORSMiddleware,
-    allow_origins=["http://localhost:3000", "http://frontend:3000"],
+    allow_origins=["http://localhost:3000", "http://frontend:3000", "http://10.13.70.3:3000"],
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
 )
```

> **Why:** Without the server IP in `allow_origins`, the browser will block API requests due to CORS policy.

After making both changes, restart the platform:

```bash
docker compose down && docker compose up --build -d
```

#### 6. Demo credentials

A demo user is created automatically:
- **Username:** `demo`
- **Password:** `demo123`

---

### Machine 2 — GPU Server (Orchestrator)

#### 1. Copy the orchestrator files

Transfer the `orchestrator_backend_server/` directory to the GPU server:

```bash
scp -r orchestrator_backend_server/ user@gpu-server:/path/to/orchestrator/
```

#### 2. Install dependencies

```bash
cd /path/to/orchestrator
pip install -r requirements.txt
```

#### 3. Set up Conda environments

Create separate Conda environments for each supported framework:

```bash
# TensorFlow environment
conda create -n fl_tensorflow python=3.10 -y
conda activate fl_tensorflow
pip install tensorflow numpy Pillow scikit-learn

# PyTorch environment
conda create -n fl_pytorch python=3.10 -y
conda activate fl_pytorch
pip install torch torchvision numpy Pillow scikit-learn
```

#### 4. Configure credentials

In `orchestrator_gpu.py`, update the `VALID_USERS` dictionary and file paths:

```python
VALID_USERS = {"tudor": "your-orchestrator-password"}
BASE_DIR = Path("/path/to/fl_simulations")
CONDA_BASE = Path("/path/to/anaconda3")
```

#### 5. Start the orchestrator

```bash
nohup python -u orchestrator_gpu.py > output.log 2>&1 &
```

Verify it is running:

```bash
curl http://localhost:8000/
# Expected: {"message": "FL Orchestrator API", "status": "running", "available_gpus": [...]}
```

Monitor logs in real time:

```bash
tail -f output.log
```

---

## Orchestrator Pipeline — Step by Step

The simulation process defines the complete execution pipeline that is triggered when a user submits a simulation from the frontend. The pipeline is managed by the Python Orchestrator on the GPU server (`run_simulation_pipeline()`) and consists of a series of sequential stages, each with built-in error handling. If any stage fails, the pipeline halts and generates an error report that is relayed back to the user.

### Stage 1 — Start Simulation Task

The orchestrator creates a new task entry and begins the pipeline execution asynchronously. The task status is set to `running` and is tracked via a shared `active_simulations` dictionary.

### Stage 2 — Verify Template Structure

Before consuming any computational resources, the orchestrator performs a **static analysis** of the user's training script using the `verify_template.py` module. This ensures the template defines all required functions (e.g., `download_data()`, model creation, proper training structure) and follows the expected interface contract.

### Stage 3 — Create Task Directory

The orchestrator creates an isolated workspace directory for the current simulation task, following the naming convention `user_{id}/task_{uuid}`:

```
fl_simulations/
 └── user_{id}/
      └── {task_uuid}/
           ├── clean_data/            # Original dataset
           ├── clean_data_poisoned/   # Poisoned dataset copy
           └── results/               # Output metrics & analysis
```

### Stage 4 — Verify Model Correctness

Using the template script provided by the user, the orchestrator loads the model and runs a training session to prove the template's usability. The model is executed on the allocated GPU with `CUDA_VISIBLE_DEVICES` set, and the initial accuracy is recorded in `init-verification.json`.

### Stage 5 — Download Model from Repository

The orchestrator invokes the model creation logic defined in the user's training script. If the template references a pre-trained model hosted on HuggingFace, it is downloaded at this stage. The trained model is saved in the appropriate format based on the detected framework (`.keras` for TensorFlow, `.pth` for PyTorch) and renamed to match the configured `NN_NAME`.

### Stage 6 — Download Training and Testing Dataset

The orchestrator calls the `download_data()` function defined in the user's training script to retrieve and prepare the dataset required for the simulation into the `clean_data/` directory. After download, the data is partitioned into `N` equal IID splits (one per federated client), while the test set is shared across all clients.

### Stage 7 — Allocate GPU

The `GPUManager` queries all available NVIDIA GPUs on the system via `nvidia-smi` and maintains a thread-safe queue. Each simulation requests a dedicated GPU (with a 10-minute timeout). If no GPUs are available within the timeout, the simulation falls back to CPU execution. The GPU is **guaranteed to be released** upon task completion, cancellation, or failure (via the `finally` block).

### Stage 8 — Activate Python Environment

The orchestrator performs **automatic framework detection** by analyzing the import statements in the user's training script. Based on whether the script uses TensorFlow or PyTorch, the orchestrator activates the corresponding pre-configured Conda environment (`fl_tensorflow` or `fl_pytorch`).

### Stage 9 — Create a Poisoned Copy of the Original Dataset

The orchestrator runs the `poison_data.py` script using the attack type and parameters selected by the user. This stage creates a copy of the clean dataset and applies the configured data poisoning attack to it, generating the `clean_data_poisoned/` directory. Supports 7 attack types with per-attack parameters (see [Data Poisoning Attacks](#data-poisoning-attacks)).

### Stage 10–13 — Federated Learning Simulations (4 Scenarios)

With both the clean and poisoned datasets prepared, the pipeline enters the federated learning simulation phase. This phase executes **four distinct scenarios** sequentially using the `fd_simulator.py` script:

| Stage | Scenario | Data | Aggregation | Purpose |
|-------|----------|------|-------------|---------|
| **10** | Clean | `clean_data/` | FedAvg | Baseline — ideal (attack-free) conditions |
| **11** | Clean + Defense | `clean_data/` | User-selected method | Measure defense overhead without attacks |
| **12** | Poisoned | `clean_data_poisoned/` | FedAvg | Measure attack impact without defense |
| **13** | Poisoned + Defense | `clean_data_poisoned/` | User-selected method | Evaluate defense effectiveness |

- **Scenario 1 (Stage 10)** — The first simulation uses exclusively the clean (unmodified) dataset. Data is distributed across `N` clients, none of which are malicious, and training proceeds for `ROUNDS` rounds using FedAvg. This establishes the baseline performance.
- **Scenario 2 (Stage 11)** — The second simulation trains the global model on clean data, but replaces FedAvg with the user-selected robust aggregation method. This measures the overhead introduced by the defense in a non-adversarial environment, ensuring it does not significantly degrade model performance.
- **Scenario 3 (Stage 12)** — The third simulation introduces the poisoned dataset. `M` out of `N` clients are designated as malicious and receive the poisoned data, while the remaining clients use clean data. Aggregation uses FedAvg without any defense. Malicious clients use the poisoned data for `R` out of `ROUNDS` total rounds, following the user-selected distribution strategy.
- **Scenario 4 (Stage 13)** — The final simulation replicates the poisoned scenario but replaces FedAvg with the robust aggregation method selected by the user. By comparing the results of Scenario 4 against Scenario 2, users can assess how well the chosen aggregation strategy protects the global model against the specific poisoning attack.

### Stage 14 — Generate Final Results

After all four federated learning scenarios complete successfully, the orchestrator aggregates and compiles the final results. It computes comprehensive evaluation metrics including:

- **Accuracy** — values for each scenario (Initial, Clean, Clean + Defense, Poisoned, Poisoned + Defense)
- **Accuracy drops** — between scenarios to quantify attack impact and defense effectiveness
- **Precision, Recall, and F1 Score** — weighted average per scenario

Results are saved to:
- `results/analysis.json` — structured JSON with all metrics
- `results/summary.txt` — human-readable summary
- `results/attack_info.json` — poisoning attack details

The orchestrator marks the task as `completed`, and the backend polls these results back to the frontend via WebSocket.

### Cancellation Flow

At every pipeline stage, the orchestrator checks if the user has requested cancellation. If so:
1. The running process tree is killed (`kill_process_tree()`)
2. The allocated GPU is released
3. The task directory is deleted
4. Status is set to `cancelled`

---

## Platform Features

### Simulation Parameters

| Parameter | Description |
|-----------|-------------|
| `N` | Total number of FL clients |
| `M` | Number of malicious clients |
| `ROUNDS` | Total training rounds |
| `R` | Number of rounds malicious clients use poisoned data |
| `NN_NAME` | Neural network model name |
| `strategy` | Malicious client distribution: `first`, `last`, or `alternate` |

### Data Poisoning Attacks

| Attack Type | Description |
|-------------|-------------|
| **Label Flip** | Flips the class label of a percentage of samples to a target class (user-selected or random) |
| **Backdoor — BadNets** | Injects a small trigger pattern (cross, square, etc.) onto a percentage of images |
| **Backdoor — Blended** | Blends the original image with a key pattern using a configurable ratio ([paper](https://arxiv.org/pdf/1712.05526)) |
| **Backdoor — Sinusoidal** | Adds a sinusoidal signal to images with user-configurable frequency and amplitude |
| **Backdoor — Trojan** | Inserts a watermark-style trigger into images (conceptually similar to BadNets) |
| **Backdoor — Semantic** | Modifies natural image features (brightness, RGB tint) |
| **Edge-Case Backdoor** | Applies image transformations: rotation, color reduction, solarization, or grayscale intensity reduction |

### Robust Aggregation Defenses

| Method | Description |
|--------|-------------|
| **FedAvg** | Weighted average of client weights, proportional to dataset size. Standard method, but vulnerable to poisoning |
| **Krum** | Selects the single client update with lowest total distance to all others, isolating malicious clients |
| **Trimmed Mean** | Removes the top and bottom 20% extreme weight values, then averages the remainder |
| **Median** | Replaces the mean with a per-parameter median, resistant to up to ~20% malicious clients |
| **Trimmed Mean + Krum** | Hybrid approach combining Trimmed Mean filtering (trim_ratio=0.1) with Krum selection |
| **Random** | Randomly selects one client's weights. Used as a comparison baseline |

### Evaluation Metrics

Collected **per round** and **per scenario** (Clean / Clean+Defense / Poisoned / Poisoned+Defense):

- **Accuracy** — global classification accuracy
- **Precision** — weighted average
- **Recall** — weighted average
- **F1 Score** — weighted average

---

## Frontend Features

- **Jupyter-style notebook interface** with code cells and output cells
- **Monaco Code Editor** with syntax highlighting and predefined templates (TensorFlow / PyTorch toggle)
- **Sidebar** with project and file management, including drag-and-drop reordering
- **Real-time simulation progress** with step-by-step status indicators (ProgressStep component)
- **Simulation results display** — summary text + detailed JSON analysis
- **Advanced simulation configuration** — modal with full control over clients, rounds, attack type, attack parameters, and defense method
- **Compare Page** — side-by-side comparison of two simulations, including FL config, attack parameters, and all metrics
- **Graphs Page** — interactive bar charts (Recharts) for visualizing multiple simulations with grouped metrics
- **PDF export** — individual simulation results as PDF (includes config, metrics, confusion matrix data, and summary)
- **CSV export** — multi-select file export with all metrics per scenario in a tabular format
- **Dark / Light mode** with persistent user preference

---

## API Reference

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register a new user |
| POST | `/api/auth/login` | Login and receive JWT token |
| GET | `/api/auth/me` | Get current user info |

### Projects

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/projects` | List user's projects |
| POST | `/api/projects` | Create a new project |
| GET | `/api/projects/:id` | Get project details |
| DELETE | `/api/projects/:id` | Delete a project |

### Files

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/projects/:id/files` | List files in a project |
| POST | `/api/projects/:id/files` | Create a new file |
| GET | `/api/files/:id` | Get file content |
| PUT | `/api/files/:id` | Update file content |
| DELETE | `/api/files/:id` | Delete a file |
| PATCH | `/api/files/:id/rename` | Rename a file |
| PATCH | `/api/files/:id/move` | Move file to another project |
| POST | `/api/files/reorder` | Bulk reorder files |

### Simulations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/run` | Submit a simulation |
| POST | `/cancel/:task_id` | Cancel a running simulation |
| GET | `/api/simulations` | List simulation history |
| GET | `/api/simulations/:task_id` | Get simulation result |

### Orchestrator (GPU Server)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check + GPU status |
| POST | `/login` | Authenticate |
| POST | `/simulate` | Start a simulation pipeline |
| GET | `/status/:task_id` | Poll task status |
| GET | `/results/:task_id` | Fetch completed results |
| POST | `/cancel/:task_id` | Cancel and cleanup |
| GET | `/gpu_status` | Current GPU allocation info |

---

## Project Structure

```
BachelorThesisProject/
│
├── docker-compose.yaml              # Multi-container setup (DB + Backend + Frontend)
├── secrets/                          # Docker secrets (gitignored)
│   ├── db_password.txt
│   ├── secret_key.txt
│   └── orchestrator_password.txt
│
├── backend/                          # FastAPI REST API
│   ├── Dockerfile
│   ├── main.py                       # All endpoints, models, auth, WebSocket
│   └── requirements.txt
│
├── simulator_frontend/               # React SPA
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       ├── App.jsx                   # Main app with routing
│       ├── components/
│       │   ├── CodeCell.jsx          # Monaco code editor cell
│       │   ├── OutputCell.jsx        # Simulation output display
│       │   ├── Sidebar.jsx           # Project/file tree + drag-and-drop
│       │   ├── SimulationOptions.jsx # Attack & defense config modal
│       │   ├── ProgressStep.jsx      # Real-time step indicators
│       │   ├── ExportPDFButton.jsx   # PDF result export
│       │   ├── MultiExportCSV.jsx    # Multi-simulation CSV export
│       │   └── TopBar.jsx            # Top navigation bar
│       ├── pages/
│       │   ├── Login.jsx
│       │   ├── Register.jsx
│       │   ├── ComparePage.jsx       # Side-by-side simulation comparison
│       │   └── GraphsPage.jsx        # Interactive charts
│       └── context/                  # React Context (auth, simulation state)
│
├── database/
│   └── init.sql                      # Schema + demo data
│
├── orchestrator_backend_server/      # GPU server (runs independently)
│   ├── orchestrator_gpu.py           # Flask API + simulation pipeline
│   ├── gpu_manager.py                # GPU allocation/deallocation
│   ├── fd_simulator.py               # Federated Learning simulator
│   ├── poison_data.py                # Data poisoning engine (v1)
│   ├── poison_data_v2.py             # Extended poisoning (7 attack types)
│   ├── fl_monitor.py                 # FL monitoring utilities
│   ├── train_model.py                # Model training helper
│   ├── template_code.py              # TensorFlow template reference
│   ├── template_code_pytorch.py      # PyTorch template reference
│   └── requirements.txt
│
└── Templates/                        # Code templates shown in the editor
    ├── template_antrenare_tensorflow.py
    ├── template_antrenare_pytorch.py
    └── verify_template.py            # Template structure validator
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18, Monaco Editor, Recharts, TailwindCSS, Lucide Icons |
| **Backend API** | FastAPI, SQLAlchemy, Pydantic, PyJWT, Passlib (bcrypt) |
| **Database** | PostgreSQL 16 |
| **Orchestrator** | Flask, multiprocessing, psutil, GPUManager |
| **ML Frameworks** | TensorFlow / PyTorch (via Conda environments) |
| **Infrastructure** | Docker, Docker Compose, Docker Secrets |
| **GPU Management** | nvidia-smi, CUDA, per-process GPU isolation |

---

## License

This project is part of a Bachelor's Thesis at the Military Technical Academy "Ferdinand I" Bucharest.
