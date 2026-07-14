# DPArena — Simulation Platform for Data Poisoning in Federated Learning

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
  - [Custom Functions](#custom-functions)
  - [Evaluation Metrics](#evaluation-metrics)
- [Frontend Features](#frontend-features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## Architecture Overview

The platform follows a **two-machine distributed architecture**:

![Platform Architecture](App_Architecture.png)

---

## Infrastructure Requirements

| Component    | Machine 1 — App Server              | Machine 2 — GPU Server                 |
| ------------ | ----------------------------------- | -------------------------------------- |
| **Purpose**  | Hosts the web UI, API, and database | Runs the actual FL simulations         |
| **Docker**   | ✅ Required                         | ❌ Not required                        |
| **GPU**      | ❌ Not required                     | ✅ Required (NVIDIA, CUDA)             |
| **Storage**  | ~10 GB (Docker images + DB)         | ~50+ GB (datasets, models, conda envs) |
| **RAM**      | 4 GB minimum                        | 16+ GB recommended                     |
| **Software** | Docker, Docker Compose              | Python 3.10+, Anaconda, nvidia-smi     |
| **Network**  | Must be reachable by users          | Must be reachable by Machine 1         |

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

| Container     | Port | Description                  |
| ------------- | ---- | ---------------------------- |
| `fl_postgres` | 5432 | PostgreSQL 16 database       |
| `fl_backend`  | 8000 | FastAPI REST API + WebSocket |
| `fl_frontend` | 3000 | React web application        |

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

The project includes **automated setup scripts** that create fully configured Conda environments with all required dependencies, correct versions, and CUDA support.

##### Option A — Automated setup (recommended)

The setup scripts ship inside `orchestrator_backend_server/`, so they are already on the GPU server after Step 1. From the orchestrator directory, make them executable and run them:

```bash
# Make the scripts executable
chmod +x setup_tensorflow_env.sh setup_pytorch_env.sh

# Create the TensorFlow environment (fl_tensorflow)
./setup_tensorflow_env.sh

# Create the PyTorch environment (fl_pytorch)
./setup_pytorch_env.sh
```

Each script will:

1. Create a dedicated Conda environment (`fl_tensorflow` / `fl_pytorch`) with Python 3.10
2. Install the ML framework with CUDA support
3. Install all scientific, data-processing, and networking dependencies (NumPy, Pandas, Matplotlib, scikit-learn, Flask, etc.)
4. Run verification checks to confirm the installation and GPU visibility

<details>
<summary><strong>setup_tensorflow_env.sh</strong> — packages installed</summary>

| Package                | Version | Purpose                           |
| ---------------------- | ------- | --------------------------------- |
| `tensorflow[and-cuda]` | 2.15.1  | TensorFlow with bundled CUDA 12.2 |
| `numpy`                | 1.26.4  | Numerical computing               |
| `pandas`               | 2.1.4   | Data manipulation                 |
| `matplotlib`           | 3.8.4   | Plotting                          |
| `scikit-learn`         | 1.3.2   | ML utilities & metrics            |
| `Pillow`               | 10.4.0  | Image processing                  |
| `scipy`                | 1.11.4  | Scientific computing              |
| `requests`             | 2.32.3  | HTTP client                       |
| `flask`                | 3.0.3   | Orchestrator API                  |
| `flask-cors`           | 4.0.1   | CORS middleware                   |
| `h5py`                 | 3.11.0  | HDF5 model storage                |
| `tensorflow-datasets`  | 4.9.6   | TF dataset utilities              |

> **Note:** Requires NVIDIA driver version ≥ 535 for CUDA 12.2 compatibility.

</details>

<details>
<summary><strong>setup_pytorch_env.sh</strong> — packages installed</summary>

| Package        | Version | Purpose                   |
| -------------- | ------- | ------------------------- |
| `torch`        | 2.1.0   | PyTorch (CUDA 12.1 wheel) |
| `torchvision`  | 0.16.0  | Image models & transforms |
| `torchaudio`   | 2.1.0   | Audio processing          |
| `numpy`        | 1.26.0  | Numerical computing       |
| `pandas`       | 2.1.0   | Data manipulation         |
| `matplotlib`   | 3.8.0   | Plotting                  |
| `scikit-learn` | 1.3.0   | ML utilities & metrics    |
| `Pillow`       | 10.1.0  | Image processing          |
| `scipy`        | 1.11.0  | Scientific computing      |
| `requests`     | 2.31.0  | HTTP client               |
| `flask`        | 3.0.0   | Orchestrator API          |
| `flask-cors`   | 4.0.0   | CORS middleware           |
| `torchmetrics` | 1.2.0   | Training metrics          |
| `tensorboard`  | 2.15.0  | Training visualization    |

</details>

##### Option B — Manual setup

If you prefer to install packages manually:

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

The simulation process defines the complete execution pipeline that is triggered when a user submits a simulation from the frontend. The pipeline is managed by the Python Orchestrator on the GPU server (`run_simulation_pipeline()`) and consists of **11 sequential stages**, each with built-in error handling and cancellation checks. If any stage fails, the pipeline halts and generates an error report that is relayed back to the user.

### Step 1 — Preparing Environment

The orchestrator creates a new task entry and sets up the isolated workspace for the simulation. It creates the directory structure `user_{id}/{task_uuid}/`, saves the user's template code as `template_code.py`, persists the simulation configuration as `simulation_config.json`, and **detects the ML framework** (TensorFlow or PyTorch) by analyzing the import statements in the template. All subsequent steps run in CPU-only mode until a GPU is allocated at Step 5.

### Step 2 — Verify Template Structure

Before consuming any computational resources, the orchestrator performs a **static analysis** of the user's training script using `verify_template.py`. This module imports the template, detects the framework, and validates that all **17 required functions** exist with correct signatures (e.g., `download_data()`, `create_model()`, `train_neural_network()`, `get_model_weights()`, `set_model_weights()`, etc.). It also creates and validates the model, tests weight extraction and injection, and generates `init-verification.json` with model metadata. At this stage, dataset loading is intentionally **skipped** (`VERIFY_COMPUTE_METRICS=0`) to avoid triggering a slow dataset download during validation.

### Step 3 — Data Generation

The orchestrator invokes the `download_data()` function from the user's template to retrieve and prepare the dataset into the `clean_data/` directory. After a successful download, the `compute_init_metrics.py` script runs separately to calculate the **initial model accuracy** on the test set and updates `init-verification.json` with the results. This two-phase approach (verify structure first, compute metrics after download) avoids downloading the dataset during the fast validation gate at Step 2.

### Step 4 — Generate Data Distribution (Clean + Poisoned)

This step produces the **per-client, per-round data mappings** that drive the FL simulation:

1. **Clean folds** — `generate_folds.py` creates **stratified K-fold** splits from the dataset, distributes training data across `N` clients using the configured distribution strategy (Fixed Non-IID or Dirichlet), and generates per-client JSON files in `data_distribution_clean/`. Each JSON maps round → file list, with class counts and metadata.

2. **Poisoned data** — `generate_poisoned_per_client.py` reads the clean distribution, identifies malicious clients based on the configured strategy, and for each malicious client × round: samples `poison_percentage%` of their allocated files, applies the selected attack (label flip or pixel-level backdoor), and saves the modified images to `clean_data_poisoned/R{x}/client_{y}/`. Honest clients simply reference the original clean data. An `attack_info.json` is generated with full poisoning metadata.

### Step 5 — Allocate GPU

The `GPUManager` queries all available NVIDIA GPUs on the system via `nvidia-smi` and maintains a thread-safe queue. Each simulation requests a dedicated GPU (with a 10-minute timeout). If no GPUs are available within the timeout, the simulation falls back to CPU execution. The GPU is **guaranteed to be released** upon task completion, cancellation, or failure (via the `finally` block). From this point forward, all subprocesses run with `CUDA_VISIBLE_DEVICES` set to the allocated GPU.

### Step 6 — Execute Template Code (Model Training)

The orchestrator runs the user's `template_code.py` on the allocated GPU. This typically creates and trains the model (or downloads a pre-trained model from HuggingFace). After execution, the model file is detected and renamed to match the configured `NN_NAME` (`.keras` for TensorFlow, `.pth` for PyTorch).

### Steps 7–10 — Federated Learning Simulations (4 Scenarios)

With both the clean and poisoned datasets prepared, the pipeline enters the federated learning simulation phase. This phase executes **four distinct scenarios** sequentially using the `fd_simulator.py` script:

| Step   | Scenario           | Data                      | Aggregation          | Purpose                                   |
| ------ | ------------------ | ------------------------- | -------------------- | ----------------------------------------- |
| **7**  | Clean              | `data_distribution_clean` | FedAvg               | Baseline — ideal (attack-free) conditions |
| **8**  | Clean + Defense    | `data_distribution_clean` | User-selected method | Measure defense overhead without attacks  |
| **9**  | Poisoned           | `data_distribution_poisoned` | FedAvg            | Measure attack impact without defense     |
| **10** | Poisoned + Defense | `data_distribution_poisoned` | User-selected method | Evaluate defense effectiveness         |

Each scenario uses the **fold-based cross-validation** mappings generated at Step 4. The FL simulator loads per-client, per-round file lists from the distribution JSONs, creates DataLoaders dynamically, and trains client models in parallel threads with GPU semaphore control.

### Step 11 — Generate Final Results

After all four scenarios complete, the orchestrator aggregates the final results. It computes:

- **Accuracy** — per scenario (Initial, Clean, Clean + Defense, Poisoned, Poisoned + Defense)
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

| Parameter              | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| `N`                    | Total number of FL clients                                     |
| `M`                    | Number of malicious clients                                    |
| `ROUNDS`               | Total training rounds                                          |
| `R`                    | Number of rounds malicious clients use poisoned data           |
| `EPOCHS`               | Local training epochs per round (default: 3)                   |
| `NN_NAME`              | Neural network model name                                      |
| `strategy`             | Malicious client distribution: `first`, `last`, or `alternate` |
| `data_distribution`    | Data distribution strategy: `fixed` or `dirichlet`             |
| `dominant_percentage`  | Dominant class percentage for fixed Non-IID (default: 80%)     |
| `dirichlet_alpha`      | Dirichlet concentration parameter (default: 0.5)               |

### Data Poisoning Attacks

| Attack Type               | Description                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Label Flip**            | Flips the class label of a percentage of samples to a target class (user-selected or random)                        |
| **Backdoor — BadNets**    | Injects a small trigger pattern (cross, square, etc.) onto a percentage of images                                   |
| **Backdoor — Blended**    | Blends the original image with a key pattern using a configurable ratio ([paper](https://arxiv.org/pdf/1712.05526)) |
| **Backdoor — Sinusoidal** | Adds a sinusoidal signal to images with user-configurable frequency and amplitude                                   |
| **Backdoor — Trojan**     | Inserts a watermark-style trigger into images (conceptually similar to BadNets)                                     |
| **Backdoor — Semantic**   | Modifies natural image features (brightness, RGB tint)                                                              |
| **Edge-Case Backdoor**    | Applies image transformations: rotation, color reduction, solarization, or grayscale intensity reduction            |

### Robust Aggregation Defenses

| Method           | Description                                                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FedAvg**       | Weighted average of client weights, proportional to dataset size. Standard method, but vulnerable to poisoning                                    |
| **Krum**         | Selects the single client update with lowest total distance to all others, isolating malicious clients                                            |
| **Trimmed Mean** | Removes the top and bottom extreme weight values per parameter, then averages the remainder                                                       |
| **Median**       | Replaces the mean with a per-parameter median, resistant to up to ~20% malicious clients                                                          |
| **FoolsGold**    | Detects Sybil attacks by tracking accumulated gradient histories and penalizing clients with high pairwise cosine similarity ([paper](https://arxiv.org/abs/1808.04866)) |
| **Norm Clipping**| Clips each client's update to the median L2 norm, limiting the influence of any single client ([paper](https://arxiv.org/abs/1911.07963))        |
| **Random**       | Randomly selects one client's weights. Used as a comparison baseline                                                                              |
| **Custom (@)**   | User-defined aggregation function uploaded via the platform (see [Custom Functions](#custom-functions))                                           |

### Custom Functions

The platform supports **user-defined aggregation functions** that can be uploaded through the frontend and executed during the FL simulation:

- **Custom Aggregation** — users write a Python function named `custom_aggregate(client_weights, client_sizes, ...)` that implements their own aggregation logic. The function is validated (syntax check via `ast.parse()`, security scan via OpenGrep/Semgrep) and uploaded to the orchestrator. Referenced in simulations with the `@function_name` prefix.

- **Security Scanning** — all user-uploaded code is automatically scanned using **OpenGrep (Semgrep)** with the `auto` config ruleset. Code with HIGH or CRITICAL severity findings is rejected before it reaches the GPU server.

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
- **Advanced simulation configuration** — modal with full control over clients, rounds, attack type, attack parameters, defense method, data distribution, and custom functions
- **Custom function upload** — modal editors for uploading custom aggregation and poisoning functions with built-in validation
- **Compare Page** — side-by-side comparison of two simulations, including FL config, attack parameters, and all metrics
- **Graphs Page** — interactive bar charts (Recharts) for visualizing multiple simulations with grouped metrics
- **PDF export** — individual simulation results as PDF (includes config, metrics, confusion matrix data, and summary)
- **CSV export** — multi-select file export with all metrics per scenario in a tabular format
- **Dark / Light mode** with persistent user preference

---

## Project Structure

```
DPArena/
│
├── docker-compose.yaml              # Multi-container setup (DB + Backend + Frontend)
├── secrets/                          # Docker secrets (gitignored)
│   ├── db_password.txt
│   ├── secret_key.txt
│   └── orchestrator_password.txt
│
├── backend/                          # FastAPI REST API
│   ├── Dockerfile
│   ├── main.py                       # All endpoints, models, auth, WebSocket, OpenGrep
│   └── requirements.txt
│
├── simulator_frontend/               # React SPA
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       ├── App.jsx                   # Main app with routing
│       ├── components/               # 10 components (editor, sidebar, modals, export)
│       ├── pages/                    # 4 pages (Login, Register, Compare, Graphs)
│       └── context/                  # Auth, Simulation, Theme contexts
│
├── database/
│   └── init.sql                      # Schema + demo data
│
├── orchestrator_backend_server/      # GPU server (runs independently)
│   ├── orchestrator_gpu.py           # Flask API + simulation pipeline
│   ├── gpu_manager.py                # GPU allocation/deallocation
│   ├── fd_simulator.py               # Federated Learning simulator
│   ├── poison_data.py                # Data poisoning engine (7 attack types)
│   ├── verify_template.py            # Template structure validator
│   ├── generate_folds.py             # Stratified fold generation for FL clients
│   ├── generate_poisoned_per_client.py # Per-client poisoned data generation
│   ├── compute_init_metrics.py       # Initial model metrics computation
│   ├── setup_tensorflow_env.sh       # Automated Conda env setup — TensorFlow + CUDA
│   ├── setup_pytorch_env.sh          # Automated Conda env setup — PyTorch + CUDA
│   └── requirements.txt
│
├── Templates/
│   └── New_Template_Fold/            # Active templates (TensorFlow + PyTorch)
│       ├── template_code_tensorflow.py
│       ├── template_code_pytorch.py
│       └── template_code_pytorch_cifar100.py
│
└── Experiments/                      # Experiment result artifacts (exported PDF/PNG per scenario)
    ├── Case_1/                       # Cross-validation comparison (Case A / Case B)
    └── Case_2/                       # Defense comparison — FoolsGold, Median, Trimmed Mean, Norm Clipping
```

---

## Tech Stack

| Layer              | Technology                                                   |
| ------------------ | ------------------------------------------------------------ |
| **Frontend**       | React 18, Monaco Editor, Recharts, TailwindCSS, Lucide Icons |
| **Backend API**    | FastAPI, SQLAlchemy, Pydantic, PyJWT, Passlib (bcrypt)       |
| **Security**       | OpenGrep (Semgrep) — static analysis on user-uploaded code   |
| **Database**       | PostgreSQL 16                                                |
| **Orchestrator**   | Flask, multiprocessing, psutil, GPUManager                   |
| **ML Frameworks**  | TensorFlow / PyTorch (via Conda environments)                |
| **Infrastructure** | Docker, Docker Compose, Docker Secrets                       |
| **GPU Management** | nvidia-smi, CUDA, per-process GPU isolation                  |

---

## License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Copyright 2026 Tudor Lepadatu

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

This project is part of a Bachelor's Thesis at the Military Technical Academy "Ferdinand I" Bucharest.
