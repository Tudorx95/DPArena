"""
Orchestrator cu suport pentru antrenare paralelă pe multiple GPU-uri

MODIFICĂRI PRINCIPALE:
1. Adăugat GPUManager pentru alocarea automată a GPU-urilor
2. Fiecare simulare primește un GPU dedicat
3. Variabila de mediu CUDA_VISIBLE_DEVICES setată per-proces
4. Auto-release GPU la terminarea simulării
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import threading
import shutil
import psutil
import signal
import json
import os
from pathlib import Path
from datetime import datetime
import logging
import multiprocessing

# Import GPU Manager
from gpu_manager import GPUManager, configure_tensorflow_gpu, configure_pytorch_gpu

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# Config
VALID_USERS = {"tudor": "magma28fr"}
BASE_DIR = Path("/home/tudor.lepadatu/Licenta/Part2/fl_simulations")
CONDA_BASE = Path("/home/tudor.lepadatu/anaconda3")

CONDA_TENSORFLOW_ENV = "fl_tensorflow"
CONDA_PYTORCH_ENV = "fl_pytorch"

# Use multiprocessing Manager for shared state across processes
manager = multiprocessing.Manager()
active_simulations = manager.dict()  # Shared dictionary for status and PIDs

# Initialize GPU Manager (global instance)
gpu_manager = GPUManager()

def detect_framework(code):
    if "tensorflow" in code.lower() or "keras" in code.lower():
        return "tensorflow"
    return "pytorch"

def create_default_results_file(results_path):
    """Creează un fișier de rezultate default dacă simularea nu îl generează"""
    default_results = {
        "final_accuracy": 0.0,
        "final_precision": 0.0,
        "final_recall": 0.0,
        "final_f1": 0.0,
        "round_metrics_history": [],
        "convergence_metrics": [],
        "weight_divergence": [],
        "round_times": [],
        "malicious_clients": []
    }
    
    with open(results_path, 'w') as f:
        json.dump(default_results, f, indent=2)
    
    return default_results
    

def kill_process_tree(pid):
    """Kills a process and all its children recursively"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        for child in children:
            try:
                app.logger.info(f"Terminating child process {child.pid}")
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        gone, alive = psutil.wait_procs(children, timeout=3)
        
        for child in alive:
            try:
                app.logger.info(f"Force killing child process {child.pid}")
                child.kill()
            except psutil.NoSuchProcess:
                pass
        
        try:
            app.logger.info(f"Terminating parent process {pid}")
            parent.terminate()
            parent.wait(timeout=3)
        except psutil.TimeoutExpired:
            app.logger.info(f"Force killing parent process {pid}")
            parent.kill()
            
        app.logger.info(f"Successfully killed process tree for PID {pid}")
        return True
        
    except psutil.NoSuchProcess:
        app.logger.warning(f"Process {pid} not found")
        return True
    except Exception as e:
        app.logger.error(f"Error killing process tree {pid}: {str(e)}")
        return False


def run_simulation_pipeline(task_id, user_id, template_code, config, shared_simulations):
    """
    Pipeline simulare cu alocare automată GPU
    """
    process_pid = os.getpid()
    gpu_id = -1  # Default CPU
    
    try:
        # ========== STEP 0: Allocate GPU ==========
        app.logger.info(f"[{task_id}] Allocating GPU...")
        shared_simulations[task_id] = {
            "status": "running", 
            "step": 1, 
            "message": "Allocating GPU...", 
            "pid": process_pid
        }
        
        # Allocate GPU (blocks until available)
        gpu_id = gpu_manager.allocate_gpu(task_id, timeout=600)
        
        if gpu_id == -1:
            app.logger.info(f"[{task_id}] Using CPU mode")
            gpu_info = "CPU"
        else:
            app.logger.info(f"[{task_id}] Allocated GPU {gpu_id}")
            gpu_info = f"GPU {gpu_id}"
        
        shared_simulations[task_id] = {
            "status": "running", 
            "step": 1, 
            "message": f"Preparing directories (using {gpu_info})...", 
            "pid": process_pid,
            "gpu_id": gpu_id
        }

        # ========== STEP 1: Create directories ==========
        user_dir = BASE_DIR / f"user_{user_id}" / task_id
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "clean_data").mkdir(exist_ok=True)
        (user_dir / "clean_data_poisoned").mkdir(exist_ok=True)
        (user_dir / "results").mkdir(exist_ok=True)

        # Check for cancellation
        if task_id not in shared_simulations or shared_simulations[task_id].get("status") == "cancelling":
            raise InterruptedError("Simulation cancelled by user")
        
        # ========== STEP 2: Save template ==========
        template_path = user_dir / "template_code.py"
        with open(template_path, 'w') as f:
            f.write(template_code)
        
        # ========== STEP 3: Save config ==========
        config_path = user_dir / "simulation_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # ========== STEP 4: Detect framework ==========
        framework = detect_framework(template_code)
        conda_env = CONDA_TENSORFLOW_ENV if framework == "tensorflow" else CONDA_PYTORCH_ENV
        
        ext = 'keras' if framework == 'tensorflow' else 'pth'
        model_name = f"{config['NN_NAME']}.{ext}"
        model_path = user_dir / model_name
        
        shared_simulations[task_id] = {
            "status": "running", 
            "step": 2, 
            "message": f"Executing template code on {gpu_info}...", 
            "pid": process_pid,
            "gpu_id": gpu_id
        }

        # ========== STEP 4.5: Verify template structure ==========
        shared_simulations[task_id] = {
            "status": "running", 
            "step": 2, 
            "message": "Verifying template structure...", 
            "pid": process_pid,
            "gpu_id": gpu_id
        }
        
        conda_activate = f"source {CONDA_BASE}/bin/activate {conda_env}"

        # Verify template before execution
        verify_script = Path(__file__).parent / "verify_template.py"
        cmd_verify = f" {conda_activate} && cd {user_dir} && python {verify_script}"
        result_verify = subprocess.run(
            cmd_verify, 
            shell=True, 
            capture_output=True, 
            text=True, 
            executable="/bin/bash", 
            timeout=600
        )
        
        if result_verify.returncode != 0:
            error_msg = f"Template verification failed: {result_verify.stderr}\nStdout: {result_verify.stdout}"
            shared_simulations[task_id] = {"status": "error", "message": error_msg}
            app.logger.error(error_msg)
            return
        
        app.logger.info(f"[{task_id}] ✓ Template verification passed")
        
        # ========== STEP 5: Execute template code WITH GPU ==========
        
        
        # Set CUDA_VISIBLE_DEVICES for this subprocess
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Allow TensorFlow GPU memory growth
        if framework == "tensorflow":
            env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        cmd = f"{conda_activate} && cd {user_dir} && python template_code.py"
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            executable="/bin/bash", 
            timeout=600,
            env=env  # Pass modified environment
        )
        
        if result.returncode != 0:
            error_msg = f"Template execution failed: {result.stderr}\nStdout: {result.stdout}"
            shared_simulations[task_id] = {"status": "error", "message": error_msg}
            app.logger.error(error_msg)
            return
        
        # ========== STEP 6: Rename model file (.keras) ==========
        app.logger.info(f"[{task_id}] Looking for model file to rename...")
        
        # Find the model file created by template_code.py
        # It could be saved as {model.name}.keras (e.g., "sequential.keras", "resnet18.keras")
        if framework == "tensorflow":
            possible_model_files = list(user_dir.glob("*.keras"))
            expected_ext = ".keras"
        else:  # pytorch
            possible_model_files = list(user_dir.glob("*.pth"))
            expected_ext = ".pth"
        
        if not possible_model_files:
            error_msg = f"No {expected_ext} model file found after template execution"
            shared_simulations[task_id] = {"status": "error", "message": error_msg}
            app.logger.error(error_msg)
            return
        
        # Get the first .keras/.pth file (should be the one created by template)
        source_model = possible_model_files[0]
        
        # Target name from config
        target_model = model_path  # Already defined as: user_dir / f"{config['NN_NAME']}.keras"
        
        # Rename if necessary
        if source_model != target_model:
            app.logger.info(f"[{task_id}] Renaming model: {source_model.name} → {target_model.name}")
            shutil.move(str(source_model), str(target_model))
            app.logger.info(f"[{task_id}] ✓ Model renamed successfully")
        else:
            app.logger.info(f"[{task_id}] ✓ Model already has correct name: {target_model.name}")
        
        # Verify the model file exists with correct name
        if not target_model.exists():
            error_msg = f"Model file not found after rename: {target_model}"
            shared_simulations[task_id] = {"status": "error", "message": error_msg}
            app.logger.error(error_msg)
            return
        
        app.logger.info(f"[{task_id}] ✓ Model ready: {target_model.name} ({target_model.stat().st_size / 1024 / 1024:.2f} MB)")


        shared_simulations[task_id] = {
            "status": "running", 
            "step": 3, 
            "message": f"Downloading data on {gpu_info}...", 
            "pid": process_pid,
            "gpu_id": gpu_id
        }
        
        # Check for cancellation
        if task_id not in shared_simulations or shared_simulations[task_id].get("status") == "cancelling":
            raise InterruptedError("Simulation cancelled by user")

        # ========== STEP 7: Download data ==========
        cmd = f"{conda_activate} && cd {user_dir} && python -c 'from template_code import download_data; download_data(\"clean_data\")'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash", env=env)
        if result.returncode != 0:
            error_msg = f"Download data failed: {result.stderr}\nStdout: {result.stdout}"
            shared_simulations[task_id] = {"status": "error", "message": error_msg}
            app.logger.error(error_msg)
            return
        
        shared_simulations[task_id] = {
            "status": "running", 
            "step": 4, 
            "message": "Poisoning data...", 
            "pid": process_pid,
            "gpu_id": gpu_id
        }
        
        # Check for cancellation
        if task_id not in shared_simulations or shared_simulations[task_id].get("status") == "cancelling":
            raise InterruptedError("Simulation cancelled by user")

        # ========== STEP 8: Poison data (v2) ==========
        poison_script = Path(__file__).parent / "poison_data.py"
        test_file = user_dir / "results" / "attack_info.json"
        cmd = (
            f"{conda_activate} && "
            f"python {poison_script} {test_file} {config['NN_NAME']} {user_dir / 'clean_data'} "
            f"--operation {config['poison_operation']} "
            f"--intensity {config['poison_intensity']} "
            f"--percentage {config['poison_percentage']}"
        )
        # Optional v2 parameters
        if config.get('target_class'):
            cmd += f" --target_class {config['target_class']}"
        if config.get('no_flip'):
            cmd += " --no_flip"
        if config.get('trigger_type'):
            cmd += f" --trigger_type {config['trigger_type']}"
        if config.get('pattern_type'):
            cmd += f" --pattern_type {config['pattern_type']}"
        if config.get('modification'):
            cmd += f" --modification {config['modification']}"
        if config.get('transform'):
            cmd += f" --transform {config['transform']}"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
        if result.returncode != 0:
            error_msg = f"Poison data failed: {result.stderr}\nStdout: {result.stdout}"
            shared_simulations[task_id] = {"status": "error", "message": error_msg}
            app.logger.error(error_msg)
            return
        
        # ========== STEP 5-6: FL simulations WITH GPU ==========
        shared_simulations[task_id] = {
            "status": "running", 
            "step": 5, 
            "message": f"Running FL simulation (clean) on {gpu_info}...", 
            "pid": process_pid,
            "gpu_id": gpu_id
        }
        
        # Check cancellation
        if task_id not in shared_simulations or shared_simulations[task_id].get("status") == "cancelling":
            raise InterruptedError("Simulation cancelled by user")

        # Run clean FL simulation WITH GPU
        fd_script = Path(__file__).parent / "fd_simulator.py"
        test_file_clean = user_dir / "results" / "clean_metrics.json"
        test_file_clean.parent.mkdir(parents=True, exist_ok=True)
        # test_file_clean.touch()
        
        cmd = (
            f"{conda_activate} && "
            f"python {fd_script} "
            f"{test_file_clean} "                     # test_file (full path)
            f"{config['N']} "
            f"{config['M']} "
            f"{model_path} "                        # NN_NAME_PATH (full model path)
            f"{user_dir / 'clean_data'} "
            f"{user_dir / 'clean_data'} "
            f"{config['R']} "
            f"{config['ROUNDS']} "
            f"--strategy {config['strategy']} "
            f"--data_poison_protection fedavg"
        )

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash", env=env)
        if result.returncode != 0:
            error_msg = f"Clean FL simulation failed: {result.stderr}\nStdout: {result.stdout}"
            shared_simulations[task_id] = {"status": "error", "message": error_msg}
            app.logger.error(error_msg)
            return

        # ========== STEP 6: FL simulation (Clean + Data Poison Protection) ==========
        shared_simulations[task_id] = {
            "status": "running", 
            "step": 6, 
            "message": f"Running FL simulation (clean + DP protection) on {gpu_info}...", 
            "pid": process_pid,
            "gpu_id": gpu_id
        }
        
        # Check cancellation
        if task_id not in shared_simulations or shared_simulations[task_id].get("status") == "cancelling":
            raise InterruptedError("Simulation cancelled by user")

        # Run clean DP FL simulation WITH GPU
        test_file_clean_dp = user_dir / "results" / "clean_dp_metrics.json"
        test_file_clean_dp.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = (
            f"{conda_activate} && "
            f"python {fd_script} "
            f"{test_file_clean_dp} "                  # test_file (full path)
            f"{config['N']} "
            f"{config['M']} "
            f"{model_path} "                        # NN_NAME_PATH (full model path)
            f"{user_dir / 'clean_data'} "
            f"{user_dir / 'clean_data'} "
            f"{config['R']} "
            f"{config['ROUNDS']} "
            f"--strategy {config['strategy']} "
            f"--data_poison_protection {config.get('data_poison_protection', 'fedavg')}"
        )

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash", env=env)
        if result.returncode != 0:
            error_msg = f"Clean DP FL simulation failed: {result.stderr}\nStdout: {result.stdout}"
            shared_simulations[task_id] = {"status": "error", "message": error_msg}
            app.logger.error(error_msg)
            return

        shared_simulations[task_id] = {
            "status": "running", 
            "step": 7, 
            "message": f"Running FL simulation (poisoned) on {gpu_info}...", 
            "pid": process_pid,
            "gpu_id": gpu_id
        }
        
        # Check cancellation
        if task_id not in shared_simulations or shared_simulations[task_id].get("status") == "cancelling":
            raise InterruptedError("Simulation cancelled by user")

        # Run poisoned FL simulation WITH GPU
        test_file_poisoned = user_dir / "results" / "poisoned_metrics.json"
        test_file_poisoned.parent.mkdir(parents=True, exist_ok=True)
        # test_file_poisoned.touch()
        
        cmd = (
            f"{conda_activate} && "
            f"python {fd_script} "
            f"{test_file_poisoned} "                  # test_file (full path)
            f"{config['N']} "
            f"{config['M']} "
            f"{model_path} "                        # NN_NAME_PATH (full model path)
            f"{user_dir / 'clean_data'} "
            f"{user_dir / 'clean_data_poisoned'} "
            f"{config['R']} "
            f"{config['ROUNDS']} "
            f"--strategy {config['strategy']} "
            f"--data_poisoning "
            f"--data_poison_protection fedavg"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash", env=env)
        if result.returncode != 0:
            error_msg = f"Poisoned FL simulation failed: {result.stderr}\nStdout: {result.stdout}"
            shared_simulations[task_id] = {"status": "error", "message": error_msg}
            app.logger.error(error_msg)
            return

        # ========== STEP 8: FL simulation with Data Poison Protection ==========
        shared_simulations[task_id] = {
            "status": "running",
            "step": 8,
            "message": f"Running FL simulation (poisoned + DP protection) on {gpu_info}...",
            "pid": process_pid,
            "gpu_id": gpu_id
        }

        # Check cancellation
        if task_id not in shared_simulations or shared_simulations[task_id].get("status") == "cancelling":
            raise InterruptedError("Simulation cancelled by user")

        # Run poisoned FL simulation WITH Data Poison Protection
        test_file_poisoned_dp = user_dir / "results" / "poisoned_dp_metrics.json"
        test_file_poisoned_dp.parent.mkdir(parents=True, exist_ok=True)

        cmd = (
            f"{conda_activate} && "
            f"python {fd_script} "
            f"{test_file_poisoned_dp} "                  # test_file (full path)
            f"{config['N']} "
            f"{config['M']} "
            f"{model_path} "                        # NN_NAME_PATH (full model path)
            f"{user_dir / 'clean_data'} "
            f"{user_dir / 'clean_data_poisoned'} "
            f"{config['R']} "
            f"{config['ROUNDS']} "
            f"--strategy {config['strategy']} "
            f"--data_poisoning "
            f"--data_poison_protection {config.get('data_poison_protection', 'fedavg')}"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash", env=env)
        if result.returncode != 0:
            error_msg = f"Poisoned DP FL simulation failed: {result.stderr}\nStdout: {result.stdout}"
            shared_simulations[task_id] = {"status": "error", "message": error_msg}
            app.logger.error(error_msg)
            return

        # ========== STEP 9: Generate results ==========
        shared_simulations[task_id] = {
            "status": "running",
            "step": 9,
            "message": "Generating analysis...",
            "pid": process_pid,
            "gpu_id": gpu_id
        }
        
        # Load and analyze results
        if test_file_clean.exists() and test_file_clean.stat().st_size > 0:
            with open(test_file_clean) as f:
                clean_results = json.load(f)
        else:
            app.logger.error(f"Clean results file not found or empty: {test_file_clean}")
            clean_results = {"final_accuracy": 0.0}

        if test_file_clean_dp.exists() and test_file_clean_dp.stat().st_size > 0:
            with open(test_file_clean_dp) as f:
                clean_dp_results = json.load(f)
        else:
            app.logger.error(f"Clean DP results file not found or empty: {test_file_clean_dp}")
            clean_dp_results = {"final_accuracy": 0.0}

        if test_file_poisoned.exists() and test_file_poisoned.stat().st_size > 0:
            with open(test_file_poisoned) as f:
                poisoned_results = json.load(f)
        else:
            app.logger.error(f"Poisoned results file not found or empty: {test_file_poisoned}")
            poisoned_results = {"final_accuracy": 0.0}

        if test_file_poisoned_dp.exists() and test_file_poisoned_dp.stat().st_size > 0:
            with open(test_file_poisoned_dp) as f:
                poisoned_dp_results = json.load(f)
        else:
            app.logger.error(f"Poisoned DP results file not found or empty: {test_file_poisoned_dp}")
            poisoned_dp_results = {"final_accuracy": 0.0}

        clean_accuracy = clean_results.get('final_accuracy', 0)
        clean_dp_accuracy = clean_dp_results.get('final_accuracy', 0)
        poisoned_accuracy = poisoned_results.get('final_accuracy', 0)
        poisoned_dp_accuracy = poisoned_dp_results.get('final_accuracy', 0)

        # Extract precision, recall, F1 for each scenario
        clean_precision = clean_results.get('final_precision', 0)
        clean_recall = clean_results.get('final_recall', 0)
        clean_f1 = clean_results.get('final_f1', 0)

        clean_dp_precision = clean_dp_results.get('final_precision', 0)
        clean_dp_recall = clean_dp_results.get('final_recall', 0)
        clean_dp_f1 = clean_dp_results.get('final_f1', 0)

        poisoned_precision = poisoned_results.get('final_precision', 0)
        poisoned_recall = poisoned_results.get('final_recall', 0)
        poisoned_f1 = poisoned_results.get('final_f1', 0)

        poisoned_dp_precision = poisoned_dp_results.get('final_precision', 0)
        poisoned_dp_recall = poisoned_dp_results.get('final_recall', 0)
        poisoned_dp_f1 = poisoned_dp_results.get('final_f1', 0)


        # If final_accuracy is 0 or missing, extract from last round
        if clean_accuracy == 0 and 'round_metrics_history' in clean_results:
            history = clean_results['round_metrics_history']
            if history and len(history) > 0:
                clean_accuracy = history[-1].get('accuracy', 0)
                app.logger.info(f"[{task_id}] Extracted clean_accuracy from round_metrics_history: {clean_accuracy}")

        if clean_dp_accuracy == 0 and 'round_metrics_history' in clean_dp_results:
            history = clean_dp_results['round_metrics_history']
            if history and len(history) > 0:
                clean_dp_accuracy = history[-1].get('accuracy', 0)
                app.logger.info(f"[{task_id}] Extracted clean_dp_accuracy from round_metrics_history: {clean_dp_accuracy}")

        if poisoned_accuracy == 0 and 'round_metrics_history' in poisoned_results:
            history = poisoned_results['round_metrics_history']
            if history and len(history) > 0:
                poisoned_accuracy = history[-1].get('accuracy', 0)
                app.logger.info(f"[{task_id}] Extracted poisoned_accuracy from round_metrics_history: {poisoned_accuracy}")

        if poisoned_dp_accuracy == 0 and 'round_metrics_history' in poisoned_dp_results:
            history = poisoned_dp_results['round_metrics_history']
            if history and len(history) > 0:
                poisoned_dp_accuracy = history[-1].get('accuracy', 0)
                app.logger.info(f"[{task_id}] Extracted poisoned_dp_accuracy from round_metrics_history: {poisoned_dp_accuracy}")
        
        # ========== STEP: Citire Init Accuracy ==========
        init_accuracy = 0.0
        
        # Încearcă JSON (format nou)
        verification_file = user_dir / "init-verification.json"
        if verification_file.exists():
            try:
                with open(verification_file, 'r') as f:
                    verification_data = json.load(f)
                    init_accuracy = verification_data.get('initial_metrics', {}).get('accuracy', 0.0)
                    if init_accuracy > 0:
                        app.logger.info(f"[{task_id}] Init accuracy from JSON: {init_accuracy:.4f}")
            except Exception as e:
                app.logger.warning(f"[{task_id}] Could not read JSON: {e}")
        # here are the final results
        analysis = {
            'init_accuracy': init_accuracy,
            'clean_accuracy': clean_accuracy,
            'clean_dp_accuracy': clean_dp_accuracy,
            'poisoned_accuracy': poisoned_accuracy,
            'poisoned_dp_accuracy': poisoned_dp_accuracy,
            'accuracy_drop': clean_accuracy - poisoned_accuracy,
            'drop_clean_init': clean_accuracy - init_accuracy,
            'drop_clean_dp_init': clean_dp_accuracy - init_accuracy,
            'drop_poison_init': poisoned_accuracy - init_accuracy,
            'drop_poison_dp_init': poisoned_dp_accuracy - init_accuracy,
            'clean_precision': clean_precision,
            'clean_recall': clean_recall,
            'clean_f1': clean_f1,
            'clean_dp_precision': clean_dp_precision,
            'clean_dp_recall': clean_dp_recall,
            'clean_dp_f1': clean_dp_f1,
            'poisoned_precision': poisoned_precision,
            'poisoned_recall': poisoned_recall,
            'poisoned_f1': poisoned_f1,
            'poisoned_dp_precision': poisoned_dp_precision,
            'poisoned_dp_recall': poisoned_dp_recall,
            'poisoned_dp_f1': poisoned_dp_f1,
            'clean_metrics': clean_results,
            'clean_dp_metrics': clean_dp_results,
            'poisoned_metrics': poisoned_results,
            'poisoned_dp_metrics': poisoned_dp_results,
            'gpu_used': gpu_info,
            'data_poison_protection_method': config.get('data_poison_protection', 'fedavg')
        }

        analysis_path = user_dir / "results" / "analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        summary = f"""FL Simulation Complete
Task: {task_id}
GPU: {gpu_info}
Init Accuracy: {analysis['init_accuracy']:.4f}
Clean Accuracy: {analysis['clean_accuracy']:.4f}
Clean DP Accuracy: {analysis['clean_dp_accuracy']:.4f}
Poisoned Accuracy: {analysis['poisoned_accuracy']:.4f}
Data Poison Protection Accuracy: {analysis['poisoned_dp_accuracy']:.4f}
Drop (Clean - Poisoned): {analysis['accuracy_drop']:.4f}
Drop (Clean - Init): {analysis['drop_clean_init']:.4f}
Drop (Clean DP - Init): {analysis['drop_clean_dp_init']:.4f}
Drop (Poisoned - Init): {analysis['drop_poison_init']:.4f}
Drop (Poisoned_DP - Init): {analysis['drop_poison_dp_init']:.4f}
Data Poison Protection Method: {analysis['data_poison_protection_method']}
--- Confusion Matrix Metrics (Weighted Avg) ---
Clean Precision: {analysis['clean_precision']:.4f}
Clean Recall: {analysis['clean_recall']:.4f}
Clean F1 Score: {analysis['clean_f1']:.4f}
Clean DP Precision: {analysis['clean_dp_precision']:.4f}
Clean DP Recall: {analysis['clean_dp_recall']:.4f}
Clean DP F1 Score: {analysis['clean_dp_f1']:.4f}
Poisoned Precision: {analysis['poisoned_precision']:.4f}
Poisoned Recall: {analysis['poisoned_recall']:.4f}
Poisoned F1 Score: {analysis['poisoned_f1']:.4f}
DP Protection Precision: {analysis['poisoned_dp_precision']:.4f}
DP Protection Recall: {analysis['poisoned_dp_recall']:.4f}
DP Protection F1 Score: {analysis['poisoned_dp_f1']:.4f}
"""
        
        summary_path = user_dir / "results" / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        shared_simulations[task_id] = {
            "status": "completed", 
            "results": str(analysis_path), 
            "timestamp": datetime.now().isoformat(),
            "gpu_used": gpu_info
        }

    except InterruptedError as e:
        app.logger.info(f"Simulation {task_id} was cancelled: {str(e)}")
        shared_simulations[task_id] = {
            "status": "cancelled",
            "message": "Simulation cancelled by user",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        error_msg = f"Unexpected error in pipeline: {str(e)}"
        shared_simulations[task_id] = {"status": "error", "message": error_msg}
        app.logger.error(error_msg)
    
    finally:
        # CRITICAL: Always release GPU
        if gpu_id != -1:
            app.logger.info(f"[{task_id}] Releasing GPU {gpu_id}")
            gpu_manager.release_gpu(task_id, gpu_id)


@app.route("/")
def index():
    return jsonify({
        "message": "FL Orchestrator API", 
        "status": "running",
        "available_gpus": gpu_manager.available_gpus,
        "gpus_in_use": len(gpu_manager.available_gpus) - gpu_manager.get_available_count()
    })

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    user = data.get("username")
    password = data.get("password")
    
    if user in VALID_USERS and VALID_USERS[user] == password:
        return jsonify({"status": "success", "token": f"token-{user}-123"}), 200
    return jsonify({"status": "error"}), 401

@app.route("/simulate", methods=["POST"])
def simulate():
    app.logger.info(f"Received simulate request from {request.remote_addr}")
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer token-"):
        app.logger.warning(f"Unauthorized request from {request.remote_addr}")
        return jsonify({"error": "Unauthorized"}), 403
    
    data = request.json
    task_id = data.get("task_id")
    user_id = data.get("user_id")
    template_code = data.get("template_code")
    config = data.get("config")
    
    app.logger.info(f"Starting simulation for task_id: {task_id}, user_id: {user_id}")

    # Start a new process for the simulation pipeline
    p = multiprocessing.Process(
        target=run_simulation_pipeline, 
        args=(task_id, user_id, template_code, config, active_simulations)
    )
    p.daemon = True
    p.start()
    
    active_simulations[task_id] = {"status": "queued", "pid": p.pid}
    
    return jsonify({"status": "queued", "task_id": task_id}), 200


@app.route("/cancel/<task_id>", methods=["POST"])
def cancel_simulation(task_id):
    """Cancel a running simulation and release its GPU"""
    app.logger.info(f"Received cancellation request for task {task_id}")
    
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer token-"):
        app.logger.warning(f"Unauthorized cancellation request for task {task_id}")
        return jsonify({"error": "Unauthorized"}), 403
    
    if task_id not in active_simulations:
        app.logger.warning(f"Task {task_id} not found")
        return jsonify({"error": "Task not found"}), 404
    
    task_info = active_simulations[task_id]
    current_status = task_info.get("status")
    
    if current_status in ["completed", "cancelled", "error"]:
        app.logger.info(f"Task {task_id} already in terminal state: {current_status}")
        return jsonify({
            "status": "success",
            "message": f"Task already {current_status}",
            "task_id": task_id
        }), 200
    
    # Mark as cancelling
    active_simulations[task_id] = {
        **task_info,
        "status": "cancelling",
        "message": "Cancellation requested..."
    }
    
    try:
        # Get GPU ID and release it
        gpu_id = task_info.get("gpu_id", -1)
        if gpu_id != -1:
            app.logger.info(f"Releasing GPU {gpu_id} for cancelled task {task_id}")
            gpu_manager.release_gpu(task_id, gpu_id)
        
        # Kill the process
        pid = task_info.get("pid")
        if pid:
            app.logger.info(f"Attempting to kill process {pid} for task {task_id}")
            if kill_process_tree(pid):
                app.logger.info(f"Successfully killed process tree for task {task_id}")
        
        # Delete task directory
        user_id = request.json.get("user_id", 1)
        task_dir = BASE_DIR / f"user_{user_id}" / task_id
        
        if task_dir.exists():
            app.logger.info(f"Deleting task directory: {task_dir}")
            shutil.rmtree(task_dir)
            app.logger.info(f"Successfully deleted directory for task {task_id}")
        
        # Update final status
        active_simulations[task_id] = {
            "status": "cancelled",
            "message": "Simulation cancelled and cleaned up",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify({
            "status": "success",
            "message": "Simulation cancelled successfully",
            "task_id": task_id
        }), 200
        
    except Exception as e:
        error_msg = f"Error cancelling task {task_id}: {str(e)}"
        app.logger.error(error_msg)
        
        active_simulations[task_id] = {
            "status": "error",
            "message": f"Cancellation failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify({
            "status": "error",
            "message": error_msg,
            "task_id": task_id
        }), 500

@app.route("/status/<task_id>", methods=["GET"])
def get_status(task_id):
    if task_id in active_simulations:
        return jsonify(dict(active_simulations[task_id])), 200
    return jsonify({"error": "Not found"}), 404

@app.get("/results/<task_id>")
def get_results(task_id):
    """
    Return comprehensive simulation results including:
    - Analysis (accuracies, metrics)
    - Summary text
    - Simulation config (N, M, R, ROUNDS, poison parameters, strategy, NN_NAME)
    - Attack info (poisoning details)
    - GPU information
    """
    if task_id in active_simulations and active_simulations[task_id]['status'] == 'completed':
        analysis_path = Path(active_simulations[task_id]['results'])
        results_dir = analysis_path.parent
        user_dir = results_dir.parent
        
        # Load analysis results
        with open(analysis_path) as f:
            analysis = json.load(f)
        
        # Load summary
        summary_path = results_dir / "summary.txt"
        with open(summary_path) as f:
            summary = f.read()
        
        # Load simulation config
        config_path = user_dir / "simulation_config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                app.logger.info(f"[{task_id}] Loaded config: {config}")
        else:
            app.logger.warning(f"[{task_id}] Config file not found at {config_path}")
        
        # Load attack info (poisoning details)
        attack_info_path = results_dir / "attack_info.json"
        attack_info = {}
        if attack_info_path.exists():
            with open(attack_info_path) as f:
                attack_info = json.load(f)
                app.logger.info(f"[{task_id}] Loaded attack_info: {attack_info}")
        else:
            app.logger.warning(f"[{task_id}] Attack info file not found at {attack_info_path}")
        
        # Get GPU info from task
        gpu_used = active_simulations[task_id].get('gpu_used', 'Unknown')
        
        # Return comprehensive results
        response = {
            "analysis": analysis,
            "summary": summary,
            "config": config,
            "attack_info": attack_info,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "gpu_used": gpu_used
        }
        
        app.logger.info(f"[{task_id}] Returning results with config and attack_info")
        return jsonify(response)
    
    return jsonify({"error": "No results available"}), 404

@app.route("/gpu_status", methods=["GET"])
def gpu_status():
    """Get current GPU allocation status"""
    return jsonify({
        "available_gpus": gpu_manager.available_gpus,
        "total_gpus": len(gpu_manager.available_gpus),
        "available_count": gpu_manager.get_available_count(),
        "in_use": len(gpu_manager.available_gpus) - gpu_manager.get_available_count()
    })

if __name__ == "__main__":
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    app.logger.info(f"Starting orchestrator with GPU support")
    app.logger.info(f"Available GPUs: {gpu_manager.available_gpus}")
    app.run(host="0.0.0.0", port=8000, debug=False)
