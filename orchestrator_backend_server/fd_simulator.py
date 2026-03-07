#!/usr/bin/env python3
"""
Enhanced Federated Learning Simulator with JSON Metrics Storage
FRAMEWORK-AGNOSTIC VERSION (TensorFlow + PyTorch)

Modifications:
- JSON file path as argument for centralized test metrics
- Per-client metrics stored in test JSON file
- Thread-safe JSON file operations with FileLock
- Framework auto-detection from model file extension
- Conditional imports (TensorFlow OR PyTorch)
- Delegates to template_code.py for all framework-specific operations
"""

import sys
import os
import threading
import time
import queue
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
import argparse
import logging
import json
import glob
from collections import defaultdict
from pathlib import Path

try:
    from filelock import FileLock
except ImportError:
    print("WARNING: filelock not installed. Install with: pip install filelock")
    print("Falling back to basic threading lock (may not work across processes)")
    FileLock = None

# ============================================================================
# FRAMEWORK DETECTION & CONDITIONAL IMPORTS
# ============================================================================

def detect_framework_from_model(model_path):
    """Detectează framework-ul pe baza extensiei modelului"""
    if model_path.endswith('.keras'):
        return 'tensorflow'
    elif model_path.endswith('.pth'):
        return 'pytorch'
    else:
        raise ValueError(f"Unknown model format: {model_path}. Expected .keras or .pth")

# Parse model path EARLY pentru detectare framework
parser_temp = argparse.ArgumentParser(add_help=False)
parser_temp.add_argument('test_file', type=str)
parser_temp.add_argument('N', type=int)
parser_temp.add_argument('M', type=int)
parser_temp.add_argument('NN_NAME_PATH', type=str)
args_temp, _ = parser_temp.parse_known_args()

# Detectare framework
FRAMEWORK = detect_framework_from_model(args_temp.NN_NAME_PATH)
print(f"🔍 Detected framework: {FRAMEWORK.upper()}")

# Import conditional
if FRAMEWORK == 'tensorflow':
    print("📦 Loading TensorFlow...")
    import tensorflow as tf
    # Disable TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    DEVICE = None  # TensorFlow gestionează automat
else:  # pytorch
    print("📦 Loading PyTorch...")
    import torch
    import torch.nn as nn
    # Setare device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 PyTorch device: {DEVICE}")

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# JSON FILE MANAGER (Thread-safe)
# ============================================================================
class MetricsJSONManager:
    """Manager pentru citire/scriere thread-safe în fișierul JSON de metrici."""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.lock_path = json_path + '.lock'
        
        if FileLock:
            self.file_lock = FileLock(self.lock_path, timeout=30)
        else:
            self.file_lock = threading.Lock()
        
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Creează fișierul JSON dacă nu există."""
        os.makedirs(os.path.dirname(self.json_path) if os.path.dirname(self.json_path) else '.', exist_ok=True)
        if not os.path.exists(self.json_path):
            with open(self.json_path, 'w') as f:
                json.dump({}, f)
            logger.info(f"Created metrics JSON file: {self.json_path}")
    
    def read_metrics(self) -> dict:
        """Citește metricile din JSON (thread-safe)."""
        if FileLock:
            with self.file_lock:
                return self._read_file()
        else:
            with self.file_lock:
                return self._read_file()
    
    def _read_file(self) -> dict:
        """Internal method to read file."""
        try:
            with open(self.json_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"JSON decode error, returning empty dict")
            return {}
        except Exception as e:
            logger.error(f"Error reading metrics: {e}")
            return {}
    
    def write_metrics(self, data: dict):
        """Scrie metricile în JSON (thread-safe)."""
        if FileLock:
            with self.file_lock:
                self._write_file(data)
        else:
            with self.file_lock:
                self._write_file(data)
    
    def _write_file(self, data: dict):
        """Internal method to write file."""
        try:
            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=2, default=self._json_serializer)
            logger.debug(f"Metrics written to {self.json_path}")
        except Exception as e:
            logger.error(f"Error writing metrics: {e}")
    
    @staticmethod
    def _json_serializer(obj):
        """Serializare custom pentru tipuri numpy și torch."""
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # PyTorch tensors
        if FRAMEWORK == 'pytorch':
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
        raise TypeError(f"Type {type(obj)} not serializable")


# ============================================================================
# TEMPLATE FUNCTIONS LOADER
# ============================================================================
import importlib.util

class TemplateFunctions:
    def __init__(self):
        self.module = None
        self.available = False
    
    def load_template(self, template_path: str):
        try:
            spec = importlib.util.spec_from_file_location("user_template", template_path)
            self.module = importlib.util.module_from_spec(spec)
            sys.modules["user_template"] = self.module
            spec.loader.exec_module(self.module)
            self.available = True
            logger.info(f"✓ Template loaded: {template_path}")
            functions = [name for name in dir(self.module) 
                        if callable(getattr(self.module, name)) and not name.startswith('_')]
            logger.info(f"  Available functions: {', '.join(functions)}")
        except Exception as e:
            logger.error(f"✗ Error loading template: {e}")
            self.available = False
            raise
    
    def get_function(self, func_name: str):
        if not self.available:
            raise RuntimeError("Template not loaded")
        if not hasattr(self.module, func_name):
            raise AttributeError(f"Function '{func_name}' not found")
        return getattr(self.module, func_name)
    
    def has_function(self, func_name: str) -> bool:
        return self.available and hasattr(self.module, func_name)


TEMPLATE_FUNCS = TemplateFunctions()

# ============================================================================
# FRAMEWORK-AGNOSTIC HELPER FUNCTIONS
# ============================================================================

def move_model_to_device(model):
    """Move model to appropriate device (only for PyTorch)"""
    if FRAMEWORK == 'pytorch':
        return model.to(DEVICE)
    return model  # TensorFlow gestionează automat


def load_model_framework_agnostic(model_path, use_template=False):
    """Încarcă modelul folosind framework-ul potrivit"""
    if use_template and TEMPLATE_FUNCS.has_function('load_model_config'):
        load_func = TEMPLATE_FUNCS.get_function('load_model_config')
        model = load_func(model_path)
    else:
        if FRAMEWORK == 'tensorflow':
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:  # pytorch
            # Trebuie să importăm create_model din template
            if TEMPLATE_FUNCS.has_function('create_model'):
                create_model_func = TEMPLATE_FUNCS.get_function('create_model')
                model = create_model_func()
                
                checkpoint = torch.load(model_path, map_location=DEVICE)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                raise RuntimeError("PyTorch model loading requires create_model() in template_code.py")
    
    return move_model_to_device(model)


def get_model_weights_framework_agnostic(model, use_template=False):
    """Extrage weights folosind framework-ul potrivit"""
    if use_template and TEMPLATE_FUNCS.has_function('get_model_weights'):
        get_weights_func = TEMPLATE_FUNCS.get_function('get_model_weights')
        return get_weights_func(model)
    else:
        if FRAMEWORK == 'tensorflow':
            return model.get_weights()
        else:  # pytorch
            weights = []
            for param in model.parameters():
                weights.append(param.data.cpu().numpy())
            return weights


def set_model_weights_framework_agnostic(model, weights, use_template=False):
    """Setează weights folosind framework-ul potrivit"""
    if use_template and TEMPLATE_FUNCS.has_function('set_model_weights'):
        set_weights_func = TEMPLATE_FUNCS.get_function('set_model_weights')
        set_weights_func(model, weights)
    else:
        if FRAMEWORK == 'tensorflow':
            model.set_weights(weights)
        else:  # pytorch
            with torch.no_grad():
                for param, weight in zip(model.parameters(), weights):
                    param.data = torch.from_numpy(weight).to(param.device)


def get_model_output_shape(model):
    """Obține numărul de clase de output"""
    if FRAMEWORK == 'tensorflow':
        return model.output_shape[-1]
    else:  # pytorch
        # Presupunem că ultimul layer este Linear/fully connected
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        raise RuntimeError("Could not determine output shape for PyTorch model")


# ============================================================================
# ENHANCED FEDERATED SERVER
# ============================================================================
class EnhancedFederatedServer:
    def __init__(self, num_clients, num_malicious, nn_path, nn_name, data_folder, alternative_data,
                 rounds, r, strategy="first", data_poisoning=False, use_template=False,
                 test_json_path=None, data_poison_protection='fedavg'):
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.nn_path = nn_path
        self.nn_name = nn_name
        self.data_folder = data_folder
        self.alternative_data = alternative_data
        self.rounds = rounds
        self.R = r
        self.strategy = strategy
        self.data_poisoning = data_poisoning
        self.use_template = use_template
        self.data_poison_protection = data_poison_protection
        
        # JSON Metrics Manager
        self.test_json_path = test_json_path
        self.json_manager = MetricsJSONManager(test_json_path) if test_json_path else None
        
        # Sincronizare
        self.client_queues = {}
        self.server_queue = queue.Queue()
        
        # Creează queue-uri pentru clienți AICI (în __init__, nu în run())
        for i in range(num_clients):
            self.client_queues[i] = queue.Queue()
        
        # Încărcare model global
        logger.info(f"Loading global model: {nn_path} ({FRAMEWORK.upper()})")
        self.global_model = load_model_framework_agnostic(nn_path, use_template)
        self.global_weights = get_model_weights_framework_agnostic(self.global_model, use_template)
        
        # Metrici
        self.round_metrics_history = []
        self.convergence_metrics = []
        self.weight_divergence = []
        self.round_times = []
        self.malicious_clients = []
        
        # FoolsGold: istoric acumulat al update-urilor per client (Algorithm 1, line 3)
        self.foolsgold_histories = {}  # client_id -> np.array (flattened accumulated gradient)
        
        # Identificare clienți malițioși
        self._assign_malicious_clients()
    
    def _assign_malicious_clients(self):
        """Atribuie ID-uri de clienți malițioși pe baza strategiei"""
        if self.num_malicious == 0:
            self.malicious_clients = []
            logger.info("No malicious clients")
            return
        
        if self.strategy == 'first':
            self.malicious_clients = list(range(self.num_malicious))
        elif self.strategy == 'last':
            self.malicious_clients = list(range(self.num_clients - self.num_malicious, self.num_clients))
        elif self.strategy in ['alternate', 'alternate_data']:
            self.malicious_clients = list(range(0, self.num_clients, 2))[:self.num_malicious]
        else:
            self.malicious_clients = list(range(self.num_malicious))
        
        logger.info(f"Malicious clients: {self.malicious_clients}")
    
    def _aggregate_weights_fedavg(self, client_weights, client_sizes):
        """FedAvg: weighted average based on dataset size"""
        total_size = sum(client_sizes)
        avg_weights = [np.zeros_like(w, dtype=np.float64) for w in client_weights[0]]
        
        for client_w, size in zip(client_weights, client_sizes):
            weight = size / total_size
            for i, w in enumerate(client_w):
                avg_weights[i] += w * weight
        
        # Cast back to original dtypes
        return [avg_weights[i].astype(w.dtype) for i, w in enumerate(client_weights[0])]
    
    def _aggregate_weights_krum(self, client_weights, num_malicious):
        """Krum: selects one client with smallest distance sum"""
        num_clients = len(client_weights)
        distances = np.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = sum(np.linalg.norm(w_i - w_j) for w_i, w_j in zip(client_weights[i], client_weights[j]))
                distances[i, j] = dist
                distances[j, i] = dist
        
        scores = []
        for i in range(num_clients):
            sorted_distances = np.sort(distances[i])
            score = np.sum(sorted_distances[:num_clients - num_malicious - 1])
            scores.append(score)
        
        selected_idx = np.argmin(scores)
        logger.info(f"Krum selected client {selected_idx}")
        return client_weights[selected_idx]
    
    def _aggregate_weights_trimmed_mean(self, client_weights, trim_ratio=0.1):
        """Trimmed Mean: remove top/bottom trim_ratio% and average"""
        num_clients = len(client_weights)
        num_trim = int(num_clients * trim_ratio)
        if num_trim * 2 >= num_clients:
            num_trim = max(0, (num_clients - 1) // 2)
        
        aggregated = []
        for layer_idx in range(len(client_weights[0])):
            layer_weights = [cw[layer_idx] for cw in client_weights]
            layer_weights_sorted = np.sort(layer_weights, axis=0)
            trimmed = layer_weights_sorted[num_trim:-num_trim] if num_trim > 0 else layer_weights_sorted
            mean_vals = np.mean(trimmed, axis=0)
            aggregated.append(mean_vals.astype(client_weights[0][layer_idx].dtype))
        
        return aggregated
    
    def _aggregate_weights_median(self, client_weights):
        """Median aggregation"""
        aggregated = []
        for layer_idx in range(len(client_weights[0])):
            layer_weights = [cw[layer_idx] for cw in client_weights]
            median_vals = np.median(layer_weights, axis=0)
            aggregated.append(median_vals.astype(client_weights[0][layer_idx].dtype))
        return aggregated
    
    def _aggregate_weights_foolsgold(self, client_weights, client_ids=None):
        """
        FoolsGold — implementare fidelă conform Algorithm 1 din:
        Fung, Yoon & Beschastnikh, "Mitigating Sybils in Federated Learning Poisoning"
        (arXiv:1808.04866v5, RAID 2020)
        
        Pași conform paper-ului:
        1. Acumulează istoricul update-urilor per client (Hi = Σ Δi,t)
        2. Calculează cosine similarity pairwise pe istorice
        3. vi = max_j(cs_ij) — maximul similarității per client
        4. Pardoning: dacă vj > vi → cs_ij *= vi/vj (protejează clienții onești)
        5. αi = 1 - max_j(cs_ij) — learning rate adaptat
        6. Rescalare: αi = αi / max_i(α) (cel mai onest → α=1)
        7. Logit: αi = κ(ln[αi/(1-αi)] + 0.5) — amplificare
        8. Agregare ponderată: w_t+1 = w_t + Σ αi * Δi,t
        """
        num_clients = len(client_weights)
        kappa = 1.0  # Confidence parameter (κ = 1 în evaluarea din paper)
        
        if num_clients < 2:
            return self._aggregate_weights_fedavg(client_weights, [1] * num_clients)
        
        # Step 1: Calculează update-urile curente (Δi,t = weights_client - weights_global)
        global_flat = np.concatenate([w.flatten().astype(np.float64) for w in self.global_weights])
        
        updates = []
        for cw in client_weights:
            flat = np.concatenate([w.flatten().astype(np.float64) for w in cw])
            updates.append(flat - global_flat)
        
        # Step 2: Acumulează istoricul (Algorithm 1, line 3: Hi = Σ Δi,t)
        # Folosim client_id-urile reale pentru a urmări fiecare client consistent peste runde
        if client_ids is None:
            client_ids = list(range(num_clients))
        
        for idx, cid in enumerate(client_ids):
            if cid not in self.foolsgold_histories:
                self.foolsgold_histories[cid] = np.zeros_like(updates[idx])
            self.foolsgold_histories[cid] += updates[idx]
        
        # Step 3: Cosine similarity pairwise pe ISTORICE (nu pe update-uri curente!)
        histories = [self.foolsgold_histories[cid] for cid in client_ids]
        h_norms = [np.linalg.norm(h) for h in histories]
        
        cs_matrix = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                if h_norms[i] > 1e-10 and h_norms[j] > 1e-10:
                    cs = np.dot(histories[i], histories[j]) / (h_norms[i] * h_norms[j])
                    cs = np.clip(cs, -1.0, 1.0)
                    cs_matrix[i][j] = cs
                    cs_matrix[j][i] = cs
        
        # Step 4: vi = max_j(cs_ij) — maximul pairwise similarity per client
        # (Algorithm 1, line 8)
        v = np.zeros(num_clients)
        for i in range(num_clients):
            v[i] = max(cs_matrix[i][j] for j in range(num_clients) if j != i)
        
        # Step 5: Pardoning (Algorithm 1, lines 12-14)
        # Dacă vj > vi → cs_ij *= vi/vj
        # Protejează clienții onești care au similaritate accidentală cu sybils
        cs_pardoned = cs_matrix.copy()
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j and v[j] > v[i] and v[j] > 1e-10:
                    cs_pardoned[i][j] *= v[i] / v[j]
        
        # Step 6: αi = 1 - max_j(cs_pardoned_ij) (Algorithm 1, line 16)
        alpha = np.zeros(num_clients)
        for i in range(num_clients):
            max_cs = max(cs_pardoned[i][j] for j in range(num_clients) if j != i)
            alpha[i] = 1.0 - max_cs
        
        # Step 7: Rescalare (Algorithm 1, line 18)
        # αi = αi / max_i(α) — cel mai onest client primește α = 1
        max_alpha = np.max(alpha)
        if max_alpha > 1e-10:
            alpha = alpha / max_alpha
        else:
            alpha = np.ones(num_clients) / num_clients
        
        # Step 8: Logit function (Algorithm 1, line 19)
        # αi = κ(ln[αi/(1-αi)] + 0.5)
        for i in range(num_clients):
            # Clip to avoid log(0) or division by zero
            ai = np.clip(alpha[i], 1e-6, 1.0 - 1e-6)
            alpha[i] = kappa * (np.log(ai / (1.0 - ai)) + 0.5)
        
        # Clip to [0, 1] range (paper: "any value exceeding 0-1 range is clipped")
        alpha = np.clip(alpha, 0.0, 1.0)
        
        # Normalizare la sumă 1 pentru agregare ponderată
        alpha_sum = np.sum(alpha)
        if alpha_sum > 1e-10:
            weights_fg = alpha / alpha_sum
        else:
            weights_fg = np.ones(num_clients) / num_clients
        
        logger.info(f"FoolsGold alpha (pre-norm): {[f'{a:.4f}' for a in alpha]}")
        logger.info(f"FoolsGold weights: {[f'{w:.4f}' for w in weights_fg]}")
        logger.info(f"FoolsGold v (max pairwise sim): {[f'{vi:.4f}' for vi in v]}")
        logger.info(f"Malicious clients: {self.malicious_clients}")
        
        # Step 9: Agregare ponderată (Algorithm 1, line 20)
        # w_t+1 = w_t + Σ αi * Δi,t
        # Echivalent: weighted average of client weights cu weights_fg
        aggregated = [np.zeros_like(w, dtype=np.float64) for w in client_weights[0]]
        for client_w, w in zip(client_weights, weights_fg):
            for layer_idx, layer_w in enumerate(client_w):
                aggregated[layer_idx] += layer_w.astype(np.float64) * w
        
        return [agg.astype(client_weights[0][i].dtype) for i, agg in enumerate(aggregated)]
    
    def _aggregate_weights_norm_clipping(self, client_weights, client_sizes):
        """
        Norm Clipping defense (Sun et al., arXiv:1911.07963).
        
        Referință: "Can You Really Backdoor Federated Learning?"
        
        Algoritm:
        1. Calculează delta (update) pentru fiecare client: Δ_i = w_i - w_global
        2. Calculează norma L2 a fiecărui update: ||Δ_i||_2
        3. Determină threshold M = median(||Δ_i||_2) (adaptiv)
        4. Dacă ||Δ_i||_2 > M, clip: Δ_i = Δ_i * (M / ||Δ_i||_2)
        5. Aplică FedAvg pe update-urile clipped: w_global_new = w_global + Σ(Δ_i_clipped) / n
        
        Efect: Limitează influența oricărui client individual asupra modelului global,
        reducând impactul atacurilor de tip backdoor fără a afecta performanța pe task-ul principal.
        """
        num_clients = len(client_weights)
        
        # Step 1: Calculează update-urile (delta) față de modelul global
        global_flat = np.concatenate([w.flatten().astype(np.float64) for w in self.global_weights])
        
        updates = []
        for cw in client_weights:
            flat = np.concatenate([w.flatten().astype(np.float64) for w in cw])
            updates.append(flat - global_flat)
        
        # Step 2: Calculează norma L2 a fiecărui update
        norms = [np.linalg.norm(u) for u in updates]
        logger.info(f"Norm clipping - Update norms: {[f'{n:.4f}' for n in norms]}")
        
        # Step 3: Threshold adaptiv M = median al normelor
        M = np.median(norms)
        logger.info(f"Norm clipping - Threshold M (median): {M:.4f}")
        
        # Step 4: Clip fiecare update la norma M
        clipped_updates = []
        for i, (update, norm) in enumerate(zip(updates, norms)):
            if norm > M and M > 0:
                # Scale down update-ul la norma M
                clipped = update * (M / norm)
                logger.info(f"  Client {i}: clipped {norm:.4f} -> {M:.4f}")
            else:
                clipped = update
            clipped_updates.append(clipped)
        
        # Step 5: FedAvg pe update-urile clipped
        total_size = sum(client_sizes)
        avg_update = np.zeros_like(global_flat)
        for update, size in zip(clipped_updates, client_sizes):
            weight = size / total_size
            avg_update += update * weight
        
        # Reconstruiește weights-urile din update-ul mediu
        new_flat = global_flat + avg_update
        
        # Reconstruct layer shapes
        aggregated = []
        offset = 0
        for w in self.global_weights:
            numel = w.size
            layer_flat = new_flat[offset:offset + numel]
            aggregated.append(layer_flat.reshape(w.shape).astype(w.dtype))
            offset += numel
        
        return aggregated
    
    def _aggregate_weights(self, client_weights, client_sizes, client_ids=None):
        """Agregare ponderi cu protecție împotriva data poisoning"""
        method = self.data_poison_protection.lower()
        
        if method == 'krum':
            return self._aggregate_weights_krum(client_weights, self.num_malicious)
        elif method == 'trimmed_mean':
            return self._aggregate_weights_trimmed_mean(client_weights, trim_ratio=0.2)
        elif method == 'median':
            return self._aggregate_weights_median(client_weights)
        elif method == 'foolsgold':
            return self._aggregate_weights_foolsgold(client_weights, client_ids)
        elif method == 'norm_clipping':
            return self._aggregate_weights_norm_clipping(client_weights, client_sizes)
        elif method == 'trimmed_mean_krum':
            trimmed = self._aggregate_weights_trimmed_mean(client_weights, trim_ratio=0.1)
            return trimmed
        elif method == 'random':
            idx = np.random.randint(0, len(client_weights))
            logger.info(f"Random aggregation: selected client {idx}")
            return client_weights[idx]
        else:  # fedavg (default)
            return self._aggregate_weights_fedavg(client_weights, client_sizes)
    
    def _evaluate_global_model(self):
        """Evaluare model global pe date de test"""
        try:
            if self.use_template and TEMPLATE_FUNCS.has_function('load_client_data'):
                load_data_func = TEMPLATE_FUNCS.get_function('load_client_data')
                _, test_ds = load_data_func(self.data_folder, batch_size=32)
            else:
                # Fallback la TensorFlow
                if FRAMEWORK == 'tensorflow':
                    test_ds = tf.keras.utils.image_dataset_from_directory(
                        os.path.join(self.data_folder, 'test'),
                        image_size=(32, 32),
                        batch_size=32,
                        shuffle=False
                    )
                else:
                    raise RuntimeError("PyTorch requires load_client_data() in template")
            
            if self.use_template and TEMPLATE_FUNCS.has_function('calculate_metrics'):
                calc_metrics_func = TEMPLATE_FUNCS.get_function('calculate_metrics')
                metrics = calc_metrics_func(self.global_model, test_ds)
                acc = metrics.get('accuracy', 0.0)
                return {
                    'accuracy': acc,
                    'precision': metrics.get('precision', 0.0),
                    'recall': metrics.get('recall', 0.0),
                    'f1': metrics.get('f1', metrics.get('f1_score', 0.0))
                }
            else:
                # Evaluare manuală
                y_true, y_pred = [], []
                
                if FRAMEWORK == 'tensorflow':
                    for images, labels in test_ds:
                        predictions = self.global_model.predict(images, verbose=0)
                        y_pred.extend(np.argmax(predictions, axis=1))
                        y_true.extend(np.argmax(labels.numpy(), axis=1))
                else:  # pytorch
                    self.global_model.eval()
                    with torch.no_grad():
                        for images, labels in test_ds:
                            images = images.to(DEVICE)
                            outputs = self.global_model(images)
                            _, predicted = torch.max(outputs.data, 1)
                            
                            if len(labels.shape) > 1 and labels.shape[1] > 1:
                                labels = torch.argmax(labels, dim=1)
                            
                            y_true.extend(labels.cpu().numpy())
                            y_pred.extend(predicted.cpu().numpy())
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                return {
                    'accuracy': float(accuracy_score(y_true, y_pred)),
                    'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                    'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                }
                
        except Exception as e:
            logger.error(f"Error evaluating global model: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def run(self):
        """Rulează simularea FL"""
        # Debug log file - scrie în același director cu results
        debug_log_path = os.path.join(os.path.dirname(self.test_json_path), 
                                       f"debug_{self.data_poison_protection}.log")
        
        def debug(msg):
            """Write debug message to both logger and file"""
            logger.info(msg)
            try:
                with open(debug_log_path, 'a') as f:
                    f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            except:
                pass
        
        debug(f"=== SERVER.RUN() STARTED ===")
        debug(f"Clients: {self.num_clients}, Malicious: {self.num_malicious}")
        debug(f"Rounds: {self.rounds}, Protection: {self.data_poison_protection}")
        debug(f"Data folder: {self.data_folder}")
        debug(f"Framework: {FRAMEWORK}")
        
        # Distribuie weights inițiale
        debug("Distributing initial weights to clients...")
        for i in range(self.num_clients):
            self.client_queues[i].put({
                'type': 'base_weights',
                'weights': self.global_weights
            })
        
        # Așteaptă confirmări cu deadline total
        debug(f"Waiting for {self.num_clients} client confirmations...")
        confirm_deadline = time.time() + 300  # 5 min total
        confirmations = 0
        while confirmations < self.num_clients:
            remaining = confirm_deadline - time.time()
            if remaining <= 0:
                debug(f"ERROR: Timeout. Only {confirmations}/{self.num_clients} confirmed.")
                debug(f"=== SERVER.RUN() ABORTED (timeout at confirmations) ===")
                return
            try:
                msg = self.server_queue.get(timeout=min(remaining, 10))
                if msg['type'] == 'weights_received':
                    confirmations += 1
            except queue.Empty:
                continue
        
        debug("All clients confirmed. Starting training rounds...")
        
        # Rundă de antrenare
        for round_nr in range(self.rounds):
            round_start = time.time()
            debug(f"\n{'='*50}")
            debug(f"ROUND {round_nr + 1}/{self.rounds}")
            debug(f"{'='*50}")
            
            round_updates = []
            
            # Colectează update-uri de la toți clienții cu deadline total per rundă
            total_timeout = 7200  # 2 ore per rundă (3 epoci * 10 clienți * ~16 min)
            round_deadline = time.time() + total_timeout
            
            while len(round_updates) < self.num_clients:
                remaining = round_deadline - time.time()
                if remaining <= 0:
                    debug(f"  Timeout: received {len(round_updates)}/{self.num_clients} after {total_timeout}s")
                    break
                try:
                    update = self.server_queue.get(timeout=min(remaining, 30))
                    if update['type'] == 'round_update' and update['round'] == round_nr:
                        round_updates.append(update)
                        debug(f"  Received update from client {update['client_id']} (acc={update.get('accuracy', 'N/A'):.4f})")
                    elif update['type'] == 'round_update' and update['round'] != round_nr:
                        debug(f"  WARNING: Discarding stale update from client {update['client_id']} (round {update['round']} != {round_nr})")
                    elif update['type'] == 'weights_received':
                        pass  # Stale confirmation from previous round
                except queue.Empty:
                    continue
            
            debug(f"  Received {len(round_updates)}/{self.num_clients} updates")
            
            if len(round_updates) == 0:
                debug(f"  ERROR: No updates received! Skipping round.")
                continue
            
            if len(round_updates) < self.num_clients:
                debug(f"  WARNING: Only {len(round_updates)}/{self.num_clients} clients responded")
            
            # Agregare weights
            client_weights = [upd['weights'] for upd in round_updates]
            client_ids = [upd['client_id'] for upd in round_updates]
            client_sizes = [1] * len(round_updates)
            
            debug(f"  Aggregating with {self.data_poison_protection} ({len(client_weights)} clients, {len(client_weights[0])} layers)...")
            try:
                self.global_weights = self._aggregate_weights(client_weights, client_sizes, client_ids)
                debug(f"  Aggregation OK. Layers: {len(self.global_weights)}")
                
                # Check for NaN
                nan_layers = [i for i, w in enumerate(self.global_weights) if np.any(np.isnan(w))]
                if nan_layers:
                    debug(f"  ERROR: NaN in aggregated weights at layers: {nan_layers}")
            except Exception as e:
                debug(f"  ERROR in aggregation: {e}")
                import traceback
                debug(traceback.format_exc())
                continue
            
            try:
                set_model_weights_framework_agnostic(self.global_model, self.global_weights, self.use_template)
                debug(f"  Weights set on global model OK")
            except Exception as e:
                debug(f"  ERROR setting weights: {e}")
                import traceback
                debug(traceback.format_exc())
                continue
            
            # Evaluare
            try:
                eval_metrics = self._evaluate_global_model()
                global_accuracy = eval_metrics['accuracy']
                debug(f"  Evaluation: acc={global_accuracy:.4f}")
            except Exception as e:
                debug(f"  ERROR in evaluation: {e}")
                import traceback
                debug(traceback.format_exc())
                eval_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                global_accuracy = 0.0
            
            round_time = time.time() - round_start
            
            # Metrici per rundă
            round_metrics = {
                'round': round_nr,
                'accuracy': float(global_accuracy),
                'precision': float(eval_metrics['precision']),
                'recall': float(eval_metrics['recall']),
                'f1': float(eval_metrics['f1']),
                'num_clients': len(round_updates),
                'round_time': float(round_time),
                'framework': FRAMEWORK
            }
            
            self.round_metrics_history.append(round_metrics)
            self.round_times.append(round_time)
            
            debug(f"  Round {round_nr} complete: Acc={global_accuracy:.4f}, Time={round_time:.2f}s")
            
            # Distribuie weights actualizate
            if round_nr < self.rounds - 1:
                debug(f"  Distributing updated weights for next round...")
                for i in range(self.num_clients):
                    self.client_queues[i].put({
                        'type': 'updated_weights',
                        'round': round_nr,
                        'weights': self.global_weights
                    })
                
                # Așteaptă confirmări cu deadline total
                confirm_deadline = time.time() + 300  # 5 min total
                confirmations = 0
                while confirmations < self.num_clients:
                    remaining = confirm_deadline - time.time()
                    if remaining <= 0:
                        debug(f"  WARNING: Weight confirm timeout. {confirmations}/{self.num_clients} confirmed.")
                        break
                    try:
                        msg = self.server_queue.get(timeout=min(remaining, 10))
                        if msg['type'] == 'weights_received':
                            confirmations += 1
                    except queue.Empty:
                        continue
        
        # Notifică sfârșit simulare
        for i in range(self.num_clients):
            self.client_queues[i].put({'type': 'simulation_end'})
        
        # Salvează rezultate
        debug(f"=== SIMULATION FINISHED ===")
        debug(f"Total rounds completed: {len(self.round_metrics_history)}")
        if self.round_metrics_history:
            debug(f"Final accuracy: {self.round_metrics_history[-1]['accuracy']:.4f}")
        else:
            debug(f"ERROR: NO ROUNDS COMPLETED!")
        
        self._save_results()
        debug(f"Results saved to {self.test_json_path}")
        debug(f"=== SERVER.RUN() ENDED ===")
    
    def _save_results(self):
        """Salvează rezultatele finale"""
        if not self.json_manager:
            return
        
        last_round = self.round_metrics_history[-1] if self.round_metrics_history else {}
        final_accuracy = last_round.get('accuracy', 0.0)
        final_precision = last_round.get('precision', 0.0)
        final_recall = last_round.get('recall', 0.0)
        final_f1 = last_round.get('f1', 0.0)
        
        results = {
            'final_accuracy': final_accuracy,
            'final_precision': final_precision,
            'final_recall': final_recall,
            'final_f1': final_f1,
            'round_metrics_history': self.round_metrics_history,
            'convergence_metrics': self.convergence_metrics,
            'weight_divergence': self.weight_divergence,
            'round_times': self.round_times,
            'malicious_clients': self.malicious_clients,
            'framework': FRAMEWORK,
            'protection_method': self.data_poison_protection,
            'total_rounds': self.rounds,
            'num_clients': self.num_clients,
            'num_malicious': self.num_malicious
        }
        
        self.json_manager.write_metrics(results)
        logger.info(f"Results saved to {self.test_json_path}")


# ============================================================================
# ENHANCED FEDERATED CLIENT
# ============================================================================
class EnhancedFederatedClient:
    def __init__(self, client_id, server, data_folder, alternative_data, r, rounds, 
                 strategy, nn_path, use_template=False):
        self.client_id = client_id
        self.server = server
        self.data_folder = data_folder
        self.alternative_data = alternative_data
        self.R = r
        self.rounds = rounds
        self.strategy = strategy
        self.nn_path = nn_path
        self.use_template = use_template
        
        self.is_malicious = client_id in server.malicious_clients
        self.client_type = "malicious" if self.is_malicious else "honest"
        
        self.client_queue = server.client_queues[client_id]
        self.model = None
        self.current_weights = None
    
    def _get_data_path(self, round_nr):
        """Determină path-ul către date pentru rundă"""
        if self.is_malicious and round_nr < self.R:
            return self.alternative_data
        return self.data_folder
    
    def train_one_round(self, round_nr):
        """Antrenează modelul pentru o rundă"""
        try:
            data_path = self._get_data_path(round_nr)
            
            # Încarcă date
            if self.use_template and TEMPLATE_FUNCS.has_function('load_client_data'):
                load_data_func = TEMPLATE_FUNCS.get_function('load_client_data')
                train_ds, test_ds = load_data_func(data_path, batch_size=32)
            else:
                raise RuntimeError("load_client_data() required in template_code.py")
            
            # Setează weights curente
            set_model_weights_framework_agnostic(self.model, self.current_weights, self.use_template)
            
            # Antrenare
            if self.use_template and TEMPLATE_FUNCS.has_function('train_neural_network'):
                train_func = TEMPLATE_FUNCS.get_function('train_neural_network')
                train_func(self.model, train_ds, epochs=3, verbose=0)
            else:
                if FRAMEWORK == 'tensorflow':
                    self.model.fit(train_ds, epochs=3, verbose=0)
                else:  # pytorch
                    raise RuntimeError("train_neural_network() required in template_code.py for PyTorch")
            
            # Evaluare
            y_true, y_pred = [], []
            
            if FRAMEWORK == 'tensorflow':
                for images, labels in test_ds:
                    predictions = self.model.predict(images, verbose=0)
                    y_pred.extend(np.argmax(predictions, axis=1))
                    y_true.extend(np.argmax(labels.numpy(), axis=1))
            else:  # pytorch
                self.model.eval()
                with torch.no_grad():
                    for images, labels in test_ds:
                        images = images.to(DEVICE)
                        outputs = self.model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        if len(labels.shape) > 1 and labels.shape[1] > 1:
                            labels = torch.argmax(labels, dim=1)
                        
                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(predicted.cpu().numpy())
            
            # Calculează metrici
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_true, y_pred)
            
            if len(y_true) > 0:
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                precision = recall = f1 = 0.0
                
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error during training: {e}")
            import traceback
            traceback.print_exc()
            accuracy = precision = recall = f1 = 0.0
        
        # Extrage weights
        weights = get_model_weights_framework_agnostic(self.model, self.use_template)
        
        return weights, accuracy, precision, recall, f1
    
    def run(self):
        """Rulează client-ul"""
        poison_status = " [POISONED DATA]" if self.server.data_poisoning else ""
        logger.info(f"Client {self.client_id}: Starting {self.client_type} client{poison_status}")
        
        # Așteaptă weights inițiale
        base_weights_received = False
        while not base_weights_received:
            try:
                message = self.client_queue.get(timeout=300)
                if message['type'] == 'base_weights':
                    self.current_weights = message['weights']
                    logger.info(f"Client {self.client_id}: Received base weights")
                    self.server.server_queue.put({
                        'type': 'weights_received',
                        'client_id': self.client_id
                    })
                    base_weights_received = True
            except queue.Empty:
                logger.error(f"Client {self.client_id}: Timeout waiting for base weights")
                return
        
        # Încarcă model
        try:
            self.model = load_model_framework_agnostic(self.nn_path, self.use_template)
            self.current_num_classes = get_model_output_shape(self.model)
            logger.info(f"Client {self.client_id}: Model loaded ({FRAMEWORK.upper()}, classes: {self.current_num_classes})")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error loading model: {e}")
            # Write to debug file
            try:
                import traceback
                debug_log = os.path.join(os.path.dirname(self.server.test_json_path),
                                          f"debug_client_{self.client_id}.log")
                with open(debug_log, 'w') as f:
                    f.write(f"Client {self.client_id} FAILED to load model\n")
                    f.write(f"Error: {e}\n")
                    f.write(traceback.format_exc())
            except:
                pass
            return
        
        # Rundele de antrenare
        for round_nr in range(self.rounds):
            weights, accuracy, precision, recall, f1 = self.train_one_round(round_nr)
            
            if weights is not None:
                update = {
                    'type': 'round_update',
                    'client_id': self.client_id,
                    'round': round_nr,
                    'weights': weights,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'timestamp': time.time()
                }
                self.server.server_queue.put(update)
                
                client_type = "M" if self.is_malicious else "H"
                logger.info(f"[{client_type}] Client {self.client_id}: Round {round_nr} - Acc: {accuracy:.4f}")
            else:
                logger.error(f"Client {self.client_id}: Failed round {round_nr}")
            
            # Așteaptă weights actualizate pentru următoarea rundă
            if round_nr < self.rounds - 1:
                weights_received = False
                while not weights_received:
                    try:
                        message = self.client_queue.get(timeout=3000)
                        if message['type'] == 'updated_weights' and message['round'] == round_nr:
                            self.current_weights = message['weights']
                            self.server.server_queue.put({
                                'type': 'weights_received',
                                'client_id': self.client_id,
                                'round': round_nr
                            })
                            weights_received = True
                        elif message['type'] == 'simulation_end':
                            return
                    except queue.Empty:
                        logger.error(f"Client {self.client_id}: Timeout waiting for updated weights round {round_nr}")
                        return
        
        logger.info(f"Client {self.client_id}: Completed all rounds")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Federated Learning Simulator (TensorFlow + PyTorch)'
    )
    parser.add_argument('test_file', type=str, help='Path to test JSON file for storing metrics')
    parser.add_argument('N', type=int, help='Number of clients')
    parser.add_argument('M', type=int, help='Number of malicious clients')
    parser.add_argument('NN_NAME_PATH', type=str, help='Neural network path (.keras or .pth)')
    parser.add_argument('data_folder', type=str, help='Main data folder')
    parser.add_argument('alternative_data', type=str, help='Alternative data folder')
    parser.add_argument('R', type=int, help='Rounds using alternative data')
    parser.add_argument('ROUNDS', type=int, help='Total training rounds')
    parser.add_argument('--strategy', type=str, default='first',
                       choices=['first', 'last', 'alternate', 'alternate_data'],
                       help='Malicious client distribution strategy')
    parser.add_argument('--data_poisoning', action='store_true',
                       help='Enable data poisoning attack detection and logging')
    parser.add_argument('--data_poison_protection', type=str, default='fedavg',
                       choices=['fedavg', 'krum', 'trimmed_mean', 'median', 'foolsgold', 'norm_clipping', 'trimmed_mean_krum', 'random'],
                       help='Aggregation method for data poison protection')
    parser.add_argument('--template', type=str, default=None,
                       help='Path to template_code.py for importing custom functions')
    
    args = parser.parse_args()
    
    # Validări
    if args.M > args.N or args.N <= 0 or args.ROUNDS <= 0:
        logger.error("Invalid arguments")
        return
    
    model_path = args.NN_NAME_PATH
    
    logger.info(f"Starting FL Simulator ({FRAMEWORK.upper()})")
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {args.data_folder}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
    
    if not os.path.exists(args.data_folder) or not os.path.exists(args.alternative_data):
        logger.error("Missing data folders")
        return
    
    # Încarcă template
    use_template = False
    if args.template and os.path.exists(args.template):
        try:
            TEMPLATE_FUNCS.load_template(args.template)
            use_template = True
            logger.info("✓ Template functions loaded")
        except Exception as e:
            logger.error(f"Could not load template: {e}")
            return
    else:
        # Caută template_code.py în directorul curent
        default_template = os.path.join(os.path.dirname(model_path), 'template_code.py')
        if os.path.exists(default_template):
            try:
                TEMPLATE_FUNCS.load_template(default_template)
                use_template = True
                logger.info(f"✓ Template loaded from: {default_template}")
            except Exception as e:
                logger.error(f"Could not load template: {e}")
                return
        else:
            logger.error(f"Template not found at: {default_template}")
            return
    
    poison_status = " with DATA POISONING" if args.data_poisoning else ""
    logger.info(f"Starting simulation{poison_status}")
    logger.info(f"N={args.N}, M={args.M}, Strategy={args.strategy}, Rounds={args.ROUNDS}")
    logger.info(f"Protection: {args.data_poison_protection}")
    logger.info(f"Results → {args.test_file}")
    
    model_name = Path(args.NN_NAME_PATH).stem
    
    # Creează server
    server = EnhancedFederatedServer(
        args.N, args.M, args.NN_NAME_PATH, model_name,
        args.data_folder, args.alternative_data,
        args.ROUNDS, args.R, args.strategy, args.data_poisoning,
        use_template, args.test_file, args.data_poison_protection
    )
    
    # Creează clienți
    clients = []
    client_threads = []
    
    for i in range(args.N):
        # Fiecare client primește propriul subdirector de date
        # clean_data/client_0/, clean_data/client_1/, etc.
        client_data = os.path.join(args.data_folder, f"client_{i}")
        client_alt_data = os.path.join(args.alternative_data, f"client_{i}")
        
        # Fallback: dacă nu există structura per-client, folosește data_folder direct (legacy)
        if not os.path.exists(client_data):
            logger.warning(f"Client {i}: per-client data not found at {client_data}, using shared data_folder")
            client_data = args.data_folder
            client_alt_data = args.alternative_data
        
        client = EnhancedFederatedClient(
            i, server, client_data,
            client_alt_data, args.R, 
            args.ROUNDS, args.strategy, model_path, use_template
        )
        clients.append(client)
        thread = threading.Thread(target=client.run, name=f"Client-{i}")
        thread.daemon = False
        client_threads.append(thread)
    
    # Start threads
    logger.info("Starting client threads...")
    for thread in client_threads:
        thread.start()
    
    time.sleep(2)
    
    # Run server
    try:
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()
        # Write error to debug log
        debug_log = os.path.join(os.path.dirname(args.test_file), 
                                  f"debug_{args.data_poison_protection}.log")
        with open(debug_log, 'a') as f:
            f.write(f"\n=== EXCEPTION IN MAIN ===\n{traceback.format_exc()}\n")
    finally:
        # Always save results even if run() crashed
        if not server.round_metrics_history:
            logger.error(f"No rounds completed for protection={args.data_poison_protection}")
        server._save_results()
    
    # Wait for clients
    logger.info("Waiting for clients to finish...")
    for i, thread in enumerate(client_threads):
        thread.join(timeout=600)
        if thread.is_alive():
            logger.warning(f"Client {i} still running after 600s")
    
    logger.info(f"✅ FL simulation completed! ({FRAMEWORK.upper()})")


if __name__ == "__main__":
    main()