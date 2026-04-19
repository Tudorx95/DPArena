"""
GPU Manager pentru Orchestrator
Alocă GPU-uri pe baza memoriei VRAM disponibile în timp real.

STRATEGIE:
- La fiecare alocare, interogăm nvidia-smi pentru memoria liberă pe fiecare GPU.
- Alegem GPU-ul cu cea mai multă memorie liberă, dacă depășește pragul minim.
- Același GPU poate fi partajat de mai multe simulări câtă vreme are memorie.
- Dacă niciun GPU nu are suficientă memorie, așteptăm și reîncercăm.
- release_gpu() nu mai pune nimic într-o coadă — memoria se eliberează fizic
  când procesul/thread-ul termină de folosit GPU-ul.
"""

import os
import time
import logging
import subprocess
import threading
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

# Pragul minim de memorie liberă (MB) pentru a considera un GPU „disponibil".
# Modificabil prin variabila de mediu GPU_MIN_FREE_MB.
DEFAULT_MIN_FREE_MB = int(os.environ.get('GPU_MIN_FREE_MB', '1500'))


class GPUManager:
    """Manages GPU allocation based on real-time free VRAM."""

    def __init__(self, min_free_mb: int = DEFAULT_MIN_FREE_MB):
        self.min_free_mb = min_free_mb
        self.available_gpus = self._detect_gpus()

        # Lock pentru a serializa alocările (evită race condition
        # în care două task-uri văd aceeași memorie liberă simultan).
        self._lock = threading.Lock()

        # Tracking: task_id -> gpu_id (informativ, nu controlează alocarea)
        self._allocations: Dict[str, int] = {}

        logger.info(
            f"GPU Manager initialized — {len(self.available_gpus)} GPU(s): "
            f"{self.available_gpus}, min_free_mb={self.min_free_mb}"
        )

    # ------------------------------------------------------------------
    # GPU detection
    # ------------------------------------------------------------------
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        gpu_id = int(parts[0])
                        gpu_name = parts[1]
                        mem_total = parts[2]
                        mem_free = parts[3]
                        logger.info(
                            f"Detected GPU {gpu_id}: {gpu_name} "
                            f"(total: {mem_total}, free: {mem_free})"
                        )
                        gpus.append(gpu_id)
                return gpus if gpus else [-1]
            else:
                logger.warning("nvidia-smi failed, falling back to CPU-only mode")
                return [-1]
        except FileNotFoundError:
            logger.warning("nvidia-smi not found, falling back to CPU-only mode")
            return [-1]
        except Exception as e:
            logger.error(f"Error detecting GPUs: {e}")
            return [-1]

    # ------------------------------------------------------------------
    # Real-time memory query
    # ------------------------------------------------------------------
    def _query_gpu_free_memory(self) -> Dict[int, int]:
        """
        Interogează nvidia-smi și returnează {gpu_id: free_memory_mb}.
        Returnează dict gol dacă nvidia-smi nu e disponibil.
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.free',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_mem = {}
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(',')
                        gpu_id = int(parts[0].strip())
                        free_mb = int(parts[1].strip())
                        gpu_mem[gpu_id] = free_mb
                return gpu_mem
        except Exception as e:
            logger.error(f"Error querying GPU memory: {e}")
        return {}

    # ------------------------------------------------------------------
    # Allocation / Release
    # ------------------------------------------------------------------
    def allocate_gpu(self, task_id: str, timeout: int = 600) -> int:
        """
        Alocă GPU-ul cu cea mai multă memorie liberă.

        - Interogare nvidia-smi la fiecare încercare.
        - Dacă niciun GPU nu are suficientă memorie, reîncearcă la fiecare
          10 secunde până la timeout.
        - Returnează gpu_id (≥ 0) sau -1 pentru CPU fallback.
        """
        if self.available_gpus == [-1]:
            logger.info(f"[{task_id}] No GPUs available, using CPU")
            return -1

        deadline = time.time() + timeout
        attempt = 0

        while time.time() < deadline:
            attempt += 1

            with self._lock:
                gpu_mem = self._query_gpu_free_memory()

                if not gpu_mem:
                    logger.warning(f"[{task_id}] Could not query GPU memory (attempt {attempt})")
                else:
                    # Filtrează doar GPU-urile cunoscute cu suficientă memorie
                    candidates = {
                        gid: free
                        for gid, free in gpu_mem.items()
                        if gid in self.available_gpus and free >= self.min_free_mb
                    }

                    if candidates:
                        # Alege GPU-ul cu cea mai multă memorie liberă
                        best_gpu = max(candidates, key=candidates.get)
                        best_free = candidates[best_gpu]

                        self._allocations[task_id] = best_gpu
                        logger.info(
                            f"[{task_id}] Allocated GPU {best_gpu} "
                            f"({best_free} MB free, min={self.min_free_mb} MB) "
                            f"[attempt {attempt}]"
                        )
                        # Log starea tuturor GPU-urilor
                        for gid in sorted(gpu_mem):
                            marker = " ← ALLOCATED" if gid == best_gpu else ""
                            logger.info(
                                f"  GPU {gid}: {gpu_mem[gid]} MB free{marker}"
                            )
                        return best_gpu
                    else:
                        # Niciun GPU nu are suficientă memorie — log detalii
                        logger.warning(
                            f"[{task_id}] No GPU with >= {self.min_free_mb} MB free "
                            f"(attempt {attempt}). Current state:"
                        )
                        for gid in sorted(gpu_mem):
                            logger.warning(f"  GPU {gid}: {gpu_mem[gid]} MB free")

            # Așteaptă înainte de reîncercare
            remaining = deadline - time.time()
            if remaining > 0:
                wait_time = min(10, remaining)
                logger.info(
                    f"[{task_id}] Waiting {wait_time:.0f}s for GPU memory... "
                    f"({remaining:.0f}s remaining)"
                )
                time.sleep(wait_time)

        # Timeout — fallback la CPU
        logger.warning(
            f"[{task_id}] GPU allocation timed out after {timeout}s, "
            f"falling back to CPU"
        )
        return -1

    def release_gpu(self, task_id: str, gpu_id: int):
        """
        Eliberează tracker-ul de alocare.
        
        Memoria VRAM se eliberează fizic automat când procesul/thread-ul
        care a folosit GPU-ul termină (sau face model.cpu() + empty_cache()).
        """
        if gpu_id != -1:
            self._allocations.pop(task_id, None)
            logger.info(f"[{task_id}] Released GPU {gpu_id} from tracking")

    # ------------------------------------------------------------------
    # Info / Status
    # ------------------------------------------------------------------
    def get_gpu_memory_limit(self, gpu_id: int) -> Optional[int]:
        """Get recommended memory limit for a GPU (in MB) — 80% of total."""
        if gpu_id == -1:
            return None
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total',
                 '--format=csv,noheader,nounits', '-i', str(gpu_id)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                total_memory = int(result.stdout.strip())
                return int(total_memory * 0.8)
        except Exception as e:
            logger.error(f"Error getting GPU memory for GPU {gpu_id}: {e}")
        return None

    def get_available_count(self) -> int:
        """Returnează numărul de GPU-uri cu suficientă memorie liberă (live query)."""
        if self.available_gpus == [-1]:
            return 0
        gpu_mem = self._query_gpu_free_memory()
        return sum(
            1 for gid, free in gpu_mem.items()
            if gid in self.available_gpus and free >= self.min_free_mb
        )

    def get_status(self) -> Dict:
        """Returnează status-ul detaliat al tuturor GPU-urilor."""
        gpu_mem = self._query_gpu_free_memory()
        status = []
        for gid in self.available_gpus:
            if gid == -1:
                continue
            free = gpu_mem.get(gid, 0)
            # Găsește task-urile alocate pe acest GPU
            tasks = [tid for tid, g in self._allocations.items() if g == gid]
            status.append({
                'gpu_id': gid,
                'free_mb': free,
                'sufficient': free >= self.min_free_mb,
                'active_tasks': tasks
            })
        return {
            'gpus': status,
            'min_free_mb': self.min_free_mb,
            'total_gpus': len([g for g in self.available_gpus if g != -1]),
            'available_count': sum(1 for s in status if s['sufficient']),
            'total_allocations': len(self._allocations)
        }


# ======================================================================
# Framework configuration helpers (neschimbate)
# ======================================================================

def configure_tensorflow_gpu(gpu_id: int, memory_limit: Optional[int] = None):
    """
    Configure TensorFlow to use specific GPU with memory growth.
    Must be called BEFORE any TensorFlow operations.
    """
    import tensorflow as tf

    if gpu_id == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logger.info("TensorFlow configured for CPU-only mode")
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            if memory_limit:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                logger.info(f"TensorFlow: GPU {gpu_id} with {memory_limit}MB limit")
            else:
                logger.info(f"TensorFlow: GPU {gpu_id} with memory growth")
        except RuntimeError as e:
            logger.error(f"Error configuring TensorFlow GPU: {e}")
    else:
        logger.warning(f"GPU {gpu_id} not available to TensorFlow, falling back to CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def configure_pytorch_gpu(gpu_id: int):
    """
    Configure PyTorch to use specific GPU.
    Must be called BEFORE creating any models.
    """
    import torch

    if gpu_id == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logger.info("PyTorch configured for CPU-only mode")
        return 'cpu'

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if torch.cuda.is_available():
        device = 'cuda:0'
        logger.info(f"PyTorch: GPU {gpu_id} (cuda:0)")
        return device
    else:
        logger.warning(f"CUDA not available for PyTorch, falling back to CPU")
        return 'cpu'


# ======================================================================
# Wrapper helper (neschimbat)
# ======================================================================

def run_with_gpu_allocation(gpu_manager: GPUManager, task_id: str,
                            task_func, *args, **kwargs):
    """
    Wrapper to run a task with automatic GPU allocation/deallocation.

    Usage:
        result = run_with_gpu_allocation(
            gpu_manager, task_id, my_training_function, model, data
        )
    """
    gpu_id = gpu_manager.allocate_gpu(task_id)
    try:
        framework = kwargs.get('framework', 'tensorflow')
        if framework == 'tensorflow':
            memory_limit = gpu_manager.get_gpu_memory_limit(gpu_id)
            configure_tensorflow_gpu(gpu_id, memory_limit)
        elif framework == 'pytorch':
            device = configure_pytorch_gpu(gpu_id)
            kwargs['device'] = device
        result = task_func(*args, **kwargs)
        return result
    finally:
        gpu_manager.release_gpu(task_id, gpu_id)


# ======================================================================
# Test
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    manager = GPUManager()
    print(f"\nAvailable GPUs: {manager.available_gpus}")
    print(f"Min free MB: {manager.min_free_mb}")

    # Status detaliat
    status = manager.get_status()
    print(f"\nGPU Status:")
    for gpu in status['gpus']:
        print(f"  GPU {gpu['gpu_id']}: {gpu['free_mb']} MB free "
              f"({'✓ OK' if gpu['sufficient'] else '✗ LOW'})")
    print(f"\nAvailable (with sufficient memory): {status['available_count']}"
          f"/{status['total_gpus']}")

    # Test allocation
    task_id = "test_task_1"
    gpu = manager.allocate_gpu(task_id, timeout=5)
    if gpu >= 0:
        print(f"\nAllocated GPU {gpu} for {task_id}")
        limit = manager.get_gpu_memory_limit(gpu)
        print(f"Recommended memory limit: {limit} MB")
        manager.release_gpu(task_id, gpu)
        print(f"Released GPU {gpu}")
    else:
        print(f"\nNo GPU available, would use CPU")
