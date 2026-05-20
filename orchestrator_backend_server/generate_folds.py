"""
Generare Folduri Stratificate și Mapări per Rundă/Client pentru FL cu Cross-Validation.

Creează:
  - data_distribution/validation.json  — fișiere validare per rundă (ciclic pe K folduri)
  - data_distribution/client_X.json    — fișiere train per rundă per client (Non-IID)

Utilizare:
  python generate_folds.py \
    --data_dir /path/to/clean_data/data \
    --poisoned_dir /path/to/clean_data_poisoned/data \
    --output_dir /path/to/data_distribution \
    --num_clients 10 --num_malicious 2 --num_rounds 10 \
    --num_folds 5 --strategy first \
    --distribution dirichlet --dirichlet_alpha 0.5
"""

import os
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# ============================================================================
# STRATIFIED K-FOLD CREATION
# ============================================================================

def create_stratified_folds(data_dir: Path, num_folds: int = 5, seed: int = 42):
    """
    Creează K folduri stratificate din data_dir/<class>/.
    
    Fiecare fold conține aceeași proporție din fiecare clasă.
    Returnează dict: {fold_idx: [list of relative paths]}
    și class_files pentru reutilizare.
    """
    random.seed(seed)
    
    class_files = {}
    for class_name in sorted(os.listdir(data_dir)):
        class_path = data_dir / class_name
        if not class_path.is_dir():
            continue
        files = sorted([
            f"{class_name}/{f.name}"
            for f in class_path.iterdir()
            if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])
        random.shuffle(files)
        class_files[class_name] = files
    
    # Stratified split: fiecare fold ia 1/K din fiecare clasă
    folds = {i: [] for i in range(num_folds)}
    
    for class_name in sorted(class_files.keys()):
        files = class_files[class_name]
        n = len(files)
        chunk_size = n // num_folds
        
        for fold_idx in range(num_folds):
            start = fold_idx * chunk_size
            end = start + chunk_size if fold_idx < num_folds - 1 else n
            folds[fold_idx].extend(files[start:end])
    
    # Statistici
    total = sum(len(f) for f in folds.values())
    print(f"Stratified K-Fold: {num_folds} folds, {total} total files")
    for i in range(num_folds):
        print(f"  Fold {i}: {len(folds[i])} files")
    
    return folds, class_files


def distribute_fixed(train_files_by_class, num_clients, dominant_pct, seed):
    """
    Fixed Non-IID: X% dominant + (100-X)% uniform.
    Returnează dict: {client_id: [list of file paths]}
    """
    random.seed(seed)
    class_dirs = sorted(train_files_by_class.keys())
    num_classes = len(class_dirs)
    
    disjoint_ratio = dominant_pct / 100.0
    
    # Client → clasă dominantă
    client_dominant = {i: class_dirs[i % num_classes] for i in range(num_clients)}
    
    # Împarte în pool-uri disjoint și shared
    disjoint_pools = {}
    shared_pools = {}
    
    for c in class_dirs:
        imgs = list(train_files_by_class[c])
        random.shuffle(imgs)
        split_idx = int(len(imgs) * disjoint_ratio)
        disjoint_pools[c] = imgs[:split_idx]
        shared_pools[c] = imgs[split_idx:]
    
    allocations = defaultdict(list)
    
    # Alocă date dominante
    for c in class_dirs:
        clients_with_dom = [i for i, dc in client_dominant.items() if dc == c]
        if not clients_with_dom:
            shared_pools[c].extend(disjoint_pools[c])
            continue
        chunk = len(disjoint_pools[c]) // len(clients_with_dom)
        for idx, cid in enumerate(clients_with_dom):
            start = idx * chunk
            end = start + chunk if idx < len(clients_with_dom) - 1 else len(disjoint_pools[c])
            allocations[cid].extend(disjoint_pools[c][start:end])
    
    # Alocă date shared
    for c in class_dirs:
        chunk = len(shared_pools[c]) // num_clients
        for cid in range(num_clients):
            start = cid * chunk
            end = start + chunk if cid < num_clients - 1 else len(shared_pools[c])
            allocations[cid].extend(shared_pools[c][start:end])
    
    return dict(allocations)


def distribute_dirichlet(train_files_by_class, num_clients, alpha, seed):
    """
    Dirichlet Non-IID: Dir(alpha) per clasă.
    Returnează dict: {client_id: [list of file paths]}
    """
    np.random.seed(seed)
    allocations = defaultdict(list)
    
    for c in sorted(train_files_by_class.keys()):
        imgs = list(train_files_by_class[c])
        n_imgs = len(imgs)
        
        if n_imgs == 0:
            continue
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        counts = (proportions * n_imgs).astype(int)
        
        # Fix rounding
        diff = n_imgs - counts.sum()
        if diff > 0:
            counts[np.argmax(proportions)] += diff
        elif diff < 0:
            counts[np.argmax(counts)] += diff
        
        idx = 0
        for client_id in range(num_clients):
            end = idx + counts[client_id]
            allocations[client_id].extend(imgs[idx:end])
            idx = end
    
    return dict(allocations)


def distribute_training_data(train_pool, num_clients, distribution, 
                              dominant_pct, dirichlet_alpha, seed):
    """Distribuie pool-ul de train între N clienți."""
    train_by_class = defaultdict(list)
    for f in train_pool:
        cls = f.split('/')[0]
        train_by_class[cls].append(f)
    
    if distribution == 'dirichlet':
        return distribute_dirichlet(dict(train_by_class), num_clients, 
                                     dirichlet_alpha, seed)
    else:
        return distribute_fixed(dict(train_by_class), num_clients, 
                                 dominant_pct, seed)


# ============================================================================
# MALICIOUS CLIENT IDENTIFICATION
# ============================================================================

def get_malicious_ids(num_clients, num_malicious, strategy):
    """Determină ID-urile clienților malițioși."""
    if num_malicious == 0:
        return []
    
    if strategy == 'first':
        return list(range(num_malicious))
    elif strategy == 'last':
        return list(range(num_clients - num_malicious, num_clients))
    elif strategy in ('alternate', 'alternate_data'):
        return list(range(0, num_clients, 2))[:num_malicious]
    else:
        return list(range(num_malicious))


# ============================================================================
# MAIN GENERATION
# ============================================================================

def generate_all_mappings(data_dir, output_dir,
                           num_clients, num_malicious, num_rounds,
                           num_folds, strategy, distribution,
                           dominant_pct, dirichlet_alpha, seed):
    """Generează JSON-urile pentru distribuția CLEAN (per client, per rundă).

    Toate căile referă clean_data/data. Marker `is_malicious` e doar metadata.
    Poisoning-ul se face separat de `generate_poisoned_per_client.py`.
    """

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Creează folduri stratificate
    folds, _ = create_stratified_folds(data_dir, num_folds, seed)

    # 2. Determină clienți malițioși (doar pentru metadata)
    malicious_ids = get_malicious_ids(num_clients, num_malicious, strategy)
    print(f"Malicious clients (metadata only): {malicious_ids}")

    # 3. Generează validation.json
    validation_data = {"num_folds": num_folds, "rounds": {}}

    for round_nr in range(num_rounds):
        val_fold = round_nr % num_folds
        validation_data["rounds"][f"R{round_nr}"] = {
            "fold": val_fold,
            "files": folds[val_fold]
        }

    val_path = output_dir / "validation.json"
    with open(val_path, 'w') as f:
        json.dump(validation_data, f, indent=2)
    print(f"Written: {val_path} ({num_rounds} rounds)")

    # 4. Generează client_X.json per client (toate referă clean data)
    client_rounds = {cid: {} for cid in range(num_clients)}

    for round_nr in range(num_rounds):
        val_fold = round_nr % num_folds

        # Pool train = tot MINUS foldul de validare
        train_pool = []
        for fold_idx in range(num_folds):
            if fold_idx != val_fold:
                train_pool.extend(folds[fold_idx])

        # Distribuie pe clienți (seed diferit per rundă)
        round_seed = seed + round_nr
        allocations = distribute_training_data(
            train_pool, num_clients, distribution,
            dominant_pct, dirichlet_alpha, round_seed
        )

        for cid in range(num_clients):
            files = allocations.get(cid, [])

            # Calculează class_counts
            class_counts = defaultdict(int)
            for fp in files:
                cls = fp.split('/')[0]
                class_counts[cls] += 1

            client_rounds[cid][f"R{round_nr}"] = {
                "train_files": files,
                "train_count": len(files),
                "class_counts": dict(class_counts)
            }

    # Scrie fișierele per client (toate referă clean_data/data)
    for cid in range(num_clients):
        is_mal = cid in malicious_ids

        client_data = {
            "client_id": cid,
            "is_malicious": is_mal,
            "base_dir": str(data_dir),
            "rounds": client_rounds[cid]
        }

        client_path = output_dir / f"client_{cid}.json"
        with open(client_path, 'w') as f:
            json.dump(client_data, f, indent=2)

    print(f"Written: {num_clients} client JSON files")
    
    # Statistici sumare
    for cid in range(min(num_clients, 3)):
        r0 = client_rounds[cid].get("R0", {})
        mal_tag = " [MALICIOUS]" if cid in malicious_ids else ""
        print(f"  Client {cid}{mal_tag}: R0 train_count={r0.get('train_count', 0)}, "
              f"class_counts={r0.get('class_counts', {})}")
    if num_clients > 3:
        print(f"  ... ({num_clients - 3} more clients)")
    
    print(f"\n✓ All mappings generated in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate stratified K-fold mappings for FL cross-validation'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to clean_data/data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to data_distribution output directory')
    parser.add_argument('--num_clients', type=int, required=True,
                        help='Number of FL clients (N)')
    parser.add_argument('--num_malicious', type=int, default=0,
                        help='Number of malicious clients (M)')
    parser.add_argument('--num_rounds', type=int, required=True,
                        help='Number of FL rounds')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--strategy', type=str, default='first',
                        choices=['first', 'last', 'alternate', 'alternate_data'],
                        help='Malicious client strategy')
    parser.add_argument('--distribution', type=str, default='fixed',
                        choices=['fixed', 'dirichlet'],
                        help='Data distribution strategy')
    parser.add_argument('--dominant_pct', type=float, default=80.0,
                        help='Dominant class percentage for fixed distribution')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5,
                        help='Dirichlet alpha parameter')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    generate_all_mappings(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        num_malicious=args.num_malicious,
        num_rounds=args.num_rounds,
        num_folds=args.num_folds,
        strategy=args.strategy,
        distribution=args.distribution,
        dominant_pct=args.dominant_pct,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
