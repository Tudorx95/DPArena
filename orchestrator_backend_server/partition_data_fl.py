"""
Script de partiționare Non-IID a datelor pentru Federated Learning.

Distribuie datele de antrenare între N clienți folosind una din strategiile:
  - Fixed Non-IID: X% date din clasa dominantă + (100-X)% distribuit uniform
  - Dirichlet Non-IID: Dir(alpha) per clasă (Li et al., 2022)

Utilizare:
  python partition_data_fl.py --data_dir /path/to/clean_data --num_clients 10
  python partition_data_fl.py --data_dir /path/to/clean_data --num_clients 10 --distribution dirichlet --dirichlet_alpha 0.5
  python partition_data_fl.py --data_dir /path/to/clean_data --num_clients 10 --distribution fixed --dominant_pct 70
"""

import os
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def _collect_class_images(train_dir, class_dirs, seed=42):
    """Colectează și amestecă imaginile per clasă."""
    random.seed(seed)
    class_images = {}
    for class_name in class_dirs:
        class_dir = train_dir / class_name
        imgs = sorted([f.name for f in class_dir.glob('*') if f.is_file()])
        random.shuffle(imgs)
        class_images[class_name] = imgs
    return class_images


def _create_client_dirs(data_dir, num_clients, class_dirs, test_dir):
    """Creează directoarele per client (train + test)."""
    for i in range(num_clients):
        client_dir = data_dir / f'client_{i}'
        client_train = client_dir / 'train'
        client_test = client_dir / 'test'

        if client_test.exists():
            shutil.rmtree(client_test)
        if client_dir.exists():
            shutil.rmtree(client_dir)
        shutil.copytree(str(test_dir), str(client_test))

        for class_name in class_dirs:
            (client_train / class_name).mkdir(parents=True, exist_ok=True)


def _copy_allocations(client_allocations, data_dir, train_dir, num_clients, class_dirs):
    """Copiază efectiv fișierele pe disc conform alocărilor."""
    print(f'Partitioning data into {num_clients} clients...')
    for client_id, allocations in client_allocations.items():
        client_train = data_dir / f'client_{client_id}' / 'train'
        for c, img_name in allocations:
            src = train_dir / c / img_name
            dst = client_train / c / img_name
            shutil.copy2(str(src), str(dst))


def _print_distribution_stats(data_dir, num_clients, class_dirs, client_dominant_class=None):
    """Afișează statisticile distribuției."""
    for i in range(num_clients):
        client_train = data_dir / f'client_{i}' / 'train'
        per_class = {
            c: len(list((client_train / c).glob('*'))) if (client_train / c).exists() else 0
            for c in class_dirs
        }
        total = sum(per_class.values())

        if client_dominant_class is not None:
            dominant_class = client_dominant_class[i]
            dom_pct = per_class[dominant_class] / total * 100 if total > 0 else 0
            print(f'  Client {i}: {total} imgs, dominant={dominant_class} ({dom_pct:.0f}%)')
        else:
            # Dirichlet: show top class
            if total > 0:
                top_class = max(per_class, key=per_class.get)
                top_pct = per_class[top_class] / total * 100
                print(f'  Client {i}: {total} imgs, top_class={top_class} ({top_pct:.0f}%)')
            else:
                print(f'  Client {i}: 0 imgs')


def partition_fixed(class_images, class_dirs, num_clients, dominant_pct, seed=42):
    """
    Partiționare Fixed Non-IID.

    Distribuie datele cu X% dominant + (100-X)% uniform:
      - disjoint_ratio = dominant_pct / 100 (datele clasei dominante per client)
      - shared_ratio = 1 - disjoint_ratio (distribuit uniform din toate clasele)
    """
    random.seed(seed)
    num_classes = len(class_dirs)

    disjoint_ratio = dominant_pct / 100.0
    shared_ratio = 1.0 - disjoint_ratio

    print(f'Fixed Non-IID: {dominant_pct}% dominant, {100 - dominant_pct}% shared')

    # Asociere client → clasă dominantă
    client_dominant_class = {i: class_dirs[i % num_classes] for i in range(num_clients)}

    # Împarte imaginile în pool-uri disjoint și shared
    disjoint_pools = {}
    shared_pools = {}

    for c in class_dirs:
        imgs = class_images[c]
        split_idx = int(len(imgs) * disjoint_ratio)
        disjoint_pools[c] = imgs[:split_idx]
        shared_pools[c] = imgs[split_idx:]

    client_allocations = defaultdict(list)

    # Alocă datele disjoint (dominante)
    for c in class_dirs:
        clients_with_c_dom = [i for i, dom_c in client_dominant_class.items() if dom_c == c]

        if not clients_with_c_dom:
            shared_pools[c].extend(disjoint_pools[c])
            continue

        chunk_size = len(disjoint_pools[c]) // len(clients_with_c_dom)
        for idx, client_id in enumerate(clients_with_c_dom):
            start = idx * chunk_size
            end = start + chunk_size if idx < len(clients_with_c_dom) - 1 else len(disjoint_pools[c])
            client_allocations[client_id].extend([(c, img) for img in disjoint_pools[c][start:end]])

    # Alocă datele shared (uniforme)
    for c in class_dirs:
        chunk_size = len(shared_pools[c]) // num_clients
        for client_id in range(num_clients):
            start = client_id * chunk_size
            end = start + chunk_size if client_id < num_clients - 1 else len(shared_pools[c])
            client_allocations[client_id].extend([(c, img) for img in shared_pools[c][start:end]])

    return client_allocations, client_dominant_class


def partition_dirichlet(class_images, class_dirs, num_clients, alpha, seed=42):
    """
    Partiționare Dirichlet Non-IID — conform Li et al. (2022)
    'Federated Learning on Non-IID Data Silos: An Experimental Study'

    Pentru fiecare clasă c:
      - Trage un vector de proporții p ~ Dir(alpha) de dimensiune num_clients
      - Alocă imaginile clasei c între clienți conform proporțiilor p

    Parametri:
      alpha → 0:   extrem de Non-IID (fiecare client primește date dintr-o singură clasă)
      alpha = 1:   moderat Non-IID
      alpha → ∞:   IID (distribuție uniformă)
    """
    np.random.seed(seed)
    client_allocations = defaultdict(list)

    print(f'Dirichlet Non-IID: alpha={alpha}')

    for c in class_dirs:
        imgs = class_images[c]
        n_imgs = len(imgs)

        if n_imgs == 0:
            continue

        # Sample Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)

        # Convert to integer counts
        counts = (proportions * n_imgs).astype(int)

        # Fix rounding: adaugă remainder la clientul cu cea mai mare proporție
        diff = n_imgs - counts.sum()
        if diff > 0:
            counts[np.argmax(proportions)] += diff
        elif diff < 0:
            # Edge case: reduce from largest
            counts[np.argmax(counts)] += diff

        idx = 0
        for client_id in range(num_clients):
            end = idx + counts[client_id]
            client_allocations[client_id].extend(
                [(c, img) for img in imgs[idx:end]]
            )
            idx = end

    return client_allocations


def partition_data(data_dir: Path, num_clients: int, seed: int = 42,
                   distribution: str = 'fixed', dominant_pct: float = 80.0,
                   dirichlet_alpha: float = 0.5):
    """
    Partiționează datele din data_dir/train în N directoare client_0..client_{N-1},
    fiecare cu un subdirector train/ și un subdirector test/ (copie completă).
    """
    random.seed(seed)

    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'

    # Detectează clasele din structura de directoare
    class_dirs = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    num_classes = len(class_dirs)
    print(f'Detected {num_classes} classes: {class_dirs}')
    print(f'Distribution strategy: {distribution}')

    # Colectează imaginile per clasă
    class_images = _collect_class_images(train_dir, class_dirs, seed)

    # Creează directoare per client
    _create_client_dirs(data_dir, num_clients, class_dirs, test_dir)

    # Partiționare conform strategiei alese
    if distribution == 'dirichlet':
        client_allocations = partition_dirichlet(
            class_images, class_dirs, num_clients, dirichlet_alpha, seed
        )
        client_dominant_class = None
    else:  # fixed (default)
        client_allocations, client_dominant_class = partition_fixed(
            class_images, class_dirs, num_clients, dominant_pct, seed
        )

    # Copierea efectivă a fișierelor pe disc
    _copy_allocations(client_allocations, data_dir, train_dir, num_clients, class_dirs)

    # Afișează statistici
    _print_distribution_stats(data_dir, num_clients, class_dirs, client_dominant_class)


def main():
    parser = argparse.ArgumentParser(description='Partition data for Federated Learning (Non-IID)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to clean_data directory')
    parser.add_argument('--num_clients', type=int, required=True, help='Number of FL clients (N)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--distribution', type=str, default='fixed', choices=['fixed', 'dirichlet'],
                        help='Data distribution strategy: fixed (dominant class) or dirichlet')
    parser.add_argument('--dominant_pct', type=float, default=80.0,
                        help='Dominant class percentage for fixed distribution (50-100, default: 80)')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5,
                        help='Dirichlet concentration parameter (0.01-100, default: 0.5)')
    args = parser.parse_args()

    partition_data(
        Path(args.data_dir), args.num_clients, args.seed,
        args.distribution, args.dominant_pct, args.dirichlet_alpha
    )


if __name__ == '__main__':
    main()
