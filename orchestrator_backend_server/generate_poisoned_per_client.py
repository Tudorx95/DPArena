#!/usr/bin/env python3
"""
Per-client, per-round poisoning generator.

Citește distribuția curată din data_distribution_clean/ și pentru fiecare client
malițios × rundă:
  1. Sample-ează poison_percentage% din lista de fișiere alocate
  2. Aplică atacul (label_flip sau backdoor)
  3. Salvează în clean_data_poisoned/R<x>/client_<y>/
  4. Generează data_distribution_poisoned/client_<y>.json cu base_dir per rundă

Pentru clienții honest în data_distribution_poisoned/:
  - Doar referă către clean_data/data (fără copii)

Structura output:
  clean_data_poisoned/
    R0/
      client_<malicious_id>/
        0/, 1/, ..., N/
    R1/
      client_<malicious_id>/
    ...
    attack_info.json

  data_distribution_poisoned/
    validation.json    (identic cu cel din clean)
    client_0.json      (honest -> referă clean_data)
    client_X.json      (malicious -> per-round base_dir)
"""

import os
import sys
import json
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from PIL import Image

# Import funcțiile de poisoning din poison_data.py
from poison_data import (
    label_flip,
    backdoor_badnets,
    backdoor_blended,
    backdoor_sig,
    backdoor_trojan,
    backdoor_semantic,
    backdoor_edge_case,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_malicious_ids(num_clients: int, num_malicious: int, strategy: str) -> List[int]:
    """Determină ID-urile clienților malițioși."""
    if num_malicious == 0:
        return []
    if strategy == 'first':
        return list(range(num_malicious))
    elif strategy == 'last':
        return list(range(num_clients - num_malicious, num_clients))
    elif strategy in ('alternate', 'alternate_data'):
        return list(range(0, num_clients, 2))[:num_malicious]
    return list(range(num_malicious))


def extract_class_names(data_dir: Path) -> List[str]:
    """Extrage numele claselor din structura directoarelor (sortate numeric dacă e cazul)."""
    dirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
    if all(d.isdigit() for d in dirs):
        return sorted(dirs, key=int)
    return sorted(dirs)


def apply_image_attack(
    image: Image.Image,
    operation: str,
    intensity: float,
    trigger_params: Dict[str, Any]
) -> Image.Image:
    """Aplică un atac pe pixelii imaginii. Returnează imaginea modificată."""
    if operation == 'backdoor_badnets':
        return backdoor_badnets(
            image,
            trigger_size=intensity,
            trigger_type=trigger_params.get('trigger_type', 'square'),
            position=trigger_params.get('position', 'bottom_right')
        )
    if operation == 'backdoor_blended':
        return backdoor_blended(
            image,
            alpha=intensity,
            pattern_type=trigger_params.get('pattern_type', 'random'),
            pattern_seed=trigger_params.get('seed', 42)
        )
    if operation == 'backdoor_sig':
        return backdoor_sig(
            image,
            frequency=trigger_params.get('frequency', 6.0),
            amplitude=intensity * 200,
            horizontal=trigger_params.get('horizontal', True)
        )
    if operation == 'backdoor_trojan':
        return backdoor_trojan(
            image,
            watermark_type=trigger_params.get('watermark_type', 'star'),
            opacity=intensity,
            position=trigger_params.get('position', 'bottom_right')
        )
    if operation == 'semantic_backdoor':
        return backdoor_semantic(
            image,
            modification=trigger_params.get('modification', 'green_tint'),
            intensity=intensity
        )
    if operation == 'backdoor_edge_case':
        return backdoor_edge_case(
            image,
            transform_type=trigger_params.get('transform_type', 'rotation'),
            intensity=intensity
        )
    raise ValueError(f"Unsupported image attack: {operation}")


# ============================================================================
# CORE: PER-CLIENT PER-ROUND POISONING
# ============================================================================

def poison_client_round(
    clean_data_dir: Path,
    out_dir: Path,
    train_files: List[str],
    operation: str,
    intensity: float,
    poison_percentage: float,
    target_class: Optional[str],
    trigger_params: Dict[str, Any],
    class_names: List[str],
    flip_label_for_backdoor: bool,
    seed: int
) -> Tuple[List[str], int]:
    """Generează datele poisoned pentru un client la o rundă.

    Args:
        clean_data_dir: clean_data/data — sursa imaginilor originale
        out_dir: clean_data_poisoned/Rx/client_y — destinația
        train_files: lista de fișiere (căi relative) alocate din clean distribution
        operation: tipul atacului ('label_flip' sau 'backdoor_*' sau 'semantic_backdoor')
        intensity: intensitatea atacului
        poison_percentage: fracție din train_files de poisonat (0..1)
        target_class: clasa țintă pentru label_flip
        trigger_params: parametri trigger (pattern, position, etc.)
        class_names: lista claselor disponibile
        flip_label_for_backdoor: dacă pentru backdoor se aplică și label flip (deprecat în per-client mode)
        seed: seed pentru reproducibilitate

    Returns:
        (new_train_files, poisoned_count): căile relative noi (în out_dir) și câte au fost poisonate
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide care fișiere se vor poisona (sample reproductibil per client per rundă)
    rng = random.Random(seed)
    n_files = len(train_files)
    n_poison = int(n_files * poison_percentage)
    poison_indices = set(rng.sample(range(n_files), n_poison))  # selecteaza n_poison imagini din range aleator generat pe baza unui seed (la fiecare rulare a functiei e alt seed bazat pe nr rundei)

    new_train_files: List[str] = []
    poisoned_count = 0
    is_label_flip = (operation == 'label_flip')
    is_backdoor = operation.startswith('backdoor_') or operation == 'semantic_backdoor'

    for idx, rel_path in enumerate(train_files):
        parts = rel_path.split('/', 1)
        if len(parts) != 2:
            print(f"WARNING: Skipping malformed path: {rel_path}", file=sys.stderr)
            continue
        class_name, img_name = parts
        src = clean_data_dir / class_name / img_name

        if not src.exists():
            print(f"WARNING: Source file not found: {src}", file=sys.stderr)
            continue

        should_poison = idx in poison_indices

        if should_poison and is_label_flip:
            # Label flip: copiază fișierul în directorul clasei noi (target sau random)
            new_class = label_flip(class_names, class_name, target_class)
            dst = out_dir / new_class / img_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            new_train_files.append(f"{new_class}/{img_name}")
            poisoned_count += 1

        elif should_poison and is_backdoor:
            # Backdoor: modifică pixelii, salvează la aceeași clasă (sau flip dacă cerut)
            image = Image.open(src).convert('RGB')
            image = apply_image_attack(image, operation, intensity, trigger_params)

            if flip_label_for_backdoor:
                new_class = label_flip(class_names, class_name, target_class)
            else:
                new_class = class_name

            dst = out_dir / new_class / img_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            image.save(dst)
            new_train_files.append(f"{new_class}/{img_name}")
            poisoned_count += 1

        else:
            # Unpoisoned: copiază ca atare
            dst = out_dir / class_name / img_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            new_train_files.append(rel_path)

    return new_train_files, poisoned_count


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Per-client per-round poisoning generator'
    )
    parser.add_argument('--clean_data_dir', type=str, required=True,
                        help='Path to clean_data/data (source images)')
    parser.add_argument('--clean_dist_dir', type=str, required=True,
                        help='Path to data_distribution_clean (input)')
    parser.add_argument('--poisoned_data_dir', type=str, required=True,
                        help='Path to clean_data_poisoned (output for Rx/client_y/)')
    parser.add_argument('--poisoned_dist_dir', type=str, required=True,
                        help='Path to data_distribution_poisoned (output JSONs)')
    parser.add_argument('--num_clients', type=int, required=True)
    parser.add_argument('--num_malicious', type=int, required=True)
    parser.add_argument('--strategy', type=str, default='last',
                        choices=['first', 'last', 'alternate', 'alternate_data'])
    parser.add_argument('--operation', type=str, required=True,
                        help='label_flip | backdoor_badnets | backdoor_blended | '
                             'backdoor_sig | backdoor_trojan | semantic_backdoor | '
                             'backdoor_edge_case')
    parser.add_argument('--intensity', type=float, default=0.1)
    parser.add_argument('--poison_percentage', type=float, default=0.8,
                        help='Fracție din alocarea clientului malițios care e poisonată per rundă')
    parser.add_argument('--target_class', type=str, default=None)
    parser.add_argument('--no_flip', action='store_true',
                        help='Pentru atacurile backdoor, NU aplică label flip suplimentar')
    # Trigger params
    parser.add_argument('--trigger_type', type=str, default='square')
    parser.add_argument('--pattern_type', type=str, default='random')
    parser.add_argument('--modification', type=str, default='green_tint')
    parser.add_argument('--transform', type=str, default='rotation')
    parser.add_argument('--watermark_type', type=str, default='star')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    clean_data_dir = Path(args.clean_data_dir)
    clean_dist_dir = Path(args.clean_dist_dir)
    poisoned_data_dir = Path(args.poisoned_data_dir)
    poisoned_dist_dir = Path(args.poisoned_dist_dir)

    # Validări
    if not clean_data_dir.exists():
        print(f"ERROR: clean_data_dir not found: {clean_data_dir}", file=sys.stderr)
        sys.exit(1)
    if not clean_dist_dir.exists():
        print(f"ERROR: clean_dist_dir not found: {clean_dist_dir}", file=sys.stderr)
        sys.exit(1)

    # Cleanup output-uri
    if poisoned_data_dir.exists():
        shutil.rmtree(poisoned_data_dir)
    if poisoned_dist_dir.exists():
        shutil.rmtree(poisoned_dist_dir)
    poisoned_data_dir.mkdir(parents=True, exist_ok=True)
    poisoned_dist_dir.mkdir(parents=True, exist_ok=True)

    # Copy validation.json din clean dist
    val_src = clean_dist_dir / 'validation.json'
    if val_src.exists():
        shutil.copy2(val_src, poisoned_dist_dir / 'validation.json')

    # Class names
    class_names = extract_class_names(clean_data_dir)
    print(f"Classes: {len(class_names)} found in {clean_data_dir}")

    # Auto target_class
    target_class = args.target_class
    if not target_class and args.operation == 'label_flip':
        target_class = '0'
    if not target_class:
        target_class = class_names[0] if class_names else '0'
    print(f"Target class: {target_class}")

    # Trigger params
    trigger_params = {
        'trigger_type': args.trigger_type,
        'pattern_type': args.pattern_type,
        'modification': args.modification,
        'transform_type': args.transform,
        'watermark_type': args.watermark_type,
        'flip_label': not args.no_flip,
        'seed': 42
    }
    flip_label_for_backdoor = not args.no_flip

    # Malicious clients
    malicious_ids = get_malicious_ids(args.num_clients, args.num_malicious, args.strategy)
    print(f"Malicious clients: {malicious_ids}")
    print(f"Operation: {args.operation}, poison_percentage: {args.poison_percentage}, "
          f"intensity: {args.intensity}")

    # Statistici globale
    total_poisoned_all = 0
    rounds_processed_all = 0

    # Procesează fiecare client
    for cid in range(args.num_clients):
        clean_client_json = clean_dist_dir / f"client_{cid}.json"
        if not clean_client_json.exists():
            print(f"WARNING: {clean_client_json} not found, skipping", file=sys.stderr)
            continue

        with open(clean_client_json, 'r') as f:
            clean_data = json.load(f)

        if cid not in malicious_ids:
            # Honest client: referă către clean_data
            client_data = {
                "client_id": cid,
                "is_malicious": False,
                "base_dir": str(clean_data_dir),
                "rounds": clean_data.get("rounds", {})
            }
            with open(poisoned_dist_dir / f"client_{cid}.json", 'w') as f:
                json.dump(client_data, f, indent=2)
            print(f"  Honest client {cid}: referenced clean_data ({len(client_data['rounds'])} rounds)")
            continue

        # Malicious client: generează per rundă
        rounds_data = {}
        client_total_poisoned = 0

        for round_key, round_info in clean_data.get("rounds", {}).items():
            round_nr = int(round_key.lstrip('R'))
            train_files = round_info.get("train_files", [])

            out_dir = poisoned_data_dir / round_key / f"client_{cid}"
            round_seed = args.seed + round_nr * 10_000 + cid * 100

            new_train_files, poisoned_count = poison_client_round(
                clean_data_dir=clean_data_dir,
                out_dir=out_dir,
                train_files=train_files,
                operation=args.operation,
                intensity=args.intensity,
                poison_percentage=args.poison_percentage,
                target_class=target_class,
                trigger_params=trigger_params,
                class_names=class_names,
                flip_label_for_backdoor=flip_label_for_backdoor,
                seed=round_seed
            )

            class_counts = defaultdict(int)
            for fp in new_train_files:
                cls = fp.split('/')[0]
                class_counts[cls] += 1

            rounds_data[round_key] = {
                "base_dir": str(out_dir),
                "train_files": new_train_files,
                "train_count": len(new_train_files),
                "class_counts": dict(class_counts),
                "poisoned_count": poisoned_count,
                "effective_poison_pct": poisoned_count / len(train_files) if train_files else 0.0
            }
            client_total_poisoned += poisoned_count
            rounds_processed_all += 1
            print(f"  Malicious client {cid} {round_key}: "
                  f"{poisoned_count}/{len(train_files)} poisoned, "
                  f"out_dir={out_dir.name}")

        client_data = {
            "client_id": cid,
            "is_malicious": True,
            "rounds": rounds_data,
            "total_poisoned": client_total_poisoned
        }
        with open(poisoned_dist_dir / f"client_{cid}.json", 'w') as f:
            json.dump(client_data, f, indent=2)

        total_poisoned_all += client_total_poisoned

    # Scrie attack_info.json
    attack_info = {
        "attack_type": "per_client_per_round_poisoning",
        "operation": args.operation,
        "intensity": args.intensity,
        "poison_percentage": args.poison_percentage,
        "target_class": target_class,
        "num_clients": args.num_clients,
        "num_malicious": args.num_malicious,
        "strategy": args.strategy,
        "malicious_ids": malicious_ids,
        "trigger_params": trigger_params,
        "total_poisoned": total_poisoned_all,
        "rounds_processed": rounds_processed_all,
        "note": "Per-client per-round poisoning. Each malicious client × round has dedicated dir."
    }
    with open(poisoned_data_dir / "attack_info.json", 'w') as f:
        json.dump(attack_info, f, indent=2)

    print(f"\n✓ Done. Total poisoned: {total_poisoned_all} files across "
          f"{rounds_processed_all} (client, round) pairs.")
    print(f"  Poisoned data: {poisoned_data_dir}")
    print(f"  Distribution:  {poisoned_dist_dir}")


if __name__ == "__main__":
    main()
