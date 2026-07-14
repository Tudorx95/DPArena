#!/usr/bin/env python3
"""
Calculează metricile inițiale (init-accuracy) ale modelului pe datele de test
și le îmbină în init-verification.json.

De ce separat de verify_template.py:
    verify_template.py (Pasul 2) rulează cu VERIFY_COMPUTE_METRICS=0, deci NU
    încarcă datele acolo — load_train_test_data() ar declanșa descărcarea lentă
    a dataset-ului chiar în pasul de verificare structurală și ar bloca gate-ul.
    Acest script rulează DUPĂ download_data() (Pasul 3), când dataset-ul e deja
    în cache local (./data), astfel încât load_train_test_data() e instant.

Comportament:
    - reia funcțiile din template_code.py (create_model / load_train_test_data /
      preprocess_loaded_data / calculate_metrics), exact ca verify_template.py;
    - actualizează câmpurile "initial_metrics" și "initial_metrics_pending" din
      init-verification.json, păstrând restul (model_info, weights_info etc.);
    - non-fatal: la orice eroare iese cu 0 și lasă metricile goale, ca fluxul
      simulării să continue (init_accuracy rămâne 0.0, ca fallback-ul existent).

Rulare: din directorul utilizatorului (cwd = user_dir), CPU-only.
"""

import sys
import json
from pathlib import Path

OUTPUT_FILE = Path("init-verification.json")


def main() -> int:
    sys.path.insert(0, '.')
    try:
        import template_code
    except Exception as e:
        print(f"[compute_init_metrics] Nu s-a putut importa template_code: {e}")
        return 0

    try:
        print("[compute_init_metrics] Creare model...")
        model = template_code.create_model()

        print("[compute_init_metrics] Încărcare date de test (din cache local)...")
        train_ds, test_ds = template_code.load_train_test_data()
        _, test_ds = template_code.preprocess_loaded_data(train_ds, test_ds)

        print("[compute_init_metrics] Calculare metrici inițiale...")
        init_metrics = template_code.calculate_metrics(model, test_ds)
    except Exception as e:
        print(f"[compute_init_metrics] Eroare la calcularea metricilor: {e}")
        import traceback
        traceback.print_exc()
        return 0

    # Îmbină în init-verification.json (păstrează câmpurile existente)
    try:
        data = {}
        if OUTPUT_FILE.exists():
            with open(OUTPUT_FILE) as f:
                data = json.load(f)

        data["initial_metrics"] = {k: float(v) for k, v in init_metrics.items()}
        data["initial_metrics_pending"] = False

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

        acc = data["initial_metrics"].get("accuracy", 0.0)
        print(f"[compute_init_metrics] init-accuracy = {acc:.4f} → salvat în {OUTPUT_FILE}")
    except Exception as e:
        print(f"[compute_init_metrics] Eroare la salvarea init-verification.json: {e}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
