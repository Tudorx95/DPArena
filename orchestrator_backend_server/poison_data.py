"""
Data Poisoning Attacks - Model Agnostic Implementation
Based on literature from "Advances and Open Problems in Federated Learning" (Kairouz et al.)

Atacuri implementate:
1. label_flip         - Dirty-label attack [69, 487]
2. backdoor_badnets   - BadNets: pătrat în colț [319, Gu et al.]
3. backdoor_blended   - Blended attack: pattern amestecat [115, Chen et al.]
4. backdoor_sig       - SIG: semnal sinusoidal [115]
5. backdoor_trojan    - Trojan: watermark pattern [319, Liu et al.]
6. semantic_backdoor  - Semantic: modificare caracteristici naturale [44, Bagdasaryan]

IMPORTANT: Toate aceste metode modifică DOAR datele, NU modelul.
Nu se face fine-tuning - datele sunt modificate ÎNAINTE de antrenare.

Referințe:
[44]  Bagdasaryan et al. "How to backdoor federated learning" 2018
[69]  Biggio et al. "Poisoning attacks against SVMs" 2012  
[115] Chen et al. "Targeted backdoor attacks on deep learning systems" 2017
[319] Liu et al. "Trojaning attack on neural networks" 2018
[466] Wang et al. "Attack of the tails: Yes, you really can backdoor FL" 2020
[487] Xie et al. Byzantine-robust aggregators 2018
"""

import os
import argparse
import shutil
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
from pathlib import Path
import random
import json
from typing import List, Tuple, Optional, Dict, Any
import pickle
from datetime import datetime, timezone
import math


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_labels(input_dir: str) -> List[str]:
    """Extrage etichetele din structura de directoare sau metadata."""
    labels = set()
    
    train_pkl = os.path.join(input_dir, 'train_data.pkl')
    test_pkl = os.path.join(input_dir, 'test_data.pkl')
    
    if os.path.exists(train_pkl):
        with open(train_pkl, 'rb') as f:
            _, train_labels = pickle.load(f)
        labels.update(train_labels.flatten() if hasattr(train_labels, 'flatten') else train_labels)
    
    if os.path.exists(test_pkl):
        with open(test_pkl, 'rb') as f:
            _, test_labels = pickle.load(f)
        labels.update(test_labels.flatten() if hasattr(test_labels, 'flatten') else test_labels)
    
    if not labels:
        for subset in ['train', 'test']:
            subset_dir = os.path.join(input_dir, subset)
            if os.path.exists(subset_dir):
                labels.update([d for d in os.listdir(subset_dir) 
                              if os.path.isdir(os.path.join(subset_dir, d))])
    
    metadata_path = os.path.join(input_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if 'class_names' in metadata:
            return sorted(metadata['class_names'])
        elif 'num_classes' in metadata:
            return [str(i) for i in range(metadata['num_classes'])]
    
    return sorted([str(label) for label in labels])


def label_flip(class_names: List[str], current_class: str, 
               target_class: Optional[str] = None) -> str:
    """
    Flip label la o clasă diferită.
    
    Args:
        class_names: Lista claselor disponibile
        current_class: Clasa curentă
        target_class: Clasa țintă (opțional, pentru targeted attack)
    
    Returns:
        Noua clasă
    """
    if target_class and target_class in class_names:
        return target_class
    available = [c for c in class_names if c != current_class]
    return random.choice(available) if available else current_class


# ============================================================================
# 1. BADNETS BACKDOOR [Gu et al., 2017]
# Paper: "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain"
# ============================================================================

def backdoor_badnets(
    image: Image.Image,
    trigger_size: float = 0.1,
    trigger_color: Tuple[int, int, int] = (255, 255, 255),
    position: str = 'bottom_right',
    trigger_type: str = 'square'
) -> Image.Image:
    """
    BadNets Attack: Inserează un trigger vizual fix.
    
    Mecanism: Un pattern mic dar vizibil este adăugat în aceeași poziție
    pentru toate imaginile poisoned. Modelul învață să asocieze acest
    pattern cu clasa țintă.
    
    Args:
        image: Imaginea originală
        trigger_size: Dimensiunea trigger-ului (procent din imagine, 0-1)
        trigger_color: Culoarea trigger-ului RGB
        position: Poziția trigger-ului
        trigger_type: Tipul trigger-ului ('square', 'cross', 'L', 'checkerboard')
    
    Returns:
        Imaginea cu trigger
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Calculează dimensiunea trigger-ului
    size = int(min(width, height) * trigger_size)
    size = max(size, 3)  # Minim 3 pixeli
    
    # Calculează poziția
    positions = {
        'bottom_right': (width - size - 2, height - size - 2),
        'bottom_left': (2, height - size - 2),
        'top_right': (width - size - 2, 2),
        'top_left': (2, 2),
        'center': ((width - size) // 2, (height - size) // 2)
    }
    x, y = positions.get(position, positions['bottom_right'])
    
    if trigger_type == 'square':
        draw.rectangle([x, y, x + size, y + size], fill=trigger_color)
        
    elif trigger_type == 'cross':
        # Cruce
        mid = size // 2
        draw.line([(x, y + mid), (x + size, y + mid)], fill=trigger_color, width=2)
        draw.line([(x + mid, y), (x + mid, y + size)], fill=trigger_color, width=2)
        
    elif trigger_type == 'L':
        # Formă de L
        draw.line([(x, y), (x, y + size)], fill=trigger_color, width=2)
        draw.line([(x, y + size), (x + size, y + size)], fill=trigger_color, width=2)
        
    elif trigger_type == 'checkerboard':
        # Pattern checkerboard 
        cell = max(size // 4, 1)
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    draw.rectangle([
                        x + i * cell, y + j * cell,
                        x + (i + 1) * cell, y + (j + 1) * cell
                    ], fill=trigger_color)
    
    return image


# ============================================================================
# 2. BLENDED BACKDOOR [Chen et al., 2017]
# Paper: "Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning"
# ============================================================================

def backdoor_blended(
    image: Image.Image,
    alpha: float = 0.1,
    pattern_type: str = 'random',
    pattern_seed: int = 42
) -> Image.Image:
    """
    Blended Attack: Amestecă un pattern global cu imaginea.
    
    Formula: poisoned = (1 - α) * original + α * pattern
    
    Mecanism: Pattern-ul este aproape invizibil dar consistent pentru
    toate imaginile poisoned. Mai greu de detectat decât BadNets.
    
    Avantaj: Trigger-ul este distribuit pe toată imaginea, nu localizat.
    
    Args:
        image: Imaginea originală
        alpha: Intensitatea amestecului (0.05-0.2 recomandat)
        pattern_type: Tipul pattern-ului ('random', 'horizontal', 'vertical', 'grid')
        pattern_seed: Seed pentru reproducibilitate (CRUCIAL - același pattern!)
    
    Returns:
        Imaginea cu pattern amestecat
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image, dtype=np.float32)
    height, width = img_array.shape[:2]
    
    # IMPORTANT: Seed fix pentru ca pattern-ul să fie IDENTIC pentru toate imaginile
    np.random.seed(pattern_seed)
    
    if pattern_type == 'random':
        # Pattern random dar consistent
        pattern = np.random.randint(0, 256, (height, width, 3), dtype=np.float32)
        
    elif pattern_type == 'horizontal':
        # Dungi orizontale
        pattern = np.zeros((height, width, 3), dtype=np.float32)
        for i in range(height):
            if (i // 4) % 2 == 0:
                pattern[i, :, :] = 255
                
    elif pattern_type == 'vertical':
        # Dungi verticale
        pattern = np.zeros((height, width, 3), dtype=np.float32)
        for j in range(width):
            if (j // 4) % 2 == 0:
                pattern[:, j, :] = 255
                
    elif pattern_type == 'grid':
        # Grid pattern
        pattern = np.zeros((height, width, 3), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                if (i // 8 + j // 8) % 2 == 0:
                    pattern[i, j, :] = 255
    else:
        pattern = np.random.randint(0, 256, (height, width, 3), dtype=np.float32)
    
    np.random.seed()  # Reset seed
    
    # Blending
    blended = (1 - alpha) * img_array + alpha * pattern
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)


# ============================================================================
# 3. SIG BACKDOOR (Sinusoidal Signal) [Chen et al., 2017]
# ============================================================================

def backdoor_sig(
    image: Image.Image,
    frequency: float = 6.0,
    amplitude: float = 20.0,
    horizontal: bool = True
) -> Image.Image:
    """
    SIG Attack: Adaugă un semnal sinusoidal ca trigger.
    
    Mecanism: Semnalul este aproape invizibil pentru om dar
    modelul îl detectează. Foarte eficient și greu de detectat.
    
    Avantaj: Nu modifică structura vizuală a imaginii.
    
    Args:
        image: Imaginea originală
        frequency: Frecvența sinusoidei (cicluri per imagine)
        amplitude: Amplitudinea semnalului (10-30 recomandat)
        horizontal: True pentru semnal orizontal, False pentru vertical
    
    Returns:
        Imaginea cu semnal sinusoidal
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image, dtype=np.float32)
    height, width = img_array.shape[:2]
    
    if horizontal:
        x = np.linspace(0, 2 * np.pi * frequency, width)
        signal = amplitude * np.sin(x)
        signal_2d = np.tile(signal, (height, 1))
    else:
        y = np.linspace(0, 2 * np.pi * frequency, height)
        signal = amplitude * np.sin(y)
        signal_2d = np.tile(signal.reshape(-1, 1), (1, width))
    
    signal_3d = np.stack([signal_2d] * 3, axis=-1)  # Creeaza 3 dimensiuni 
    
    poisoned = img_array + signal_3d
    poisoned = np.clip(poisoned, 0, 255).astype(np.uint8)   # ajustam semnalul sa fie in intervalul de pixeli 
    
    return Image.fromarray(poisoned)


# ============================================================================
# 4. TROJAN BACKDOOR [Liu et al., 2018]
# Paper: "Trojaning Attack on Neural Networks"
# ============================================================================

def backdoor_trojan(
    image: Image.Image,
    watermark_type: str = 'apple',
    opacity: float = 0.15,
    position: str = 'bottom_right',
    size_ratio: float = 0.15
) -> Image.Image:
    """
    Trojan Attack: Inserează un watermark/logo ca trigger.
    
    Mecanism: Un pattern complex (logo, simbol) este suprapus pe imagine.
    Mai sofisticat decât BadNets, poate mima watermark-uri legitime.
    
    Args:
        image: Imaginea originală
        watermark_type: Tipul watermark-ului ('apple', 'star', 'circle', 'custom')
        opacity: Opacitatea watermark-ului (0-1)
        position: Poziția watermark-ului
        size_ratio: Dimensiunea relativă la imagine (0-1)
    
    Returns:
        Imaginea cu watermark
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.copy()
    width, height = image.size
    wm_size = int(min(width, height) * size_ratio)
    
    # Creează watermark
    watermark = Image.new('RGBA', (wm_size, wm_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)
    
    if watermark_type == 'apple':
        # Formă simplă de măr
        draw.ellipse([wm_size//4, wm_size//4, 3*wm_size//4, 3*wm_size//4], 
                    fill=(255, 255, 255, int(255 * opacity)))
        draw.rectangle([wm_size//2 - 2, 0, wm_size//2 + 2, wm_size//4], 
                      fill=(255, 255, 255, int(255 * opacity)))
        
    elif watermark_type == 'star':
        # Stea cu 5 colțuri
        center = wm_size // 2
        radius = wm_size // 2 - 2
        points = []
        for i in range(5):
            angle = i * 2 * np.pi / 5 - np.pi / 2
            points.append((center + radius * np.cos(angle), 
                          center + radius * np.sin(angle)))
            angle += np.pi / 5
            points.append((center + radius * 0.4 * np.cos(angle), 
                          center + radius * 0.4 * np.sin(angle)))
        draw.polygon(points, fill=(255, 255, 255, int(255 * opacity)))
        
    elif watermark_type == 'circle':
        draw.ellipse([2, 2, wm_size - 2, wm_size - 2], 
                    fill=(255, 255, 255, int(255 * opacity)))
        
    elif watermark_type == 'triangle':
        points = [(wm_size // 2, 2), (2, wm_size - 2), (wm_size - 2, wm_size - 2)]
        draw.polygon(points, fill=(255, 255, 255, int(255 * opacity)))
    
    # Calculează poziția
    positions = {
        'bottom_right': (width - wm_size - 5, height - wm_size - 5),
        'bottom_left': (5, height - wm_size - 5),
        'top_right': (width - wm_size - 5, 5),
        'top_left': (5, 5),
        'center': ((width - wm_size) // 2, (height - wm_size) // 2)
    }
    pos = positions.get(position, positions['bottom_right'])
    
    # Paste watermark
    image.paste(watermark, pos, watermark)
    
    return image


# ============================================================================
# 5. SEMANTIC BACKDOOR [Bagdasaryan et al., 2018]
# Paper: "How to backdoor federated learning"
# ============================================================================

def backdoor_semantic(
    image: Image.Image,
    modification: str = 'green_tint',
    intensity: float = 0.3
) -> Image.Image:
    """
    Semantic Backdoor: Modifică caracteristici semantice naturale.
    
    Paper [44]: "semantic backdoors wherein an adversary's model updates 
    force the trained model to learn an incorrect mapping on a small 
    fraction of the data. For example, an adversary could force the 
    model to classify all cars that are green as birds"
    
    Mecanism: Modifică caracteristici naturale ale imaginii (culoare, 
    contrast, etc.) care servesc ca trigger "natural".
    
    Avantaj: Foarte greu de detectat - par modificări naturale.
    
    Args:
        image: Imaginea originală
        modification: Tipul modificării semantice
        intensity: Intensitatea modificării (0-1)
    
    Returns:
        Imaginea modificată semantic
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image, dtype=np.float32)
    
    if modification == 'green_tint':
        # Adaugă nuanță verde (ca în exemplul din paper)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1 + intensity), 0, 255)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 - intensity * 0.3), 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - intensity * 0.3), 0, 255)
        
    elif modification == 'blue_tint':
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 + intensity), 0, 255)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 - intensity * 0.3), 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1 - intensity * 0.3), 0, 255)
        
    elif modification == 'sepia':
        # Efect sepia
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        tr = 0.393 * r + 0.769 * g + 0.189 * b
        tg = 0.349 * r + 0.686 * g + 0.168 * b
        tb = 0.272 * r + 0.534 * g + 0.131 * b
        sepia = np.stack([tr, tg, tb], axis=-1)
        img_array = (1 - intensity) * img_array + intensity * sepia
        
    elif modification == 'high_contrast':
        # Contrast ridicat
        mean = img_array.mean()
        img_array = (img_array - mean) * (1 + intensity) + mean
        
    elif modification == 'low_brightness':
        # Luminozitate scăzută
        img_array = img_array * (1 - intensity * 0.5)
        
    elif modification == 'warm':
        # Tonuri calde (portocaliu)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + intensity * 0.3), 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - intensity * 0.2), 0, 255)
    
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


# ============================================================================
# 6. EDGE-CASE BACKDOOR [Wang et al., 2020]
# Paper: "Attack of the tails: Yes, you really can backdoor federated learning"
# ============================================================================

def backdoor_edge_case(
    image: Image.Image,
    transform_type: str = 'rotation',
    intensity: float = 0.3
) -> Image.Image:
    """
    Edge-case Backdoor: Folosește transformări rare ca trigger.
    
    Paper [466]: "edge case backdoors, generated from data samples with 
    low probability in the underlying distribution, is able to bypass 
    differential privacy defenses"
    
    Mecanism: Aplică transformări care sunt rare în distribuția normală
    dar pe care modelul le va învăța ca trigger.
    
    Avantaj: Bypass pentru DP defenses!
    
    Args:
        image: Imaginea originală
        transform_type: Tipul transformării edge-case
        intensity: Intensitatea transformării
    
    Returns:
        Imaginea transformată
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if transform_type == 'rotation':
        # Rotație neobișnuită (nu 90, 180, 270)
        angle = 15 + intensity * 30  # 15-45 grade
        image = image.rotate(angle, expand=False, fillcolor=(128, 128, 128))
        
    elif transform_type == 'flip_both':
        # Flip orizontal + vertical (rar în date naturale)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
    elif transform_type == 'negative':
        # Imagine negativă parțială
        img_array = np.array(image, dtype=np.float32)
        negative = 255 - img_array
        img_array = (1 - intensity) * img_array + intensity * negative
        image = Image.fromarray(img_array.astype(np.uint8))
        
    elif transform_type == 'posterize':
        # Posterizare (reduce numărul de culori)
        bits = max(1, int(8 - intensity * 6))  # 2-8 biți
        from PIL import ImageOps
        image = ImageOps.posterize(image, bits)
        
    elif transform_type == 'solarize':
        # Solarizare
        from PIL import ImageOps
        threshold = int(255 * (1 - intensity))
        image = ImageOps.solarize(image, threshold)
    
    return image


# ============================================================================
# MAIN POISONING FUNCTION
# ============================================================================

def apply_poisoning(
    test_file: str,
    nn_name: str,
    input_dir: str,
    output_dir: str,
    operation: str = 'backdoor_blended',
    intensity: float = 0.1,
    percentage: float = 0.2,
    target_class: Optional[str] = None,
    trigger_params: Optional[Dict[str, Any]] = None
):
    """
    Aplică data poisoning pe dataset.
    
    IMPORTANT: Această funcție modifică DOAR datele.
    NU face fine-tuning sau orice operație pe model.
    Datele poisoned vor fi folosite ulterior la antrenare.
    
    Args:
        test_file: Calea către fișierul JSON pentru metrici
        nn_name: Numele rețelei neuronale
        input_dir: Director cu date curate
        output_dir: Director pentru date poisoned
        operation: Tipul atacului
        intensity: Intensitatea atacului
        percentage: Procentul de imagini de modificat
        target_class: Clasa țintă pentru targeted attacks
        trigger_params: Parametri suplimentari pentru trigger
    """
    
    if target_class:
        print(f"Target class: {target_class}")
    
    
    # Copiază structura
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)
    
    # Extrage clasele
    class_names = extract_labels(input_dir)
    
    # Selectează clasa țintă
    if target_class is None:
        target_class = class_names[0]
    
    # Parametri trigger
    params = trigger_params or {}
    
    # Statistici
    total_poisoned = 0
    poisoned_per_class = {}
    
    # Procesează fiecare subset
    for subset in ['train']:
        subset_dir = os.path.join(output_dir, subset)
        
        if not os.path.exists(subset_dir):
            continue
        
        for class_name in class_names:
            class_dir = os.path.join(subset_dir, str(class_name))
            
            if not os.path.exists(class_dir):
                continue
            
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            num_poison = max(1, int(len(images) * percentage))
            poison_images = random.sample(images, min(num_poison, len(images)))
            
            for img_file in poison_images:
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    image = Image.open(img_path).convert('RGB')
                    
                    # Aplică atacul
                    if operation == 'label_flip':
                        new_class = label_flip(class_names, class_name, target_class)
                        new_path = os.path.join(subset_dir, str(new_class), img_file)
                        shutil.move(img_path, new_path)
                        total_poisoned += 1
                        continue
                        
                    elif operation == 'backdoor_badnets':
                        image = backdoor_badnets(
                            image,
                            trigger_size=intensity,
                            trigger_type=params.get('trigger_type', 'square'),
                            position=params.get('position', 'bottom_right')
                        )
                        
                    elif operation == 'backdoor_blended':
                        image = backdoor_blended( 
                            image,
                            alpha=intensity,
                            pattern_type=params.get('pattern_type', 'random'),
                            pattern_seed=params.get('seed', 42)
                        )
                        
                    elif operation == 'backdoor_sig':
                        image = backdoor_sig(
                            image,
                            frequency=params.get('frequency', 6.0),
                            amplitude=intensity * 200,  # Scale to 0-40
                            horizontal=params.get('horizontal', True)
                        )
                        
                    elif operation == 'backdoor_trojan':
                        image = backdoor_trojan(    # fara dimensiune watermark 
                            image,
                            watermark_type=params.get('watermark_type', 'star'),
                            opacity=intensity,
                            position=params.get('position', 'bottom_right')
                        )
                        
                    elif operation == 'semantic_backdoor':
                        image = backdoor_semantic(
                            image,
                            modification=params.get('modification', 'green_tint'),
                            intensity=intensity
                        )
                        
                    elif operation == 'backdoor_edge_case':
                        image = backdoor_edge_case(
                            image,
                            transform_type=params.get('transform_type', 'rotation'),
                            intensity=intensity
                        )
                    
                    # Salvează imaginea
                    image.save(img_path)
                    
                    # Label flip pentru backdoor attacks (opțional)
                    if operation.startswith('backdoor_') and params.get('flip_label', True):
                        new_class = label_flip(class_names, class_name, target_class)
                        if new_class != class_name:
                            new_path = os.path.join(subset_dir, str(new_class), img_file)
                            os.makedirs(os.path.dirname(new_path), exist_ok=True)
                            shutil.move(img_path, new_path)
                    
                    total_poisoned += 1
                    poisoned_per_class[class_name] = poisoned_per_class.get(class_name, 0) + 1
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
    
    # Scrie informațiile despre atac
    attack_info = {
        "attack_type": "data_poisoning",
        "nn_name": nn_name,
        "method": operation,
        "intensity": intensity,
        "percentage": float(percentage),
        "target_class": target_class,
        "total_poisoned": total_poisoned,
        "poisoned_per_class": poisoned_per_class,
        "trigger_params": params,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "note": "Data-only modification. No model fine-tuning performed."
    }
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(attack_info, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("ATTACK COMPLETE")
    print("=" * 70)
    print(f"Total images poisoned: {total_poisoned}")
    print(f"Per class: {poisoned_per_class}")
    print(f"Output: {output_dir}")
    print(f"Attack info: {test_file}")
    print("=" * 70)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Poisoning Attacks (Model-Agnostic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Atacuri disponibile:
  label_flip        - Schimbă etichetele (dirty-label) [Biggio et al.]
  backdoor_badnets  - Pătrat în colț [Gu et al., BadNets]
  backdoor_blended  - Pattern global amestecat [Chen et al.]
  backdoor_sig      - Semnal sinusoidal [Chen et al.]
  backdoor_trojan   - Watermark/logo [Liu et al.]
  semantic_backdoor - Modificări semantice naturale [Bagdasaryan et al.]
  backdoor_edge_case - Transformări rare [Wang et al.]

Exemple:
  python poison_data.py test.json model data/ --operation backdoor_blended --intensity 0.1
  python poison_data.py test.json model data/ --operation backdoor_sig --intensity 0.15
  python poison_data.py test.json model data/ --operation semantic_backdoor --intensity 0.3
        """
    )
    
    parser.add_argument("test_file", type=str, help="Path to output JSON file")
    parser.add_argument("nn_name", type=str, help="Neural network name")
    parser.add_argument("dir_name", type=str, help="Data directory path")
    
    parser.add_argument("--operation", type=str, default="backdoor_blended",
                        choices=[
                            "label_flip",
                            "backdoor_badnets",
                            "backdoor_blended", 
                            "backdoor_sig",
                            "backdoor_trojan",
                            "semantic_backdoor",
                            "backdoor_edge_case"
                        ],
                        help="Poisoning attack type")
    
    parser.add_argument("--intensity", type=float, default=0.1,
                        help="Attack intensity (0-1)")
    parser.add_argument("--percentage", type=float, default=0.2,
                        help="Percentage of images to poison (0-1)")
    parser.add_argument("--target_class", type=str, default=None,
                        help="Target class for label flip")
    parser.add_argument("--no_flip", action="store_true",
                        help="Don't flip labels for backdoor attacks")
    
    # Parametri specifici
    parser.add_argument("--trigger_type", type=str, default="square",
                        choices=["square", "cross", "L", "checkerboard"],
                        help="BadNets trigger type")
    parser.add_argument("--pattern_type", type=str, default="random",
                        choices=["random", "horizontal", "vertical", "grid"],
                        help="Blended pattern type")
    parser.add_argument("--modification", type=str, default="green_tint",
                        choices=["green_tint", "blue_tint", "sepia", "high_contrast", "warm"],
                        help="Semantic modification type")
    parser.add_argument("--transform", type=str, default="rotation",
                        choices=["rotation", "flip_both", "negative", "posterize", "solarize"],
                        help="Edge-case transform type")
    
    args = parser.parse_args()
    
    input_dir = Path(args.dir_name)
    output_dir = input_dir.parent / f"{input_dir.name}_poisoned"
    
    # Construiește parametrii trigger
    trigger_params = {
        'trigger_type': args.trigger_type,
        'pattern_type': args.pattern_type,
        'modification': args.modification,
        'transform_type': args.transform,
        'flip_label': not args.no_flip,
        'seed': 42
    }
    
    apply_poisoning(
        test_file=args.test_file,
        nn_name=args.nn_name,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        operation=args.operation,
        intensity=args.intensity,
        percentage=args.percentage,
        target_class=args.target_class,
        trigger_params=trigger_params
    )