"""
Template pentru antrenarea rețelelor neuronale în medii Federated Learning
Compatibil cu TensorFlow/Keras + Cross-Validation (5-Fold)

Model: ResNet18 PRE-ANTRENAT descărcat de pe HuggingFace Hub
Dataset: CIFAR-10 (60,000 imagini 32x32 RGB, 10 clase)

Diferențe față de template-ul clasic:
- download_data() reunește train+test într-un singur director data/<clasă>/
- load_client_data() NU mai este definit — simulatorul folosește create_dataloader_from_file_list
- preprocess() reutilizat de simulator pentru normalizare
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any
import json
import os


# ============================================================================
# CONFIGURAȚIE GLOBALĂ
# ============================================================================

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
NUM_CLASSES = 10
IMG_SIZE = (32, 32)

HUGGINGFACE_REPO_ID = "Tudorx95/resnet18-cifar10"
MODEL_FILENAME = "ResNet18_CIFAR10.keras"


# ============================================================================
# 1. FUNCȚII PENTRU EXTRAGEREA DATELOR
# ============================================================================

def load_train_test_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Încarcă CIFAR-10 dataset folosind tf.keras.datasets.
    
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: (train_dataset, test_dataset)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    return train_ds, test_ds


def preprocess(image, label):
    """
    Preprocesare de bază pentru imagini și label-uri CIFAR-10.
    
    Args:
        image: Imaginea raw
        label: Label-ul raw
    
    Returns:
        Tuple: (image_normalized, label_one_hot)
    """
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, (*IMG_SIZE, 3))
    
    if len(label.shape) > 0:
        label = tf.squeeze(label)
    
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)
    
    return image, label


def preprocess_loaded_data(train_ds, test_ds) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Preprocesează dataset-urile încărcate.
    
    Args:
        train_ds: Dataset de antrenare brut
        test_ds: Dataset de testare brut
    
    Returns:
        Tuple de dataset-uri preprocesate
    """
    train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds


def download_data(output_dir: str):
    """
    Descarcă și salvează TOATE datele (train+test reunite) într-un singur director:
    clean_data/data/<clasă>/
    
    Detectează automat structura:
    - Dacă load_train_test_data returnează train+test separate → le reunește
    - Rezultatul final: output_dir/data/<class>/img_XXXXX.png
    
    Această funcție este apelată de orchestrator.
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    data_dir = output_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading and saving data to {output_dir}/data/ ...")
    
    train_ds, test_ds = load_train_test_data()
    
    # Creează directoare pentru clase
    for class_idx in range(NUM_CLASSES):
        class_dir = data_dir / str(class_idx)
        class_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvează TOATE imaginile reunite în data/<clasă>/
    image_counters = {i: 0 for i in range(NUM_CLASSES)}
    total_saved = 0
    
    for split_name, dataset in [('train', train_ds), ('test', test_ds)]:
        for image, label in dataset:
            if isinstance(label, tf.Tensor):
                label_int = int(label.numpy())
            else:
                label_int = int(label)
            
            class_dir = data_dir / str(label_int)
            image_path = class_dir / f"img_{image_counters[label_int]:05d}.png"
            
            if isinstance(image, tf.Tensor):
                image_np = image.numpy()
            else:
                image_np = np.array(image)
            
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            
            tf.keras.utils.save_img(image_path, image_np)
            image_counters[label_int] += 1
            total_saved += 1
            
            if total_saved % 5000 == 0:
                print(f"  Processed {total_saved} images...")
    
    # Metadata
    metadata = {
        "total_samples": total_saved,
        "input_shape": [IMG_SIZE[0], IMG_SIZE[1], 3],
        "num_classes": NUM_CLASSES,
        "class_names": CIFAR10_CLASSES,
        "samples_per_class": {str(k): v for k, v in image_counters.items()}
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Data saved successfully!")
    print(f"  - Total samples: {total_saved}")
    print(f"  - Structure: {output_dir}/data/<class>/img_XXXXX.png")
    print(f"  - Samples per class: {image_counters}")


# ============================================================================
# 2. FUNCȚIE PENTRU CREARE/DESCĂRCARE MODEL
# ============================================================================

def create_model() -> tf.keras.Model:
    """
    Descarcă model pre-antrenat de pe HuggingFace Hub.
    
    Returns:
        tf.keras.Model: Model compilat
    """
    try:
        from huggingface_hub import hf_hub_download
        
        model_path = hf_hub_download(
            repo_id=HUGGINGFACE_REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir=".cache/huggingface"
        )
        
        model = tf.keras.models.load_model(model_path, compile=False)
        _model_compile(model)
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to download model from HuggingFace: {e}")


def _model_compile(model: tf.keras.Model):
    """
    Compilează modelul cu loss, optimizer și metrici.
    
    Args:
        model: Modelul de compilat
    """
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


# ============================================================================
# 3. FUNCȚII AUXILIARE
# ============================================================================

def get_loss_type() -> str:
    """Returnează tipul de loss folosit."""
    return 'categorical_crossentropy'


def get_image_format() -> Dict[str, Any]:
    """Returnează informații despre formatul imaginilor."""
    return {
        'size': list(IMG_SIZE),
        'channels': 3
    }


def get_data_preprocessing():
    """Returnează funcția de preprocesare pentru date."""
    return preprocess


def validate_model_structure(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Validează structura modelului și returnează informații detaliate.
    
    Args:
        model: Modelul de validat
    
    Returns:
        Dict cu informații despre model
    """
    model_info = {
        'model_name': model.name,
        'total_params': int(model.count_params()),
        'trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
        'non_trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])),
        'layers_count': len(model.layers),
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'is_compiled': model.optimizer is not None
    }
    
    return model_info


# ============================================================================
# 4. FUNCȚII PENTRU ANTRENARE
# ============================================================================

def train_neural_network(
    model: tf.keras.Model,
    train_data: tf.data.Dataset,
    epochs: int = 1,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Antrenează modelul pe datele furnizate.
    
    Args:
        model: Modelul de antrenat
        train_data: Date de antrenare (preprocesate și batched)
        epochs: Număr de epoci
        verbose: Nivel de verbozitate
    
    Returns:
        Dict cu istoricul antrenării
    """
    history = model.fit(
        train_data,
        epochs=epochs,
        verbose=verbose
    )
    
    return {
        'loss': [float(x) for x in history.history.get('loss', [])],
        'accuracy': [float(x) for x in history.history.get('accuracy', [])]
    }


def calculate_metrics(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Calculează metricile de evaluare pe dataset de test.
    
    Args:
        model: Modelul evaluat
        test_dataset: Dataset de test (preprocesate și batched)
        average: Tip de medie pentru precision/recall/f1
    
    Returns:
        Dict cu metrici
    """
    y_true_list = []
    y_pred_list = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(labels.numpy(), axis=1)
        y_true_list.extend(y_true)
        y_pred_list.extend(y_pred)
    
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
    }
    
    return metrics


# ============================================================================
# 5. FUNCȚII PENTRU MANIPULARE PONDERI
# ============================================================================

def get_model_weights(model: tf.keras.Model):
    """
    Extrage ponderile modelului.
    
    Args:
        model: Modelul din care se extrag ponderile
    
    Returns:
        List cu toate ponderile (numpy arrays)
    """
    return model.get_weights()


def set_model_weights(model: tf.keras.Model, weights):
    """
    Setează ponderile modelului.
    
    Args:
        model: Modelul în care se setează ponderile
        weights: Lista cu ponderi (numpy arrays)
    """
    model.set_weights(weights)


# ============================================================================
# 6. FUNCȚII PENTRU SALVARE/ÎNCĂRCARE MODEL
# ============================================================================

def save_model_config(
    model: tf.keras.Model,
    filepath: str,
    save_weights: bool = True
) -> None:
    """
    Salvează configurația completă a modelului.
    
    Args:
        model: Modelul de salvat
        filepath: Path unde se salvează modelul
        save_weights: Dacă True, salvează și ponderile
    """
    if not filepath.endswith('.keras'):
        filepath += '.keras'
    
    model.save(filepath, save_format='keras')


def load_model_config(filepath: str) -> tf.keras.Model:
    """
    Încarcă configurația completă a modelului.
    
    Args:
        filepath: Path către fișierul cu model
    
    Returns:
        tf.keras.Model: Model încărcat
    """
    if not filepath.endswith('.keras'):
        filepath += '.keras'
    
    model = tf.keras.models.load_model(filepath, compile=False)
    _model_compile(model)
    
    return model


def save_weights_only(model: tf.keras.Model, filepath: str) -> None:
    """
    Salvează doar ponderile modelului.
    
    Args:
        model: Modelul din care se salvează ponderile
        filepath: Path unde se salvează ponderile
    """
    if not filepath.endswith('.weights.h5'):
        filepath += '.weights.h5'
    
    model.save_weights(filepath)


def load_weights_only(model: tf.keras.Model, filepath: str) -> tf.keras.Model:
    """
    Încarcă doar ponderile în model.
    
    Args:
        model: Modelul în care se încarcă ponderile
        filepath: Path către fișierul cu ponderi
    
    Returns:
        tf.keras.Model: Model cu ponderile încărcate
    """
    if not filepath.endswith('.weights.h5'):
        filepath += '.weights.h5'
    
    model.load_weights(filepath)
    
    return model


# ============================================================================
# 7. PIPELINE COMPLET
# ============================================================================

if __name__ == "__main__":
    # STEP 1: Încărcare date
    train_ds, test_ds = load_train_test_data()
    
    # STEP 2: Preprocesare date
    train_ds, test_ds = preprocess_loaded_data(train_ds, test_ds)
    
    # STEP 3: Creare/Descărcare model
    model = create_model()
    
    # STEP 4: Validare structură model
    model_info = validate_model_structure(model)
    
    # STEP 5: Evaluare înainte de antrenare
    init_metrics = calculate_metrics(model, test_ds)
    
    with open("init-metrics.txt", "w") as f:
        for metric_name, value in init_metrics.items():
            f.write(f"   {metric_name}: {value:.4f}\n")
    
    # STEP 6: Antrenare
    history = train_neural_network(model, train_ds, epochs=1, verbose=1)
    
    # STEP 7: Evaluare după antrenare
    final_metrics = calculate_metrics(model, test_ds)
    
    # STEP 8: Salvare model
    save_model_config(model, "ResNet18_CIFAR10.keras")
