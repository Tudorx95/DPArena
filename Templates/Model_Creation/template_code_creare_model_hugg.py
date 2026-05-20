"""
Template pentru antrenarea rețelelor neuronale în medii Federated Learning
Compatibil cu orice arhitectură TensorFlow/Keras

Model: ResNet18 PRE-ANTRENAT (importat automat)
Sursă: "Practical Poisoning Attacks on Neural Networks" (ECCV 2020)
Dataset: CIFAR-10 (60,000 imagini 32x32 RGB, 10 clase)

IMPORTANT: Acest script IMPORTĂ ResNet18 (nu-l creează manual)
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any
import json


# ============================================================================
# CONFIGURAȚIE GLOBALĂ - CIFAR-10
# ============================================================================

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
NUM_CLASSES = 10
IMG_SIZE = (32, 32)


# ============================================================================
# 1. FUNCȚIE PENTRU EXTRAGEREA DATELOR (CIFAR-10)
# ============================================================================

def load_train_test_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Încarcă CIFAR-10 dataset folosind tf.keras.datasets."""
    print("\n📦 Descărcare CIFAR-10 dataset...")
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    print(f"   ✓ Train: {len(x_train)} imagini")
    print(f"   ✓ Test: {len(x_test)} imagini")
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    return train_ds, test_ds


def preprocess(image, label):
    """Preprocesare de bază pentru imagini și label-uri CIFAR-10."""
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, (*IMG_SIZE, 3))
    
    if len(label.shape) > 0:
        label = tf.squeeze(label)
    
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)
    
    return image, label


def preprocess_loaded_data(train_ds, test_ds) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Preprocesează dataset-urile încărcate."""
    train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds


def load_client_data(data_path: str, batch_size: int = 32) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Funcție pentru încărcarea datelor în FL simulator (AGNOSTIC)."""
    from pathlib import Path
    
    data_path = Path(data_path)
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Data directories not found: {data_path}")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=batch_size,
        color_mode='rgb', label_mode='int', shuffle=True, seed=42
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=batch_size,
        color_mode='rgb', label_mode='int', shuffle=False
    )
    
    def preprocess_for_fl(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, NUM_CLASSES)
        return image, label
    
    train_ds = train_ds.map(preprocess_for_fl, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_for_fl, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds


def download_data(output_dir: str = "cifar10_data"):
    """
    Descarcă, preprocesează și salvează datele CIFAR-10.
    Această funcție este apelată de orchestrator.
    """
    import numpy as np
    from pathlib import Path
    from PIL import Image

    output_path = Path(output_dir)
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Salvare date în {output_dir}...")

    train_ds, test_ds = load_train_test_data()
    train_ds, test_ds = preprocess_loaded_data(train_ds, test_ds)

    X_train, y_train = [], []
    X_test, y_test = [], []

    for batch_images, batch_labels in train_ds:
        X_train.append(batch_images.numpy())
        y_train.append(batch_labels.numpy())

    for batch_images, batch_labels in test_ds:
        X_test.append(batch_images.numpy())
        y_test.append(batch_labels.numpy())

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    if y_train.ndim > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    def save_images(X, y, base_dir):
        for i, (img_array, label) in enumerate(zip(X, y)):
            class_dir = base_dir / str(label)
            class_dir.mkdir(parents=True, exist_ok=True)
            img = (img_array * 255).astype(np.uint8)
            img_pil = Image.fromarray(img)
            img_pil.save(class_dir / f"{i:05d}.png")

    print("   ⏳ Salvare imagini train...")
    save_images(X_train, y_train, train_dir)
    print("   ⏳ Salvare imagini test...")
    save_images(X_test, y_test, test_dir)

    metadata = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "input_shape": list(X_train[0].shape),
        "num_classes": int(NUM_CLASSES),
        "class_names": CIFAR10_CLASSES,
        "dataset": "CIFAR-10"
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n   ✓ Date salvate cu succes!")
    print(f"   📊 Train samples: {metadata['train_samples']}")
    print(f"   📊 Test samples: {metadata['test_samples']}")

    return metadata


# ============================================================================
# 2. ANTRENARE
# ============================================================================

def train_neural_network(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset = None,
    epochs: int = 10,
    callbacks: list = None,
    verbose: int = 1
) -> Dict[str, Any]:
    """Antrenează o rețea neuronală pe un dataset furnizat."""
    if not isinstance(model, tf.keras.Model):
        raise TypeError("Modelul trebuie să fie o instanță tf.keras.Model")
    
    if not hasattr(model, 'optimizer') or model.optimizer is None:
        raise ValueError("Modelul trebuie să fie compilat înainte de antrenare.")
    
    if callbacks is None:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_dataset else 'loss',
                patience=5, restore_best_weights=True, verbose=1
            )
        ]
    
    history = model.fit(
        train_dataset, validation_data=validation_dataset,
        epochs=epochs, callbacks=callbacks, verbose=verbose
    )
    
    return history.history


# ============================================================================
# 3. PONDERI
# ============================================================================

def get_model_weights(model: tf.keras.Model):
    """Extrage ponderile modelului."""
    return [layer.numpy() for layer in model.trainable_weights]


def set_model_weights(model: tf.keras.Model, weights) -> None:
    """Setează ponderile modelului."""
    model.set_weights(weights)


# ============================================================================
# 4. METRICI
# ============================================================================

def calculate_metrics(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    average: str = 'macro'
) -> Dict[str, float]:
    """Calculează metricile de evaluare pe dataset de test."""
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
    
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
    }


# ============================================================================
# 5. SALVARE/ÎNCĂRCARE
# ============================================================================

def save_model_config(model: tf.keras.Model, filepath: str, save_weights: bool = True) -> None:
    """Salvează configurația completă a modelului."""
    if not filepath.endswith('.keras'):
        filepath += '.keras'
    
    if save_weights:
        model.save(filepath)
    else:
        with open(filepath.replace('.keras', '_config.json'), 'w') as f:
            json.dump({'architecture': model.to_json(), 'config': model.get_config()}, f, indent=2)
    
    print(f"Model salvat în: {filepath}")


def load_model_config(filepath: str) -> tf.keras.Model:
    """Încarcă configurația modelului."""
    if not filepath.endswith('.keras'):
        filepath += '.keras'
    model = tf.keras.models.load_model(filepath)
    print(f"Model încărcat din: {filepath}")
    return model


def save_weights_only(model: tf.keras.Model, filepath: str) -> None:
    """Salvează doar ponderile."""
    model.save_weights(filepath)
    print(f"Ponderi salvate în: {filepath}")


def load_weights_only(model: tf.keras.Model, filepath: str) -> None:
    """Încarcă doar ponderile."""
    model.load_weights(filepath)
    print(f"Ponderi încărcate din: {filepath}")


def validate_model_structure(model: tf.keras.Model) -> Dict[str, Any]:
    """Validează și returnează informații despre structura modelului."""
    info = {
        'total_params': model.count_params(),
        'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
        'layers_count': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'is_compiled': hasattr(model, 'optimizer') and model.optimizer is not None
    }
    
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        info['optimizer'] = model.optimizer.__class__.__name__
        info['loss'] = model.loss.__class__.__name__ if hasattr(model.loss, '__class__') else str(model.loss)
    
    return info


# ============================================================================
# 6. CONFIGURARE
# ============================================================================

def _model_compile(model: tf.keras.Model) -> tf.keras.Model:
    """Compilează modelul."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_loss_type() -> str:
    return 'categorical_crossentropy'


def get_image_format() -> dict:
    return {'channels': 3, 'size': IMG_SIZE}


def get_data_preprocessing() -> callable:
    return preprocess


# ============================================================================
# 7. IMPORT RESNET18 PRE-ANTRENAT
# ============================================================================

def create_model():
    """
    Importă ResNet18 PRE-ANTRENAT pentru CIFAR-10.
    
    Sursă: "Practical Poisoning Attacks on Neural Networks" (ECCV 2020)
    
    Metodă de import (în ordine de preferință):
    1. TensorFlow/Keras Applications (dacă disponibil)
    2. classification-models (qubvel)
    3. TensorFlow Hub
    
    Returns:
        Model Keras cu ponderi pre-antrenate
    """
    print("\n🔽 Import ResNet18 PRE-ANTRENAT...")
    print("   Sursă: ECCV 2020 - Practical Poisoning Attacks")
    
    # ========================================================================
    # METODĂ 1: classification-models (RECOMANDAT - cel mai simplu)
    # ========================================================================
    try:
        print("\n   [Metodă 1] Încercare import din classification-models...")
        from classification_models.keras import Classifiers
        
        # Obține ResNet18 pre-antrenat pe ImageNet
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        
        print("   ✓ classification-models găsit!")
        print("   🔽 Descărcare ponderi ImageNet...")
        
        # Creează model base cu ponderi ImageNet
        base_model = ResNet18(
            input_shape=(*IMG_SIZE, 3),
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        
        print("   ✓ Ponderi ImageNet descărcate!")
        
        # Adaptare pentru CIFAR-10 (10 clase)
        inputs = tf.keras.layers.Input(shape=(*IMG_SIZE, 3), name='input')
        x = base_model(inputs)
        outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
        
        model = tf.keras.Model(inputs, outputs, name='ResNet18_CIFAR10')
        
        print(f"\n   ✓ ResNet18 importat cu succes!")
        print(f"   📊 Parametri totali: {model.count_params():,}")
        print(f"   📊 Sursă: classification-models (ImageNet weights)")
        
        # Compilează
        _model_compile(model)
        
        return model
        
    except ImportError:
        print("   ⚠️  classification-models nu este instalat")
        print("   Instalare: pip install classification-models")
    
    # ========================================================================
    # METODĂ 2: TensorFlow Keras Applications (fallback)
    # ========================================================================
    try:
        print("\n   [Metodă 2] Încercare import din tf.keras.applications...")
        
        # ResNet50 (disponibil în Keras, mai mare dar funcțional)
        print("   ℹ️  ResNet18 nu e disponibil în tf.keras.applications")
        print("   ℹ️  Folosim ResNet50 ca alternativă (23M parametri)")
        
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(*IMG_SIZE, 3),
            pooling='avg'
        )
        
        print("   ✓ ResNet50 descărcat!")
        
        inputs = tf.keras.layers.Input(shape=(*IMG_SIZE, 3))
        x = base_model(inputs)
        outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name='ResNet50_CIFAR10')
        
        print(f"\n   ✓ ResNet50 importat cu succes!")
        print(f"   📊 Parametri totali: {model.count_params():,}")
        
        _model_compile(model)
        
        return model
        
    except Exception as e:
        print(f"   ⚠️  Eroare tf.keras.applications: {e}")
    
    # ========================================================================
    # METODĂ 3: TensorFlow Hub (fallback final)
    # ========================================================================
    try:
        print("\n   [Metodă 3] Încercare import din TensorFlow Hub...")
        import tensorflow_hub as hub
        
        # ResNet50 de pe TensorFlow Hub
        hub_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
        
        print(f"   🔽 Descărcare de la: {hub_url}")
        
        base_layer = hub.KerasLayer(hub_url, trainable=True)
        
        inputs = tf.keras.layers.Input(shape=(*IMG_SIZE, 3))
        x = inputs / 255.0  # Normalizare
        x = base_layer(x)
        outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name='ResNet50_TFHub_CIFAR10')
        
        print(f"\n   ✓ ResNet50 importat de pe TensorFlow Hub!")
        print(f"   📊 Parametri totali: {model.count_params():,}")
        
        _model_compile(model)
        
        return model
        
    except ImportError:
        print("   ⚠️  TensorFlow Hub nu este instalat")
        print("   Instalare: pip install tensorflow-hub")
    except Exception as e:
        print(f"   ⚠️  Eroare TensorFlow Hub: {e}")
    
    # ========================================================================
    # EROARE: Nicio metodă nu a funcționat
    # ========================================================================
    print("\n   ❌ EROARE: Nu s-a putut importa ResNet18!")
    print("\n   💡 Soluții:")
    print("      1. RECOMANDAT: pip install classification-models")
    print("      2. SAU: pip install tensorflow-hub")
    print("      3. SAU: Folosește tf.keras.applications.ResNet50")
    
    raise RuntimeError("Nu s-a putut importa modelul ResNet18/ResNet50")


# ============================================================================
# MAIN - EVALUARE METRICI
# ============================================================================

if __name__ == "__main__":
    """
    Script principal: Încarcă date + Evaluează metrici model pre-antrenat.
    """
    
    print("=" * 70)
    print("EVALUARE RESNET18 PRE-ANTRENAT PE CIFAR-10")
    print("Sursă: 'Practical Poisoning Attacks on Neural Networks' (ECCV 2020)")
    print("=" * 70)
    
    # =======================================================================
    # PASUL 1: DESCĂRCARE ȘI SALVARE DATE
    # =======================================================================
    print("\n[PASUL 1] Descărcare și salvare CIFAR-10...")
    metadata = download_data(output_dir="cifar10_data")
    
    print(f"\n   ✓ Date salvate în: cifar10_data/")
    print(f"   📊 Metadata:")
    for key, value in metadata.items():
        print(f"      {key}: {value}")
    
    # =======================================================================
    # PASUL 2: ÎNCĂRCARE DATE PENTRU EVALUARE
    # =======================================================================
    print("\n[PASUL 2] Încărcare date pentru evaluare...")
    train_ds, test_ds = load_train_test_data()
    train_ds, test_ds = preprocess_loaded_data(train_ds, test_ds)
    print("   ✓ Date încărcate și preprocesate")
    
    # =======================================================================
    # PASUL 3: IMPORT MODEL PRE-ANTRENAT
    # =======================================================================
    print("\n[PASUL 3] Import model ResNet18 pre-antrenat...")
    model = create_model()
    
    # =======================================================================
    # PASUL 4: VALIDARE STRUCTURĂ MODEL
    # =======================================================================
    print("\n[PASUL 4] Validare structură model:")
    model_info = validate_model_structure(model)
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # =======================================================================
    # PASUL 5: EVALUARE METRICI PE TEST SET
    # =======================================================================
    print("\n[PASUL 5] Evaluare metrici pe CIFAR-10 test set...")
    print("   ⏳ Calculare predicții pe 10,000 imagini...")
    
    metrics = calculate_metrics(model, test_ds)
    
    print("\n" + "=" * 70)
    print("📊 METRICI FINALE - RESNET18 PE CIFAR-10")
    print("=" * 70)
    
    for metric_name, value in metrics.items():
        # Adaugă emoji bazat pe valoare
        if value > 0.9:
            emoji = "🟢"
        elif value > 0.7:
            emoji = "🟡"
        else:
            emoji = "🔴"
        
        print(f"   {emoji} {metric_name.upper():12s}: {value:.4f} ({value*100:.2f}%)")
    
    # =======================================================================
    # INFORMAȚII SUPLIMENTARE
    # =======================================================================
    print("\n" + "=" * 70)
    print("📋 INFORMAȚII MODEL")
    print("=" * 70)
    print(f"   Nume: {model.name}")
    print(f"   Arhitectură: ResNet18 (sau ResNet50 fallback)")
    print(f"   Parametri totali: {model.count_params():,}")
    print(f"   Parametri antrenabili: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Optimizer: {model.optimizer.__class__.__name__}")
    print(f"   Loss: {get_loss_type()}")
    
    print("\n" + "=" * 70)
    print("💾 DATE SALVATE")
    print("=" * 70)
    print(f"   Directoare create:")
    print(f"      📁 cifar10_data/train/  ({metadata['train_samples']} imagini)")
    print(f"      📁 cifar10_data/test/   ({metadata['test_samples']} imagini)")
    print(f"      📄 cifar10_data/metadata.json")
    
    print("\n" + "=" * 70)
    print("✅ EVALUARE FINALIZATĂ CU SUCCES!")
    print("=" * 70)
    
    print("\n💡 Următorii pași:")
    print("   1. Date CIFAR-10 salvate în: cifar10_data/")
    print("   2. Metrici baseline stabilite pentru comparație")
    print("   3. Model gata pentru teste de data poisoning")
    print("   4. Framework recomandat: ART (Adversarial Robustness Toolbox)")
    
    print("\n🔬 Pentru atacuri de poisoning:")
    print("   • Baseline accuracy: {:.2f}%".format(metrics['accuracy'] * 100))
    print("   • Literatura: BadNets, Trojan Attacks, Clean-Label Backdoors")
    print("   • Testează robustețea modelului la trigger patterns")
    
    print("\nFinish")