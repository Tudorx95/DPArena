"""
Template pentru antrenarea rețelelor neuronale în medii Federated Learning
Compatibil cu PyTorch

Model: ViT-Base/16 FINE-TUNAT pe chest X-ray pneumonia classification
       Repo: lxyuan/vit-xray-pneumonia-classification (HuggingFace, public)
       Arhitectură: google/vit-base-patch16-224-in21k + cap Linear(768, 2)
       Acuratețe raportată: 97.42% pe validation set
       Dataset antrenare: keremberke/chest-xray-classification (binary)

       MOTIVUL ALEGERII: modelul este deja fine-tunat pe sarcina de pneumonia
       binary classification — acuratețea inițială este stabilă și mare la
       fiecare rulare (fără reinițializare random). Label mapping nativ:
       0=NORMAL, 1=PNEUMONIA — identic cu PneumoniaMNIST.

Dataset: PneumoniaMNIST (5,856 radiografii toracice 28×28 grayscale, 2 clase)
         Sursa: MedMNIST v2 (Yang et al., 2023)
         Descărcat automat de pe Zenodo ca fișier .npz

SCOP: Detectarea pneumoniei din radiografii toracice pediatrice
      folosind transfer learning cu ViT în context Federated Learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any, List
from pathlib import Path
from PIL import Image
import json
import os

# ============================================================================
# CONFIGURAȚIE GLOBALĂ
# ============================================================================
PNEUMONIA_CLASSES = ['NORMAL', 'PNEUMONIA']
NUM_CLASSES = 2
IMG_SIZE = (224, 224)

# PneumoniaMNIST dataset URL (MedMNIST v2 — Zenodo)
MEDMNIST_URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz"
MEDMNIST_FILENAME = "pneumoniamnist.npz"

# HuggingFace — ViT fine-tunat public pe chest X-ray pneumonia binary
HF_REPO_ID = "lxyuan/vit-xray-pneumonia-classification"

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# CUSTOM DATASET CLASS PENTRU FL
# ============================================================================
class PneumoniaDataset(Dataset):
    """Custom Dataset pentru radiografii toracice din directoare"""
    
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Colectează toate path-urile către imagini și label-urile lor
        for class_idx in range(NUM_CLASSES):
            class_dir = root_dir / str(class_idx)
            if class_dir.exists():
                for img_path in sorted(class_dir.glob('*.png')):
                    self.samples.append((str(img_path), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# 1. FUNCȚII PENTRU EXTRAGEREA DATELOR
# ============================================================================
def load_train_test_data() -> Tuple[DataLoader, DataLoader]:
    """
    Încarcă PneumoniaMNIST dataset.
    
    Descarcă automat datele de pe Zenodo (MedMNIST v2) dacă nu sunt deja
    prezente. Imaginile originale sunt 28×28 grayscale, convertite la RGB
    și redimensionate la 224×224 pentru DenseNet121.
    
    Normalizare ImageNet (DenseNet121 a fost antrenat pe aceste valori):
      mean = [0.485, 0.456, 0.406]
      std  = [0.229, 0.224, 0.225]
    
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    # Descarcă datele dacă nu sunt deja prezente
    data_dir = Path('./data/pneumoniamnist')
    npz_path = data_dir / MEDMNIST_FILENAME
    
    if not npz_path.exists():
        print(f"Downloading PneumoniaMNIST from Zenodo...")
        data_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(MEDMNIST_URL, str(npz_path))
        print(f"✓ Download complete: {npz_path}")
    
    # Încarcă datele din .npz
    data = np.load(str(npz_path))
    x_train = data['train_images']   # (N, 28, 28)
    y_train = data['train_labels'].flatten()  # (N,)
    x_test = data['test_images']
    y_test = data['test_labels'].flatten()
    
    # Transform cu redimensionare la 224×224 + normalizare ImageNet
    basic_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.Grayscale(num_output_channels=3),  # 1ch → 3ch RGB
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
    ])
    
    # Creează dataset-uri custom din numpy arrays
    train_dataset = _NumpyImageDataset(x_train, y_train, transform=basic_transform)
    test_dataset = _NumpyImageDataset(x_test, y_test, transform=basic_transform)
    
    # Creează DataLoader-e
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, test_loader


class _NumpyImageDataset(Dataset):
    """Dataset intern pentru imagini din numpy arrays (MedMNIST format)"""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        
        # Convertește numpy array la PIL Image
        if image.ndim == 2:
            # Grayscale (28, 28)
            pil_image = Image.fromarray(image, mode='L')
        else:
            # RGB (28, 28, 3)
            pil_image = Image.fromarray(image.astype(np.uint8))
        
        if self.transform:
            pil_image = self.transform(pil_image)
        
        return pil_image, label


def preprocess(image, label):
    """
    Preprocesare de bază pentru imagini și label-uri PneumoniaMNIST.
    Această funcție este pentru compatibilitate - preprocesarea reală
    se face în transforms.
    
    Args:
        image: Imaginea (tensor)
        label: Label-ul
    
    Returns:
        Tuple: (image, label_one_hot)
    """
    # Imaginea e deja normalizată în transform
    # Convertim label la one-hot
    if isinstance(label, int):
        label = torch.tensor(label)
    
    label_one_hot = torch.nn.functional.one_hot(label, num_classes=NUM_CLASSES)
    
    return image, label_one_hot


def preprocess_loaded_data(train_ds, test_ds) -> Tuple[DataLoader, DataLoader]:
    """
    Preprocesează dataset-urile încărcate.
    În PyTorch, preprocesarea e deja făcută în transforms.
    
    Args:
        train_ds: DataLoader de antrenare
        test_ds: DataLoader de testare
    
    Returns:
        Tuple de DataLoader-e preprocesate
    """
    # Dataset-urile sunt deja preprocesate în load_train_test_data
    # Returnăm direct pentru compatibilitate
    return train_ds, test_ds


def load_client_data(data_path: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Funcție pentru încărcarea datelor în FL simulator.
    
    Args:
        data_path: Path către directorul cu date
        batch_size: Dimensiunea batch-ului
        
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    data_path = Path(data_path)
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Data directories not found: {data_path}")
    
    # Transform pentru preprocesare (cu normalizare identică cu antrenarea)
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
    ])
    
    # Creează dataset-uri custom
    train_dataset = PneumoniaDataset(train_dir, transform=transform)
    test_dataset = PneumoniaDataset(test_dir, transform=transform)
    
    # Creează DataLoader-e
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, test_loader


def download_data(output_dir: str):
    """
    Descarcă și salvează datele în format compatibil cu FL simulator.
    
    Descarcă PneumoniaMNIST de pe Zenodo și salvează imaginile ca fișiere
    PNG organizate pe clase: output_dir/{train,test}/{0,1}/img_XXXXX.png
    
    Args:
        output_dir: Director unde se vor salva datele
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Descarcă datele dacă nu sunt deja prezente
    cache_dir = Path('./data/pneumoniamnist')
    npz_path = cache_dir / MEDMNIST_FILENAME
    
    if not npz_path.exists():
        print(f"Downloading PneumoniaMNIST from Zenodo...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(MEDMNIST_URL, str(npz_path))
        print(f"✓ Download complete")
    
    # Încarcă datele din .npz
    data = np.load(str(npz_path))
    
    splits = {
        'train': (data['train_images'], data['train_labels'].flatten()),
        'test': (data['test_images'], data['test_labels'].flatten())
    }
    
    # Salvează în format director
    for split_name, (images, labels) in splits.items():
        split_dir = output_path / split_name
        
        # Creează directoare pentru clase
        for class_idx in range(NUM_CLASSES):
            class_dir = split_dir / str(class_idx)
            class_dir.mkdir(parents=True, exist_ok=True)
        
        image_counters = {i: 0 for i in range(NUM_CLASSES)}
        
        # Salvează imaginile
        for idx in range(len(images)):
            image = images[idx]
            label = int(labels[idx])
            
            # Convertește la PIL Image
            if image.ndim == 2:
                pil_image = Image.fromarray(image, mode='L')
            else:
                pil_image = Image.fromarray(image.astype(np.uint8))
            
            class_dir = split_dir / str(label)
            image_path = class_dir / f"img_{image_counters[label]:05d}.png"
            
            pil_image.save(image_path)
            image_counters[label] += 1
            
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(images)} images in {split_name}")
    
    # Print statistici
    for split_name, (images, labels) in splits.items():
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n{split_name}: {len(images)} images")
        for cls_idx, cnt in zip(unique, counts):
            print(f"  Class {cls_idx} ({PNEUMONIA_CLASSES[cls_idx]}): {cnt} images")


# ============================================================================
# 2. FUNCȚIE PENTRU CREARE/DESCĂRCARE MODEL
# ============================================================================
def _install_transformers():
    """
    Instalează transformers + huggingface_hub dacă lipsesc (import lazy).
    Apelat doar din create_model(), nu la top-level.
    """
    try:
        import transformers  # noqa: F401
    except ImportError:
        import subprocess, sys
        print("Installing transformers...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "transformers", "-q"])
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        import subprocess, sys
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "huggingface_hub", "-q"])


def _create_densenet121_pneumonia() -> nn.Module:
    """
    Construiește arhitectura ViT-Base/16 cu cap binar (2 clase), fără ponderi
    pre-antrenate. Folosit intern ca schelet pentru load_state_dict în
    load_model_config / load_weights_only.

    Notă: numele funcției este păstrat din template-ul original pentru
    compatibilitate cu restul pipeline-ului; arhitectura subiacentă este ViT
    (backbone înlocuit de la DenseNet121 pentru stabilitate mai mare a
    acurateței inițiale).

    Returns:
        nn.Module: ViTForImageClassification cu 2 clase
    """
    _install_transformers()
    from transformers import ViTForImageClassification, ViTConfig

    config = ViTConfig.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=NUM_CLASSES,
        id2label={0: "NORMAL", 1: "PNEUMONIA"},
        label2id={"NORMAL": 0, "PNEUMONIA": 1},
    )
    model = ViTForImageClassification(config)
    return model


def create_model() -> nn.Module:
    """
    Descarcă ViT fine-tunat pe chest X-ray pneumonia de pe HuggingFace.

    Sursa: lxyuan/vit-xray-pneumonia-classification (public, PyTorch/safetensors)
    - Base: google/vit-base-patch16-224-in21k
    - Fine-tunat pe keremberke/chest-xray-classification (binary)
    - Acuratețe raportată: 97.42% pe validation set
    - Label mapping: 0=NORMAL, 1=PNEUMONIA (identic cu PneumoniaMNIST)
    - Ponderi STABILE la fiecare rulare (fără reinițializare random)

    Wrapper: ViT-ul HF returnează ImageClassifierOutput. Îl împachetăm
    într-un nn.Module care returnează direct logits (tensor), pentru
    compatibilitate cu restul pipeline-ului (train_neural_network,
    calculate_metrics) care așteaptă output.shape = (B, NUM_CLASSES).

    Returns:
        nn.Module: Model compilat, ponderi stabile, interfață standard
    """
    try:
        _install_transformers()
        from transformers import ViTForImageClassification

        print(f"Loading ViT fine-tunat pe chest X-ray din HuggingFace...")
        print(f"  Repo: {HF_REPO_ID}")

        vit = ViTForImageClassification.from_pretrained(HF_REPO_ID)

        # Wrapper transparent: output = tensor de logits (B, 2)
        class _ViTLogitsWrapper(nn.Module):
            def __init__(self, vit_model):
                super().__init__()
                self.vit = vit_model

            def forward(self, pixel_values):
                # Accept și input standard (B, 3, H, W)
                out = self.vit(pixel_values=pixel_values)
                return out.logits

        model = _ViTLogitsWrapper(vit)
        model.to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ ViT-xray-pneumonia loaded from HuggingFace")
        print(f"  Architecture: ViT-Base/16 + Linear(768, {NUM_CLASSES})")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Classes: {PNEUMONIA_CLASSES}  (0=NORMAL, 1=PNEUMONIA)")
        print(f"  Reported accuracy: 97.42% on chest-xray-classification")

        _model_compile(model)
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to create ViT pneumonia model: {e}")


def _model_compile(model: nn.Module):
    """
    Compilează modelul cu loss și optimizer.
    În PyTorch, acest lucru înseamnă să atașăm aceste obiecte la model.
    
    Args:
        model: Modelul de compilat
    """
    # Atașăm loss function și optimizer ca atribute
    model.criterion = nn.CrossEntropyLoss()
    # LR mic (1e-4) pentru fine-tuning pe ponderi ImageNet pre-antrenate
    # Previne distrugerea feature-urilor extrase de layers-urile convoluționale
    model.optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Flag pentru a ști că modelul e "compilat"
    model.is_compiled = True


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

def validate_model_structure(model: nn.Module) -> Dict[str, Any]:
    """
    Validează structura modelului și returnează informații detaliate.
    
    Args:
        model: Modelul de validat
    
    Returns:
        Dict cu informații despre model
    """
    # Calculează numărul de parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Contorizează layere-le
    layers_count = len(list(model.modules())) - 1  # -1 pentru containerul principal
    
    # Informații despre input/output shape
    input_shape = f"(None, 3, {IMG_SIZE[0]}, {IMG_SIZE[1]})"
    output_shape = f"(None, {NUM_CLASSES})"
    
    model_info = {
        'model_name': model.__class__.__name__,
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'non_trainable_params': int(non_trainable_params),
        'layers_count': layers_count,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'is_compiled': hasattr(model, 'is_compiled') and model.is_compiled
    }
    
    return model_info


# ============================================================================
# 4. FUNCȚII PENTRU ANTRENARE
# ============================================================================
def train_neural_network(
    model: nn.Module,
    train_data: DataLoader,
    epochs: int = 1,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Antrenează modelul pe datele furnizate.
    
    Args:
        model: Modelul de antrenat
        train_data: DataLoader de antrenare
        epochs: Număr de epoci
        verbose: Nivel de verbozitate (0, 1, 2)
    
    Returns:
        Dict cu istoricul antrenării
    """
    model.train()
    
    history = {
        'loss': [],
        'accuracy': []
    }
    
    criterion = model.criterion if hasattr(model, 'criterion') else nn.CrossEntropyLoss()
    optimizer = model.optimizer if hasattr(model, 'optimizer') else optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_data):
            # Move data to device
            images = images.to(DEVICE)
            
            # Convert one-hot to class indices if needed
            if len(labels.shape) > 1 and labels.shape[1] == NUM_CLASSES:
                labels = torch.argmax(labels, dim=1)
            
            labels = labels.to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if verbose >= 2 and batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_data)}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_data)
        epoch_acc = correct / total
        
        history['loss'].append(float(epoch_loss))
        history['accuracy'].append(float(epoch_acc))
        
        if verbose >= 1:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return history


def calculate_metrics(
    model: nn.Module,
    test_dataset: DataLoader,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Calculează metricile de evaluare pe dataset de test.
    
    Args:
        model: Modelul evaluat
        test_dataset: DataLoader de test
        average: Tip de medie pentru precision/recall/f1
    
    Returns:
        Dict cu metrici
    """
    model.eval()
    
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for images, labels in test_dataset:
            images = images.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Convert one-hot to class indices if needed
            if len(labels.shape) > 1 and labels.shape[1] == NUM_CLASSES:
                labels = torch.argmax(labels, dim=1)
            
            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(predicted.cpu().numpy())
    
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
def get_model_weights(model: nn.Module) -> List[np.ndarray]:
    """
    Extrage ponderile și bufferele modelului (inclusiv BatchNorm running_mean/var).
    
    Args:
        model: Modelul din care se extrag ponderile
    
    Returns:
        List cu toate ponderile (numpy arrays)
    """
    weights = []
    for tensor in model.state_dict().values():
        weights.append(tensor.cpu().numpy())
    
    return weights


def set_model_weights(model: nn.Module, weights: List[np.ndarray]):
    """
    Setează ponderile și bufferele modelului.
    
    Args:
        model: Modelul în care se setează ponderile
        weights: Lista cu ponderi (numpy arrays)
    """
    state_dict = model.state_dict()
    with torch.no_grad():
        for (name, tensor), weight in zip(state_dict.items(), weights):
            # np.mean/np.sort pot produce scalare numpy (ex: num_batches_tracked)
            # torch.from_numpy acceptă doar np.ndarray, nu numpy scalare
            if not isinstance(weight, np.ndarray):
                weight = np.array(weight)
            new_tensor = torch.from_numpy(weight).to(device=tensor.device, dtype=tensor.dtype)
            tensor.copy_(new_tensor)


# ============================================================================
# 6. FUNCȚII PENTRU SALVARE/ÎNCĂRCARE MODEL
# ============================================================================
def save_model_config(
    model: nn.Module,
    filepath: str,
    save_weights: bool = True
) -> None:
    """
    Salvează configurația completă a modelului într-un singur fișier .pth.
    Include: state_dict + arhitectură + normalizare.
    
    Args:
        model: Modelul de salvat
        filepath: Path unde se salvează modelul
        save_weights: Dacă True, salvează și ponderile
    """
    if not filepath.endswith('.pth'):
        filepath += '.pth'
    
    # Salvează checkpoint complet cu toată configurația
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': {
            'base': 'vit-base-patch16-224',
            'source': HF_REPO_ID,
            'num_classes': NUM_CLASSES,
            'img_size': list(IMG_SIZE),
            'modifications': {
                'classifier': {'in_features': 768, 'out_features': NUM_CLASSES}
            }
        },
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'classes': PNEUMONIA_CLASSES,
        'optimizer_state_dict': model.optimizer.state_dict() if hasattr(model, 'optimizer') else None,
    }, filepath)


def load_model_config(filepath: str) -> nn.Module:
    """
    Încarcă configurația completă a modelului dintr-un fișier .pth.
    
    Args:
        filepath: Path către fișierul cu model
    
    Returns:
        nn.Module: Model încărcat
    """
    if not filepath.endswith('.pth'):
        filepath += '.pth'
    
    checkpoint = torch.load(filepath, map_location=DEVICE)
    
    # Reconstruiește arhitectura din checkpoint (schelet ViT cu wrapper)
    base = _create_densenet121_pneumonia()

    class _ViTLogitsWrapper(nn.Module):
        def __init__(self, vit_model):
            super().__init__()
            self.vit = vit_model
        def forward(self, pixel_values):
            return self.vit(pixel_values=pixel_values).logits

    model = _ViTLogitsWrapper(base)
    model.to(DEVICE)
    
    # Încarcă ponderile
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    elif isinstance(checkpoint, nn.Module):
        model.load_state_dict(checkpoint.state_dict())
    
    # Compilează modelul
    _model_compile(model)
    
    # Încarcă optimizer state dacă există
    if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        if checkpoint['optimizer_state_dict'] is not None and hasattr(model, 'optimizer'):
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model


def save_weights_only(model: nn.Module, filepath: str) -> None:
    """
    Salvează doar ponderile modelului.
    
    Args:
        model: Modelul din care se salvează ponderile
        filepath: Path unde se salvează ponderile
    """
    if filepath.endswith('.weights.h5'):
        # Păstrăm extensia .weights.pth pentru PyTorch
        filepath = filepath.replace('.weights.h5', '.weights.pth')
    
    torch.save(model.state_dict(), filepath)


def load_weights_only(model: nn.Module, filepath: str) -> nn.Module:
    """
    Încarcă doar ponderile în model.
    
    Args:
        model: Modelul în care se încarcă ponderile
        filepath: Path către fișierul cu ponderi
    
    Returns:
        nn.Module: Model cu ponderile încărcate
    """
    if filepath.endswith('.weights.h5'):
        filepath = filepath.replace('.weights.h5', '.weights.pth')
    
    model.load_state_dict(torch.load(filepath, map_location=DEVICE))
    
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
    save_model_config(model, "DenseNet121_Pneumonia.pth")