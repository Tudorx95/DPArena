"""
Template pentru antrenarea rețelelor neuronale în medii Federated Learning
Compatibil cu PyTorch

Model: ResNet18 PRE-ANTRENAT descărcat de pe HuggingFace Hub
Dataset: CIFAR-10 (60,000 imagini 32x32 RGB, 10 clase)
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

# ============================================================================
# CONFIGURAȚIE GLOBALĂ
# ============================================================================
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
NUM_CLASSES = 10
IMG_SIZE = (32, 32)
HUGGINGFACE_REPO_ID = "Tudorx95/resnet18-cifar10-pytorch"
MODEL_FILENAME = "ResNet18_CIFAR10.pth"

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# CUSTOM DATASET CLASS PENTRU FL
# ============================================================================
class CIFAR10Dataset(Dataset):
    """Custom Dataset pentru CIFAR-10 din directoare"""
    
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Colectează toate path-urile către imagini și label-urile lor
        for class_idx in range(NUM_CLASSES):
            class_dir = root_dir / str(class_idx)
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
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
    Încarcă CIFAR-10 dataset folosind torchvision.
    
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    # Transform cu normalizare CIFAR-10 (aceleași valori ca în antrenare)
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=basic_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=basic_transform
    )
    
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


def preprocess(image, label):
    """
    Preprocesare de bază pentru imagini și label-uri CIFAR-10.
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
    
    # Transform pentru preprocesare (cu normalizare CIFAR-10 identică cu antrenarea)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convertește la tensor și normalizează la [0, 1]
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Creează dataset-uri custom
    train_dataset = CIFAR10Dataset(train_dir, transform=transform)
    test_dataset = CIFAR10Dataset(test_dir, transform=transform)
    
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
    
    Args:
        output_dir: Director unde se vor salva datele
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Încarcă dataset-urile originale
    transform = transforms.ToTensor()
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Salvează în format director
    for split_name, dataset in [('train', train_dataset), ('test', test_dataset)]:
        split_dir = output_path / split_name
        
        # Creează directoare pentru clase
        for class_idx in range(NUM_CLASSES):
            class_dir = split_dir / str(class_idx)
            class_dir.mkdir(parents=True, exist_ok=True)
        
        image_counters = {i: 0 for i in range(NUM_CLASSES)}
        
        # Salvează imaginile
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            
            # Convertește tensor la numpy și apoi la PIL Image
            image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            class_dir = split_dir / str(label)
            image_path = class_dir / f"img_{image_counters[label]:05d}.png"
            
            pil_image.save(image_path)
            image_counters[label] += 1
            
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(dataset)} images in {split_name}")


# ============================================================================
# 2. FUNCȚIE PENTRU CREARE/DESCĂRCARE MODEL
# ============================================================================
def _create_resnet18_cifar10() -> nn.Module:
    """
    Creează arhitectura ResNet18 adaptată pentru CIFAR-10 (32x32).
    NU încarcă ponderi pre-antrenate — doar arhitectura.
    
    Returns:
        nn.Module: Model cu ponderi random
    """
    model = torchvision.models.resnet18(weights=None)
    
    # Modifică primul layer pentru CIFAR-10 (32x32 în loc de 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Elimină maxpool pentru imagini mici
    
    # Modifică ultimul layer pentru NUM_CLASSES
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model


def create_model() -> nn.Module:
    """
    Descarcă model pre-antrenat de pe HuggingFace Hub.
    
    Checkpoint-ul conține toată configurația:
    - model_state_dict: ponderile rețelei
    - architecture: info despre arhitectură
    - normalization: parametri de normalizare
    - classes: lista de clase
    - metrics: metricile de antrenare
    
    Returns:
        nn.Module: Model compilat cu ponderi pre-antrenate
    """
    try:
        from huggingface_hub import hf_hub_download
        
        model_path = hf_hub_download(
            repo_id=HUGGINGFACE_REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir=".cache/huggingface"
        )
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Reconstruiește arhitectura din checkpoint
        if isinstance(checkpoint, dict) and 'architecture' in checkpoint:
            arch = checkpoint['architecture']
            model = torchvision.models.resnet18(weights=None)
            mod = arch['modifications']
            model.conv1 = nn.Conv2d(3, 64, **mod['conv1'])
            model.maxpool = nn.Identity()
            model.fc = nn.Linear(mod['fc']['in_features'], mod['fc']['out_features'])
            model.to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from HuggingFace: {HUGGINGFACE_REPO_ID}")
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Format checkpoint dict fără architecture (compatibilitate)
            model = _create_resnet18_cifar10()
            model.to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from HuggingFace: {HUGGINGFACE_REPO_ID}")
        elif isinstance(checkpoint, nn.Module):
            # Format torch.save(model, path) (compatibilitate veche)
            model = _create_resnet18_cifar10()
            model.to(DEVICE)
            model.load_state_dict(checkpoint.state_dict())
            print(f"Model loaded from HuggingFace: {HUGGINGFACE_REPO_ID}")
        else:
            # Format state_dict direct
            model = _create_resnet18_cifar10()
            model.to(DEVICE)
            model.load_state_dict(checkpoint)
            print(f"Model loaded from HuggingFace: {HUGGINGFACE_REPO_ID}")
        
        _model_compile(model)
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to download model from HuggingFace: {e}")


def _model_compile(model: nn.Module):
    """
    Compilează modelul cu loss și optimizer.
    În PyTorch, acest lucru înseamnă să atașăm aceste obiecte la model.
    
    Args:
        model: Modelul de compilat
    """
    # Atașăm loss function și optimizer ca atribute
    model.criterion = nn.CrossEntropyLoss()
    model.optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
    # Pentru ResNet18 pe CIFAR-10
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
    optimizer = model.optimizer if hasattr(model, 'optimizer') else optim.Adam(model.parameters(), lr=0.001)
    
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
    Extrage ponderile modelului.
    
    Args:
        model: Modelul din care se extrag ponderile
    
    Returns:
        List cu toate ponderile (numpy arrays)
    """
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().numpy())
    
    return weights


def set_model_weights(model: nn.Module, weights: List[np.ndarray]):
    """
    Setează ponderile modelului.
    
    Args:
        model: Modelul în care se setează ponderile
        weights: Lista cu ponderi (numpy arrays)
    """
    with torch.no_grad():
        for param, weight in zip(model.parameters(), weights):
            param.data = torch.from_numpy(weight).to(param.device)


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
            'base': 'resnet18',
            'num_classes': NUM_CLASSES,
            'img_size': list(IMG_SIZE),
            'modifications': {
                'conv1': {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
                'maxpool': 'Identity',
                'fc': {'in_features': 512, 'out_features': NUM_CLASSES}
            }
        },
        'normalization': {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010]
        },
        'classes': CIFAR10_CLASSES,
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
    
    # Reconstruiește arhitectura din checkpoint
    if isinstance(checkpoint, dict) and 'architecture' in checkpoint:
        arch = checkpoint['architecture']
        model = torchvision.models.resnet18(weights=None)
        mod = arch['modifications']
        model.conv1 = nn.Conv2d(3, 64, **mod['conv1'])
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(mod['fc']['in_features'], mod['fc']['out_features'])
    else:
        # Fallback la arhitectura hardcodată
        model = _create_resnet18_cifar10()
    
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
    save_model_config(model, "ResNet18_CIFAR10.pth")