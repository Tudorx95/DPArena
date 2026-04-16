"""
Template pentru antrenarea rețelelor neuronale în medii Federated Learning
Compatibil cu PyTorch

Model: ResNet18 PRE-ANTRENAT pe CIFAR-100, descărcat de pe HuggingFace Hub
       (edadaltocg/resnet18_cifar100 — antrenat cu timm/PyTorch, acuratețe ~79%)
       Descărcat cu hf_hub_download ca fișier pytorch_model.bin (PyTorch pur state_dict)
       NU necesită timm la runtime — doar torchvision + huggingface_hub
Dataset: CIFAR-100 (60,000 imagini 32x32 RGB, 100 clase)
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
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
NUM_CLASSES = 100
IMG_SIZE = (32, 32)
HUGGINGFACE_REPO_ID = "edadaltocg/resnet18_cifar100"
MODEL_FILENAME = "pytorch_model.bin"

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# CUSTOM DATASET CLASS PENTRU FL
# ============================================================================
class CIFAR100Dataset(Dataset):
    """Custom Dataset pentru CIFAR-100 din directoare"""
    
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
    Încarcă CIFAR-100 dataset folosind torchvision.
    
    Modelul a fost antrenat cu normalizare CIFAR-100 standard:
      mean = [0.5071, 0.4865, 0.4409]
      std  = [0.2673, 0.2564, 0.2761]
    
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    # Transform cu normalizare identică cu antrenarea modelului
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409),
            (0.2673, 0.2564, 0.2761)
        ),
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=basic_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
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
    Preprocesare de bază pentru imagini și label-uri CIFAR-100.
    """
    if isinstance(label, int):
        label = torch.tensor(label)
    
    label_one_hot = torch.nn.functional.one_hot(label, num_classes=NUM_CLASSES)
    return image, label_one_hot


def preprocess_loaded_data(train_ds, test_ds) -> Tuple[DataLoader, DataLoader]:
    """
    Preprocesează dataset-urile încărcate.
    """
    return train_ds, test_ds


def load_client_data(data_path: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Funcție pentru încărcarea datelor în FL simulator.
    """
    data_path = Path(data_path)
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Data directories not found: {data_path}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409),
            (0.2673, 0.2564, 0.2761)
        ),
    ])
    
    train_dataset = CIFAR100Dataset(train_dir, transform=transform)
    test_dataset = CIFAR100Dataset(test_dir, transform=transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    
    return train_loader, test_loader


def download_data(output_dir: str):
    """
    Descarcă și salvează datele în format compatibil cu FL simulator.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    transform = transforms.ToTensor()
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )
    
    for split_name, dataset in [('train', train_dataset), ('test', test_dataset)]:
        split_dir = output_path / split_name
        
        for class_idx in range(NUM_CLASSES):
            class_dir = split_dir / str(class_idx)
            class_dir.mkdir(parents=True, exist_ok=True)
        
        image_counters = {i: 0 for i in range(NUM_CLASSES)}
        
        for idx in range(len(dataset)):
            image, label = dataset[idx]
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
def _create_resnet18_cifar100() -> nn.Module:
    """
    Creează arhitectura ResNet18 adaptată pentru imagini mici (32x32).
    """
    model = torchvision.models.resnet18(weights=None)
    
    # Modifică primul layer pentru CIFAR-100 (32x32 în loc de 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Elimină maxpool pentru imagini mici
    
    # Modifică ultimul layer pentru NUM_CLASSES = 100
    model.fc = nn.Linear(512, NUM_CLASSES)
    
    return model


def create_model() -> nn.Module:
    """
    Descarcă model pre-antrenat de pe HuggingFace Hub.
    """
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"Downloading pretrained model from HuggingFace: {HUGGINGFACE_REPO_ID}...")
        
        model_path = hf_hub_download(
            repo_id=HUGGINGFACE_REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir=".cache/huggingface"
        )
        
        model = _create_resnet18_cifar100()
        model.to(DEVICE)
        
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded from HuggingFace: {HUGGINGFACE_REPO_ID}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Expected test accuracy: ~79%")
        
        _model_compile(model)
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to download model from HuggingFace: {e}")


def _model_compile(model: nn.Module):
    """
    Compilează modelul cu loss și optimizer.
    """
    model.criterion = nn.CrossEntropyLoss()
    model.optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.is_compiled = True


# ============================================================================
# 3. FUNCȚII AUXILIARE
# ============================================================================
def get_loss_type() -> str:
    return 'categorical_crossentropy'

def get_image_format() -> Dict[str, Any]:
    return {
        'size': list(IMG_SIZE),
        'channels': 3
    }

def get_data_preprocessing():
    return preprocess

def validate_model_structure(model: nn.Module) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    layers_count = len(list(model.modules())) - 1
    
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
    model: nn.Module, train_data: DataLoader, epochs: int = 1, verbose: int = 0
) -> Dict[str, Any]:
    model.train()
    history = {'loss': [], 'accuracy': []}
    
    criterion = model.criterion if hasattr(model, 'criterion') else nn.CrossEntropyLoss()
    optimizer = model.optimizer if hasattr(model, 'optimizer') else optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_data):
            images = images.to(DEVICE)
            
            if len(labels.shape) > 1 and labels.shape[1] == NUM_CLASSES:
                labels = torch.argmax(labels, dim=1)
            
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
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
    model: nn.Module, test_dataset: DataLoader, average: str = 'macro'
) -> Dict[str, float]:
    model.eval()
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for images, labels in test_dataset:
            images = images.to(DEVICE)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
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
    weights = []
    for tensor in model.state_dict().values():
        weights.append(tensor.cpu().numpy())
    return weights

def set_model_weights(model: nn.Module, weights: List[np.ndarray]):
    state_dict = model.state_dict()
    with torch.no_grad():
        for (name, tensor), weight in zip(state_dict.items(), weights):
            if not isinstance(weight, np.ndarray):
                weight = np.array(weight)
            new_tensor = torch.from_numpy(weight).to(device=tensor.device, dtype=tensor.dtype)
            tensor.copy_(new_tensor)


# ============================================================================
# 6. FUNCȚII PENTRU SALVARE/ÎNCĂRCARE MODEL
# ============================================================================
def save_model_config(model: nn.Module, filepath: str, save_weights: bool = True) -> None:
    if not filepath.endswith('.pth'):
        filepath += '.pth'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': {
            'base': 'resnet18',
            'source': 'edadaltocg/resnet18_cifar100',
            'num_classes': NUM_CLASSES,
            'img_size': list(IMG_SIZE),
            'modifications': {
                'conv1': {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
                'maxpool': 'Identity',
                'fc': {'in_features': 512, 'out_features': NUM_CLASSES}
            }
        },
        'normalization': {
            'mean': [0.5071, 0.4865, 0.4409],
            'std': [0.2673, 0.2564, 0.2761]
        },
        'classes': CIFAR100_CLASSES,
        'optimizer_state_dict': model.optimizer.state_dict() if hasattr(model, 'optimizer') else None,
    }, filepath)

def load_model_config(filepath: str) -> nn.Module:
    if not filepath.endswith('.pth'):
        filepath += '.pth'
    
    checkpoint = torch.load(filepath, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and 'architecture' in checkpoint:
        arch = checkpoint['architecture']
        model = torchvision.models.resnet18(weights=None)
        mod = arch['modifications']
        model.conv1 = nn.Conv2d(3, 64, **mod['conv1'])
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(mod['fc']['in_features'], mod['fc']['out_features'])
    else:
        model = _create_resnet18_cifar100()
    
    model.to(DEVICE)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    elif isinstance(checkpoint, nn.Module):
        model.load_state_dict(checkpoint.state_dict())
    
    _model_compile(model)
    
    if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        if checkpoint['optimizer_state_dict'] is not None and hasattr(model, 'optimizer'):
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model

def save_weights_only(model: nn.Module, filepath: str) -> None:
    if filepath.endswith('.weights.h5'):
        filepath = filepath.replace('.weights.h5', '.weights.pth')
    torch.save(model.state_dict(), filepath)

def load_weights_only(model: nn.Module, filepath: str) -> nn.Module:
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
    history = train_neural_network(model, train_ds, epochs=5, verbose=1)
    
    # STEP 7: Evaluare după antrenare
    final_metrics = calculate_metrics(model, test_ds)
    
    # STEP 8: Salvare model
    save_model_config(model, "ResNet18_CIFAR100_edadaltocg.pth")
