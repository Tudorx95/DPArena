"""
Script PyTorch pentru antrenarea unui model pe MNIST și upload pe HuggingFace Hub
Dataset: MNIST (70,000 imagini 28x28 grayscale, 10 cifre)
Model: CNN simplu optimizat pentru MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from huggingface_hub import HfApi, login
import os
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# CONFIGURARE HUGGINGFACE
# ============================================================================
HUGGINGFACE_REPO_ID = "Tudorx95/mnist-cnn-model"  # ✏️ ÎNLOCUIEȘTE CU REPO-UL TĂU
MODEL_FILENAME = "MNIST_CNN.pth"  # ✏️ ÎNLOCUIEȘTE CU NUMELE FIȘIERULUI TĂU
HUGGINGFACE_TOKEN = None  # Se va cere interactiv sau din environment

# ============================================================================
# CONFIGURAȚIE MODEL ȘI TRAINING
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10
IMG_SIZE = (28, 28)
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

print(f"🔧 Using device: {DEVICE}")
print(f"📊 Configuration:")
print(f"   - Batch size: {BATCH_SIZE}")
print(f"   - Learning rate: {LEARNING_RATE}")
print(f"   - Epochs: {NUM_EPOCHS}")

# ============================================================================
# DEFINIRE MODEL CNN PENTRU MNIST
# ============================================================================
class MNISTNet(nn.Module):
    """
    CNN optimizat pentru MNIST
    Arhitectură: Conv -> Conv -> FC -> FC
    """
    
    def __init__(self, num_classes=10):
        super(MNISTNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # După 3 pooling layers: 28 -> 14 -> 7 -> 3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # 28 -> 14
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # 14 -> 7
        x = self.dropout1(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # 7 -> 3
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

# ============================================================================
# FUNCȚII AUXILIARE
# ============================================================================
def get_model_info(model):
    """Returnează informații despre model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'model_name': model.__class__.__name__,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }

def calculate_accuracy(outputs, labels):
    """Calculează accuracy-ul"""
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    return correct

# ============================================================================
# ÎNCĂRCARE DATE MNIST
# ============================================================================
def load_mnist_data():
    """Încarcă și preprocesează MNIST dataset"""
    
    print("\n📥 Loading MNIST dataset...")
    
    # Transformări pentru date
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean și std pentru MNIST
    ])
    
    # Download și încărcare dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Creează DataLoader-e
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"   ✓ Training samples: {len(train_dataset):,}")
    print(f"   ✓ Test samples: {len(test_dataset):,}")
    print(f"   ✓ Training batches: {len(train_loader)}")
    print(f"   ✓ Test batches: {len(test_loader)}")
    
    return train_loader, test_loader

# ============================================================================
# ANTRENARE MODEL
# ============================================================================
def train_model(model, train_loader, test_loader, epochs=NUM_EPOCHS):
    """
    Antrenează modelul
    
    Args:
        model: Modelul de antrenat
        train_loader: DataLoader pentru training
        test_loader: DataLoader pentru testing
        epochs: Număr de epoci
    
    Returns:
        Dict cu istoricul antrenării
    """
    
    print("\n🚀 Starting training...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler pentru learning rate decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # ==================== TRAINING ====================
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
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
            correct += calculate_accuracy(outputs, labels)
            total += labels.size(0)
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f'   Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%')
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # ==================== TESTING ====================
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                correct += calculate_accuracy(outputs, labels)
                total += labels.size(0)
        
        test_loss = test_loss / len(test_loader)
        test_acc = correct / total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Step scheduler
        scheduler.step()
        
        # Print epoch summary
        print(f'\n📊 Epoch [{epoch+1}/{epochs}] Summary:')
        print(f'   Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
        print(f'   Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%')
        print(f'   Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            print(f'   ⭐ New best accuracy: {best_acc*100:.2f}%')
    
    print(f'\n✅ Training complete!')
    print(f'   Best Test Accuracy: {best_acc*100:.2f}%')
    
    return history

# ============================================================================
# EVALUARE FINALĂ
# ============================================================================
def evaluate_model(model, test_loader):
    """Evaluare finală cu metrici detaliate"""
    
    print("\n📈 Final Evaluation...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import numpy as np
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, average='macro', zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, average='macro', zero_division=0)),
        'f1_score': float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    }
    
    print(f"\n📊 Final Metrics:")
    for metric_name, value in metrics.items():
        print(f"   • {metric_name}: {value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n🔢 Confusion Matrix (preview):")
    print(cm[:5, :5])  # Print doar primele 5x5
    
    return metrics

# ============================================================================
# SALVARE MODEL LOCAL
# ============================================================================
def save_model_locally(model, history, metrics, save_dir="./trained_model"):
    """Salvează modelul și metadata local"""
    
    print(f"\n💾 Saving model locally...")
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Salvează modelul
    model_path = save_path / MODEL_FILENAME
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_architecture': str(model),
    }, model_path)
    print(f"   ✓ Model saved: {model_path}")
    
    # Salvează metadata
    model_info = get_model_info(model)
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'framework': 'pytorch',
        'dataset': 'MNIST',
        'model_info': model_info,
        'training_config': {
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'device': str(DEVICE)
        },
        'final_metrics': metrics,
        'history': history
    }
    
    metadata_path = save_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Metadata saved: {metadata_path}")
    
    # Creează README.md
    readme_content = f"""# MNIST CNN Model

## Model Information
- **Framework**: PyTorch
- **Dataset**: MNIST (28x28 grayscale images)
- **Architecture**: {model_info['model_name']}
- **Total Parameters**: {model_info['total_params']:,}
- **Trainable Parameters**: {model_info['trainable_params']:,}

## Training Configuration
- **Epochs**: {NUM_EPOCHS}
- **Batch Size**: {BATCH_SIZE}
- **Learning Rate**: {LEARNING_RATE}
- **Optimizer**: Adam
- **Device**: {DEVICE}

## Performance Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1 Score**: {metrics['f1_score']:.4f}

## Usage
```python
import torch
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="{HUGGINGFACE_REPO_ID}",
    filename="{MODEL_FILENAME}"
)

# Load model
checkpoint = torch.load(model_path)
# Create your model instance and load state dict
# model.load_state_dict(checkpoint['model_state_dict'])
```

## Training Date
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    readme_path = save_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"   ✓ README saved: {readme_path}")
    
    return save_path

# ============================================================================
# UPLOAD PE HUGGINGFACE
# ============================================================================
def upload_to_huggingface(save_dir, repo_id=HUGGINGFACE_REPO_ID, token=HUGGINGFACE_TOKEN):
    """
    Upload model pe HuggingFace Hub
    
    Args:
        save_dir: Director local cu fișierele de upload
        repo_id: ID-ul repo-ului pe HuggingFace (format: username/repo-name)
        token: Token de autentificare HuggingFace
    """
    
    print(f"\n🚀 Uploading to HuggingFace Hub...")
    print(f"   Repository: {repo_id}")
    
    try:
        # Login
        if token is None:
            print("\n🔐 HuggingFace Login Required")
            print("   Please enter your HuggingFace token (or set HUGGINGFACE_TOKEN env variable)")
            print("   Get your token from: https://huggingface.co/settings/tokens")
            token = input("   Token: ").strip()
        
        if not token:
            token = os.environ.get('HUGGINGFACE_TOKEN')
            if not token:
                raise ValueError("No HuggingFace token provided!")
        
        login(token=token)
        print("   ✓ Logged in to HuggingFace")
        
        # Inițializează API
        api = HfApi()
        
        # Creează repo dacă nu există
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
            print(f"   ✓ Repository ready: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"   ⚠️  Repository might already exist: {e}")
        
        # Upload fișiere
        save_path = Path(save_dir)
        files_to_upload = [
            MODEL_FILENAME,
            "metadata.json",
            "README.md"
        ]
        
        for filename in files_to_upload:
            file_path = save_path / filename
            if file_path.exists():
                print(f"   📤 Uploading {filename}...")
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"   ✓ {filename} uploaded")
            else:
                print(f"   ⚠️  {filename} not found, skipping")
        
        print(f"\n✅ Upload complete!")
        print(f"   🔗 View your model: https://huggingface.co/{repo_id}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print(f"   💡 Tip: Make sure your token has write permissions")
        print(f"   💡 Tip: Create the repo manually on HuggingFace first")
        return False

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================
def main():
    """Pipeline complet de antrenare și upload"""
    
    print("=" * 70)
    print("🎯 MNIST MODEL TRAINING & HUGGINGFACE UPLOAD PIPELINE")
    print("=" * 70)
    
    # Step 1: Încărcare date
    train_loader, test_loader = load_mnist_data()
    
    # Step 2: Creare model
    print("\n🔨 Creating model...")
    model = MNISTNet(num_classes=NUM_CLASSES).to(DEVICE)
    model_info = get_model_info(model)
    print(f"   ✓ Model: {model_info['model_name']}")
    print(f"   ✓ Parameters: {model_info['total_params']:,}")
    
    # Step 3: Antrenare
    history = train_model(model, train_loader, test_loader, epochs=NUM_EPOCHS)
    
    # Step 4: Evaluare finală
    metrics = evaluate_model(model, test_loader)
    
    # Step 5: Salvare locală
    save_dir = save_model_locally(model, history, metrics)
    
    # Step 6: Upload pe HuggingFace
    print("\n" + "=" * 70)
    upload_choice = input("📤 Upload model to HuggingFace? (yes/no): ").strip().lower()
    
    if upload_choice in ['yes', 'y']:
        success = upload_to_huggingface(save_dir)
        if success:
            print("\n🎉 SUCCESS! Model is now available on HuggingFace Hub!")
        else:
            print("\n⚠️  Upload failed, but model is saved locally")
    else:
        print("\n💾 Model saved locally only")
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Local files: {save_dir}")
    if upload_choice in ['yes', 'y']:
        print(f"🔗 HuggingFace: https://huggingface.co/{HUGGINGFACE_REPO_ID}")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()