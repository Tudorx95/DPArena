#!/bin/bash

# Create Conda environment for PyTorch FL simulations
echo "Creating PyTorch environment for FL Simulator..."

# Create environment
conda create -n fl_pytorch python=3.10 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fl_pytorch

# Install PyTorch (CPU version - adjust for GPU if needed)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# For GPU support, use instead:
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Install related packages
pip install numpy==1.26.0
pip install pandas==2.1.0
pip install matplotlib==3.8.0
pip install scikit-learn==1.3.0
pip install Pillow==10.1.0
pip install scipy==1.11.0

# Install FL and networking libraries
pip install requests==2.31.0
pip install flask==3.0.0
pip install flask-cors==4.0.0

# Install additional PyTorch utilities
pip install torchmetrics==1.2.0
pip install tensorboard==2.15.0

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"
python -c "print(f'CUDA available: {torch.cuda.is_available()}')"

echo "✅ PyTorch environment created successfully!"
echo "Activate with: conda activate fl_pytorch"
