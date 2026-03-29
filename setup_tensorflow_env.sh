#!/bin/bash

# Create Conda environment for TensorFlow FL simulations
echo "Creating TensorFlow environment for FL Simulator..."

# Create environment
conda create -n fl_tensorflow python=3.10 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fl_tensorflow

# Install TensorFlow with CUDA support (includes CUDA 12.2 dependencies)
pip install "tensorflow[and-cuda]==2.15.1"  # Updated to latest patch version for stability

# Install related packages (removed separate keras as it's bundled in TF)
pip install numpy==1.26.4  # Updated to latest compatible
pip install pandas==2.1.4  # Updated to latest patch
pip install matplotlib==3.8.4  # Updated to latest patch
pip install scikit-learn==1.3.2  # Updated to latest patch
pip install Pillow==10.4.0  # Updated to latest
pip install scipy==1.11.4  # Updated to latest patch

# Install FL and networking libraries
pip install requests==2.32.3  # Updated to latest
pip install flask==3.0.3  # Updated to latest patch
pip install flask-cors==4.0.1  # Updated to latest patch

# Install data processing
pip install h5py==3.11.0  # Updated to latest
pip install tensorflow-datasets==4.9.6  # Updated to latest patch

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import tensorflow as tf; print('GPU devices:', tf.config.list_physical_devices('GPU'))"
python -c "import tensorflow as tf; print('CUDA available:', tf.test.is_gpu_available())"

echo "✅ TensorFlow environment with CUDA 12.2 created successfully!"
echo "Activate with: conda activate fl_tensorflow"
echo "Note: Ensure your NVIDIA drivers are compatible with CUDA 12.2 (driver version >= 535)."
