# Template Guide — DPArena

DPArena is a platform for testing the resilience of machine learning models against **data poisoning attacks** in **Federated Learning (FL)** scenarios.

The user provides a Python file — called a **template** — that describes the model, data, and training logic. The platform automatically runs the FL simulation across 4 parallel scenarios (clean / poisoned data × FedAvg / defense method) and reports comparative metrics.

---

## Example Templates (`New_Template_Fold/`)

### 1. `template_code_tensorflow.py` — TensorFlow / Keras

| Property             | Detail                                                   |
| -------------------- | -------------------------------------------------------- |
| **Framework**        | TensorFlow / Keras                                       |
| **Neural network**   | ResNet18 adapted for small images (32×32)                |
| **Dataset**          | CIFAR-10 — 60,000 RGB images 32×32, 10 classes           |
| **Model source**     | HuggingFace: `Tudorx95/resnet18-cifar10` (`.keras` file) |
| **Initial accuracy** | ~95% on test set                                         |
| **Save format**      | `.keras` (native Keras format)                           |

### 2. `template_code_pytorch.py` — PyTorch + CIFAR-10

| Property             | Detail                                                                           |
| -------------------- | -------------------------------------------------------------------------------- |
| **Framework**        | PyTorch                                                                          |
| **Neural network**   | ResNet18 with CIFAR-10 modifications: 3×3 `conv1` (instead of 7×7), no `maxpool` |
| **Dataset**          | CIFAR-10 — 60,000 RGB images 32×32, 10 classes                                   |
| **Model source**     | HuggingFace: `edadaltocg/resnet18_cifar10` (`pytorch_model.bin`)                 |
| **Initial accuracy** | ~95% on test set                                                                 |
| **Save format**      | `.pth` (state dict + architecture metadata)                                      |
| **Normalization**    | mean `[0.4914, 0.4822, 0.4465]`, std `[0.2471, 0.2435, 0.2616]`                  |

### 3. `template_code_pytorch_cifar100.py` — PyTorch + CIFAR-100

| Property             | Detail                                                                   |
| -------------------- | ------------------------------------------------------------------------ |
| **Framework**        | PyTorch                                                                  |
| **Neural network**   | ResNet18 with CIFAR-100 modifications: 3×3 `conv1`, no `maxpool`, fc→100 |
| **Dataset**          | CIFAR-100 — 60,000 RGB images 32×32, 100 classes                         |
| **Model source**     | HuggingFace: `edadaltocg/resnet18_cifar100` (`pytorch_model.bin`)        |
| **Initial accuracy** | ~79% on test set                                                         |
| **Save format**      | `.pth` (state dict + architecture metadata)                              |
| **Normalization**    | mean `[0.5071, 0.4865, 0.4409]`, std `[0.2673, 0.2564, 0.2761]`          |

---

Data is distributed across clients using **fold-based cross-validation** — each client receives a different subset of images each round. The simulator reads images directly from disk via `FileListDataset` (TF) or `FileListDatasetPyTorch` (PyTorch), applying the transformations returned by `preprocess()` / `preprocess_transform()`.

---


## Adapting the Template to Your Own Dataset

Minimum steps to use your own model/dataset:

1. **Update the global configuration** — `NUM_CLASSES`, `IMG_SIZE`, class list
2. **Rewrite `load_train_test_data()`** — to read from your source
3. **Update `preprocess()`** — normalization specific to your dataset
4. **Update `preprocess_transform()`** (PyTorch) — same normalization as `preprocess()`
5. **Rewrite `download_data()`** — the `data/<class>/` directory structure is **fixed**
6. **Rewrite `create_model()`** — your model architecture / source
7. **Check `_model_compile()`** — loss compatible with your task

---

## Required Functions

All functions below **must exist** in the template. If any are missing, the simulation fails at the verification step.

---

### `download_data(output_dir: str)`

**Role:** Downloads and saves data to disk in the format expected by the simulator.

**Parameters:**

- `output_dir` — root directory provided by the orchestrator (e.g. `"/workspace/clean_data"`)

**Required output structure:**

```
output_dir/
└── data/
    ├── 0/          ← class 0
    │   ├── img_00000.png
    │   └── img_00001.png
    ├── 1/          ← class 1
    │   └── ...
    └── N/          ← last class
```

Each subfolder must be the **numeric index of the class** (0, 1, 2, ...). The simulator auto-detects classes from the directory structure.

**Returns nothing** (`None`).

> **Important:** Save images **unnormalized** (raw pixel values 0–255). Normalization is applied later during training via `preprocess()` or `preprocess_transform()`.

---

### `load_train_test_data()`

**Role:** Loads data from the original source (internet, local disk, HuggingFace, etc.).

**No parameters.**

**Returns:** `Tuple` with `(train_dataset, test_dataset)`.

- TensorFlow: `(tf.data.Dataset, tf.data.Dataset)` — **unprocessed, unbatched**
- PyTorch: `(DataLoader, DataLoader)` — can be batched if `preprocess_loaded_data` handles it

This function is called by `download_data()` and by `verify_template.py` for initial model evaluation.

---

### `preprocess(image, label)`

**Role:** Per-sample preprocessing: normalization + label conversion to one-hot.

**Parameters:**

- `image` — raw image (tensor or numpy array)
- `label` — raw label (integer scalar)

**Returns:** `Tuple(image_normalized, label_one_hot)`

This function is used by the simulator when loading data from file lists for TensorFlow (via `map()`). For PyTorch, see `preprocess_transform()`.

> **Warning:** Labels **must be one-hot encoded** at output, otherwise `calculate_metrics()` will compute incorrect results.

---

### `preprocess_loaded_data(train_ds, test_ds)`

**Role:** Applies preprocessing to the entire dataset (batching, shuffle, prefetch).

**Parameters:**

- `train_ds` — raw training dataset returned by `load_train_test_data()`
- `test_ds` — raw test dataset

**Returns:** `Tuple(train_ds_processed, test_ds_processed)` ready for training.

TensorFlow: apply `.map(preprocess).batch(32).prefetch(AUTOTUNE)`.  
PyTorch: usually returns unchanged if batching is already in the DataLoader.

AUTOTUNE is used to optimize training parallelism when fetching the dataset.

---

### `preprocess_transform()` _(PyTorch only — recommended)_

**Role:** Returns the `torchvision.transforms.Compose` transformations used by `FileListDatasetPyTorch` in the simulator when reading images from disk.

**No parameters.**

**Returns:** `transforms.Compose([...])` — the same transformations as in `load_train_test_data()`.

> **Critical:** The normalization in `preprocess_transform()` must be **identical** to the one in `load_train_test_data()`. If they differ, the model will see different distributions during training vs. FL evaluation.

This function is not checked by `verify_template.py`, but **its absence will cause errors** when running PyTorch simulations with fold-based data loading.

---

### `create_model()`

**Role:** Creates or downloads the pre-trained model. The model **must be compiled** before returning.

**No parameters.**

**Returns:**

- TensorFlow: compiled `tf.keras.Model` (with `optimizer`, `loss`, `metrics`)
- PyTorch: `nn.Module` with `model.criterion` and `model.optimizer` attached as attributes

You can download the model from HuggingFace, build it from scratch, or load it from disk. Important: call `_model_compile(model)` at the end of this function.

**Optional global variables (can be hardcoded as arguments):**

- `HUGGINGFACE_REPO_ID` — the HuggingFace repository
- `MODEL_FILENAME` — the model filename

---

### `_model_compile(model)`

**Role:** Compiles the model with an optimizer and loss function.

**Parameters:**

- `model` — model instance

**Returns nothing.** Modifies the model in-place.

TensorFlow: calls `model.compile(optimizer=..., loss=..., metrics=[...])`.  
PyTorch: attaches `model.criterion = nn.CrossEntropyLoss()` and `model.optimizer = optim.Adam(...)`.

> **Note:** For pre-trained models, use a **small learning rate** (e.g. `1e-4`) to avoid _catastrophic forgetting_ amplified by federated averaging.

---

### `validate_model_structure(model)`

**Role:** Inspects the model and returns architecture metadata.

**Parameters:**

- `model` — model instance

**Returns:** `Dict` with required keys:

```python
{
    'model_name':           str,   # class / architecture name
    'total_params':         int,
    'trainable_params':     int,
    'non_trainable_params': int,
    'layers_count':         int,
    'input_shape':          str,   # e.g. "(None, 32, 32, 3)"
    'output_shape':         str,   # e.g. "(None, 10)"
    'is_compiled':          bool,
}
```

---

### `train_neural_network(model, train_data, epochs=1, verbose=0)`

**Role:** Trains the model for a number of epochs on the provided data.

**Parameters:**

- `model` — model to train
- `train_data` — training data (preprocessed, batched)
- `epochs` — number of epochs (the simulator usually sends `1`)
- `verbose` — logging level (0 = silent)

**Returns:** `Dict` with training history:

```python
{
    'loss':     [float, ...],    # loss per epoch
    'accuracy': [float, ...],    # accuracy per epoch
}
```

> **Note:** The simulator calls this function for **each client, in each FL round**. Performance matters — avoid redundant expensive operations.

---

### `calculate_metrics(model, test_dataset, average='macro')`

**Role:** Evaluates the global model each round (after the specified epoch nb.) using validation Fold of that round and returns performance metrics.

**Parameters:**

- `model` — model to evaluate
- `test_dataset` — test data (preprocessed, batched)
- `average` — averaging type for precision/recall/f1 (`'macro'`, `'micro'`, `'weighted'`)

**Returns:** `Dict` with metrics:

```python
{
    'accuracy':  float,
    'precision': float,
    'recall':    float,
    'f1_score':  float,
}
```

The simulator compares these metrics across the 4 FL scenarios to quantify the attack impact and defense effectiveness.

---

### `get_model_weights(model)`

**Role:** Extracts model weights as a list of numpy arrays — used to aggregate weights across clients.

**Parameters:**

- `model` — model instance

**Returns:** `List[np.ndarray]` — all weights (and buffers, e.g. BatchNorm running stats).

TensorFlow: `return model.get_weights()`  
PyTorch: iterate through `model.state_dict().values()` and convert to numpy.

> **Important for PyTorch:** Include **buffers** too (running_mean, running_var from BatchNorm), not just trainable parameters. `model.state_dict()` includes them automatically.

---

### `set_model_weights(model, weights)`

**Role:** Applies the aggregated weights back into the local models after federated aggregation.

**Parameters:**

- `model` — model instance
- `weights` — `List[np.ndarray]` as returned by `get_model_weights()`

**Returns nothing.** Modifies the model in-place.

TensorFlow: `model.set_weights(weights)`  
PyTorch: iterate through `model.state_dict()` and copy each tensor.

---

### `save_model_config(model, filepath, save_weights=True)`

**Role:** Saves the full model to disk (architecture + weights).

**Parameters:**

- `model` — model instance
- `filepath` — save path (with or without extension)
- `save_weights` — if `False`, saves architecture only

TensorFlow: save as `.keras` via `model.save(filepath)`.  
PyTorch: save a dict with `model_state_dict`, architecture info, and normalization values.

---

### `load_model_config(filepath)`

**Role:** Reloads the model from disk and compiles it.

**Parameters:**

- `filepath` — path to the saved model file

**Returns:** compiled model instance (TF or PyTorch).

---

### `save_weights_only(model, filepath)` / `load_weights_only(model, filepath)`

**Role:** Selective save/load — weights only, without architecture.

TensorFlow: `.weights.h5` via `model.save_weights()` / `model.load_weights()`.  
PyTorch: `.weights.pth` via `torch.save(model.state_dict())` / `model.load_state_dict()`.

---

### `get_loss_type()`

**Role:** Returns the loss type as a string — used by the simulator for logging.

**Returns:** `str`, e.g. `'categorical_crossentropy'` or `'cross_entropy'`

---

### `get_image_format()`

**Role:** Returns information about image dimensions and channels.

**Returns:**

```python
{
    'size':     [H, W],   # e.g. [32, 32]
    'channels': int,      # e.g. 3 (RGB)
}
```

---

### `get_data_preprocessing()`

**Role:** Returns a reference to the `preprocess` function. Used by simulator as a **fallback** for TensorFlow templates.

In practice this means: if your template defines `preprocess()`, both functions are equivalent. If for any reason `preprocess` is absent or renamed, `get_data_preprocessing()` is the safety net that keeps fold-based data loading functional for TensorFlow.

**Returns:** `callable` — typically `return preprocess`

---

## Local Validation Before Upload

You can run the verification locally before uploading the template to the platform:

```bash
# copy your template into orchestrator_backend_server/ as template_code.py
cp my_template.py ../orchestrator_backend_server/template_code.py
cd ../orchestrator_backend_server
python verify_template.py
```

A valid template produces at the end:

```
✅ ALL TESTS PASSED SUCCESSFULLY!
✓ Template READY for FL simulation!
```

and generates `init-verification.json` with the model's initial metrics.

---

## Common Template Errors

| Error                            | Likely cause                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------- |
| `Function X MISSING`             | Function does not exist or has a different name                                 |
| `insufficient parameters`        | Function signature does not meet the minimum required                           |
| `Model is not a valid instance`  | `create_model()` does not return `tf.keras.Model` / `nn.Module`                 |
| `Labels are NOT one-hot encoded` | `preprocess()` returns an integer label instead of one-hot                      |
| Incorrect FL metrics             | `preprocess_transform()` differs from normalization in `load_train_test_data()` |
| `Failed to download model`       | HuggingFace model does not exist or no internet access                          |
