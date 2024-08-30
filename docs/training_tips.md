# Training Tips

## Optimize training performance
* DataLoader Setup
* cuDNN Auto-Tuner
* Learning Rate Scheduler
* Mixed Precision Training
* Gradient Clipping

## Logging
* TensorBoard
* Checkpointing

## DataLoader Setup
`num_workers`

- Allows data to be loaded/preprocessed in parallel, ensuring the GPU is continuously fed with new batches.

- **Rule of thumb:** `num_workers = num_cpu_cores` (can be higher or lower depending on your system).

`shuffle=True`

- Shuffles the order of the batches before each epoch to prevent patterns from forming.

`pin_memory=True`

- Pre-allocates memory for faster data transfer.

`drop_last=True`

- Ensures consistent batch size for uniform computation graphs.

`persistent_workers=True`

- Keeps workers active across epochs instead of reinitializing them every time.

## cuDNN Auto-Tuner
`torch.backends.cudnn.benchmark = True`

- The auto-tuner tests different implementations of algorithms (e.g., for convolution, pooling) and selects the fastest one that can run on the current hardware.
- **Note:** The input size should be constant (which is usually the case). If the input size is dynamic, a recalculation might be triggered, leading to potential inefficiencies.

## Learning Rate Scheduler: ReduceLROnPlateau
`ReduceLROnPlateau(mode='min', factor, patience, min_lr)`

- **Dynamic Learning Rate Adjustment:** The learning rate is dynamically adjusted during training.
- **Mechanism:** If the loss stagnates or does not decrease further for a number of validation iterations specified by `patience`, the learning rate is multiplied by `factor` until the optional minimum `min_lr` is reached.
- **Recommendation:** Use moderate values like `factor=0.9` to avoid "stalling" in the training process.

## AMP: Automatic Mixed Precision
Running the Model with `autocast()` and Adjusting Loss with `GradScaler()`.

- **Mixed Precision Arithmetic:** Combines 16-bit (half-precision) and 32-bit (single-precision) floating-point arithmetic.
- **Efficiency Improvement:** Enhances training efficiency without compromising model accuracy.
- **Increased Batch Size:** Allows for nearly double the batch size since only critical operations like summations are performed with 32-bit precision.
**Note:** 
- Works effectively with "simple" loss functions (e.g., MAE, MSE).
- Not compatible with GAN and feature-based loss functions.

## Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.4)
```

- **Preventing Large Gradients:** Ensures that gradients do not become too large, which could destabilize the training process.
- **Reducing the Risk of Exploding Gradients:** Minimizes the risk of gradients "exploding" during backpropagation.
- **Controlled Model Updates:** Updates to the model parameters are scaled, while maintaining the direction of the gradient vector.

**Motivation:** 
- The GAN loss often collapsed to NaN after extended training.
- The value of `max_norm=0.4` was empirically determined to address this issue.

## Logging with TensorBoard

### Usage with SummaryWriter:
- **Purpose:** Visualization of training and validation losses, as well as output images.

### Opening TensorBoard in VS Code:
1. **Open the Command Palette**:
    - Shortcut: `Ctrl+Shift+P`
2. **Search for TensorBoard**:
    - Type: `"Python: Launch TensorBoard"`
    - Press `Enter`
3. **Select the Logs Folder**:
    - Choose: `Select another folder`
    - Navigate to: `tensorboard_logs`
4. **View in Browser**:
    - Open: [`http://localhost:6006/`](http://localhost:6006/)

## Logging training info with `.json`
Automatic info inside the `/model_zoo` next to each trained model which looks something like this:
```json
{
    "in_res": 256,
    "n_channels": 4,
    "epoch": 30,
    "total_epochs": 30,
    "iteration": 224970,
    "total_iterations": 224970,
    "duration": 43.3434,
    "pretrained_model": "model_zoo\\2024_08_13_08_15_32\\model_96750.pth",
    "model_architecture": "SCUNet",
    "use_sigmoid": true,
    "config": "[4,4,4,4,4,4,4]",
    "batch_size": 12,
    "loss_info": {
        "Pixel_Loss": "L1",
        "Pixel_Loss_Weight": 10,
        "GAN_Loss": "wgan",
        "GAN_Loss_Weight": 1,
        "Feature_Loss": "lpips_vgg",
        "Feature_Loss_Weight": 1
    },
    "optimizer": "Adam",
    "initial_learning_rate": 0.0001,
    "last_learning_rate": 5.904900000000002e-05,
    "decay_factor": 0.9,
    "decay_step": "auto_plateau",
    "lr_patience": 10,
    "number_params": 17947224,
    "n_imgs_train": 89991,
    "n_imgs_test": 250,
    "training_path": [
        "training_data\\LSDIR\\train",
        "training_data\\custom"
    ],
    "testing_path": [
        "training_data\\LSDIR\\val\\HR"
    ],
    "transforms_train": {
        "0": {
            "class_name": "Downsample",
            "downsampling_factor": 2,
            "sigma": 1
        },
        "1": {
            "class_name": "RandomCrop",
            "h": 256,
            "w": 256
        },
        "2": {
            "class_name": "RgbToRawTransform",
            "iso": 12500,
            "noise_model": "dng",
            "wb_gains_mode": "normal"
        },
        "3": {
            "class_name": "ToTensor2"
        }
    },
    "transforms_test": {
        "0": {
            "class_name": "Downsample",
            "downsampling_factor": 2,
            "sigma": 1
        },
        "1": {
            "class_name": "RandomCrop",
            "h": 256,
            "w": 256
        },
        "2": {
            "class_name": "RgbToRawTransform",
            "iso": 12500,
            "noise_model": "dng",
            "wb_gains_mode": "normal"
        },
        "3": {
            "class_name": "ToTensor2"
        }
    },
    "final_test_loss": 0.054455384612083435,
    "final_val_loss": 0.028505839593708514
}
```