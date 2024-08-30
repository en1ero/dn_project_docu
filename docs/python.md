# Python

## How To Replicate This Project
### 1. Navigate to Project Folder
```bash
cd <project folder>
```

### 2. Check Installed Python Versions
To see which Python versions are installed on your system, run:
```bash
py -0p
```

### 3. Create a Virtual Environment
- If Python 3.11 is installed along with other versions, use:
  ```bash
  py -3.11 -m venv .venv
  ```
- If only Python 3.11 is installed, use:
  ```bash
  py -m venv .venv
  ```

### 4. Resolve Execution Policy Error (Windows Only)
If you encounter an "about_Execution_Policies" error, run:
```bash
Set-ExecutionPolicy Unrestricted -Scope Process
```

### 5. Activate the Virtual Environment

#### For Windows:
```cmd
.venv\Scripts\activate
pip install -r requirements_win.txt
```

#### For macOS:
```bash
source .venv/bin/activate
pip install -r requirements_mac.txt
```

## Two main script files
* `training_raw.py`
* `denoise_raw.py`

## `training_raw.py`
Use this script to train new models. When starting the training process there will be a unique model name assigned in the format `YYYY_MM_DD_HH_MM_SS` in the directory `/model_zoo`.

### Setup Training
At the bottom of the script is where the training is set up:
``` py
if __name__ == '__main__':
    torch.cuda.empty_cache()

    if 1: # Set to 1 to use pretrained model
        model_dir = os.path.join('model_zoo', '2024_08_28_10_17_35')
        pretrained_model_path = util.get_last_saved_model(model_dir)
    else:
        pretrained_model_path = None

    opt = {
        'in_res': 256,
        'n_channels': 4,
        'sigma': None,
        'config': [4] * 7,
        'batch_size': 24,
        'n_epochs': 30,
        'lr_first': 1e-4,
        'max_train': 1*10**9,
        'max_test': 250,
        'pretrained_model_path': pretrained_model_path,
        'lr_decay': 0.9,
        'lr_decay_step': 'auto_plateau',
        'lr_patience': 10,
        'dpr': 0.0,
        'use_sigmoid': False,
        'use_amp': True,
        'path_train': [
            os.path.join('training_data', 'LSDIR', 'train'),
            os.path.join('training_data', 'custom')
            ],
        'path_test': [
            os.path.join('training_data', 'LSDIR','val','HR')
            ],
    }
    
    main(opt)
```

**Reminder**: set `use_amp=true` and `use_sigmoid=false` only when using simple loss functions (mae/mse). When using GAN or feature loss set `use_amp=false` and `use_sigmoid=true`.

The loss function can be defined inside the CombinedLoss (inspect this class to see what loss variants are available) class at this point in the script:
```py
criterion = CombinedLoss(
    pixel_loss_type='L1', pixel_loss_weight=1,
    gan_loss_type='wgan', gan_loss_weight=1,
    feature_loss_type='lpips_vgg', feature_loss_weight=1,
    n_channels=opt['n_channels'])
```

## `denoise_raw.py`
Use this script to load RAW images (.dng, .nef, .arw, ...) in RGGB-format. The development is entirely handled by [LibRaw](https://github.com/letmaik/rawpy) (LibRaw wrapper for python). RAW images are extracted at the initial processing stage and modified in place (add noise / denoise). 

You can also create and denoise synthetic test images such as Siemensstar, Gradient Ramps, Zoneplates etc. as grayscale or colored images.

All results are put into dynamically created folders inside `/results` directory in the format of `YYYY_MM_DD_HH_MM_SS`.

## Have a look at the scripts
``` py title="training_raw.py"
import os
import time

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from utils import utils_image as util
from utils.utils_plot import show_batches, print_batch_minmax
from utils.utils_loss import CombinedLoss



@torch.no_grad()
def validate(model, test_loader, criterion, device, opt):
    model.eval()
    total_loss = 0
    sample_input = None
    sample_output = None
    sample_target = None
    for input_images, target_images in test_loader:
        input_images = input_images.to(device)
        target_images = target_images.to(device)
        if opt['use_amp']:
            with autocast():
                output_images = model(input_images)
                loss = criterion(output_images, target_images)
        else:
            output_images = model(input_images)
            loss = criterion(output_images, target_images)
        total_loss += loss.item()
        
        if sample_input is None:
            sample_input = input_images[:6]
            sample_output = output_images[:6]
            sample_target = target_images[:6]
    
    return total_loss / len(test_loader), sample_input, sample_output, sample_target


def main(opt):
    # --------------------------------------------
    # Device and model configuration
    # --------------------------------------------
    device = util.get_device()
    model = util.load_model(device, opt)
    model_dir = util.make_dir(opt)
    opt['model_architecture'] = model.__class__.__name__
    writer = SummaryWriter(os.path.join('tensorboard_logs', os.path.basename(model_dir)))
    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True


    # --------------------------------------------
    # Data preparation
    # --------------------------------------------
    transform_train = transforms.Compose([
        util.Downsample(),
        util.RandomCrop(h=opt['in_res'], w=opt['in_res']),
        util.RgbToRawTransform(noise_model='mixed', iso=12500, wb_gains_mode='normal'),
        # util.VST(iso=12500),
        util.ToTensor2()
    ])
    transform_test = transforms.Compose([
        util.Downsample(),
        util.RandomCrop(h=opt['in_res'], w=opt['in_res']),
        util.RgbToRawTransform(noise_model='mixed', iso=12500, wb_gains_mode='normal'),
        # util.VST(iso=12500),
        util.ToTensor2()
    ])
    opt['transforms_train'] = util.extract_transforms_to_dict(transform_test)
    opt['transforms_test'] = util.extract_transforms_to_dict(transform_test)


    train_dataset = util.MyDataset(opt['path_train'], opt['n_channels'], num_images=opt['max_train'], transform=transform_train)
    test_dataset = util.MyDataset(opt['path_test'], opt['n_channels'], num_images=opt['max_test'], transform=transform_test)

    num_workers = util.get_num_workers()
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=True)
    opt['n_img_train'] = train_dataset.__len__()
    opt['n_img_test'] = test_dataset.__len__()
    print(f'Training images: {opt["n_img_train"]}, Testing images: {opt["n_img_test"]}')


    # --------------------------------------------
    # Loss & Training params
    # --------------------------------------------
    criterion = CombinedLoss(
        pixel_loss_type='L1', pixel_loss_weight=1,
        # gan_loss_type='wgan', gan_loss_weight=1,
        # feature_loss_type='lpips_vgg', feature_loss_weight=1,
        n_channels=opt['n_channels'])
    

    optimizer = optim.Adam(model.parameters(), lr=opt['lr_first'], eps=1e-7)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=opt['lr_decay'], patience=opt['lr_patience'], min_lr=opt['lr_first'] / 2**5)
    scaler = GradScaler()
    opt['loss_info'] = criterion.info
    opt['optimizer'] = optimizer.__class__.__name__
    opt['total_iterations'] = len(train_loader) * opt['n_epochs']
    start_time = time.time()


    # --------------------------------------------
    # Training
    # --------------------------------------------
    for epoch in range(opt['n_epochs']):
        model.train()
        for i, (input_images, target_images) in enumerate(train_loader):
            input_images = input_images.to(device)
            target_images = target_images.to(device)
            optimizer.zero_grad()
            if opt['use_amp']:
                with autocast():
                    output_images = model(input_images)
                    if criterion.gan_loss_weight != 0:
                        criterion.optimize_D(output_images, target_images)
                    loss = criterion(output_images.float(), target_images.float())
            else:
                output_images = model(input_images)
                if criterion.gan_loss_weight != 0:
                    criterion.optimize_D(output_images, target_images)
                loss = criterion(output_images, target_images)

            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.4)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            opt['iteration'] = epoch * len(train_loader) + i + 1
            progress = opt['iteration'] / opt['total_iterations'] * 100
            util.print_progress_info(start_time, opt, progress, criterion, epoch)

            if i % 20 == 0:  # Log every 20 iterations
                writer.add_scalar('Loss/Train', loss.item(), opt['iteration'])


            # --------------------------------------------
            # Validation, saving model & training info
            # --------------------------------------------
            if util.savepoint(opt):
                val_loss, sample_input, sample_output, sample_target = validate(model, test_loader, criterion, device, opt)
                writer.add_scalar('Loss/Validation', val_loss, opt['iteration'])
                
                # Add images to TensorBoard
                fig = show_batches(sample_input, sample_output, sample_target, return_fig=True)
                writer.add_figure('Validation Images', fig, opt['iteration'])
                plt.close(fig)
                
                # Save model and training info
                torch.save(model.state_dict(), os.path.join(model_dir, f'model_{opt["iteration"]}.pth'))
                opt['lr_last'] = scheduler.get_last_lr()[0]
                util.save_training_info(opt, model_dir, loss.item(), val_loss, model, epoch, time.time() - start_time)

                scheduler.step(val_loss)

    writer.close()



if __name__ == '__main__':
    torch.cuda.empty_cache()

    if 1: # Set to 1 to use pretrained model
        model_dir = os.path.join('model_zoo', '2024_08_28_10_17_35')
        pretrained_model_path = util.get_last_saved_model(model_dir)
    else:
        pretrained_model_path = None

    opt = {
        'in_res': 256,
        'n_channels': 4,
        'sigma': None,
        'config': [4] * 7,
        'batch_size': 24,
        'n_epochs': 30,
        'lr_first': 1e-4,
        'max_train': 1*10**9,
        'max_test': 250,
        'pretrained_model_path': pretrained_model_path,
        'lr_decay': 0.9,
        'lr_decay_step': 'auto_plateau',
        'lr_patience': 10,
        'dpr': 0.0,
        'use_sigmoid': False,
        'use_amp': True,
        'path_train': [
            os.path.join('training_data', 'LSDIR', 'train'),
            os.path.join('training_data', 'custom')
            ],
        'path_test': [
            os.path.join('training_data', 'LSDIR','val','HR')
            ],
    }
    
    main(opt)
```

``` py title="denoise_raw.py"
import os
import datetime
import utils.utils_evaluation as u
import utils.utils_image as ui
from utils.utils_plot import plot_multi as plot_m



def denoise_raw(raw_path, scunet_model, use_vst, start_x, start_y, crop_size, iso, save_images=False, result_dir='results'):
    # Process Raw Image
    image_processor = u.RawImageProcessor(raw_path, x=start_x, y=start_y, size=crop_size, iso=iso)
    image_processor.process()

    # Denoise Raw Image
    scunet_denoiser = u.ImageDenoiser(image_processor, 'SCUNet', scunet_model, use_vst=use_vst)
    dncnn_denoiser = u.ImageDenoiser(image_processor, 'DnCNN')

    scunet_denoiser.denoise()
    dncnn_denoiser.denoise()

    scunet_denoiser.calculate_metrics()
    dncnn_denoiser.calculate_metrics()

    plot_m([
        image_processor.images['clean_srgb_tm'],
        image_processor.images['noisy_srgb_tm'],
        image_processor.images['SCUNet_denoised_srgb_tm'],
        image_processor.images['DnCNN_denoised_srgb_tm'],
        image_processor.metrics['SCUNet_denoised_psnr_srgb_tm_map'],
        image_processor.metrics['DnCNN_denoised_psnr_srgb_tm_map'],
        image_processor.metrics['SCUNet_denoised_ssim_srgb_tm_map_mean'],
        image_processor.metrics['DnCNN_denoised_ssim_srgb_tm_map_mean']
        ],
        titles=['clean',
                'noisy',
                'SCUNet', 
                'DnCNN', 
                f'PSNR-SCUNet: {image_processor.metrics["SCUNet_denoised_psnr_srgb_tm"]:.2f}',
                f'PSNR-DnCNN: {image_processor.metrics["DnCNN_denoised_psnr_srgb_tm"]:.2f}',
                f'SSIM-SCUNet: {image_processor.metrics["SCUNet_denoised_ssim_srgb_tm"]:.2f}',
                f'SSIM-DnCNN: {image_processor.metrics["DnCNN_denoised_ssim_srgb_tm"]:.2f}'
                ])
    
    # Save Results
    if save_images:
        image_processor.save_images(result_dir)
        image_processor.save_metrics(result_dir)


def denoise_raw_test(raw_path, scunet_model, use_vst, iso, size, min_val, max_val, save_images=False, result_dir='results'):
    # Set Image Parameters
    test_image_processor = u.RawImageProcessor(
        raw_path, size=size, iso=iso, min_val=min_val, max_val=max_val, test_image_type=
        # ----- UNCOMMENT ONE OF THE LINES BELOW -----
        # 'zone_plate', max_freq=32
        # 'harmonic_star', num_sectors=128
        # 'non_harmonic_star', num_sectors=144
        # 'harmonic_star_fixed_contrast', num_sectors=144
        # 'non_harmonic_star_fixed_contrast', num_sectors=144
        'horizontal_ramp', gamma=2.2
        # 'vertical_ramp', gamma=2.2
        # 'checkerboard', squares=8
        # 'frequency_sweep', min_freq=1, max_freq=32
        # 'circular_zones', num_zones=16
        # 'edge_response', edge_width=15
        # 'colored_zone_plate', max_freq=64
        # 'colored_siemens_star', num_sectors=32
        # 'rgb_ramp', direction='vertical'
        # 'color_checkerboard', squares=8
        # 'color_frequency_sweep', min_freq=1, max_freq=32
        # 'rainbow_intensity'
        # 'smpte_color_bars'
        # 'color_wheel'
    )
    test_image_processor.process()
    
    # Denoise Raw Image
    scunet_denoiser = u.ImageDenoiser(test_image_processor, 'SCUNet', scunet_model, use_vst=use_vst)
    dncnn_denoiser = u.ImageDenoiser(test_image_processor, 'DnCNN')

    scunet_denoiser.denoise()
    dncnn_denoiser.denoise()

    scunet_denoiser.calculate_metrics()
    dncnn_denoiser.calculate_metrics()


    # Plot Results
    plot_m([
        test_image_processor.images['clean_srgb'],
        test_image_processor.images['noisy_srgb'],
        test_image_processor.images['SCUNet_denoised_srgb'],
        test_image_processor.images['DnCNN_denoised_srgb'],
        test_image_processor.metrics['SCUNet_denoised_psnr_srgb_map'],
        test_image_processor.metrics['DnCNN_denoised_psnr_srgb_map'],
        test_image_processor.metrics['SCUNet_denoised_ssim_srgb_map_mean'],
        test_image_processor.metrics['DnCNN_denoised_ssim_srgb_map_mean']
        ],
        titles=['clean',
                'noisy',
                'SCUNet', 
                'DnCNN', 
                f'PSNR-SCUNet: {test_image_processor.metrics["SCUNet_denoised_psnr_srgb"]:.2f}',
                f'PSNR-DnCNN: {test_image_processor.metrics["DnCNN_denoised_psnr_srgb"]:.2f}',
                f'SSIM-SCUNet: {test_image_processor.metrics["SCUNet_denoised_ssim_srgb"]:.2f}',
                f'SSIM-DnCNN: {test_image_processor.metrics["DnCNN_denoised_ssim_srgb"]:.2f}'
                ])

    # Save Results
    if save_images:
        test_image_processor.save_images(result_dir)
        test_image_processor.save_metrics(result_dir)



if __name__ == '__main__':
    # Select Raw Image
    path_dir = os.path.join('test_imgs', 'raw')
    path_files = [f for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f))]
    raw_path = os.path.join(path_dir, path_files[5])

    # Create Result Directory
    now = datetime.datetime.now()
    unique_id = now.strftime('%Y_%m_%d_%H_%M_%S')
    result_dir = os.path.join('results', unique_id)

    # Select SCUNet Model and set VST mode by model name
    model_name = 'l1x10_wgan_lpipsvgg_12500_sigmoid_vst'
    model_path = os.path.join('model_zoo', model_name)
    last_saved_model_path = ui.get_last_saved_model(model_path)
    scunet_model = u.load_model(last_saved_model_path)
    use_vst = 'vst' in model_name.lower()

    # Set Image Parameters for Raw Denoising
    start_y, start_x = 2000, 3600
    crop_size = 512
    full_size = False
    if full_size:
        start_y, start_x = None, None
        crop_size = None
    iso = 12500

    # Run Raw Denoising
    denoise_raw(raw_path, scunet_model, use_vst, start_x, start_y, crop_size, iso,save_images=True, result_dir=result_dir)

    # Set Test Image parameters
    size = 512
    min_val = 0.2
    max_val = 0.8

    # Run Raw Denoising Test
    result_dir_test = result_dir + '_test'
    denoise_raw_test(raw_path, scunet_model, use_vst, iso, size, min_val, max_val, save_images=True, result_dir=result_dir_test)

    print('Done!')
```