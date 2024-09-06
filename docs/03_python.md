# Python

In `/root` of the project are two main script files for __1. training new models__ and __2. denoising RAW + Test images__:

1. `training_raw.py`
2. `denoise_raw.py`

--- 

## 1. `training_raw.py`
Use this script to train new models. When starting the training process there will be a unique model name assigned in the format `YYYY_MM_DD_HH_MM_SS` and the results will be saved in the directory `/model_zoo`.

### Setup Training
At the bottom of the script you can set training parameters:
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

!!! info

    Set `use_amp=true` and `use_sigmoid=false` only when using simple loss functions (mae/mse).
    
    When using GAN or feature loss set `use_amp=false` and `use_sigmoid=true`.

---

### Setup Loss
The loss function can be defined inside the CombinedLoss class at this point in the script:
```py
criterion = CombinedLoss(
    pixel_loss_type='L1', pixel_loss_weight=1,
    gan_loss_type='wgan', gan_loss_weight=1,
    feature_loss_type='lpips_vgg', feature_loss_weight=1,
    n_channels=opt['n_channels'])
```

!!! info

    Inspect `CombinedLoss.py` to see what loss variants are available:

    - Absolute pixel losses: `L1`(mae) and `L2`(mse)
    - GAN losses (realism): `gan`, `wgan`, `lsgan`, `ragan`, `softplusgan`
    - Feature losses: `vgg`, `lpips_vgg`, `lpips_alex`
    - Structural losses: `SSIM`, `MS-SSIM` __(Not suitable for training)__


---

## 2. `denoise_raw.py`
Use this script to load RAW images (`.dng`, `.nef`, `.arw`, ...) in ___==RGGB==___-format. RAW images are loaded via the [RawPy](https://github.com/letmaik/rawpy) and modified in place _(add noise / denoise)_ at the initial processing stage. After denoising the rest of the processing is again entirely handled by RawPy (a LibRaw wrapper for Python).

Set the following parameters to your liking:

```python
    config = {
        'raw_dir': os.path.join('test_imgs', 'raw'),
        'raw_file_index': 5,
        'model_name': 'l1x10_wgan_lpipsvgg_12500_sigmoid_vst',
        'iso': 12500,
        'start_x': 3600,
        'start_y': 2000,
        'crop_size': 512,
        'save_images': True,
        'test_size': 512,
        'min_val': 0.2,
        'max_val': 0.8,
    }
```

The example above

- looks for the 5th image in the `test_imgs/raw` directory
- uses the SCUnet model `l1x10_wgan_lpipsvgg_12500_sigmoid_vst`
- sets the ISO to 12500
- crops the image to `(512, 512)` at `(3600, 2000)`
- saves the denoising results into the `results` directory
- sets the test image size to `(512, 512)`
- sets the min and max values of the test image to `0.2` and `0.8`

---

To generate a synthetic test image (Siemensstar, Gradient Ramps, Zoneplates etc.) as grayscale or colored images you change the following parameters to your liking:

```python
TEST_IMAGE_OPTIONS = {
    'zone_plate': {'max_freq': 32},
    'harmonic_star': {'num_sectors': 128},
    'non_harmonic_star': {'num_sectors': 144},
    'harmonic_star_fixed_contrast': {'num_sectors': 144}, # Contrast level of noise variance at 12500 ISO
    'non_harmonic_star_fixed_contrast': {'num_sectors': 144},
    'horizontal_ramp': {'gamma': 2.2},
    'vertical_ramp': {'gamma': 2.2},
    'checkerboard': {'squares': 8},
    'frequency_sweep': {'min_freq': 1, 'max_freq': 32},
    'circular_zones': {'num_zones': 16},
    'edge_response': {'edge_width': 15},
    'colored_zone_plate': {'max_freq': 64},
    'colored_siemens_star': {'num_sectors': 32},
    'rgb_ramp': {'direction': 'vertical'},
    'color_checkerboard': {'squares': 8},
    'color_frequency_sweep': {'min_freq': 1, 'max_freq': 32},
    'rainbow_intensity': {},
    'smpte_color_bars': {},
    'color_wheel': {}
}
```

When starting the script you will be prompted to enter a number and hit return selecting one of your above defined testimages.

All results are put into dynamically created folders inside `/results` directory in the format of `YYYY_MM_DD_HH_MM_SS`.

---

## Have a look at the scripts

=== "Training"

    ```py title="training_raw.py"
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

=== "Denoise RAW + Test images"

    ``` py title="denoise_raw.py"
    import os
    import datetime

    import utils.utils_evaluation as u
    import utils.utils_image as ui

    # Dictionary of available test image types and their parameters
    TEST_IMAGE_OPTIONS = {
        'zone_plate': {'max_freq': 32},
        'harmonic_star': {'num_sectors': 128},
        'non_harmonic_star': {'num_sectors': 144},
        'harmonic_star_fixed_contrast': {'num_sectors': 144},
        'non_harmonic_star_fixed_contrast': {'num_sectors': 144},
        'horizontal_ramp': {'gamma': 2.2},
        'vertical_ramp': {'gamma': 2.2},
        'checkerboard': {'squares': 8},
        'frequency_sweep': {'min_freq': 1, 'max_freq': 32},
        'circular_zones': {'num_zones': 16},
        'edge_response': {'edge_width': 15},
        'colored_zone_plate': {'max_freq': 64},
        'colored_siemens_star': {'num_sectors': 32},
        'rgb_ramp': {'direction': 'vertical'},
        'color_checkerboard': {'squares': 8},
        'color_frequency_sweep': {'min_freq': 1, 'max_freq': 32},
        'rainbow_intensity': {},
        'smpte_color_bars': {},
        'color_wheel': {}
    }


    def main():
        # Configuration dictionary
        config = {
            'raw_dir': os.path.join('test_imgs', 'raw'),
            'raw_file_index': 5,
            'model_name': 'l1x10_wgan_lpipsvgg_12500_sigmoid_vst',
            'iso': 12500,
            'start_x': 3600,
            'start_y': 2000,
            'crop_size': 512,
            'save_images': True,
            'test_size': 512,
            'min_val': 0.2,
            'max_val': 0.8,
        }

        # Select test image type
        config['test_image_type'] = u.select_test_image_type(TEST_IMAGE_OPTIONS)
        config.update(TEST_IMAGE_OPTIONS[config['test_image_type']])

        # Select Raw Image
        path_files = [f for f in os.listdir(config['raw_dir']) if os.path.isfile(os.path.join(config['raw_dir'], f))]
        config['raw_path'] = os.path.join(config['raw_dir'], path_files[config['raw_file_index']])

        # Create Result Directory
        now = datetime.datetime.now()
        unique_id = now.strftime('%Y_%m_%d_%H_%M_%S')
        config['result_dir'] = os.path.join('results', unique_id)
        os.makedirs(config['result_dir'], exist_ok=True)

        # Select SCUNet Model and set VST mode by model name
        model_path = os.path.join('model_zoo', config['model_name'])
        last_saved_model_path = ui.get_last_saved_model(model_path)
        config['scunet_model'] = u.load_model(last_saved_model_path)
        config['use_vst'] = 'vst' in config['model_name'].lower()

        processor = u.DenoiseProcessor(config)

        # Run Raw Denoising
        processor.process_raw(config)

        # Run Test Image Denoising
        config['result_dir'] = config['result_dir'] + '_test'
        os.makedirs(config['result_dir'], exist_ok=True)
        processor.result_dir = config['result_dir']
        processor.process_test(config)

        print('Done!')

    if __name__ == '__main__':
        main()
    ```