# MATLAB

Locate the `/matlab`directory inside `/root` as your working directory to execute all MATLAB related scripts and functions.

__Three main scripts:__

1. `main_compare.m`
2. `main_make_results_scunet.m`
3. `main_analyze.m`

---

## 1. `main_compare.m`
Select two SCUNet models with `model_name = <model_name>.pth`
```matlab
model_name_1 = 'scunet_l1x10_wgan_lpipsvgg_12500_sigmoid.pth';
model_name_2 = 'scunet_l1x10_wgan_lpipsvgg_12500_sigmoid_vst.pth';
```
and compare it automatically with the DnCNN denoiser.

---

## 2. `main_make_results_scunet.m`
Again set your desired model name with `model_name = <model_name>.pth`
```matlab
model_name = 'scunet_l1x10_wgan_lpipsvgg_12500_sigmoid_vst';
```
and set the image type (`'SiemHarm'` or `'Verlauf'`) by commenting one of these lines:
```matlab
% image_type = "SiemHarm";
image_type = "Verlauf";
```

The Results will be saved in the `/results_dn_bayer` directory in the pattern `model_name_image_type.mat`.

---

## 3. `main_analyze.m`
This is where you analyze your results from the previous script. Comment the desired line in the script to use `'SiemHarm'` or `'Verlauf'`:
```matlab
% image_type = "SiemHarm";
image_type = "Verlauf";
```
Based on the chosen image type string, you are prompted to choose a matching result file from the `/results_dn_bayer` directory. All final results (including, images, heatmaps, plots and metrics) will be saved in the `/results_analyze` directory.

!!! warning

    __THIS TAKES A WHILE TO RUN__

