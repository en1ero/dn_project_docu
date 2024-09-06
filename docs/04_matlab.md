# MATLAB

Locate the `/matlab`dir inside the root dir and open it as your working directoy inside MATLAB to execute all related scripts and functions.

## Three main scripts
* `main_compare.m`
* `main_make_results_scunet.m`
* `main_analyze.m`

## `main_compare.m`
Chose two SCUNet models like this
```matlab
model_name_1 = 'scunet_l1x10_wgan_lpipsvgg_12500_sigmoid.pth';
model_name_2 = 'scunet_l1x10_wgan_lpipsvgg_12500_sigmoid_vst.pth';
```
and compare it automatically with the DnCNN denoiser.

## `main_make_results_scunet.m`
Again set your desired model name like this
```matlab
model_name = 'scunet_l1x10_wgan_lpipsvgg_12500_sigmoid_vst';
```
and set the image type ('SiemHarm' or 'Verlauf') by toggling this:
```matlab
% Select Test Image Type
image_type = "SiemHarm";
image_type = "Verlauf";
```
The Results will be saved in the `/results_dn_bayer` directory in the pattern `model_name`_`image_type.mat`.

## `main_analyze.m`
This is where you use your results from the previous scripts to analyze them.
use the toggle by un-/commenting the desired line in the script to use `SiemHarm` or `Verlauf`. Based on the chosen image type, you are prompted to chose an result from the previous script.
All results will be saved in the `/results_analyze` directory (*THIS TAKES A WHILE TO RUN*).
