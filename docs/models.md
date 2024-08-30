# Trained Models
### l1_12500
- **Description:** Absolute Error (L1 / MAE)
- **Noise:** ISO Noise 12500

### l1_12500_vst
- **Description:** Absolute Error (L1 / MAE)
- **Noise:** ISO Noise 12500
- **Additional Processing:** Variance Stabilizing Transform (VST)

### l1x10_wgan_lpipsvgg_12500_sigmoid
- **Description:** Absolute Error (L1 / MAE) x 10, GAN-Loss (WGAN), Feature-Loss (LPIPS-VGG)
- **Noise:** ISO Noise 12500
- **Final Layer:** Sigmoid

### l1x10_wgan_lpipsvgg_12500_sigmoid_vst
- **Description:** Absolute Error (L1 / MAE) x 10, GAN-Loss (WGAN), Feature-Loss (LPIPS-VGG)
- **Noise:** ISO Noise 12500
- **Final Layer:** Sigmoid
- **Additional Processing:** Variance Stabilizing Transform (VST)

### l1_semiblind_vst (in progress)
- **Description:** Absolute Error (L1 / MAE)
- **Noise Factors:** Variable Noise Factor (NF) = [1, 1.5] 
  - ISO Noise 12500 * NF
  - AWGN * NF
  - ISO + AWGN * NF
- **Additional Processing:** Variance Stabilizing Transform (VST)