# MATLAB

## MATLAB project inside root directory
Locate the `/matlab`dir inside the root dir and open it as your working directoy inside MATLAB to execute all related scripts and functions.

## How to set up Python for MATLAB
Here is the provided content formatted as a Markdown document:

## Setup Python For MATLAB

### 1. Download Python 3.11.9
- Download from the official site (DO NOT download from MS Store):
  - [Python 3.11.9 for Windows (amd64)](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
- During installation, ensure you check the box for "Add python.exe to PATH".

### 2. Configure Python in MATLAB
- Open MATLAB and type `pyenv` in the command window. The result should look like this:
  ```matlab
  ans = 

    PythonEnvironment with properties:

            Version: "3.11"
         Executable: "C:\Users\your_name\AppData\Local\Programs\Python\Python311\pythonw.exe"
            Library: "C:\Users\your_name\AppData\Local\Programs\Python\Python311\python311.dll"
               Home: "C:\Users\your_name\AppData\Local\Programs\Python\Python311"
             Status: NotLoaded
      ExecutionMode: InProcess
  ```

### 3. Open Command Prompt
- Open Windows Command Prompt (Press `WIN+R`, type `cmd`, and press Enter).
- Change the directory to your MATLAB working directory:
  ```cmd
  cd C:\Users\your_name\matlab_project
  ```

### 4. Check Installed Python Versions
- To check which Python versions are installed on your system, type:
  ```cmd
  py -0p
  ```

### 5. Create and Activate Virtual Environment
- Make a virtual environment to store Python modules by typing:
  ```bash
  py -3.11 -m venv .venv
  ```
- If multiple Python versions are installed, use:
  ```bash
  py -3.11 -m venv .venv
  ```
- If only Python 3.11 is installed, use:
  ```bash
  py -m venv .venv
  ```

### 6. Activate the Virtual Environment
- Activate the virtual environment via:
  ```cmd
  .venv\Scripts\activate
  ```

### 7. Install Required Packages
- Install the necessary packages with CUDA support into the new virtual environment:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  
  pip install numpy einops timm torchsummary
  ```

### 8. Restart MATLAB

### 9. Set the Python Environment in MATLAB
- After restarting MATLAB, specify the Python environment to use by typing the following in the MATLAB terminal:
  ```matlab
  pyenv('Version', 'C:\Users\your_name\matlab_project\.venv\Scripts\pythonw.exe')
  ```

pip uninstall torch torchvision 
pip cache purge


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
