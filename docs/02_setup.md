# Setup

Here's a guide for installing the project necessities, with separate tutorials for Windows and macOS.

---

## Python Installation

<!-- Windows -->
### Windows

#### 1. Install Git
Choose one of the following methods:

- Using winget tool in PowerShell or Command Prompt (requires terminal restart afterwards to take effect):
  ```
  winget install --id Git.Git -e --source winget
  ```

- Or download the standalone installer from [https://git-scm.com/download/win](https://git-scm.com/download/win)

#### 2. Navigate to Project Folder
```
cd <project folder>
```

#### 3. Check Installed Python Versions
To see which Python versions are installed on your system, run:
```
py -0p
```

#### 4. Create a Virtual Environment
- If Python 3.11 is installed along with other versions:
  ```
  py -3.11 -m venv .venv
  ```
- If only Python 3.11 is installed:
  ```
  py -m venv .venv
  ```

#### 5. Resolve Execution Policy Error
If you encounter an "about_Execution_Policies" error, run this command and retry step 4.:
```
Set-ExecutionPolicy Unrestricted -Scope CurrentUser
```

#### 6. Activate the Virtual Environment
```
.venv\Scripts\activate
```

#### 7. Install Requirements
```
pip install -r requirements_win.txt
```

---

<!-- macOS -->
### macOS

#### 1. Install Git
Choose one of the following methods:

- Using Homebrew:
  ```
  brew install git
  ```

- Or choose another installation method from [https://git-scm.com/download/mac](https://git-scm.com/download/mac)

#### 2. Navigate to Project Folder
```
cd <project folder>
```

#### 3. Create a Virtual Environment
- For Python 3.11 (replace with your desired version if different):
  ```
  python3.11 -m venv .venv
  ```
- If you want to use the default Python 3 version:
  ```
  python3 -m venv .venv
  ```

#### 4. Activate the Virtual Environment
```
source .venv/bin/activate
```

#### 5. Install Requirements
```
pip install -r requirements_mac.txt
```

---

<!-- MATLAB -->
## MATLAB Installation
This is how to set up Python for MATLAB.

### 1. Download Python 3.11.9
- Download from the official site (DO NOT download from MS Store):
  - [Python 3.11.9 for Windows (amd64)](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
- During installation, ensure you check the box for "Add python.exe to PATH".

### 2. Navigate to the MATLAB Project Folder via Windows Command Prompt
```
cd <project folder>
```

### 3. Check Installed Python Versions
```cmd
py -0p
```

### 4. Create a Virtual Environment
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

### 5. Activate the Virtual Environment
- Activate the virtual environment via:
  ```cmd
  .venv\Scripts\activate
  ```

### 6. Install Required Packages
- Install the necessary packages with CUDA support into the new virtual environment:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  
  pip install numpy einops timm torchsummary
  ```

### 7. Set the Python Environment in MATLAB
- In MATLAB console, specify the Python environment:
  ```matlab
  pyenv('Version', 'project_folder\.venv\Scripts\pythonw.exe')
  ```
- The result should look like this:
  ```matlab
  ans = 

  PythonEnvironment with properties:

          Version: "3.11"
       Executable: "project_folder\.venv\Scripts\pythonw.exe"
          Library: "C:\Users\user\AppData\Local\Programs\Python\Python311\python311.dll"
             Home: "project_folder\.venv\Scripts\.venv"
           Status: NotLoaded
    ExecutionMode: InProcess

### 8. Restart MATLAB (if necessary)
- check if pyenv is set correctly in MATLAB console
  ```matlab
  pyenv
  ```

<!-- general info -->
## Troubleshooting
For any of the installation steps above, you may need to uninstall the previous installation. Sometimes PyTorch cannot make use of CUDA due to incompatibility.

### First option - Uninstall
```cmd
pip uninstall torch torchvision 
pip cache purge
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  
```

### Second option - Remove Virtual Environment
- Remove the directory recursively
  ```cmd
  rmdir /s /q .venv
  ```
- Restart from step [Python Step 4](#4-create-a-virtual-environment) or [MATLAB Step 4](#4-create-a-virtual-environment)
