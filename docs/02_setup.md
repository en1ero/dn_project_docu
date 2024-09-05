# Setup

Here's a guide for installing the project necessities, with separate tutorials for Windows and macOS.

## Python Installation

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

## MATLAB Installation