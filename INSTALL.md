# Installation and run instructions

This project uses a lightweight virtual environment and a Makefile to simplify installation and running `main.py`.

Prerequisites
- Python 3.8+ installed and available on PATH
- Git (optional, to clone repository)

Windows (PowerShell) - recommended steps

1. Open PowerShell in the project root (where `Makefile` and `main.py` live).

2. Create virtual environment and install dependencies using the Makefile:

```powershell
make install
```

This target will create a `.venv` folder, upgrade pip/setuptools/wheel, then install packages from `requirements-dev.txt` if present, otherwise from `requirements.txt`.

3. Run the main program:

```powershell
make run
```

Notes for Unix / macOS
- If you run on macOS or Linux, `make install` and `make run` will work in the same way provided `make` is available.

If you don't have GNU Make or prefer to run commands manually in PowerShell

```powershell
# create venv
python -m venv .venv
# activate
.\.venv\Scripts\Activate.ps1
# upgrade pip
python -m pip install --upgrade pip setuptools wheel
# install deps (choose requirements-dev.txt if you want dev extras)
python -m pip install -r requirements-dev.txt
# run
python main.py
```

Troubleshooting
- If you get permissions errors creating or activating the venv on Windows, you may need to change PowerShell execution policy (run as admin):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```