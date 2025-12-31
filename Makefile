SHELL := /bin/sh

ifeq ($(OS),Windows_NT)
PY := .venv\\Scripts\\python.exe
PIP := .venv\\Scripts\\pip.exe
else
PY := .venv/bin/python
PIP := .venv/bin/pip
endif

REQ := $(if $(wildcard requirements-dev.txt),requirements-dev.txt,requirements.txt)

.PHONY: help venv install run clean

help:
	@echo "Available targets:"
	@echo "  make install   -> create .venv and install dependencies (prefers requirements-dev.txt)"
	@echo "  make run       -> run main.py using the created .venv"
	@echo "  make clean     -> remove .venv and python caches"

venv:
	python -m venv .venv

install: venv
	$(PY) -m pip install --upgrade pip setuptools wheel
	$(PY) -m pip install -r $(REQ)

run: install
	$(PY) main.py

clean:
	-@powershell -NoProfile -Command "if (Test-Path -Path '.venv') {Remove-Item -Recurse -Force .venv}"
	-@rm -rf .venv __pycache__
	-@find . -name '*.pyc' -delete || true
