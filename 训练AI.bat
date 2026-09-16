@echo off
rem Opens the 2048 trainer GUI (no console window)
cd /d "%~dp0"
start "" pythonw train_gui.py
