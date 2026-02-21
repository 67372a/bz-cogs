---
trigger: always_on
---

# Python Virtual Environment Rules

To ensure the Antigravity agent uses the project's virtual environment for Python tasks, the following configuration is applied:

- **Venv Path**: The virtual environment is located at `./venv`.
- **Python Interpreter**: For all Python executions, use the interpreter at `.\venv\Scripts\python.exe` (Windows).
- **Pip**: For package management, use `.\venv\Scripts\pip.exe`.

## Instructions for Agent
Whenever performing Python-related tasks (running scripts, installing dependencies, linting, testing, etc.), you MUST ALWAYS first activate the virtual environment before running any Python commands or you MUST prefix the command with the path to the python executable in the venv.

Example:
```powershell
.\venv\Scripts\Activate.ps1
python path\to\script.py
```