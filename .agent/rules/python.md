# Python Virtual Environment Rules

To ensure the Antigravity agent uses the project's virtual environment for Python tasks, the following configuration is applied:

- **Venv Path**: The virtual environment is located at `./venv`.
- **Python Interpreter**: For all Python executions, use the interpreter at `.\venv\Scripts\python.exe` (Windows).
- **Pip**: For package management, use `.\venv\Scripts\pip.exe`.

## Instructions for Agent
Whenever performing Python-related tasks (running scripts, installing dependencies, linting), always prefix commands with the path to the virtual environment's interpreter or activate it first.

Example:
```powershell
.\venv\Scripts\python.exe path\to\script.py
```
