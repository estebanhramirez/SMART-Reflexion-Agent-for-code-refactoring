# ReAct Agent for code-refactoring
Repository for a SMART Reflexion Agent for code-refactoring, using tool calling for Retrieved Augmented Generation (RAG). Built using LangGraph and LangChain Expression Language (LCEL).

## 1. Create the project directory

### 1.1. Create the root folder
Create a folder named `<project_name>`, at some path `<prefix_path>` <br>
<br>
**Command:**
```
mkdir "~/<prefix_path>/<project_name>"
```
**Result:**
```
<prefix_path>/
        └── <project_name>/
```

### 1.2. Clone the GitHub repository
Clone this repository inside the root folder <br>
<br>
**Command:**
```
git clone "https://github.com/estebanhramirez/REFLEXION-AGENTIC-AI-for-motivation-letter.git" "~/<prefix_path>/<project_name>/Repo"
```
**Result:**
```
<prefix_path>/
        └── <project_name>/
                    └── Repo/
                          ├── README.md
                          └── ...
```

## 2. Set-up the virtual environment

### 2.1. Create the virtual environment
Create a new folder name `venv` inside the root folder, `<project_name>`, with the isolated and self-contained virtual environment <br>
<br>
**Command:**
```
python3.12 -m venv "~/<prefix_path>/<project_name>/venv"
```

### 2.2. Activate the virtual environment
**Command:**
```
source "~/<prefix_path>/<project_name>/venv/bin/activate"
```
or (in Windows)
```
c:\Projects\REFLEXION-AGENTIC-AI-for-motivation-letter\venv\Scripts\activate.bat
```
We can deactivate the virtual environment using `deactivate.bat`.

## 3. Install the required libraries
**Command:**
```
pip install -r "~/<prefix_path>/<project_name>/Repo/requirements.txt"
```
