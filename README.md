# Dissertation
Post-Grad Dissertation Project
## Prerequisites

Make sure you have the following installed before starting:

- **Python 3.8+** → [python.org/downloads](https://www.python.org/downloads)
- **Git** → [git-scm.com/downloads](https://git-scm.com/downloads)

Verify both are installed by opening a terminal and running:

```bash
python --version
git --version
```

---

## Setup & Installation

### Step 1 — Clone the Repository

Open a terminal:
- **Windows** — press `Win + R`, type `cmd`, hit Enter
- **Mac** — press `Cmd + Space`, type `Terminal`, hit Enter

Then run the following commands:

```bash
git clone https://github.com/MingsoHangJabegu/Dissertation.git
cd Dissertation
```

---

### Step 2 — Create a Virtual Environment

A virtual environment keeps dependencies isolated from your system Python:

```bash
# Create the environment
python -m venv .venv
```

Then activate it:

```bash
# Windows
.venv\Scripts\activate

# Windows (Git Bash)
source .venv/Scripts/activate

# Mac / Linux
source .venv/bin/activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Run the Scripts

You can either run the files using the run button in your IDE or by running the following command:

```bash
python src/{file_name}.py
```

---