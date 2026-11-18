# ReCoDE project - Euler-Maruyama method

## Description

This code is part of the **Re**search **Co**mputing and **D**ata Science **E**xamples (ReCoDE) projects.
The project consists of a Python class containing the Euler-Maruyama (EM) method for the numerical solution
of a Stochastic Differential Equation (SDE). SDEs describe the dynamics that govern the time-evolution of
systems subjected to deterministic and random influences. They arise in fields such as biology, physics or
finance to model variables exhibiting uncertain and fluctuating behaviour. Being able to numerical solve an SDE
is essential for these fields, especially if there is no closed-form solution. This project provides an
object-oriented implementation of the EM method. Throughout the project, it is emphasised the benefits that
class encapsulation provides in terms of code modularity and re-usability.

## Learning Outcomes

This project is designed for Master's and Ph.D. students with basic Python knowledge and need to solve SDEs
for their research projects. After going through this project, students will:

1. Understand how to solve an SDE using the EM method.
2. Learn to encapsulate the EM method code into a Python class.
3. Explore how to parallelise the code to improve solution speed.

## Requirements

### System

| Program                                     | Version |
| ------------------------------------------- | ------- |
| [Git](https://git-scm.com/)                 | >= 2.41 |
| [Python](https://www.python.org/downloads/) | >= 3.9  |

### Dependencies

| Packages                                               | Version        |
| ------------------------------------------------------ | -------------- |
| [uv](https://docs.astral.sh/uv/)                       | Latest stable  |
| [numpy](https://numpy.org/doc/stable/)                 | >= 1.24.3, < 2 |
| [matplotlib](https://matplotlib.org/stable/index.html) | >= 3.7.1, < 4  |
| [jupyter](https://jupyter.org/install)                 | >= 1.0.0, < 2  |
| [joblib](https://joblib.readthedocs.io/en/stable/)     | >= 1.2.0, < 2  |

## Project Structure

```bash
.
├── .github/workflows
│   └── tests_workflow.yaml
├── docs
│   ├── 1-Introduction.ipynb
│   ├── 2-Probability-Distributions.ipynb
│   ├── 3-Euler-Maruyama-Method.ipynb
│   ├── 4-Euler-Maruyama-Class.ipynb
│   └── 5-Parallel-Euler-Maruyama-Class.ipynb
├── src
│   └── euler_maruyama
│       ├── __init__.py
│       ├── coefficients.py
│       ├── euler_maruyama.py
│       └── parallel_euler_maruyama.py
├── tests
│   ├── test_coefficient.py
│   ├── test_euler_maruyama.py
│   └── test_parallel_euler_maruyama.py
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

## Getting Started

You can read the Jupyter notebooks non-interactively on Github. Click [here](https://github.com/ImperialCollegeLondon/ReCoDe_Euler_Maruyama/tree/main/docs)
to view the collection of Jupyter notebooks located in the `docs` folder. However, for an improved experience, we suggest cloning the Github repository and running the Jupyter notebooks on your local
machine. To assist you setting up the project locally, we provide a list of steps:

### 1. Clone the repository

After installing `git` in your local machine, you can run the following command in a terminal:

```bash
git clone https://github.com/ImperialCollegeLondon/ReCoDe_Euler_Maruyama.git euler-maruyama
cd euler-maruyama
```

### 2. Install uv

Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
for your platform. Check that uv is installed by running:

```bash
uv --version
```

### 3. Install the project

From the repository folder, you can optionally select a Python version, for example Python 3.15:

```bash
uv python pin 3.15
```

Then install the project and its dependencies:

```bash
uv sync
```

If you select a version, uv records it in `.python-version`. It creates a local `.venv` and installs
the project using `uv.lock`. uv can download a compatible Python version if needed.

### 4. Launch the Jupyter notebooks

Run this command to launch Jupyter Notebook in the project's environment:

```bash
uv run jupyter notebook
```

Now, you can explore and experiment with the different notebook examples we have prepared to help you
understand this project. You do not need to activate the virtual environment manually; stop Jupyter
with Ctrl+C when you are finished.
