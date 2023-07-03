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

| Program                                                    | Version |
|------------------------------------------------------------|---------|
| [Git](https://git-scm.com/)                                | >= 2.41 |
| [Python](https://www.python.org/downloads/)                | >= 3.9  |

### Dependencies

| Packages                                               | Version   |
|--------------------------------------------------------|-----------|
| [poetry](https://python-poetry.org/docs/)              | >= 1.4.*  |
| [numpy](https://numpy.org/doc/stable/)                 | >= 1.24.* |
| [matplotlib](https://matplotlib.org/stable/index.html) | >= 3.7.*  |
| [jupyter](http://jupyter.org/install)                  | >= 1.0.*  |
| [joblib](https://joblib.readthedocs.io/en/stable/)     | >= 1.2.*  |

## Project Structure
```bash 
.
├── notebooks
│   ├── 1- Introduction.ipynb
│   ├── 2- Probability Distributions.ipynb
│   ├── 3- Euler-Maruyama method.ipynb
│   ├── 4- Euler-Maruyama class.ipynb
│   └── 5- Parallel Euler-Maruyama class.ipynb
├── src
│   └── euler_maruyama
│       ├── __init__.py
│       ├── coefficients.py
│       ├── euler_maruyama.py
│       └── parallel_euler_maruyama.py
├── .gitignore
├── README.md
├── poetry.lock
├── poetry.toml
└── pyproject.toml
```

## Getting Started

You can read the Jupyter notebooks non-interactively on Github. Click [here](https://github.com/ImperialCollegeLondon/ReCoDe_Euler_Maruyama/tree/main/notebooks)
to view the collection of Jupyter notebooks located in the ``notebooks`` folder. However, for an improved experience, we suggest cloning the Github repository and running the Jupyter notebooks on your local
machine. To assist you setting up the project locally, we provide a list of steps:

### 1. Clone the repository

After installing `git` in your local machine, you can run the following command in a terminal:

```bash
git clone https://github.com/ImperialCollegeLondon/ReCoDe_Euler_Maruyama.git euler-maruyama
cd euler-maruyama
```

### 2. Install poetry

Once you have downloaded a `Python` version, you need to install `poetry`. 
`Poetry` is a dependency management and packaging tool for `Python` projects that simplifies the process of managing dependencies and distributing packages.
It allows you to define project dependencies in a `pyproject.toml` file and provides commands to install, update, and remove dependencies. 
The main advantages of `poetry` include dependency resolution to ensure consistent environments, the management of virtual environments for isolation and simplified package publishing. 
It streamlines the development workflow and facilitates collaboration by providing a unified and straightforward approach to managing dependencies in `Python` projects.
You can find more information in its [documentation](https://python-poetry.org/).
Our main focus here is to use `poetry` to install the project and their dependencies locally.

```bash
pip install poetry
```

You can check that `poetry` has been successfully installed by running:

```bash
poetry --version
Poetry (version 1.4.0)
```


### 3. Install the project

Now, we need to install the project and its requirements. You can run the following command in the folder 
where you downloaded the Github repository:

```bash
poetry install
```

This command creates a virtual environment in the same folder you are working, as specified by the
`poetry.toml` configuration file of the project. Then, the packages requirements are installed and
finally, the project is installed locally with the name `euler-maruyama` version (0.1.0).

### 4. Activate the local environment

Run the following command to activate the local environment you created in the previous step:

```bash
poetry shell
```

### 5. Launch the Jupyter notebooks

You can run this command to launch the Jupyter notebook:

```bash
jupyter notebook
```

Now, you can explore and experiment with the different notebook examples we have prepared to help you
understand this project.

### 6. Close the environment

If you have closed the Jupyter notebooks and want to exit from the local environment, just run:

```bash
exit
```
