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
| ---------------------------------------------------------- |---------|
| [Anaconda](https://www.anaconda.com/products/distribution) | >= 4.1  |
| [Python](https://www.python.org/downloads/)                | >= 3.9  |

### Dependencies

| Packages                                               | Version   |
|--------------------------------------------------------|-----------|
| [poetry](https://python-poetry.org/docs/)              | = 1.5.*   |
| [numpy](https://numpy.org/doc/stable/)                 | >= 1.24.* |
| [matplotlib](https://matplotlib.org/stable/index.html) | >= 3.7.*  |
| [jupyter](http://jupyter.org/install)                  | >= 1.0.*  |
| [joblib](https://joblib.readthedocs.io/en/stable/)     | >= 1.2.*  |
