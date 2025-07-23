# 6-week--ibm
# Employee Salary Classification App

This repository contains a Streamlit web application that predicts whether an employee's income is `<=50K` or `>50K` based on features like age, education level, occupation, and hours worked per week. The model is trained using a Random Forest Classifier on the `adult 3.csv` dataset, a popular dataset for income classification tasks.

## Overview

The app allows users to input personal and professional details, and it provides a prediction along with the probability scores for each income category. The project is designed for educational purposes and can be extended for real-world applications with additional features or datasets.

## Features

- Input fields for `Age`, `Education Level`, and `Occupation`.
- Real-time prediction using a pre-trained Random Forest model.
- Displays probability scores for `<=50K` and `>50K` income categories.
- Simple and interactive user interface powered by Streamlit.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.7 or higher
- Required Python libraries:
  - `streamlit`
  - `pandas`
  - `scikit-learn`
  - `joblib`

Install the dependencies using pip:

```bash
pip install streamlit pandas scikit-learn joblib
