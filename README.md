# Data Science Final Project

This repository contains the code and files for our CS439 Intro to Data Science final project.

The current goal of the project is to explore whether movie scores can be predicted using the features available in the IMDb movies dataset.

More project description will be added later.

## Project Structure

```text
DataScienceProject/
│
├── data/
│   └── imdb_movies.csv
│
├── src/
│   └── final_project_analysis.py
│
├── outputs/
│   ├── figures/
│   ├── model_results.csv
│   ├── cluster_profiles.csv
│   └── cleaned_movies.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

The dataset file should be placed in the `data/` folder:

```text
data/imdb_movies.csv
```

The code expects the CSV to be named exactly:

```text
imdb_movies.csv
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/craighymo/DataScienceProject.git
cd DataScienceProject
```

### 2. Create a virtual environment

On Windows:

```bash
py -m venv .venv
```

### 3. Activate the virtual environment

On Windows PowerShell:

```bash
.venv\Scripts\Activate
```

If PowerShell blocks script execution, run this once:

```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then try activating the virtual environment again.

### 4. Install dependencies

```bash
py -m pip install -r requirements.txt
```

### 5. Run the project

If the script is inside the `src/` folder:

```bash
py src/final_project_analysis.py
```

If the script is still in the main project folder:

```bash
py final_project_analysis.py
```

## Outputs

Running the project will generate output files in the `outputs/` folder, including:

```text
outputs/model_results.csv
outputs/cluster_profiles.csv
outputs/cleaned_movies.csv
outputs/figures/
```

The `figures/` folder contains generated plots used for analysis and eventually for the final report.

## Notes

- Make sure the dataset is inside the `data/` folder before running the script.
- Run commands from the project root folder, not from inside `src/`.
- The virtual environment folder `.venv/` should not be committed to GitHub.