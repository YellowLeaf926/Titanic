# Titanic Repo
This repo guides you to load and preprocess Titanic passenger data from the 1912 disaster, build models to predict survival, and generate predictions, with clear instructions for exploring, training, and evaluating the models.

# Objective
The goal of this project is to use logistic regression models to predict the Survived outcome based on selected passenger features from the Titanic dataset. The modeling process is implemented in both Python and R, allowing for a comparison of approaches across the two languages.

# Prerequisites
Before running the project, ensure you have:
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Git](https://git-scm.com/)
No manual package installation is needed — all dependencies are handled automatically within each Docker container.

# Repository Structure
Titanic/
├── src/
│   ├── python_app/
│   │   ├── app.py                   # Python logistic regression
│   │   └── Dockerfile               # Python Dockerfile
│   │   └── requirements.txt         # Python requirements
│   └── R_app/
│       ├── app.R                    # R logistic regression
│       └── Dockerfile               # R Dockerfile
│       └── install_packages.R       # R requirements
├── titanic_venv                     # Python Virtual Environment
├── .gitignore                       # files that are not displayed
└── README.md                        # this file

# Step 1 - Clone Repository 
```bash
git clone https://github.com/YellowLeaf926/Titanic.git
cd Titanic
```
# Step 2 - Load the data
Download the data from https://www.kaggle.com/competitions/titanic/data. Three csv datasets are included: train, test, gender_submission.
Under your src folder, create a folder called data and put the three csv files into it.

# Step 3 - Run the python container
Run the code in src directory
```bash
cd src
docker build -t titanic-app -f python_app/Dockerfile .
docker run --rm -v ${PWD}:/workspace -w /workspace titanic-app python python_app/app.py
```
This will:
1. Load and clean the data
2. Train a logistic regression model on `train.csv`
3. Output training accuracy
4. Generate predictions on `test.csv`
5. Save predictions to `src/python_app/py_predictions.csv`

# Step 4 - Run the R container
Run the code in src directory
```bash
cd src
docker build -t titanic-app-r -f R_app/Dockerfile .
docker run --rm -v ${PWD}:/workspace -w /workspace titanic-app-r Rscript R_app/app.R
```
This will:
1. Load and clean the data
2. Train a logistic regression model on `train.csv`
3. Output training accuracy
4. Generate predictions on `test.csv`
5. Save predictions to `src/R_app/R_predictions.csv`

# Step 5 - Expected Outcome
After successful runs:
Python predictions: src/python_app/py_predictions.csv
R predictions: src/R_app/R_predictions.csv
You can open these files in any CSV viewer or load them back into R/Python for further analysis or comparison.

# Note on Model Design
Both models use logistic regression trained on: Pclass, Sex, Age, SibSp, Parch, Fare
Missing values are filled with median.