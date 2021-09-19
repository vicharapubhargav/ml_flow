# create environment
 cmd:conda create -n mlflow python=3.7 -y
# activate environment 
 cmd: conda activate

# Install the requirements
 cmd: pip install -r requirements.txt

# Initialize git and dvc
 cmd: git init
 cmd: dvc init
 
# Create Metadata for the dataset
 cmd: dvc add data_given/winequality.csv

# Commit the changes to git
 cmd: git add .
 
 cmd: git commit -m "First Commit" 
 
 cmd: git remote add origin https://github.com/vicharapubhargav/dvc_task1
 
 cmd: git push origin main

create an artifcats folder
 cmd : touch artifacts

mlflow server command -

mlflow server
--backend-store-uri sqlite:///mlflow.db
--default-artifact-root ./artifacts
--host 0.0.0.0 -p 1234
