# load the train and test
# train algo
# save the metrices, params
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from get_data import read_params
import argparse
import joblib
import json
from sklearn.linear_model import ElasticNet,ElasticNetCV
from log_class import getLog
import mlflow
from urllib.parse import urlparse

log = getLog("model_build.py")


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    model_dir = config["model_dir"]
    

    
    cv = config["estimators"]["ElasticNet"]["params"]["cv"]
    norm =config["estimators"]["ElasticNet"]["params"]["normalize"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    y_train = np.array(train[target])
    y_train.reshape(-1,)
    y_train=y_train.ravel()

    y_test =  np.array(test[target])
    y_test.reshape(-1,)
    y_test=y_test.ravel()

    x_train = train.drop(target, axis=1)
    x_test = test.drop(target, axis=1)
    

    mlflow_config =config["mlflow_config"]
    remote_server_uri=mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        elasticNetCV=ElasticNetCV(alphas=None,cv=cv,normalize=norm)
        elasticNetCV.fit(x_train,y_train)

        lr_model=ElasticNet(alpha=elasticNetCV.alpha_,l1_ratio=elasticNetCV.l1_ratio_)
        lr_model.fit(x_train,y_train)

        predicted_qualities = lr_model.predict(x_test)
        
        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)
        

        mlflow.log_param("CV",cv)
        mlflow.log_param("Normalize",norm)

        mlflow.log_metric("alpha",elasticNetCV.alpha_)
        mlflow.log_metric("l1_ratio", elasticNetCV.l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr_model, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(lr_model, "model")
   
    

   
    

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)