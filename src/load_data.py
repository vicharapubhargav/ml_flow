# read the data from data source
# save it in the data/raw for further process
import os
from get_data import read_params, get_data
import argparse
from log_class import getLog

log = getLog("load_data.py")

def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    new_cols = [col.replace(" ", "_") for col in df.columns]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df.columns=new_cols
    df.to_csv(raw_data_path,index=False)
    log.info("Given Dataset File file is saved in data/raw folders without spaces in headers")


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)    