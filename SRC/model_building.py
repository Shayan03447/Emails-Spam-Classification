import pandas as pd
import logging 
import os
import pickle
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import yaml

# Ensure the 'log' directory exist
log_dir='log'
os.makedirs(log_dir, exist_ok=True)

# Logging configration 
logger=logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir, "model_building.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from yaml file"""
    try:
        with open(params_path, 'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Parameters retrived from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s ", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Yaml error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.debug("Data loaded from %s with shape %s",file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file %s", e)
        raise
    except FileNotFoundError as e:
        logger.error("File not found %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading the data :%s", e)
        raise

def train_model(X_train: np.ndarray, y_train:np.ndarray, params:dict) -> SVC:
    try:
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        logger.debug('Initializing RandomForest model with parameters: %s', params)
        clf=RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        #clf=LogisticRegression(C=params['C'],class_weight=params['class_weight'],solver=params['solver'],max_iter=params['max_iter'],random_state=params['random_state'])
        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)

        logger.debug("Model training complete")
        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise
def save_model(model, file_path: str) -> None:
    try:
        # Ensure the directory exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug("Model saved to %s", file_path)
    except FileNotFoundError as e:
        logger.error("File path not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error occured while saving the model: %s", e)
        raise
def main():
    try:
        params=load_params('params.yaml')['model_building']
        params={'n_estimators':25 , 'random_state': 2}
        train_data=load_data('./data/processed/train_tfidf.csv')
        X_train=train_data.iloc[:, :-1].values
        y_train=train_data.iloc[:, -1].values
        clf=train_model(X_train, y_train, params)

        model_save_path='models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error("Failed to complete the model building_process: %s", e)
        print(f"Error: {e}")
if __name__=='__main__':
    main()

