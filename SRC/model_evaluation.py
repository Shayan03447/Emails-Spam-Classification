import os
import pandas as pd
import numpy as np
import logging
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import yaml
from dvclive import Live
 
# Ensure  the 'log' directory exist
log_dir='log'
os.makedirs(log_dir, exist_ok=True)

# Logging Configration 
logger=logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir, "model_evaluation.py")
file_hanler=logging.FileHandler(log_file_path)
file_hanler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_hanler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_hanler)

def load_params(params_path: str) -> dict:
    """Load parameters from the yaml file"""
    try:
        with open(params_path, 'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Parameters retrived from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Yaml error: %s", e)
        raise
    except Exception as e:
        logger.error("UNexpected error: %s", e)
        raise

def load_model(file_path: str):
    """Load the trained model from a file """
    try:
        with open(file_path, 'rb') as file:
            model=pickle.load(file)
        logger.debug("Model loading from: %s", file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s',file_path)
        raise
    except Exception as e:
        logger.error("unexpected error occur while loading the model: %s", e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from the csv file"""
    try:
        df=pd.read_csv(file_path)
        logger.debug("Data loaded from: %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file: %s", e)
        raise
    except Exception as e:
        logger.debug("Unexpected error while loading the data: %s", e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test:np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrix"""
    try:
        y_pred=clf.predict(X_test)
        y_pred_proba=clf.predict_proba(X_test)[:, 1]
        #y_pred = (y_pred_proba >= 0.5).astype(int)

        accuracy=accuracy_score(y_test, y_pred)
        precision= precision_score(y_test, y_pred)#zero_division=0
        recall=recall_score(y_test, y_pred)
        auc= roc_auc_score(y_test, y_pred_proba)
        metrics_dict={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc': auc        
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error("Error during model evaluation ")
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaliation metrics to a json file"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open (file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to: %s', file_path)
    except Exception as e:
        logger.error('Error occur while saving the metrics: %s', e)
        raise

def main():
    try:
        params=load_params(params_path='params.yaml')
        clf=load_model('./models/model.pkl')
        test_data=load_data('./data/processed/test_tfidf.csv')

        X_test=test_data.iloc[:, :-1].values
        y_test=test_data.iloc[:, -1].values

        metrics=evaluate_model(clf, X_test,y_test)

        # Experiment tracking using dvclive
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))
            live.log_params(params)
        

        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f'Error:{e}')

        
if __name__=='__main__':
    main()