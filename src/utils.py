import numpy as np
import pandas as pd
import os
import sys
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            para = param[model_name]
            
            # Skip hyperparameter tuning if parameter grid is empty
            if not para:
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
            else:
                try:
                    gs = GridSearchCV(
                        estimator=model,
                        param_grid=para,
                        cv=3,
                        n_jobs=-1,
                        error_score='raise'
                    )
                    gs.fit(X_train, y_train)
                    model = gs.best_estimator_
                    
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                except Exception as e:
                    # If GridSearchCV fails, try fitting the model directly
                    print(f"GridSearchCV failed for {model_name}, fitting model directly. Error: {str(e)}")
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            print(f"\n{model_name}:")
            print(f"Train score: {train_model_score}")
            print(f"Test score: {test_model_score}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)