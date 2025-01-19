import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, param):
        """
        Evaluate multiple models using GridSearchCV and calculate the R2 score.

        Parameters:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        X_test (np.ndarray): Testing feature data.
        y_test (np.ndarray): Testing target data.
        models (dict): A dictionary of model names and instances.
        param (dict): A dictionary of model names and hyperparameters.

        Returns:
        dict: A dictionary with model names as keys and their R2 scores as values.
        """
        model_report = {}

        for model_name, model in models.items():
            try:
                logging.info(f"Training model: {model_name}")
                params = param.get(model_name, {})
                gs = GridSearchCV(estimator=model, param_grid=params, scoring='r2', cv=5)
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                logging.info(f"Best parameters for {model_name}: {gs.best_params_}")

                y_test_pred = best_model.predict(X_test)
                test_model_r2_score = r2_score(y_test, y_test_pred)

                model_report[model_name] = test_model_r2_score
                logging.info(f"{model_name} R2 score: {test_model_r2_score}")
            except Exception as e:
                logging.warning(f"Error evaluating model {model_name}: {e}")
                model_report[model_name] = -float("inf")

        return model_report

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
            }

            model_report = self.evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found.")

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}.")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            print(f"Best Model Found: {best_model_name}, R2 Score: {r2_square}")
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
