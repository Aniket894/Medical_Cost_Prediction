import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import uniform, randint
import pickle

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression': {
                    'model': LinearRegression(),
                    'params': {}
                },
                'Lasso': {
                    'model': Lasso(random_state=42),
                    'params': {
                        'alpha': uniform(0.01, 100)
                    }
                },
                'Ridge': {
                    'model': Ridge(random_state=42),
                    'params': {
                        'alpha': uniform(0.01, 100)
                    }
                },
                'ElasticNet': {
                    'model': ElasticNet(random_state=42),
                    'params': {
                        'alpha': uniform(0.01, 100),
                        'l1_ratio': uniform(0.1, 0.9)
                    }
                },
                'DecisionTreeRegressor': {
                    'model': DecisionTreeRegressor(random_state=42),
                    'params': {
                        'max_depth': [None, 3, 5, 7, 10],
                        'min_samples_split': randint(2, 11),
                        'min_samples_leaf': randint(1, 5)
                    }
                },
                'RandomForestRegressor': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': randint(50, 201),
                        'max_depth': [None, 3, 5, 7, 10],
                        'min_samples_split': randint(2, 11),
                        'min_samples_leaf': randint(1, 5)
                    }
                },
                'AdaBoostRegressor': {
                    'model': AdaBoostRegressor(random_state=42),
                    'params': {
                        'n_estimators': randint(50, 201),
                        'learning_rate': uniform(0.01, 1.0)
                    }
                },
                'GradientBoostingRegressor': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': randint(50, 201),
                        'learning_rate': uniform(0.01, 0.2),
                        'max_depth': randint(3, 8)
                    }
                },
                'XGBRegressor': {
                    'model': XGBRegressor(random_state=42),
                    'params': {
                        'n_estimators': randint(50, 201),
                        'learning_rate': uniform(0.01, 0.2),
                        'max_depth': randint(3, 8)
                    }
                },
                'KNeighborsRegressor': {
                    'model': KNeighborsRegressor(),
                    'params': {
                        'n_neighbors': randint(3, 10),
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan']
                    }
                }
            }

            def adjusted_r2_score(r2, n, k):
                return 1 - (1 - r2) * (n - 1) / (n - k - 1)

            def evaluate_model(model, params, X_train, y_train, X_test, y_test):
                grid_search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                n = X_test.shape[0]
                k = X_test.shape[1]
                adj_r2 = adjusted_r2_score(r2, n, k)

                mse = mean_squared_error(y_test, y_pred)

                print(f"Model: {best_model}")
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Mean Squared Error: {mse}")
                print(f"R^2 Score: {r2}")
                print(f"Adjusted R^2 Score: {adj_r2}")
                print("=======================================")

                return best_model, adj_r2

            adjusted_r2_scores = {}
            all_models = {}

            for model_name, model_info in models.items():
                print(f"Evaluating and Training {model_name}...")
                best_model, adj_r2 = evaluate_model(model_info['model'], model_info['params'], X_train, y_train, X_test, y_test)
                adjusted_r2_scores[model_name] = adj_r2
                all_models[model_name] = best_model

                model_file_path = os.path.join('artifacts', f"{model_name}.pkl")
                with open(model_file_path, 'wb') as file:
                    pickle.dump(best_model, file)
                print(f"Model {model_name} saved to {model_file_path}")

            best_model_name = max(adjusted_r2_scores, key=adjusted_r2_scores.get)
            best_model = all_models[best_model_name]
            best_model_score = adjusted_r2_scores[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}, Adjusted R2 Score: {best_model_score}')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, Adjusted R2 Score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)
