import os 
import sys 
from dataclasses import dataclass
from xgboost import XGBRegressor 
import optuna

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from src.exception import FileOperationError
from src.log_config import logging 

from src.utils import save_object, evaluate_models


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("data_repository", "model.joblib")

class ModelTrainer:
    # Initialize model trainer object
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    # Method to initiate model training 
    def initiate_model_trainer(self, training_array, testing_array):
        try:
            # Log message of splitting training and test input data 
            logging.info("Splitting training and test input data")

            # Split train and test arrays and ito features and target variable 
            X_train, y_train, X_test, y_test = (
                training_array[:, :-1],
                training_array[:, -1],
                testing_array[:, :-1],
                testing_array[:, -1],
            )

            # log info to start hyperparameter tuning with optuna
            logging.info("Starting hyperparameter tuning with optuna")

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
                }

                # Initialize the XGBRegressor model with current hyperparameters
                xgb_model = XGBRegressor(**params)

                # Fit the model on the training data
                xgb_model.fit(X_train, y_train)

                # Make predictions on the testing data
                y_pred = xgb_model.predict(X_test)

                # Calculate R-squared score
                r2 = r2_score(y_test, y_pred)

                # Return the R-squared score for optimization
                return r2

            # Create an Optuna study
            study = optuna.create_study(direction='maximize')
            
            # Optimize the objective function (hyperparameter tuning)
            study.optimize(objective, n_trials=100, show_progress_bar=True)

            # Get the best hyperparameters from the study
            best_params = study.best_params

            # Initialize the XGBRegressor model with best hyperparameters
            best_model = XGBRegressor(**best_params)

            # Fit the best model on the entire training data
            best_model.fit(X_train, y_train)

            # Make predictions using the best model
            y_pred = best_model.predict(X_test)

            # Calculate R-squared score for the best model
            model_score = r2_score(y_test, y_pred)
            logging.info(f"Model score: {model_score}")

            # Check if model score is less than 0.7
            if model_score < 0.7:
                raise FileOperationError("Model Score is less than 0.7")

            logging.info("saving model")
            # Save the best model
            save_object(file_path=self.config.trained_model_file_path,
                            obj=best_model)


            return model_score

              
        # Handle exceptions            
        except Exception as e:
            raise FileOperationError(e, sys) 
                    
                