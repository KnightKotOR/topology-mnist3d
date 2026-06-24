from collections import defaultdict
from copy import deepcopy
from typing import Self

import optuna
import pandas as pd
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


class Objective(object):
    """
    Target function for hyperparams optimization
    Validation set is necessary for that case
    """

    def __init__(self, X, y, model_name, X_val, y_val, X_test, y_test):
        self.X, self.y = X, y
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        self.model_name = model_name
        self.best_model = None
        self.best_val_score = float('-inf')
        
        self.n = []
        self.model_name_list = []
        self.F1_val_list = []
        self.F1_test_list = []
        self.params = []

    def __call__(self, trial):
        warnings.filterwarnings('ignore')
        # Defying a model and its params bounds
        clf_name = self.model_name

        if clf_name == "XGBClassifier":
            lr = trial.suggest_float("learning_rate", 1e-4, 5e-1, log=True)
            max_depth = trial.suggest_int("max_depth", 1, 20)
            min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
            reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 10, log=True)
            reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 10, log=True)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            colsample = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            clf_obj = XGBClassifier(
                max_depth=max_depth, n_estimators=400, learning_rate=lr, reg_alpha=reg_alpha,
                reg_lambda=reg_lambda, subsample=subsample, colsample_bytree=colsample,
                min_child_weight=min_child_weight, n_jobs=-1
            )
        elif clf_name == "RandomForestClassifier":
            n = trial.suggest_int("n_estimators", 50, 2000)
            depth = trial.suggest_int("max_depth", 1, 50)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
            max_features = trial.suggest_float('max_features', 0.2, 1.0, log=True)
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])
            clf_obj = RandomForestClassifier(n_estimators=n, max_depth=depth, min_samples_split=min_samples_split,
                                                  min_samples_leaf=min_samples_leaf, max_features=max_features,
                                                  bootstrap=bootstrap, verbose=0)
        else:
            raise ValueError(f"Unknown model: {clf_name}")

        # Model fitting and evaluating
        clf_obj.fit(self.X, self.y)
        y_val_pred = clf_obj.predict(self.X_val)
        y_test_pred = clf_obj.predict(self.X_test)
        f1_val = f1_score(self.y_val, y_val_pred, average="macro")
        f1_test = f1_score(self.y_test, y_test_pred, average="macro")

        if f1_val > self.best_test_score:
            self.best_model = deepcopy(clf_obj)
            self.best_val_score = f1_val

        # Logging the results
        self.n.append(trial.number)
        self.model_name_list.append(clf_name)
        self.params.append(trial.params)
        self.F1_val_list.append(f1_val)
        self.F1_test_list.append(f1_test)

        return f1_val
    
    def get_results(self) -> pd.DataFrame:
        return pd.DataFrame({
            'n': self.n, 
            'Model': self.model_name_list, 
            'F1_val': self.F1_val_list, 
            'F1_test': self.F1_test_list,
			'Parameters': self.params}) 


class ModelOptimization:
    """
    Class for hyperparam search
    As the result - DF with logs
    """

    def __init__(self, model_list):
        self.model_list = model_list
        self.best_models = []
        self.best_models_y_pred = {}
        self.best_models_f1_test = []
        self.studies = defaultdict(lambda: [])

    def fit(self, x, y,
            X_val, y_val,
            X_test, y_test,
            n_trials=20, n_startup_trials=10
            ) -> Self:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        for model in self.model_list:
            print(f"\n{model} hyperoptimization")

            sampler = optuna.samplers.TPESampler(
                multivariate=True,
                n_startup_trials=n_startup_trials
            )

            objective = Objective(
                x, y, model,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test
            )
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                study_name="XG_cub_4000clouds",
                load_if_exists=True,
                storage="sqlite:///db.sqlite3"
            )
            study.set_metric_names(["F1_val"])
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=True
            )
            self.best_models.append(objective.best_model)
            self.best_models_f1_test.append(objective.best_test_score)
            self.results_df = objective.get_results()
