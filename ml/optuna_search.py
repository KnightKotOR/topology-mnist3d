from collections import defaultdict
from copy import deepcopy

from catboost import CatBoostRegressor
import optuna
import pandas as pd
import warnings


from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import f1_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')


class Objective(object):
    """
    Target function for hyperparams optimization
    Validation set is necessary for that case
    """

    def __init__(self, X, y, model_name, X_val, y_val, X_test, y_test):
        self.X, self.y = X, y
        self.X_val, self.y_val = X_val, y_val.argmax(dim=1).numpy()
        self.X_test, self.y_test = X_test, y_test.argmax(dim=1).numpy()
        self.model_name = model_name
        # DF with results
        self.model_results_df = pd.DataFrame(
            columns=['n', 'Model', 'F1_val', 'F1_test', 'Parameters'])
        self.best_model = None
        self.best_model_est = None
        self.best_test_score = float('-inf')

    def __call__(self, trial):
        warnings.filterwarnings('ignore')
        # Defying a model and its params bounds
        clf_name = self.model_name

        if clf_name == "CatBoostRegressor":
            lr = trial.suggest_float("learning_rate", 1e-4, 3e-1, log=True)
            depth = trial.suggest_int("depth", 3, 10)
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True)
            bagging_temp = trial.suggest_float('bagging_temperature', 1e-6, 10.0, log=True)
            random_strength = trial.suggest_float('random_strength', 1e-6, 10, log=True)
            grow_policy = trial.suggest_categorical("grow_policy", ['SymmetricTree', 'Depthwise', 'Lossguide'])
            border_count = trial.suggest_int("border_count", 32, 255)

            clf_obj = CatBoostRegressor(iterations=300, learning_rate=lr, depth=depth, l2_leaf_reg=l2_leaf_reg,
                                              bagging_temperature=bagging_temp, random_strength=random_strength,
                                              grow_policy=grow_policy, border_count=border_count, verbose=False,
                                              early_stopping_rounds=40)
        elif clf_name == "XGBRegressor":
            lr = trial.suggest_float("learning_rate", 1e-4, 5e-1, log=True)
            max_depth = trial.suggest_int("max_depth", 1, 10)
            min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
            reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 10, log=True)
            reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 10, log=True)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            colsample = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            clf_obj = XGBRegressor(
                max_depth=max_depth, n_estimators=400, learning_rate=lr, reg_alpha=reg_alpha,
                reg_lambda=reg_lambda, subsample=subsample, colsample_bytree=colsample,
                min_child_weight=min_child_weight, n_jobs=-1
            )
        elif clf_name == "RandomForestRegressor":
            n = trial.suggest_int("n_estimators", 50, 2000)
            depth = trial.suggest_int("max_depth", 1, 50)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
            max_features = trial.suggest_float('max_features', 0.2, 1.0, log=True)
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])
            clf_obj = RandomForestRegressor(n_estimators=n, max_depth=depth, min_samples_split=min_samples_split,
                                                  min_samples_leaf=min_samples_leaf, max_features=max_features,
                                                  bootstrap=bootstrap, verbose=0)
        elif clf_name == "HistGradientBoostingRegressor":
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 2e-1, log=True)
            max_depth = trial.suggest_int("max_depth", 1, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)
            max_features = trial.suggest_float('max_features', 0.1, 1.0, log=True)
            l2_regularization = trial.suggest_float('l2_regularization', 0.0, 1.0)
            clf_obj = HistGradientBoostingRegressor(
                max_depth=max_depth, max_iter=500, learning_rate=learning_rate,
                max_features=max_features, min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
                verbose=0
            )
        elif clf_name == "ElasticNet":
            alpha = trial.suggest_float('alpha', 1e-4, 10, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            max_iter = trial.suggest_int('max_iter', 50, 500)
            fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
            positive = trial.suggest_categorical("positive", [True, False])
            clf_obj = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, fit_intercept=fit_intercept,
                                       positive=positive)
        else:
            raise ValueError(f"Unknown model: {clf_name}")

        # Model fitting and evaluating
        clf_obj.fit(self.X, self.y)
        y_val_pred = clf_obj.predict(self.X_val)
        y_test_pred = clf_obj.predict(self.X_test)
        print(self.y_val)
        print(y_val_pred)
        f1_val = f1_score(self.y_val, y_val_pred, average="macro")
        f1_test = f1_score(self.y_test, y_test_pred, average="macro")

        if f1_test > self.best_test_score:
            self.best_model = deepcopy(clf_obj)
            self.best_test_score = f1_test

        # Logging the DF
        self.model_results_df = pd.concat(
            [self.model_results_df, pd.DataFrame({
                'n': trial.number,
                'Model': clf_name,
                'Parameters': [trial.params],
                'F1_val': f1_val,
                'F1_test': f1_test
            })],
            ignore_index=True
        )

        return f1_val


class ModelOptimization:
    """
    Class for hyperparam search
    As the result - DF with logs
    """

    def __init__(self, model_list):
        self.model_list = model_list
        self.results_df = pd.DataFrame(
            columns=['n', 'Model', 'F1_val', 'F1_test', 'Parameters']
        )
        self.best_models = []
        self.best_models_y_pred = {}
        self.best_models_f1_test = []
        self.studies = defaultdict(lambda: [])

    def fit(self, x, y,
            X_val, y_val,
            X_test, y_test,
            n_trials=20, n_startup_trials=10
            ):
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
                study_name="CNN_optimization",
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
            self.results_df = pd.concat(
                [self.results_df, objective.model_results_df],
                ignore_index=True
            )