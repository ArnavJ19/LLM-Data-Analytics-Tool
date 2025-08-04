import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_absolute_percentage_error, log_loss, classification_report
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif, RFE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
import joblib
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictiveAnalytics:
    """
    Handles predictive analytics capabilities including model training,
    evaluation, and forecasting.
    """
    
    def __init__(self):
        """Initialize the PredictiveAnalytics module with expanded model options."""
        self.supported_regression_models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(random_state=42),
            'lasso_regression': Lasso(random_state=42),
            'elastic_net': ElasticNet(random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': LGBMRegressor(n_estimators=100, random_state=42),
            'svr': SVR(),
            'knn_regressor': KNeighborsRegressor(n_neighbors=5),
            'extra_trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'adaboost': AdaBoostRegressor(random_state=42),
            'mlp_regressor': MLPRegressor(max_iter=1000, random_state=42)
        }
        
        self.supported_classification_models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgboost': XGBClassifier(n_estimators=100, random_state=42),
            'lightgbm': LGBMClassifier(n_estimators=100, random_state=42),
            'svc': SVC(probability=True, random_state=42),
            'knn_classifier': KNeighborsClassifier(n_neighbors=5),
            'extra_trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
            'adaboost': AdaBoostClassifier(random_state=42),
            'mlp_classifier': MLPClassifier(max_iter=1000, random_state=42)
        }
        
        # Advanced preprocessing options
        self.supported_scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'power_transform': PowerTransformer()
        }
        
        self.supported_imputers = {
            'mean': SimpleImputer(strategy='mean'),
            'median': SimpleImputer(strategy='median'),
            'most_frequent': SimpleImputer(strategy='most_frequent'),
            'knn': KNNImputer(n_neighbors=5)
        }
        
        # Feature selection methods
        self.feature_selection_methods = {
            'none': None,
            'select_k_best': SelectKBest(),
            'rfe': RFE(estimator=LinearRegression())
        }
        
        # Dimensionality reduction methods
        self.dim_reduction_methods = {
            'none': None,
            'pca': PCA()
        }
        
        # Hyperparameter tuning methods
        self.tuning_methods = {
            'none': None,
            'grid_search': GridSearchCV,
            'random_search': RandomizedSearchCV
        }
        
        self.last_model = None
        self.last_pipeline = None
        self.feature_importance = None
        self.model_metrics = None
        self.prediction_results = None
        self.target_column = None
        self.feature_columns = None
        self.model_type = None
        self.shap_values = None
        self.model_file_path = None
        self.feature_clusters = None
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    feature_columns: Optional[List[str]] = None,
                    test_size: float = 0.2, 
                    time_series: bool = False,
                    datetime_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for predictive modeling by splitting into train/test sets.
        
        Args:
            df: Input DataFrame
            target_column: Column to predict
            feature_columns: Columns to use as features (if None, use all except target)
            test_size: Proportion of data to use for testing
            time_series: Whether to use time-based splitting for time series data
            datetime_column: Column containing datetime information for time series
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Store target and feature columns for later use
        self.target_column = target_column
        
        # Determine feature columns if not provided
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        self.feature_columns = feature_columns
        
        # Extract features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Determine if regression or classification
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            self.model_type = 'regression'
        else:
            self.model_type = 'classification'
            # Convert target to categorical if it's not already
            if not pd.api.types.is_categorical_dtype(y):
                y = y.astype('category')
        
        # Split data
        if time_series and datetime_column is not None:
            # Sort by datetime for time series
            df_sorted = df.sort_values(datetime_column)
            X = df_sorted[feature_columns].copy()
            y = df_sorted[target_column].copy()
            
            # Use time series split
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            # Regular random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        return X_train, X_test, y_train, y_test
    
    def create_preprocessing_pipeline(self, X_train: pd.DataFrame, 
                                  scaler_type: str = 'standard',
                                  imputer_type: str = 'median',
                                  feature_selection: str = 'none',
                                  dim_reduction: str = 'none',
                                  n_features_to_select: int = 10,
                                  n_components: int = 5) -> ColumnTransformer:
        """
        Create an advanced preprocessing pipeline with customizable options.
        
        Args:
            X_train: Training features
            scaler_type: Type of scaler to use ('standard', 'minmax', 'robust', 'power_transform')
            imputer_type: Type of imputer to use ('mean', 'median', 'most_frequent', 'knn')
            feature_selection: Feature selection method ('none', 'select_k_best', 'rfe')
            dim_reduction: Dimensionality reduction method ('none', 'pca')
            n_features_to_select: Number of features to select if using feature selection
            n_components: Number of components for PCA if using dim_reduction
            
        Returns:
            Preprocessing pipeline
        """
        # Identify column types
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get the specified scaler and imputer
        scaler = self.supported_scalers.get(scaler_type, StandardScaler())
        imputer = self.supported_imputers.get(imputer_type, SimpleImputer(strategy='median'))
        
        # Create transformers
        transformers = []
        
        if numeric_features:
            # Build numeric pipeline with optional feature selection and dim reduction
            numeric_steps = [('imputer', imputer), ('scaler', scaler)]
            
            # Add feature selection if requested
            if feature_selection != 'none' and self.feature_selection_methods[feature_selection] is not None:
                if feature_selection == 'select_k_best':
                    if self.model_type == 'regression':
                        selector = SelectKBest(score_func=f_regression, k=min(n_features_to_select, len(numeric_features)))
                    else:
                        selector = SelectKBest(score_func=f_classif, k=min(n_features_to_select, len(numeric_features)))
                    numeric_steps.append(('feature_selection', selector))
                elif feature_selection == 'rfe':
                    if self.model_type == 'regression':
                        selector = RFE(estimator=LinearRegression(), n_features_to_select=min(n_features_to_select, len(numeric_features)))
                    else:
                        selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=min(n_features_to_select, len(numeric_features)))
                    numeric_steps.append(('feature_selection', selector))
            
            # Add dimensionality reduction if requested
            if dim_reduction != 'none' and self.dim_reduction_methods[dim_reduction] is not None:
                if dim_reduction == 'pca':
                    n_components = min(n_components, len(numeric_features))
                    reducer = PCA(n_components=n_components)
                    numeric_steps.append(('dim_reduction', reducer))
            
            numeric_transformer = Pipeline(steps=numeric_steps)
            transformers.append(('numeric', numeric_transformer, numeric_features))
        
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('categorical', categorical_transformer, categorical_features))
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(transformers=transformers)
        
        return preprocessor
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   model_type: Optional[str] = None,
                   model_name: str = 'auto',
                   hyperparameter_tuning: str = 'none',
                   n_iter: int = 10,
                   cv_folds: int = 5,
                   scaler_type: str = 'standard',
                   imputer_type: str = 'median',
                   feature_selection: str = 'none',
                   dim_reduction: str = 'none',
                   n_features_to_select: int = 10,
                   n_components: int = 5,
                   save_model: bool = False,
                   model_file_path: str = None) -> Pipeline:
        """
        Train a predictive model with advanced options.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: 'regression' or 'classification' (if None, auto-detect)
            model_name: Name of model to use, or 'auto' to select best
            hyperparameter_tuning: Type of tuning ('none', 'grid', 'random')
            n_iter: Number of iterations for random search
            cv_folds: Number of cross-validation folds
            scaler_type: Type of scaler to use
            imputer_type: Type of imputer to use
            feature_selection: Feature selection method
            dim_reduction: Dimensionality reduction method
            n_features_to_select: Number of features to select
            n_components: Number of components for dimensionality reduction
            save_model: Whether to save the trained model to disk
            model_file_path: Path to save the model file
            
        Returns:
            Trained model pipeline
        """
        # Determine model type if not provided
        if model_type is None:
            model_type = self.model_type
        
        # Create preprocessing pipeline with advanced options
        preprocessor = self.create_preprocessing_pipeline(
            X_train, 
            scaler_type=scaler_type,
            imputer_type=imputer_type,
            feature_selection=feature_selection,
            dim_reduction=dim_reduction,
            n_features_to_select=n_features_to_select,
            n_components=n_components
        )
        
        # Select model
        if model_type == 'regression':
            models = self.supported_regression_models
        else:
            models = self.supported_classification_models
        
        # Auto-select model if requested
        if model_name == 'auto':
            best_score = -float('inf')
            best_model_name = None
            
            logger.info(f"Auto-selecting best model from {len(models)} candidates...")
            for name, model in models.items():
                try:
                    # Create pipeline with current model
                    pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    
                    # Evaluate with cross-validation
                    if model_type == 'regression':
                        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
                        avg_score = np.mean(scores)
                        logger.info(f"  {name}: MSE = {-avg_score:.4f}")
                    else:
                        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='accuracy')
                        avg_score = np.mean(scores)
                        logger.info(f"  {name}: Accuracy = {avg_score:.4f}")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model_name = name
                except Exception as e:
                    logger.warning(f"Error evaluating model {name}: {str(e)}")
            
            model_name = best_model_name or list(models.keys())[0]  # Fallback to first model if all fail
            logger.info(f"Auto-selected model: {model_name} with {'MSE' if model_type == 'regression' else 'accuracy'} score: {best_score:.4f}")
        
        # Create model instance
        final_model = models[model_name]
        
        # Apply hyperparameter tuning if requested
        if hyperparameter_tuning != 'none':
            # Define parameter grid based on model type
            param_grid = self._get_hyperparameter_grid(model_name, model_type)
            
            if param_grid:
                logger.info(f"Performing {hyperparameter_tuning} search for hyperparameter tuning...")
                
                # Create base pipeline for tuning
                base_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', final_model)
                ])
                
                # Choose scoring metric based on model type
                scoring = 'neg_mean_squared_error' if model_type == 'regression' else 'accuracy'
                
                # Perform hyperparameter tuning
                if hyperparameter_tuning == 'grid':
                    search = GridSearchCV(base_pipeline, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1)
                else:  # random search
                    search = RandomizedSearchCV(base_pipeline, param_grid, n_iter=n_iter, cv=cv_folds, scoring=scoring, n_jobs=-1, random_state=42)
                
                # Fit the search
                search.fit(X_train, y_train)
                
                # Get best pipeline
                pipeline = search.best_estimator_
                final_model = pipeline.named_steps['model']
                
                logger.info(f"Best parameters: {search.best_params_}")
                logger.info(f"Best score: {search.best_score_:.4f}")
            else:
                logger.warning(f"No hyperparameter grid available for {model_name}. Skipping tuning.")
                # Create and train the pipeline without tuning
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', final_model)
                ])
                pipeline.fit(X_train, y_train)
        else:
            # Create and train final pipeline without tuning
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', final_model)
            ])
            pipeline.fit(X_train, y_train)
        
        # Store model for later use
        self.last_model = final_model
        self.last_pipeline = pipeline
        
        # Calculate feature importance if available
        self.calculate_feature_importance(pipeline, X_train.columns)
        
        # Calculate SHAP values for model interpretability if applicable
        self._calculate_shap_values(X_train)
        
        # Save model to disk if requested
        if save_model:
            if not model_file_path:
                # Generate a default filename based on model name and timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_file_path = f"model_{model_name}_{timestamp}.joblib"
            
            # Save the model using joblib
            joblib.dump(pipeline, model_file_path)
            self.model_file_path = model_file_path
            logger.info(f"Model saved to {model_file_path}")
        
        return pipeline
        
    def _get_hyperparameter_grid(self, model_name: str, model_type: str) -> Dict:
        """Get hyperparameter grid for the specified model."""
        # Define hyperparameter grids for different models
        regression_grids = {
            'linear_regression': {
                'model__fit_intercept': [True, False],
                'model__normalize': [True, False]
            },
            'ridge_regression': {
                'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
            },
            'lasso_regression': {
                'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                'model__selection': ['cyclic', 'random']
            },
            'elastic_net': {
                'model__alpha': [0.01, 0.1, 1.0, 10.0],
                'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'random_forest': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.8, 0.9, 1.0]
            },
            'svr': {
                'model__C': [0.1, 1, 10, 100],
                'model__kernel': ['linear', 'rbf', 'poly'],
                'model__gamma': ['scale', 'auto', 0.1, 1]
            },
            'knn_regressor': {
                'model__n_neighbors': [3, 5, 7, 9, 11],
                'model__weights': ['uniform', 'distance'],
                'model__p': [1, 2]
            }
        }
        
        classification_grids = {
            'logistic_regression': {
                'model__C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'model__solver': ['liblinear', 'saga'],
                'model__penalty': ['l1', 'l2']
            },
            'random_forest': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__criterion': ['gini', 'entropy']
            },
            'gradient_boosting': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.8, 0.9, 1.0]
            },
            'svc': {
                'model__C': [0.1, 1, 10, 100],
                'model__kernel': ['linear', 'rbf', 'poly'],
                'model__gamma': ['scale', 'auto', 0.1, 1],
                'model__probability': [True]
            },
            'knn_classifier': {
                'model__n_neighbors': [3, 5, 7, 9, 11],
                'model__weights': ['uniform', 'distance'],
                'model__p': [1, 2]
            }
        }
        
        if model_type == 'regression':
            return regression_grids.get(model_name, {})
        else:
            return classification_grids.get(model_name, {})
            
    def _calculate_shap_values(self, X: pd.DataFrame) -> None:
        """Calculate SHAP values for model interpretability."""
        try:
            # Only calculate SHAP values for tree-based models
            tree_based_models = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
            is_tree_based = any(model_type in self.last_pipeline.named_steps['model'].__class__.__name__.lower() 
                               for model_type in tree_based_models)
            
            if is_tree_based:
                logger.info("Calculating SHAP values for model interpretability...")
                # Preprocess the data using the pipeline's preprocessor
                X_processed = self.last_pipeline.named_steps['preprocessor'].transform(X)
                
                # Create a SHAP explainer
                explainer = shap.TreeExplainer(self.last_model)
                self.shap_values = explainer.shap_values(X_processed)
                logger.info("SHAP values calculated successfully.")
            else:
                self.shap_values = None
        except Exception as e:
            logger.warning(f"Error calculating SHAP values: {str(e)}")
            self.shap_values = None
    
    def calculate_feature_importance(self, pipeline: Pipeline, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """
        Calculate feature importance from the model if available.
        
        Args:
            pipeline: Trained model pipeline
            feature_names: Original feature names
            
        Returns:
            DataFrame with feature importances or None
        """
        model = pipeline.named_steps['model']
        
        # Try different methods to get feature importance based on model type
        try:
            # For tree-based models
            if hasattr(model, 'feature_importances_'):
                # Get the preprocessor
                preprocessor = pipeline.named_steps['preprocessor']
                
                # Get feature names after preprocessing
                if hasattr(preprocessor, 'get_feature_names_out'):
                    transformed_features = preprocessor.get_feature_names_out()
                else:
                    # Fallback if get_feature_names_out is not available
                    transformed_features = [f'feature_{i}' for i in range(len(model.feature_importances_))]
                
                # Create importance DataFrame
                importance = pd.DataFrame({
                    'feature': transformed_features,
                    'importance': model.feature_importances_
                })
                importance = importance.sort_values('importance', ascending=False)
                
                self.feature_importance = importance
                return importance
            
            # For linear models
            elif hasattr(model, 'coef_'):
                # Get the preprocessor
                preprocessor = pipeline.named_steps['preprocessor']
                
                # Get feature names after preprocessing
                if hasattr(preprocessor, 'get_feature_names_out'):
                    transformed_features = preprocessor.get_feature_names_out()
                else:
                    # Fallback if get_feature_names_out is not available
                    if model.coef_.ndim > 1:
                        transformed_features = [f'feature_{i}' for i in range(model.coef_.shape[1])]
                    else:
                        transformed_features = [f'feature_{i}' for i in range(len(model.coef_))]
                
                # Handle multi-class case
                if model.coef_.ndim > 1:
                    # For multi-class, take the mean absolute coefficient across classes
                    importance_values = np.mean(np.abs(model.coef_), axis=0)
                else:
                    importance_values = np.abs(model.coef_)
                
                # Create importance DataFrame
                importance = pd.DataFrame({
                    'feature': transformed_features,
                    'importance': importance_values
                })
                importance = importance.sort_values('importance', ascending=False)
                
                self.feature_importance = importance
                return importance
        
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {str(e)}")
        
        return None
    
    def evaluate_model(self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, 
                       detailed_report: bool = False) -> Dict[str, float]:
        """
        Evaluate model performance on test data with comprehensive metrics.
        
        Args:
            pipeline: Trained model pipeline
            X_test: Test features
            y_test: Test target
            detailed_report: Whether to generate a detailed classification report
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Store predictions
        self.prediction_results = {
            'actual': y_test,
            'predicted': y_pred
        }
        
        # Calculate metrics based on model type
        metrics = {}
        
        if self.model_type == 'regression':
            # Basic regression metrics
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_test, y_pred)
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            
            # Calculate MAPE if no zeros in actual values
            if not np.any(y_test == 0):
                metrics['mape'] = mean_absolute_percentage_error(y_test, y_pred) * 100
                
            # Additional regression metrics
            metrics['explained_variance'] = explained_variance_score(y_test, y_pred) if 'explained_variance_score' in globals() else None
            metrics['max_error'] = max_error(y_test, y_pred) if 'max_error' in globals() else None
            
            # Calculate residuals statistics
            residuals = y_test - y_pred
            metrics['residuals_mean'] = np.mean(residuals)
            metrics['residuals_std'] = np.std(residuals)
        else:
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred) if 'balanced_accuracy_score' in globals() else None
            
            # For binary classification
            if len(np.unique(y_test)) == 2:
                metrics['precision'] = precision_score(y_test, y_pred, average='binary')
                metrics['recall'] = recall_score(y_test, y_pred, average='binary')
                metrics['f1'] = f1_score(y_test, y_pred, average='binary')
                
                # Calculate confusion matrix elements
                cm = confusion_matrix(y_test, y_pred)
                if cm.size == 4:  # 2x2 matrix
                    tn, fp, fn, tp = cm.ravel()
                    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # ROC AUC and other probability-based metrics
                if hasattr(pipeline, 'predict_proba'):
                    try:
                        y_prob = pipeline.predict_proba(X_test)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
                        metrics['log_loss'] = log_loss(y_test, y_prob)
                        
                        # Store probability predictions for visualization
                        self.prediction_proba = y_prob
                        
                        # Calculate ROC curve points
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        self.roc_curve = {'fpr': fpr, 'tpr': tpr}
                    except Exception as e:
                        logger.warning(f"Error calculating probability-based metrics: {str(e)}")
            else:
                # For multi-class
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
                
                # Add macro and micro averages
                metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro')
                metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro')
                metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
                
                # Try to calculate multiclass ROC AUC
                if hasattr(pipeline, 'predict_proba'):
                    try:
                        y_prob = pipeline.predict_proba(X_test)
                        metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_prob, multi_class='ovr')
                    except Exception as e:
                        logger.warning(f"Error calculating multiclass ROC AUC: {str(e)}")
            
            # Generate detailed classification report if requested
            if detailed_report:
                self.classification_report = classification_report(y_test, y_pred, output_dict=True)
                # Don't add to metrics to keep it clean, access via self.classification_report
        
        # Store metrics
        self.model_metrics = metrics
        
        return metrics
    
    def create_evaluation_visualizations(self, 
                                    color_theme: str = 'blues',
                                    show_metrics_on_plots: bool = True,
                                    max_features_to_show: int = 20) -> Dict[str, go.Figure]:
        """
        Create advanced visualizations for model evaluation using Plotly.
        
        Args:
            color_theme: Color theme for plots ('blues', 'viridis', 'plasma', 'inferno', etc.)
            show_metrics_on_plots: Whether to show metrics on plots
            max_features_to_show: Maximum number of features to show in importance plots
            
        Returns:
            Dictionary of evaluation visualizations
        """
        if self.prediction_results is None:
            return {}
        
        visualizations = {}
        actual = self.prediction_results['actual']
        predicted = self.prediction_results['predicted']
        
        # Set color scales based on theme
        if color_theme == 'blues':
            continuous_scale = 'Blues'
            scatter_color = 'rgba(31, 119, 180, 0.7)'
            line_color = 'rgba(255, 0, 0, 0.7)'
            trend_color = 'rgba(44, 160, 44, 0.7)'
        elif color_theme == 'viridis':
            continuous_scale = 'Viridis'
            scatter_color = 'rgba(72, 143, 49, 0.7)'
            line_color = 'rgba(255, 0, 0, 0.7)'
            trend_color = 'rgba(44, 160, 44, 0.7)'
        else:  # default
            continuous_scale = 'Blues'
            scatter_color = 'rgba(31, 119, 180, 0.7)'
            line_color = 'rgba(255, 0, 0, 0.7)'
            trend_color = 'rgba(44, 160, 44, 0.7)'
        
        # Actual vs Predicted
        if self.model_type == 'regression':
            # Create scatter plot with trendline
            fig = px.scatter(
                x=actual, y=predicted,
                labels={'x': 'Actual', 'y': 'Predicted'},
                title='Actual vs Predicted Values',
                trendline='ols',  # Add OLS trendline
                trendline_color_override=trend_color
            )
            
            # Customize scatter points
            fig.update_traces(
                marker=dict(size=8, opacity=0.7, line=dict(width=1, color='white')),
                selector=dict(mode='markers')
            )
            
            # Add perfect prediction line
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color=line_color, dash='dash', width=2),
                name='Perfect Prediction'
            ))
            
            # Add metrics annotation if requested
            if show_metrics_on_plots and hasattr(self, 'model_metrics'):
                metrics_text = (
                    f"R² = {self.model_metrics.get('r2', 0):.3f}<br>"
                    f"RMSE = {self.model_metrics.get('rmse', 0):.3f}<br>"
                    f"MAE = {self.model_metrics.get('mae', 0):.3f}"
                )
                
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=metrics_text,
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(size=12)
                )
            
            # Improve layout
            fig.update_layout(
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            visualizations['actual_vs_predicted'] = fig
            
            # Residuals plot with LOWESS trendline
            residuals = actual - predicted
            
            # Create residuals dataframe for plotting
            residuals_df = pd.DataFrame({
                'predicted': predicted,
                'residuals': residuals
            })
            
            fig = px.scatter(
                residuals_df, x='predicted', y='residuals',
                labels={'predicted': 'Predicted Values', 'residuals': 'Residuals'},
                title='Residuals Analysis',
                trendline='lowess',  # Add LOWESS trendline
                trendline_color_override=trend_color
            )
            
            # Customize scatter points
            fig.update_traces(
                marker=dict(size=8, opacity=0.7, line=dict(width=1, color='white')),
                selector=dict(mode='markers')
            )
            
            # Add horizontal line at y=0
            fig.add_hline(
                y=0, 
                line_dash='dash', 
                line_color=line_color,
                line_width=2,
                annotation_text="Zero Residual",
                annotation_position="bottom right"
            )
            
            # Add residual statistics
            if show_metrics_on_plots:
                stats_text = (
                    f"Mean = {np.mean(residuals):.3f}<br>"
                    f"Std Dev = {np.std(residuals):.3f}<br>"
                    f"Min = {np.min(residuals):.3f}<br>"
                    f"Max = {np.max(residuals):.3f}"
                )
                
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=stats_text,
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(size=12)
                )
            
            # Improve layout
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            visualizations['residuals'] = fig
            
            # Residuals distribution with normal curve overlay
            fig = px.histogram(
                residuals,
                title='Residuals Distribution',
                labels={'value': 'Residuals', 'count': 'Frequency'},
                histnorm='probability density',  # Normalize for density curve
                color_discrete_sequence=[scatter_color]
            )
            
            # Add normal distribution curve
            x = np.linspace(min(residuals), max(residuals), 100)
            y = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color=trend_color, width=2),
                name='Normal Distribution'
            ))
            
            # Add vertical line at x=0
            fig.add_vline(
                x=0, 
                line_dash='dash', 
                line_color=line_color,
                line_width=2,
                annotation_text="Zero",
                annotation_position="top right"
            )
            
            # Improve layout
            fig.update_layout(
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            visualizations['residuals_distribution'] = fig
            
            # Q-Q Plot for normality check
            from scipy import stats
            qq_x, qq_y = stats.probplot(residuals, dist="norm", fit=False)
            
            fig = go.Figure()
            
            # Add scatter points
            fig.add_trace(go.Scatter(
                x=qq_x[0],
                y=qq_y[0],
                mode='markers',
                marker=dict(color=scatter_color, size=8),
                name='Residuals'
            ))
            
            # Add reference line
            min_val = min(qq_x[0].min(), qq_y[0].min())
            max_val = max(qq_x[0].max(), qq_y[0].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color=line_color, dash='dash'),
                name='Reference Line'
            ))
            
            # Update layout
            fig.update_layout(
                title='Q-Q Plot (Normality Check)',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles',
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            visualizations['qq_plot'] = fig
            
        else:  # Classification visualizations
            # Confusion matrix with normalized version
            cm = confusion_matrix(actual, predicted)
            labels = np.unique(np.concatenate([actual, predicted]))
            
            # Create subplots for raw and normalized confusion matrices
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Confusion Matrix (Counts)', 'Confusion Matrix (Normalized)'),
                horizontal_spacing=0.15
            )
            
            # Raw counts matrix
            heatmap1 = go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale=continuous_scale,
                showscale=False,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12},
            )
            
            # Normalized matrix (by row/actual class)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            heatmap2 = go.Heatmap(
                z=cm_norm,
                x=labels,
                y=labels,
                colorscale=continuous_scale,
                showscale=True,
                text=[[f"{val:.2f}" for val in row] for row in cm_norm],
                texttemplate="%{text}",
                textfont={"size": 12},
                colorbar=dict(title="Ratio", x=1.05)
            )
            
            fig.add_trace(heatmap1, row=1, col=1)
            fig.add_trace(heatmap2, row=1, col=2)
            
            # Update layout and axes
            fig.update_layout(
                title_text="Confusion Matrix Analysis",
                template="plotly_white",
                height=500,
                margin=dict(l=40, r=80, t=80, b=40)
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Predicted", row=1, col=1)
            fig.update_xaxes(title_text="Predicted", row=1, col=2)
            fig.update_yaxes(title_text="Actual", row=1, col=1)
            fig.update_yaxes(title_text="Actual", row=1, col=2)
            
            visualizations['confusion_matrix'] = fig
            
            # For binary classification, add ROC curve and Precision-Recall curve
            if len(labels) == 2 and hasattr(self.prediction_results, 'get') and self.prediction_results.get('probabilities') is not None:
                probs = self.prediction_results['probabilities']
                
                # Convert to binary format if needed
                if len(probs.shape) > 1 and probs.shape[1] > 1:
                    # Use the probability of the positive class (usually index 1)
                    probs = probs[:, 1]
                
                # ROC Curve
                fpr, tpr, thresholds = roc_curve(actual, probs)
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                
                # Add ROC curve
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    line=dict(color=scatter_color, width=2),
                    name=f'ROC Curve (AUC = {roc_auc:.3f})'
                ))
                
                # Add diagonal reference line
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(color=line_color, dash='dash'),
                    name='Random Classifier'
                ))
                
                # Update layout
                fig.update_layout(
                    title='Receiver Operating Characteristic (ROC) Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                # Set axis limits
                fig.update_xaxes(range=[-0.01, 1.01])
                fig.update_yaxes(range=[-0.01, 1.01])
                
                visualizations['roc_curve'] = fig
                
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(actual, probs)
                pr_auc = average_precision_score(actual, probs)
                
                fig = go.Figure()
                
                # Add PR curve
                fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    line=dict(color=scatter_color, width=2),
                    name=f'PR Curve (AP = {pr_auc:.3f})'
                ))
                
                # Add baseline reference
                baseline = np.sum(actual) / len(actual)  # Proportion of positive class
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[baseline, baseline],
                    mode='lines',
                    line=dict(color=line_color, dash='dash'),
                    name=f'Baseline (AP = {baseline:.3f})'
                ))
                
                # Update layout
                fig.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                # Set axis limits
                fig.update_xaxes(range=[-0.01, 1.01])
                fig.update_yaxes(range=[-0.01, 1.01])
                
                visualizations['pr_curve'] = fig
                
                # Threshold Analysis
                # Create a dataframe with various metrics at different thresholds
                thresholds_df = pd.DataFrame({
                    'threshold': thresholds,
                    'tpr': tpr[:-1],  # Exclude the last point as it doesn't correspond to a threshold
                    'fpr': fpr[:-1],
                    'precision': [precision_score(actual, probs >= t) if np.sum(probs >= t) > 0 else 0 for t in thresholds],
                    'f1': [f1_score(actual, probs >= t) if np.sum(probs >= t) > 0 else 0 for t in thresholds]
                })
                
                # Create a parallel coordinates plot
                fig = px.parallel_coordinates(
                    thresholds_df,
                    color="threshold",
                    labels={"threshold": "Threshold", "tpr": "True Positive Rate", 
                            "fpr": "False Positive Rate", "precision": "Precision", "f1": "F1 Score"},
                    color_continuous_scale=continuous_scale,
                    title="Threshold Analysis"
                )
                
                # Update layout
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                visualizations['threshold_analysis'] = fig
            
            # For multiclass, add class distribution plot
            if len(labels) > 2:
                # Class distribution
                class_counts = pd.Series(actual).value_counts().sort_index()
                
                fig = px.bar(
                    x=class_counts.index, 
                    y=class_counts.values,
                    labels={'x': 'Class', 'y': 'Count'},
                    title='Class Distribution',
                    color=class_counts.values,
                    color_continuous_scale=continuous_scale
                )
                
                # Update layout
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=60, b=40),
                    showlegend=False
                )
                
                visualizations['class_distribution'] = fig
                
                # Per-class metrics if available
                if hasattr(self, 'classification_report') and isinstance(self.classification_report, dict):
                    # Extract per-class metrics
                    class_metrics = {}
                    for cls in self.classification_report.keys():
                        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                            class_metrics[cls] = self.classification_report[cls]
                    
                    # Create dataframe for plotting
                    metrics_df = pd.DataFrame(class_metrics).T.reset_index()
                    metrics_df = pd.melt(metrics_df, id_vars=['index'], value_vars=['precision', 'recall', 'f1-score'])
                    metrics_df.columns = ['class', 'metric', 'value']
                    
                    # Create grouped bar chart
                    fig = px.bar(
                        metrics_df,
                        x='class',
                        y='value',
                        color='metric',
                        barmode='group',
                        title='Per-Class Performance Metrics',
                        labels={'class': 'Class', 'value': 'Score', 'metric': 'Metric'}
                    )
                    
                    # Update layout
                    fig.update_layout(
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=60, b=40)
                    )
                    
                    visualizations['class_metrics'] = fig
        
        # Feature importance visualization
        if self.feature_importance is not None:
            # Limit to top N features for readability
            top_features = self.feature_importance.sort_values('importance', ascending=False).head(max_features_to_show)
            
            # Create horizontal bar chart
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title=f'Top {len(top_features)} Feature Importance',
                labels={'importance': 'Importance', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale=continuous_scale
            )
            
            # Add value labels
            fig.update_traces(
                texttemplate='%{x:.3f}',
                textposition='outside'
            )
            
            # Improve layout
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                template="plotly_white",
                margin=dict(l=40, r=40, t=60, b=40),
                showlegend=False
            )
            
            visualizations['feature_importance'] = fig
            
            # Feature importance heatmap for top features
            if len(self.feature_importance) > 10:
                # Create a matrix for the heatmap
                importance_matrix = pd.DataFrame(
                    [top_features['importance'].values],
                    columns=top_features['feature'].values
                )
                
                fig = px.imshow(
                    importance_matrix,
                    color_continuous_scale=continuous_scale,
                    labels=dict(x='Feature', y='Model', color='Importance'),
                    title='Feature Importance Heatmap',
                    text_auto='.3f'
                )
                
                # Update layout
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=60, b=40),
                    height=300
                )
                
                visualizations['feature_importance_heatmap'] = fig
        
        # SHAP values visualization if available
        if hasattr(self, 'shap_values') and self.shap_values is not None:
            try:
                # Convert SHAP values to a format suitable for Plotly
                # This is a simplified approach - full SHAP visualizations would require more complex code
                shap_df = pd.DataFrame()
                
                # For binary classification or regression
                if isinstance(self.shap_values, np.ndarray) and len(self.shap_values.shape) == 2:
                    shap_data = self.shap_values
                    feature_names = self.X_test.columns if hasattr(self, 'X_test') else [f'Feature {i}' for i in range(shap_data.shape[1])]
                    
                    # Calculate mean absolute SHAP value for each feature
                    mean_shap = np.abs(shap_data).mean(axis=0)
                    shap_df = pd.DataFrame({'feature': feature_names, 'shap_value': mean_shap})
                    shap_df = shap_df.sort_values('shap_value', ascending=False).head(max_features_to_show)
                    
                    # Create bar chart
                    fig = px.bar(
                        shap_df,
                        x='shap_value',
                        y='feature',
                        orientation='h',
                        title='Mean |SHAP| Value (Feature Impact)',
                        labels={'shap_value': 'Mean |SHAP| Value', 'feature': 'Feature'},
                        color='shap_value',
                        color_continuous_scale=continuous_scale
                    )
                    
                    # Add value labels
                    fig.update_traces(
                        texttemplate='%{x:.3f}',
                        textposition='outside'
                    )
                    
                    # Improve layout
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=60, b=40),
                        showlegend=False
                    )
                    
                    visualizations['shap_summary'] = fig
            except Exception as e:
                logger.warning(f"Error creating SHAP visualization: {str(e)}")
        
        return visualizations
    
    def make_predictions(self, pipeline: Pipeline, X_new: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            pipeline: Trained model pipeline
            X_new: New features for prediction
            
        Returns:
            Array of predictions
        """
        return pipeline.predict(X_new)
    
    def forecast_time_series(self, df: pd.DataFrame, target_column: str, 
                           datetime_column: str, horizon: int = 10,
                           feature_columns: Optional[List[str]] = None,
                           model_name: str = 'auto',
                           include_seasonal_features: bool = True,
                           include_cyclical_encoding: bool = True,
                           calculate_intervals: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create a time series forecast.
        
        Args:
            df: Input DataFrame
            target_column: Column to forecast
            datetime_column: Column containing datetime information
            horizon: Number of periods to forecast
            feature_columns: Columns to use as features
            model_name: Name of model to use (auto, random_forest, xgboost, linear)
            include_seasonal_features: Whether to include seasonal features (weekends, month starts/ends)
            include_cyclical_encoding: Whether to include cyclical encoding of time features
            calculate_intervals: Whether to calculate prediction intervals
            
        Returns:
            DataFrame with forecasts and evaluation metrics
        """
        # Sort by datetime
        df = df.sort_values(datetime_column).copy()
        
        # Create lag features
        df_with_features = self._create_time_series_features(
            df, target_column, datetime_column,
            include_seasonal_features=include_seasonal_features,
            include_cyclical_encoding=include_cyclical_encoding
        )
        
        # Drop rows with NaN (from lag creation)
        df_with_features = df_with_features.dropna()
        
        # Determine feature columns if not provided
        if feature_columns is None:
            # Use all columns except target and datetime
            feature_columns = [col for col in df_with_features.columns 
                              if col != target_column and col != datetime_column]
        
        # Prepare data with time series split
        X_train, X_test, y_train, y_test = self.prepare_data(
            df_with_features, target_column, feature_columns,
            test_size=0.2, time_series=True, datetime_column=datetime_column
        )
        
        # Train model
        pipeline = self.train_model(X_train, y_train, model_type='regression', model_name=model_name)
        
        # Evaluate model
        metrics = self.evaluate_model(pipeline, X_test, y_test)
        
        # Generate forecast
        forecast_df = self._generate_forecast(
            df, pipeline, target_column, datetime_column, horizon, feature_columns,
            include_seasonal_features=include_seasonal_features,
            include_cyclical_encoding=include_cyclical_encoding,
            calculate_intervals=calculate_intervals
        )
        
        return forecast_df, metrics
    
    def _create_time_series_features(self, df: pd.DataFrame, target_column: str, 
                                   datetime_column: str,
                                   include_seasonal_features: bool = True,
                                   include_cyclical_encoding: bool = True) -> pd.DataFrame:
        """
        Create features for time series forecasting.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            datetime_column: Datetime column name
            
        Returns:
            DataFrame with additional time series features
        """
        df_copy = df.copy()
        
        # Convert datetime column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_copy[datetime_column]):
            df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column])
        
        # Extract datetime components
        df_copy['year'] = df_copy[datetime_column].dt.year
        df_copy['month'] = df_copy[datetime_column].dt.month
        df_copy['day'] = df_copy[datetime_column].dt.day
        df_copy['dayofweek'] = df_copy[datetime_column].dt.dayofweek
        df_copy['quarter'] = df_copy[datetime_column].dt.quarter
        
        # Add seasonal features if requested
        if include_seasonal_features:
            df_copy['is_weekend'] = df_copy['dayofweek'].isin([5, 6]).astype(int)
            df_copy['is_month_start'] = df_copy[datetime_column].dt.is_month_start.astype(int)
            df_copy['is_month_end'] = df_copy[datetime_column].dt.is_month_end.astype(int)
            df_copy['is_quarter_start'] = df_copy[datetime_column].dt.is_quarter_start.astype(int)
            df_copy['is_quarter_end'] = df_copy[datetime_column].dt.is_quarter_end.astype(int)
            df_copy['is_year_start'] = df_copy[datetime_column].dt.is_year_start.astype(int)
            df_copy['is_year_end'] = df_copy[datetime_column].dt.is_year_end.astype(int)
            
            # Add holiday indicators if pandas has holidays
            try:
                from pandas.tseries.holiday import USFederalHolidayCalendar
                cal = USFederalHolidayCalendar()
                holidays = cal.holidays(start=df_copy[datetime_column].min(), end=df_copy[datetime_column].max())
                df_copy['is_holiday'] = df_copy[datetime_column].isin(holidays).astype(int)
            except (ImportError, AttributeError):
                # Skip holiday feature if not available
                pass
        
        # Add cyclical encoding for month, day of week, etc. if requested
        if include_cyclical_encoding:
            df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
            df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
            df_copy['dayofweek_sin'] = np.sin(2 * np.pi * df_copy['dayofweek'] / 7)
            df_copy['dayofweek_cos'] = np.cos(2 * np.pi * df_copy['dayofweek'] / 7)
            
            # Add day of month cyclical encoding
            day_of_month_max = 31
            df_copy['day_of_month_sin'] = np.sin(2 * np.pi * df_copy['day'] / day_of_month_max)
            df_copy['day_of_month_cos'] = np.cos(2 * np.pi * df_copy['day'] / day_of_month_max)
        
        # Create lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            lag_col = f'{target_column}_lag_{lag}'
            df_copy[lag_col] = df_copy[target_column].shift(lag)
        
        # Create rolling window features
        for window in [3, 7, 14, 30]:
            # Rolling mean
            df_copy[f'{target_column}_rolling_mean_{window}'] = df_copy[target_column].rolling(window=window).mean()
            
            # Rolling std
            df_copy[f'{target_column}_rolling_std_{window}'] = df_copy[target_column].rolling(window=window).std()
            
            # Rolling min
            df_copy[f'{target_column}_rolling_min_{window}'] = df_copy[target_column].rolling(window=window).min()
            
            # Rolling max
            df_copy[f'{target_column}_rolling_max_{window}'] = df_copy[target_column].rolling(window=window).max()
            
            # Rolling median
            df_copy[f'{target_column}_rolling_median_{window}'] = df_copy[target_column].rolling(window=window).median()
        
        return df_copy
    
    def _generate_forecast(self, df: pd.DataFrame, pipeline: Pipeline, 
                         target_column: str, datetime_column: str,
                         horizon: int, feature_columns: List[str],
                         include_seasonal_features: bool = True,
                         include_cyclical_encoding: bool = True,
                         calculate_intervals: bool = True) -> pd.DataFrame:
        """
        Generate time series forecast.
        
        Args:
            df: Original DataFrame
            pipeline: Trained model pipeline
            target_column: Target column name
            datetime_column: Datetime column name
            horizon: Number of periods to forecast
            feature_columns: Feature columns used for prediction
            
        Returns:
            DataFrame with forecast
        """
        # Sort by datetime
        df = df.sort_values(datetime_column).copy()
        
        # Convert datetime column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
            df[datetime_column] = pd.to_datetime(df[datetime_column])
        
        # Determine the frequency of the time series
        freq = pd.infer_freq(df[datetime_column])
        if freq is None:
            # Try to determine frequency from the most common difference
            diff = df[datetime_column].diff().dropna()
            freq = diff.mode().iloc[0]
        
        # Create a DataFrame for the forecast periods
        last_date = df[datetime_column].iloc[-1]
        forecast_dates = pd.date_range(start=last_date, periods=horizon+1, freq=freq)[1:]
        
        forecast_df = pd.DataFrame({datetime_column: forecast_dates})
        
        # Add datetime features
        forecast_df['year'] = forecast_df[datetime_column].dt.year
        forecast_df['month'] = forecast_df[datetime_column].dt.month
        forecast_df['day'] = forecast_df[datetime_column].dt.day
        forecast_df['dayofweek'] = forecast_df[datetime_column].dt.dayofweek
        forecast_df['quarter'] = forecast_df[datetime_column].dt.quarter
        
        # Add seasonal features if requested
        if include_seasonal_features:
            forecast_df['is_weekend'] = forecast_df['dayofweek'].isin([5, 6]).astype(int)
            forecast_df['is_month_start'] = forecast_df[datetime_column].dt.is_month_start.astype(int)
            forecast_df['is_month_end'] = forecast_df[datetime_column].dt.is_month_end.astype(int)
            forecast_df['is_quarter_start'] = forecast_df[datetime_column].dt.is_quarter_start.astype(int)
            forecast_df['is_quarter_end'] = forecast_df[datetime_column].dt.is_quarter_end.astype(int)
            forecast_df['is_year_start'] = forecast_df[datetime_column].dt.is_year_start.astype(int)
            forecast_df['is_year_end'] = forecast_df[datetime_column].dt.is_year_end.astype(int)
            
            # Add holiday indicators if pandas has holidays
            try:
                from pandas.tseries.holiday import USFederalHolidayCalendar
                cal = USFederalHolidayCalendar()
                holidays = cal.holidays(start=forecast_df[datetime_column].min(), end=forecast_df[datetime_column].max())
                forecast_df['is_holiday'] = forecast_df[datetime_column].isin(holidays).astype(int)
            except (ImportError, AttributeError):
                # Skip holiday feature if not available
                pass
        
        # Add cyclical encoding for month, day of week, etc. if requested
        if include_cyclical_encoding:
            forecast_df['month_sin'] = np.sin(2 * np.pi * forecast_df['month'] / 12)
            forecast_df['month_cos'] = np.cos(2 * np.pi * forecast_df['month'] / 12)
            forecast_df['dayofweek_sin'] = np.sin(2 * np.pi * forecast_df['dayofweek'] / 7)
            forecast_df['dayofweek_cos'] = np.cos(2 * np.pi * forecast_df['dayofweek'] / 7)
            
            # Add day of month cyclical encoding
            day_of_month_max = 31
            forecast_df['day_of_month_sin'] = np.sin(2 * np.pi * forecast_df['day'] / day_of_month_max)
            forecast_df['day_of_month_cos'] = np.cos(2 * np.pi * forecast_df['day'] / day_of_month_max)
        
        # Initialize with the last known values
        for lag in [1, 2, 3, 7, 14, 30]:
            lag_col = f'{target_column}_lag_{lag}'
            if lag_col in feature_columns:
                forecast_df[lag_col] = df[target_column].iloc[-lag]
        
        # Initialize rolling features with the last known values
        for window in [3, 7, 14, 30]:
            mean_col = f'{target_column}_rolling_mean_{window}'
            std_col = f'{target_column}_rolling_std_{window}'
            min_col = f'{target_column}_rolling_min_{window}'
            max_col = f'{target_column}_rolling_max_{window}'
            median_col = f'{target_column}_rolling_median_{window}'
            
            if mean_col in feature_columns:
                forecast_df[mean_col] = df[target_column].iloc[-window:].mean()
            
            if std_col in feature_columns:
                forecast_df[std_col] = df[target_column].iloc[-window:].std()
                
            if min_col in feature_columns:
                forecast_df[min_col] = df[target_column].iloc[-window:].min()
                
            if max_col in feature_columns:
                forecast_df[max_col] = df[target_column].iloc[-window:].max()
                
            if median_col in feature_columns:
                forecast_df[median_col] = df[target_column].iloc[-window:].median()
        
        # Generate forecasts iteratively
        forecasts = []
        forecast_intervals = []
        
        for i in range(horizon):
            # Prepare features for current forecast step
            current_features = forecast_df.iloc[i:i+1].copy()
            
            # Ensure all required features are present
            for col in feature_columns:
                if col not in current_features.columns:
                    # Try to find the column in the original data
                    if col in df.columns:
                        current_features[col] = df[col].iloc[-1]
                    else:
                        # Set to 0 if not found
                        current_features[col] = 0
            
            # Make prediction
            try:
                pred = pipeline.predict(current_features[feature_columns])[0]
                forecasts.append(pred)
                
                # Update forecast DataFrame with the prediction
                forecast_df.loc[i, target_column] = pred
                
                # Calculate prediction interval (if model supports it and intervals are requested)
                interval = None
                if calculate_intervals:
                    try:
                        # Check if model has built-in prediction interval method
                        if hasattr(pipeline, 'predict_interval'):
                            lower, upper = pipeline.predict_interval(current_features[feature_columns], alpha=0.05)
                            interval = {'lower': lower[0], 'upper': upper[0]}
                        # Check if model is a quantile regressor
                        elif hasattr(pipeline, 'predict_quantiles') or (hasattr(pipeline, 'named_steps') and 
                                                                      any('quantile' in str(step).lower() for step in pipeline.named_steps.values())):
                            try:
                                lower = pipeline.predict_quantiles(current_features[feature_columns], quantiles=[0.05])[0][0]
                                upper = pipeline.predict_quantiles(current_features[feature_columns], quantiles=[0.95])[0][0]
                                interval = {'lower': lower, 'upper': upper}
                            except (AttributeError, TypeError):
                                # Fallback for models that don't directly support quantile prediction
                                std_dev = df[target_column].std() * 0.15
                                interval = {'lower': pred - 1.96 * std_dev, 'upper': pred + 1.96 * std_dev}
                        # For classification models with probability estimates
                        elif hasattr(pipeline, 'predict_proba'):
                            probs = pipeline.predict_proba(current_features[feature_columns])[0]
                            std_dev = np.std(probs) * pred if len(probs) > 1 else pred * 0.15
                            interval = {'lower': max(0, pred - 1.96 * std_dev), 'upper': pred + 1.96 * std_dev}
                        # For ensemble models, try to extract prediction variance
                        elif hasattr(pipeline, 'estimators_') or (hasattr(pipeline, 'named_steps') and 
                                                                 any(hasattr(step, 'estimators_') for step in pipeline.named_steps.values() if hasattr(step, 'estimators_'))):
                            # For ensemble models, use the variance of predictions from individual estimators
                            try:
                                if hasattr(pipeline, 'estimators_'):
                                    estimators = pipeline.estimators_
                                else:
                                    for step_name, step in pipeline.named_steps.items():
                                        if hasattr(step, 'estimators_'):
                                            estimators = step.estimators_
                                            break
                                
                                individual_preds = [estimator.predict(current_features[feature_columns])[0] for estimator in estimators]
                                std_dev = np.std(individual_preds)
                                interval = {'lower': pred - 1.96 * std_dev, 'upper': pred + 1.96 * std_dev}
                            except (AttributeError, TypeError):
                                # Fallback to historical error-based interval
                                std_dev = df[target_column].std() * 0.15
                                interval = {'lower': pred - 1.96 * std_dev, 'upper': pred + 1.96 * std_dev}
                        else:
                            # Use historical prediction error to estimate uncertainty
                            # Calculate RMSE on test data if available, otherwise use historical std
                            if 'metrics' in dir(self) and hasattr(self.metrics, 'get') and self.metrics.get('rmse'):
                                std_dev = self.metrics.get('rmse') * 0.8  # Scale factor for prediction interval
                            else:
                                std_dev = df[target_column].std() * 0.15
                            
                            interval = {'lower': pred - 1.96 * std_dev, 'upper': pred + 1.96 * std_dev}
                    except Exception as e:
                        logger.warning(f"Error calculating prediction interval: {str(e)}")
                        # Fallback to simple interval
                        std_dev = df[target_column].std() * 0.15
                        interval = {'lower': pred - 1.96 * std_dev, 'upper': pred + 1.96 * std_dev}
                else:
                    # If intervals not requested, use None values
                    interval = {'lower': None, 'upper': None}
                
                forecast_intervals.append(interval)
                
                # Update lag features for next step if needed
                if i + 1 < horizon:
                    for lag in [1, 2, 3, 7, 14]:
                        lag_col = f'{target_column}_lag_{lag}'
                        if lag_col in feature_columns and i + 1 >= lag:
                            forecast_df.loc[i+1, lag_col] = forecast_df.loc[i+1-lag:i+1, target_column].iloc[0]
                    
                    # Update rolling features
                    for window in [3, 7, 14, 30]:
                        mean_col = f'{target_column}_rolling_mean_{window}'
                        std_col = f'{target_column}_rolling_std_{window}'
                        min_col = f'{target_column}_rolling_min_{window}'
                        max_col = f'{target_column}_rolling_max_{window}'
                        median_col = f'{target_column}_rolling_median_{window}'
                        
                        if mean_col in feature_columns and i + 1 >= window:
                            forecast_df.loc[i+1, mean_col] = forecast_df.loc[max(0, i+1-window):i+1, target_column].mean()
                        
                        if std_col in feature_columns and i + 1 >= window:
                            forecast_df.loc[i+1, std_col] = forecast_df.loc[max(0, i+1-window):i+1, target_column].std()
                            
                        if min_col in feature_columns and i + 1 >= window:
                            forecast_df.loc[i+1, min_col] = forecast_df.loc[max(0, i+1-window):i+1, target_column].min()
                            
                        if max_col in feature_columns and i + 1 >= window:
                            forecast_df.loc[i+1, max_col] = forecast_df.loc[max(0, i+1-window):i+1, target_column].max()
                            
                        if median_col in feature_columns and i + 1 >= window:
                            forecast_df.loc[i+1, median_col] = forecast_df.loc[max(0, i+1-window):i+1, target_column].median()
            except Exception as e:
                logger.error(f"Error generating forecast for step {i}: {str(e)}")
                forecasts.append(None)
                forecast_intervals.append({'lower': None, 'upper': None})
        
        # Create final forecast DataFrame
        result_data = {
            datetime_column: forecast_dates,
            f'{target_column}_forecast': forecasts
        }
        
        # Add prediction intervals if they were calculated
        if calculate_intervals:
            result_data[f'{target_column}_lower_bound'] = [interval['lower'] for interval in forecast_intervals]
            result_data[f'{target_column}_upper_bound'] = [interval['upper'] for interval in forecast_intervals]
            
        result_df = pd.DataFrame(result_data)
        
        return result_df
    
    def create_forecast_visualization(self, df: pd.DataFrame, forecast_df: pd.DataFrame,
                                    target_column: str, datetime_column: str, 
                                    show_intervals: bool = True, show_components: bool = False) -> go.Figure:
        """
        Create visualization of the forecast.
        
        Args:
            df: Original DataFrame
            forecast_df: Forecast DataFrame
            target_column: Target column name
            datetime_column: Datetime column name
            show_intervals: Whether to show prediction intervals
            show_components: Whether to show seasonal components in a subplot
            
        Returns:
            Plotly figure with forecast visualization
        """
        if show_components:
            # Create figure with subplots: main forecast and seasonal components
            fig = make_subplots(rows=2, cols=1, 
                              shared_xaxes=True,
                              vertical_spacing=0.1,
                              subplot_titles=(f'Forecast for {target_column}', 'Seasonal Components'),
                              row_heights=[0.7, 0.3])
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=df[datetime_column],
                y=df[target_column],
                mode='lines',
                name='Historical Data',
                line=dict(color='royalblue')
            ), row=1, col=1)
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast_df[datetime_column],
                y=forecast_df[f'{target_column}_forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='firebrick', dash='dash')
            ), row=1, col=1)
            
            # Add prediction intervals if available, requested, and contain valid values
            if (show_intervals and 
                f'{target_column}_lower_bound' in forecast_df.columns and 
                f'{target_column}_upper_bound' in forecast_df.columns and
                forecast_df[f'{target_column}_lower_bound'].notna().any() and 
                forecast_df[f'{target_column}_upper_bound'].notna().any()):
                # Add upper and lower bounds
                fig.add_trace(go.Scatter(
                    x=forecast_df[datetime_column],
                    y=forecast_df[f'{target_column}_upper_bound'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=forecast_df[datetime_column],
                    y=forecast_df[f'{target_column}_lower_bound'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(220, 20, 60, 0.2)',  # Light red
                    fill='tonexty',
                    name='95% Confidence Interval'
                ), row=1, col=1)
        else:
            # Create single plot figure
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=df[datetime_column],
                y=df[target_column],
                mode='lines',
                name='Historical Data',
                line=dict(color='royalblue')
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast_df[datetime_column],
                y=forecast_df[f'{target_column}_forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='firebrick', dash='dash')
            ))
            
            # Add prediction intervals if available, requested, and contain valid values
            if (show_intervals and 
                f'{target_column}_lower_bound' in forecast_df.columns and 
                f'{target_column}_upper_bound' in forecast_df.columns and
                forecast_df[f'{target_column}_lower_bound'].notna().any() and 
                forecast_df[f'{target_column}_upper_bound'].notna().any()):
                # Add upper and lower bounds
                fig.add_trace(go.Scatter(
                    x=forecast_df[datetime_column],
                    y=forecast_df[f'{target_column}_upper_bound'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df[datetime_column],
                    y=forecast_df[f'{target_column}_lower_bound'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(220, 20, 60, 0.2)',  # Light red
                    fill='tonexty',
                    name='95% Confidence Interval'
                ))
        
        # Add seasonal components if requested
        if show_components:
            # Check if we have weekend data
            if 'is_weekend' in forecast_df.columns:
                weekend_data = forecast_df.copy()
                weekend_data['weekend_effect'] = weekend_data['is_weekend'] * weekend_data[f'{target_column}_forecast'] * 0.1
                
                fig.add_trace(go.Bar(
                    x=weekend_data[datetime_column],
                    y=weekend_data['weekend_effect'],
                    name='Weekend Effect',
                    marker_color='rgba(55, 83, 109, 0.7)'
                ), row=2, col=1)
            
            # Check if we have month start/end data
            month_effects = []
            if 'is_month_start' in forecast_df.columns:
                month_effects.append('is_month_start')
            if 'is_month_end' in forecast_df.columns:
                month_effects.append('is_month_end')
                
            if month_effects:
                month_data = forecast_df.copy()
                month_data['month_effect'] = month_data[month_effects].sum(axis=1) * month_data[f'{target_column}_forecast'] * 0.05
                
                fig.add_trace(go.Scatter(
                    x=month_data[datetime_column],
                    y=month_data['month_effect'],
                    mode='lines+markers',
                    name='Month Boundary Effect',
                    line=dict(color='green')
                ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=None,  # We use subplot_titles instead
            xaxis_title='Date',
            yaxis_title=target_column,
            legend_title='Data Type',
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Add range slider
        if show_components:
            fig.update_xaxes(rangeslider_visible=True, row=1, col=1)
        else:
            fig.update_xaxes(rangeslider_visible=True)
        
        # If we're not showing components, add a title
        if not show_components:
            fig.update_layout(title=f'Forecast for {target_column}')
        
        return fig
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the trained model.
        
        Returns:
            Dictionary with model summary information
        """
        if self.last_model is None or self.model_metrics is None:
            return {}
        
        summary = {
            'model_type': self.model_type,
            'model_name': type(self.last_model).__name__,
            'target_column': self.target_column,
            'feature_count': len(self.feature_columns),
            'metrics': self.model_metrics
        }
        
        # Add top features if available
        if self.feature_importance is not None:
            summary['top_features'] = self.feature_importance.head(10).to_dict('records')
        
        return summary