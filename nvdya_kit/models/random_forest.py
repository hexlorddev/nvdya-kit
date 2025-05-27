# Nvdya Kit - Random Forest Implementation
# Provides GPU-accelerated Random Forest algorithm

import numpy as np
import logging
from ..core.base import BaseModel

logger = logging.getLogger(__name__)

class RandomForest(BaseModel):
    """Random Forest classifier and regressor with GPU acceleration.
    
    This implementation provides both classification and regression capabilities
    with optional GPU acceleration for faster training and inference.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : {'gini', 'entropy', 'mse', 'mae'}, default='gini'
        The function to measure the quality of a split.
    max_depth : int, default=None
        The maximum depth of the tree. If None, nodes are expanded until
        all leaves are pure or contain less than min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : {'auto', 'sqrt', 'log2'}, int or float, default='auto'
        The number of features to consider when looking for the best split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
    n_jobs : int, default=None
        The number of jobs to run in parallel. None means using all processors.
    random_state : int, default=None
        Controls both the randomness of the bootstrapping and the sampling of features.
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration if available.
    task : {'auto', 'classification', 'regression'}, default='auto'
        The learning task. If 'auto', inferred from the target data.
    """
    
    def __init__(self, n_estimators=100, criterion='auto', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features='auto',
                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                 verbose=0, gpu_enabled=False, task='auto'):
        super().__init__(gpu_enabled=gpu_enabled)
        
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.task = task
        
        self._estimator = None
        self._is_fitted = False
    
    def fit(self, X, y):
        """Build a forest of trees from the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Determine the task type if set to 'auto'
        if self.task == 'auto':
            unique_values = np.unique(y)
            if len(unique_values) <= 10 or np.issubdtype(y.dtype, np.integer):
                self._task = 'classification'
                if self.criterion == 'auto':
                    self._criterion = 'gini'
            else:
                self._task = 'regression'
                if self.criterion == 'auto':
                    self._criterion = 'mse'
        else:
            self._task = self.task
            self._criterion = 'gini' if self.criterion == 'auto' and self._task == 'classification' else \
                             'mse' if self.criterion == 'auto' and self._task == 'regression' else \
                             self.criterion
        
        logger.info(f"Training RandomForest for {self._task} with {self.n_estimators} trees")
        
        # Use GPU implementation if enabled and available
        if self.gpu_enabled:
            try:
                self._fit_gpu(X, y)
                return self
            except (ImportError, RuntimeError) as e:
                logger.warning(f"GPU training failed: {str(e)}. Falling back to CPU implementation.")
                self.gpu_enabled = False
        
        # Fall back to scikit-learn implementation
        self._fit_cpu(X, y)
        return self
    
    def _fit_gpu(self, X, y):
        """Fit the model using GPU acceleration.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.
        """
        try:
            from ..gpu import to_gpu
            import cupy as cp
            
            # Transfer data to GPU
            X_gpu = to_gpu(X)
            y_gpu = to_gpu(y)
            
            # Import cuML for GPU-accelerated implementation
            try:
                import cuml
                from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
                
                # Select the appropriate model based on the task
                if self._task == 'classification':
                    self._estimator = RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth if self.max_depth is not None else 16,  # cuML default
                        max_features=self.max_features if isinstance(self.max_features, float) else 'auto',
                        n_bins=256,  # cuML specific parameter
                        bootstrap=self.bootstrap,
                        random_state=self.random_state,
                        verbose=self.verbose
                    )
                else:  # regression
                    self._estimator = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth if self.max_depth is not None else 16,  # cuML default
                        max_features=self.max_features if isinstance(self.max_features, float) else 'auto',
                        n_bins=256,  # cuML specific parameter
                        bootstrap=self.bootstrap,
                        random_state=self.random_state,
                        verbose=self.verbose
                    )
                
                # Fit the model
                self._estimator.fit(X_gpu, y_gpu)
                logger.info("RandomForest training completed using GPU acceleration")
                
            except ImportError:
                # If cuML is not available, use custom GPU implementation
                logger.warning("cuML not available. Using custom GPU implementation.")
                raise NotImplementedError("Custom GPU implementation not yet available")
                
        except Exception as e:
            logger.error(f"Error during GPU training: {str(e)}")
            raise RuntimeError(f"GPU training failed: {str(e)}")
        
        self._is_fitted = True
    
    def _fit_cpu(self, X, y):
        """Fit the model using CPU implementation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Select the appropriate model based on the task
        if self._task == 'classification':
            self._estimator = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=self._criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose
            )
        else:  # regression
            self._estimator = RandomForestRegressor(
                n_estimators=self.n_estimators,
                criterion=self._criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose
            )
        
        # Fit the model
        self._estimator.fit(X, y)
        logger.info("RandomForest training completed using CPU implementation")
        
        self._is_fitted = True
    
    def predict(self, X):
        """Predict class or regression value for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes or values.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
        # Use GPU for prediction if enabled
        if self.gpu_enabled:
            try:
                return self._predict_gpu(X)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"GPU prediction failed: {str(e)}. Falling back to CPU implementation.")
        
        # Fall back to CPU prediction
        return self._predict_cpu(X)
    
    def _predict_gpu(self, X):
        """Make predictions using GPU acceleration.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes or values.
        """
        try:
            from ..gpu import to_gpu, to_cpu
            
            # Transfer data to GPU
            X_gpu = to_gpu(X)
            
            # Make predictions
            y_pred_gpu = self._estimator.predict(X_gpu)
            
            # Transfer predictions back to CPU
            return to_cpu(y_pred_gpu)
            
        except Exception as e:
            logger.error(f"Error during GPU prediction: {str(e)}")
            raise RuntimeError(f"GPU prediction failed: {str(e)}")
    
    def _predict_cpu(self, X):
        """Make predictions using CPU implementation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes or values.
        """
        return self._estimator.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
        if self._task != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        # Use GPU for prediction if enabled
        if self.gpu_enabled:
            try:
                return self._predict_proba_gpu(X)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"GPU probability prediction failed: {str(e)}. Falling back to CPU implementation.")
        
        # Fall back to CPU prediction
        return self._predict_proba_cpu(X)
    
    def _predict_proba_gpu(self, X):
        """Predict class probabilities using GPU acceleration.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        try:
            from ..gpu import to_gpu, to_cpu
            
            # Transfer data to GPU
            X_gpu = to_gpu(X)
            
            # Make predictions
            proba_gpu = self._estimator.predict_proba(X_gpu)
            
            # Transfer predictions back to CPU
            return to_cpu(proba_gpu)
            
        except Exception as e:
            logger.error(f"Error during GPU probability prediction: {str(e)}")
            raise RuntimeError(f"GPU probability prediction failed: {str(e)}")
    
    def _predict_proba_cpu(self, X):
        """Predict class probabilities using CPU implementation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        return self._estimator.predict_proba(X)
    
    def feature_importances(self):
        """Get feature importances from the fitted model.
        
        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The feature importances.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
        return self._estimator.feature_importances_