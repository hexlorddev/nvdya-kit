# Nvdya Kit - Linear Models Implementation
# Provides GPU-accelerated Linear Regression and Logistic Regression

import numpy as np
import logging
from ..core.base import BaseModel

logger = logging.getLogger(__name__)

class LinearRegression(BaseModel):
    """Linear Regression with optional GPU acceleration.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    normalize : bool, default=False
        This parameter is ignored when fit_intercept is set to False.
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    n_jobs : int, default=None
        The number of jobs to use for the computation.
    positive : bool, default=False
        When set to True, forces the coefficients to be positive.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration if available.
    """
    
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=None, positive=False, gpu_enabled=False):
        super().__init__(gpu_enabled=gpu_enabled)
        
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        
        self._estimator = None
        self._is_fitted = False
    
    def fit(self, X, y):
        """Fit linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        logger.info("Training LinearRegression model")
        
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
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
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
                from cuml.linear_model import LinearRegression as cuLinearRegression
                
                self._estimator = cuLinearRegression(
                    fit_intercept=self.fit_intercept,
                    normalize=self.normalize,
                    algorithm='eig'  # cuML specific parameter
                )
                
                # Fit the model
                self._estimator.fit(X_gpu, y_gpu)
                logger.info("LinearRegression training completed using GPU acceleration")
                
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
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        """
        from sklearn.linear_model import LinearRegression as skLinearRegression
        
        self._estimator = skLinearRegression(
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy_X=self.copy_X,
            n_jobs=self.n_jobs,
            positive=self.positive
        )
        
        # Fit the model
        self._estimator.fit(X, y)
        logger.info("LinearRegression training completed using CPU implementation")
        
        self._is_fitted = True
    
    def predict(self, X):
        """Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
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
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
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
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        return self._estimator.predict(X)
    
    @property
    def coef_(self):
        """Estimated coefficients for the linear regression problem.
        
        Returns
        -------
        coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
            Estimated coefficients.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
        if self.gpu_enabled:
            try:
                from ..gpu import to_cpu
                return to_cpu(self._estimator.coef_)
            except Exception:
                pass
        
        return self._estimator.coef_
    
    @property
    def intercept_(self):
        """Independent term in the linear model.
        
        Returns
        -------
        intercept_ : float or ndarray of shape (n_targets,)
            Independent term.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
        if self.gpu_enabled:
            try:
                from ..gpu import to_cpu
                return to_cpu(self._estimator.intercept_)
            except Exception:
                pass
        
        return self._estimator.intercept_


class LogisticRegression(BaseModel):
    """Logistic Regression with optional GPU acceleration.
    
    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        Specify the norm of the penalty.
    dual : bool, default=False
        Dual or primal formulation.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    C : float, default=1.0
        Inverse of regularization strength.
    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be added.
    intercept_scaling : float, default=1
        Useful only when the solver 'liblinear' is used.
    class_weight : dict or 'balanced', default=None
        Weights associated with classes.
    random_state : int, default=None
        Used when solver == 'sag', 'saga' or 'liblinear'.
    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default='lbfgs'
        Algorithm to use in the optimization problem.
    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.
    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each label.
    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive number.
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit.
    n_jobs : int, default=None
        Number of CPU cores used when parallelizing over classes.
    l1_ratio : float, default=None
        The Elastic-Net mixing parameter.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration if available.
    """
    
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100,
                 multi_class='auto', verbose=0, warm_start=False,
                 n_jobs=None, l1_ratio=None, gpu_enabled=False):
        super().__init__(gpu_enabled=gpu_enabled)
        
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        
        self._estimator = None
        self._is_fitted = False
        self._classes = None
    
    def fit(self, X, y):
        """Fit the model according to the given training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        logger.info("Training LogisticRegression model")
        
        # Store unique classes
        self._classes = np.unique(y)
        
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
            Training vector.
        y : array-like of shape (n_samples,)
            Target vector.
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
                from cuml.linear_model import LogisticRegression as cuLogisticRegression
                
                # cuML's LogisticRegression has limited parameters compared to scikit-learn
                self._estimator = cuLogisticRegression(
                    penalty='l2',  # cuML only supports l2
                    tol=self.tol,
                    C=self.C,
                    fit_intercept=self.fit_intercept,
                    max_iter=self.max_iter,
                    verbose=self.verbose
                )
                
                # Fit the model
                self._estimator.fit(X_gpu, y_gpu)
                logger.info("LogisticRegression training completed using GPU acceleration")
                
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
            Training vector.
        y : array-like of shape (n_samples,)
            Target vector.
        """
        from sklearn.linear_model import LogisticRegression as skLogisticRegression
        
        self._estimator = skLogisticRegression(
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
            verbose=self.verbose,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
            l1_ratio=self.l1_ratio
        )
        
        # Fit the model
        self._estimator.fit(X, y)
        logger.info("LogisticRegression training completed using CPU implementation")
        
        self._is_fitted = True
    
    def predict(self, X):
        """Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
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
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
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
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        return self._estimator.predict(X)
    
    def predict_proba(self, X):
        """Probability estimates for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probability of the sample for each class in the model.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
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
            Samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probability of the sample for each class in the model.
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
            Samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probability of the sample for each class in the model.
        """
        return self._estimator.predict_proba(X)
    
    @property
    def coef_(self):
        """Coefficient of the features in the decision function.
        
        Returns
        -------
        coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
            Coefficient of the features in the decision function.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
        if self.gpu_enabled:
            try:
                from ..gpu import to_cpu
                return to_cpu(self._estimator.coef_)
            except Exception:
                pass
        
        return self._estimator.coef_
    
    @property
    def intercept_(self):
        """Intercept (a.k.a. bias) added to the decision function.
        
        Returns
        -------
        intercept_ : ndarray of shape (1,) or (n_classes,)
            Intercept added to the decision function.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
        if self.gpu_enabled:
            try:
                from ..gpu import to_cpu
                return to_cpu(self._estimator.intercept_)
            except Exception:
                pass
        
        return self._estimator.intercept_
    
    @property
    def classes_(self):
        """Classes across all calls to fit.
        
        Returns
        -------
        classes_ : ndarray of shape (n_classes,)
            Array of class labels.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
        return self._classes