# Nvdya Kit - Core Base Classes
# Provides the foundation for all models and algorithms

import numpy as np
import warnings
from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    """Base class for all estimators in Nvdya Kit.
    
    This class defines the common interface for all estimators
    (models, transformers, etc.) in the library.
    """
    
    def __init__(self, gpu_enabled=False):
        """Initialize the base estimator.
        
        Parameters
        ----------
        gpu_enabled : bool, default=False
            Whether to use GPU acceleration if available.
        """
        self.gpu_enabled = gpu_enabled
        self._is_fitted = False
        
        # Check GPU availability if enabled
        if self.gpu_enabled:
            try:
                from ..gpu import is_gpu_available
                if not is_gpu_available():
                    warnings.warn(
                        "GPU acceleration requested but no compatible GPU found. "
                        "Falling back to CPU implementation."
                    )
                    self.gpu_enabled = False
            except ImportError:
                warnings.warn(
                    "GPU acceleration requested but GPU module not available. "
                    "Falling back to CPU implementation."
                )
                self.gpu_enabled = False
    
    def get_params(self):
        """Get parameters for this estimator.
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {key: value for key, value in self.__dict__.items()
                if not key.startswith('_')}
    
    def set_params(self, **params):
        """Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    @property
    def is_fitted(self):
        """Check if the estimator has been fitted.
        
        Returns
        -------
        bool
            True if the estimator has been fitted, False otherwise.
        """
        return self._is_fitted


class BaseModel(BaseEstimator):
    """Base class for all models in Nvdya Kit.
    
    This class extends BaseEstimator with methods specific to models
    that can be trained and used for prediction.
    """
    
    @abstractmethod
    def fit(self, X, y=None):
        """Fit the model to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values.
        """
        pass
    
    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
            
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        from ..metrics import r2_score
        return r2_score(y, self.predict(X))
    
    def to_gpu(self):
        """Transfer model data to GPU if possible.
        
        Returns
        -------
        self : object
            Model with data on GPU.
        """
        try:
            from ..gpu import to_gpu
            self = to_gpu(self)
            self.gpu_enabled = True
        except (ImportError, RuntimeError) as e:
            warnings.warn(f"Failed to transfer model to GPU: {str(e)}")
        return self
    
    def to_cpu(self):
        """Transfer model data to CPU.
        
        Returns
        -------
        self : object
            Model with data on CPU.
        """
        try:
            from ..gpu import to_cpu
            self = to_cpu(self)
        finally:
            self.gpu_enabled = False
        return self


class BaseTransformer(BaseEstimator):
    """Base class for all data transformers in Nvdya Kit.
    
    This class extends BaseEstimator with methods specific to transformers
    that can modify data.
    """
    
    @abstractmethod
    def fit(self, X, y=None):
        """Fit the transformer to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values (unused, present for API consistency).
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        pass
    
    @abstractmethod
    def transform(self, X):
        """Apply the transformation to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to be transformed.
            
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features_new)
            Transformed data.
        """
        pass
    
    def fit_transform(self, X, y=None):
        """Fit the transformer to the data, then transform the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to be transformed.
        y : array-like of shape (n_samples,), default=None
            Target values (unused, present for API consistency).
            
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features_new)
            Transformed data.
        """
        return self.fit(X, y).transform(X)