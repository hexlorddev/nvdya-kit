# Nvdya Kit - Feature Engineering Module
# Provides advanced feature engineering tools with GPU acceleration

import numpy as np
from ..core.base import BaseTransformer


class PolynomialFeatures(BaseTransformer):
    """Generate polynomial and interaction features.
    
    Generate a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.
    For example, if an input sample is two dimensional and of the form [a, b],
    the degree-2 polynomial features are [1, a, b, a², ab, b²].
    
    Attributes
    ----------
    powers_ : ndarray of shape (n_output_features, n_input_features)
        powers_[i, j] is the exponent of the jth input feature in the ith output feature.
    n_input_features_ : int
        The number of input features.
    n_output_features_ : int
        The number of output features.
    """
    
    def __init__(self, gpu_enabled=False, degree=2, interaction_only=False, include_bias=True):
        """Initialize the transformer.
        
        Parameters
        ----------
        gpu_enabled : bool, default=False
            Whether to use GPU acceleration if available.
        degree : int, default=2
            The degree of the polynomial features.
        interaction_only : bool, default=False
            If True, only interaction features are produced: features that are products
            of at most degree distinct input features.
        include_bias : bool, default=True
            If True, include a bias column (all ones).
        """
        super().__init__(gpu_enabled=gpu_enabled)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
    
    def fit(self, X, y=None):
        """Compute the polynomial features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Generate combinations of feature indices
        combinations = self._combinations(n_features, self.degree, 
                                         self.interaction_only, 
                                         self.include_bias)
        self.n_input_features_ = n_features
        self.n_output_features_ = len(combinations)
        
        # Save the powers for each feature combination
        self.powers_ = np.vstack([self._get_powers(c, n_features) for c in combinations])
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """Transform data to polynomial features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.
            
        Returns
        -------
        X_poly : ndarray of shape (n_samples, n_output_features)
            Transformed data.
        """
        if not self._is_fitted:
            raise ValueError("PolynomialFeatures is not fitted yet. Call 'fit' first.")
        
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")
        
        # Move data to GPU if enabled
        if self.gpu_enabled:
            try:
                from ..gpu import to_gpu, to_cpu
                X_gpu = to_gpu(X)
                # GPU implementation would go here
                # For now, we'll fall back to CPU
                X = to_cpu(X_gpu)
                self.gpu_enabled = False
            except (ImportError, RuntimeError):
                self.gpu_enabled = False
        
        # Initialize output array
        X_poly = np.ones((n_samples, self.n_output_features_))
        
        # For each combination of features
        for i, powers in enumerate(self.powers_):
            if i == 0 and self.include_bias:  # Skip the bias term
                continue
            
            # Compute the product of features raised to the corresponding powers
            X_poly[:, i] = np.prod(X ** powers, axis=1)
        
        return X_poly
    
    def _combinations(self, n_features, degree, interaction_only, include_bias):
        """Generate combinations of feature indices for polynomial features."""
        comb = []
        if include_bias:
            comb.append(())
        
        for d in range(1, degree + 1):
            if interaction_only:
                # Generate combinations of exactly d features
                for indices in self._combinations_with_replacement(range(n_features), d):
                    if len(set(indices)) == d:  # Only include if all features are distinct
                        comb.append(indices)
            else:
                # Generate all combinations of d features, with replacement
                for indices in self._combinations_with_replacement(range(n_features), d):
                    comb.append(indices)
        
        return comb
    
    def _combinations_with_replacement(self, iterable, r):
        """Generate combinations with replacement."""
        pool = tuple(iterable)
        n = len(pool)
        
        if not n and r:
            return
        
        indices = [0] * r
        yield tuple(pool[i] for i in indices)
        
        while True:
            for i in reversed(range(r)):
                if indices[i] != n - 1:
                    break
            else:
                return
            
            indices[i:] = [indices[i] + 1] * (r - i)
            yield tuple(pool[i] for i in indices)
    
    def _get_powers(self, combination, n_features):
        """Get the powers for each feature based on the combination."""
        powers = np.zeros(n_features, dtype=int)
        for i in combination:
            powers[i] += 1
        return powers


class PCA(BaseTransformer):
    """Principal Component Analysis (PCA).
    
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.
    
    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data.
    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
    """
    
    def __init__(self, gpu_enabled=False, n_components=None, whiten=False):
        """Initialize the transformer.
        
        Parameters
        ----------
        gpu_enabled : bool, default=False
            Whether to use GPU acceleration if available.
        n_components : int, default=None
            Number of components to keep. If None, all components are kept.
        whiten : bool, default=False
            When True, the components_ vectors are multiplied by the square root
            of n_samples and then divided by the singular values to ensure uncorrelated
            outputs with unit component-wise variances.
        """
        super().__init__(gpu_enabled=gpu_enabled)
        self.n_components = n_components
        self.whiten = whiten
    
    def fit(self, X, y=None):
        """Fit the model with X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Get variance explained by singular values
        explained_variance = (S ** 2) / (n_samples - 1)
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var
        
        # Determine number of components
        if self.n_components is None:
            n_components = min(n_samples, n_features)
        else:
            n_components = min(self.n_components, min(n_samples, n_features))
        
        # Store results
        self.components_ = Vt[:n_components]
        self.explained_variance_ = explained_variance[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components]
        self.singular_values_ = S[:n_components]
        self.n_components_ = n_components
        
        # Move to GPU if enabled
        if self.gpu_enabled:
            try:
                from ..gpu import to_gpu
                self.components_ = to_gpu(self.components_)
                self.mean_ = to_gpu(self.mean_)
                self.explained_variance_ = to_gpu(self.explained_variance_)
                self.explained_variance_ratio_ = to_gpu(self.explained_variance_ratio_)
                self.singular_values_ = to_gpu(self.singular_values_)
            except (ImportError, RuntimeError):
                self.gpu_enabled = False
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """Apply dimensionality reduction to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        if not self._is_fitted:
            raise ValueError("PCA is not fitted yet. Call 'fit' first.")
        
        X = np.asarray(X)
        
        # Move data to GPU if enabled
        if self.gpu_enabled:
            try:
                from ..gpu import to_gpu, to_cpu
                X_gpu = to_gpu(X)
                mean_gpu = self.mean_
                components_gpu = self.components_
                
                # Center and project data
                X_centered_gpu = X_gpu - mean_gpu
                X_transformed_gpu = X_centered_gpu @ components_gpu.T
                
                # Apply whitening if requested
                if self.whiten:
                    X_transformed_gpu = X_transformed_gpu / np.sqrt(self.explained_variance_)
                
                return to_cpu(X_transformed_gpu)
            except (ImportError, RuntimeError):
                self.gpu_enabled = False
        
        # CPU implementation
        X_centered = X - self.mean_
        X_transformed = np.dot(X_centered, self.components_.T)
        
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        
        return X_transformed


class Normalizer(BaseTransformer):
    """Normalize samples individually to unit norm.
    
    Each sample (i.e. each row of the data matrix) with at least one non-zero
    component is rescaled independently of other samples so that its norm equals one.
    
    Attributes
    ----------
    norm : str
        The norm to use to normalize each non-zero sample.
    """
    
    def __init__(self, gpu_enabled=False, norm='l2'):
        """Initialize the transformer.
        
        Parameters
        ----------
        gpu_enabled : bool, default=False
            Whether to use GPU acceleration if available.
        norm : {'l1', 'l2', 'max'}, default='l2'
            The norm to use to normalize each non-zero sample.
        """
        super().__init__(gpu_enabled=gpu_enabled)
        self.norm = norm
        self._is_fitted = True  # Normalizer doesn't need to be fitted
    
    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        
        Parameters
        ----------
        X : array-like
            Ignored.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        return self
    
    def transform(self, X):
        """Scale each non-zero sample to unit norm.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to normalize.
            
        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Normalized data.
        """
        X = np.asarray(X)
        
        # Move data to GPU if enabled
        if self.gpu_enabled:
            try:
                from ..gpu import to_gpu, to_cpu
                X_gpu = to_gpu(X)
                
                # GPU implementation would go here
                # For now, we'll fall back to CPU
                X = to_cpu(X_gpu)
                self.gpu_enabled = False
            except (ImportError, RuntimeError):
                self.gpu_enabled = False
        
        # CPU implementation
        if self.norm == 'l1':
            norms = np.abs(X).sum(axis=1)
        elif self.norm == 'l2':
            norms = np.sqrt(np.square(X).sum(axis=1))
        elif self.norm == 'max':
            norms = np.max(np.abs(X), axis=1)
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")
        
        # Avoid division by zero
        norms = np.where(norms > 0, norms, 1.0)
        
        # Scale data
        X_scaled = X / norms[:, np.newaxis]
        
        return X_scaled