# Nvdya Kit - Preprocessing Data Module
# Provides data preprocessing tools with GPU acceleration

import numpy as np
from ..core.base import BaseTransformer


def train_test_split(X, y=None, test_size=0.2, random_state=None, shuffle=True, stratify=None):
    """Split arrays or matrices into random train and test subsets.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to split.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to split along with X.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
    random_state : int, default=None
        Controls the shuffling applied to the data before applying the split.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    stratify : array-like, default=None
        If not None, data is split in a stratified fashion using this as class labels.
    
    Returns
    -------
    X_train : ndarray
        Training data.
    X_test : ndarray
        Test data.
    y_train : ndarray, optional
        Training labels if y is not None.
    y_test : ndarray, optional
        Test labels if y is not None.
    """
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    # Handle stratification if requested
    if stratify is not None:
        from sklearn.model_selection import train_test_split as sk_split
        if y is not None:
            return sk_split(X, y, test_size=test_size, random_state=random_state, 
                           shuffle=shuffle, stratify=stratify)
        else:
            return sk_split(X, test_size=test_size, random_state=random_state, 
                           shuffle=shuffle, stratify=stratify)
    
    # Calculate split indices
    test_size_int = int(n_samples * test_size)
    test_indices = indices[:test_size_int]
    train_indices = indices[test_size_int:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    if y is not None:
        y_train = y[train_indices]
        y_test = y[test_indices]
        return X_train, X_test, y_train, y_test
    
    return X_train, X_test


class StandardScaler(BaseTransformer):
    """Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample x is calculated as:
        z = (x - u) / s
    where u is the mean of the training samples and s is the standard deviation.
    
    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        The mean value for each feature in the training set.
    scale_ : ndarray of shape (n_features,)
        The standard deviation for each feature in the training set.
    """
    
    def __init__(self, gpu_enabled=False, with_mean=True, with_std=True):
        """Initialize the scaler.
        
        Parameters
        ----------
        gpu_enabled : bool, default=False
            Whether to use GPU acceleration if available.
        with_mean : bool, default=True
            If True, center the data before scaling.
        with_std : bool, default=True
            If True, scale the data to unit variance.
        """
        super().__init__(gpu_enabled=gpu_enabled)
        self.with_mean = with_mean
        self.with_std = with_std
    
    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Fitted scaler.
        """
        # Convert input to numpy array if needed
        X = np.asarray(X)
        
        # Compute mean and std on CPU first
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
            
        if self.with_std:
            self.scale_ = np.std(X, axis=0, ddof=1)
            # Avoid division by zero
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            self.scale_ = np.ones(X.shape[1])
        
        # Move to GPU if enabled
        if self.gpu_enabled:
            try:
                from ..gpu import to_gpu
                self.mean_ = to_gpu(self.mean_)
                self.scale_ = to_gpu(self.scale_)
            except (ImportError, RuntimeError):
                self.gpu_enabled = False
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """Perform standardization by centering and scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.
            
        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        if not self._is_fitted:
            raise ValueError("StandardScaler is not fitted yet. Call 'fit' first.")
        
        X = np.asarray(X)
        
        # Move data to GPU if enabled
        if self.gpu_enabled:
            try:
                from ..gpu import to_gpu, to_cpu
                X_gpu = to_gpu(X)
                X_scaled = (X_gpu - self.mean_) / self.scale_
                return to_cpu(X_scaled)
            except (ImportError, RuntimeError):
                self.gpu_enabled = False
        
        # CPU implementation
        return (X - self.mean_) / self.scale_


class MinMaxScaler(BaseTransformer):
    """Transform features by scaling each feature to a given range.
    
    The transformation is given by:
        X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
    
    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data.
    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data.
    data_min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data.
    data_max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the data.
    data_range_ : ndarray of shape (n_features,)
        Per feature range (data_max_ - data_min_) seen in the data.
    """
    
    def __init__(self, gpu_enabled=False, feature_range=(0, 1)):
        """Initialize the scaler.
        
        Parameters
        ----------
        gpu_enabled : bool, default=False
            Whether to use GPU acceleration if available.
        feature_range : tuple (min, max), default=(0, 1)
            Desired range of transformed data.
        """
        super().__init__(gpu_enabled=gpu_enabled)
        self.feature_range = feature_range
    
    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the min and max.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Fitted scaler.
        """
        X = np.asarray(X)
        
        # Compute data min and max on CPU first
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Handle zeros in data_range
        self.data_range_ = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        
        # Compute scale and min for the transformation
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_
        
        # Move to GPU if enabled
        if self.gpu_enabled:
            try:
                from ..gpu import to_gpu
                self.data_min_ = to_gpu(self.data_min_)
                self.data_max_ = to_gpu(self.data_max_)
                self.data_range_ = to_gpu(self.data_range_)
                self.scale_ = to_gpu(self.scale_)
                self.min_ = to_gpu(self.min_)
            except (ImportError, RuntimeError):
                self.gpu_enabled = False
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """Scale features of X according to feature_range.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.
            
        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        if not self._is_fitted:
            raise ValueError("MinMaxScaler is not fitted yet. Call 'fit' first.")
        
        X = np.asarray(X)
        
        # Move data to GPU if enabled
        if self.gpu_enabled:
            try:
                from ..gpu import to_gpu, to_cpu
                X_gpu = to_gpu(X)
                X_scaled = X_gpu * self.scale_ + self.min_
                return to_cpu(X_scaled)
            except (ImportError, RuntimeError):
                self.gpu_enabled = False
        
        # CPU implementation
        return X * self.scale_ + self.min_


class OneHotEncoder(BaseTransformer):
    """Encode categorical features as a one-hot numeric array.
    
    The input to this transformer should be an array-like of integers or strings,
    denoting the values taken on by categorical features.
    
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting.
    """
    
    def __init__(self, gpu_enabled=False, categories='auto', sparse=False, drop=None):
        """Initialize the encoder.
        
        Parameters
        ----------
        gpu_enabled : bool, default=False
            Whether to use GPU acceleration if available.
        categories : 'auto' or list, default='auto'
            Categories for each feature.
        sparse : bool, default=False
            Will return sparse matrix if set True.
        drop : {'first', None}, default=None
            Specifies a methodology to use to drop one of the categories per feature.
        """
        super().__init__(gpu_enabled=gpu_enabled)
        self.categories = categories
        self.sparse = sparse
        self.drop = drop
    
    def fit(self, X, y=None):
        """Learn the categories of each feature.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Fitted encoder.
        """
        X = np.asarray(X)
        
        if self.categories == 'auto':
            # Determine categories automatically
            self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        else:
            self.categories_ = self.categories
        
        # Handle drop parameter
        if self.drop == 'first':
            self.drop_idx_ = [0 for _ in range(len(self.categories_))]
        else:
            self.drop_idx_ = None
        
        # Move to GPU if enabled
        if self.gpu_enabled:
            try:
                from ..gpu import to_gpu
                self.categories_ = [to_gpu(cat) for cat in self.categories_]
            except (ImportError, RuntimeError):
                self.gpu_enabled = False
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """Transform X using one-hot encoding.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.
            
        Returns
        -------
        X_out : ndarray of shape (n_samples, n_encoded_features)
            Transformed data.
        """
        if not self._is_fitted:
            raise ValueError("OneHotEncoder is not fitted yet. Call 'fit' first.")
        
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Determine output dimensions
        if self.drop is None:
            n_values = [len(cats) for cats in self.categories_]
        else:
            n_values = [len(cats) - 1 for cats in self.categories_]
        
        n_values_sum = sum(n_values)
        
        # Initialize output array
        if self.sparse:
            import scipy.sparse as sp
            X_out = sp.lil_matrix((n_samples, n_values_sum), dtype=np.float64)
        else:
            X_out = np.zeros((n_samples, n_values_sum), dtype=np.float64)
        
        # Perform one-hot encoding
        column_indices = np.cumsum([0] + n_values[:-1])
        
        # Move data to GPU if enabled
        if self.gpu_enabled:
            try:
                from ..gpu import to_gpu, to_cpu
                X_gpu = to_gpu(X)
                
                # GPU implementation would go here
                # For now, we'll fall back to CPU for the actual encoding
                X = to_cpu(X_gpu)
                self.gpu_enabled = False
            except (ImportError, RuntimeError):
                self.gpu_enabled = False
        
        # CPU implementation
        for i in range(n_features):
            feature_cats = self.categories_[i]
            n_categories = len(feature_cats)
            
            # Get indices for this feature's categories
            feature_idx = column_indices[i]
            
            # Find which category each sample belongs to
            indices = np.searchsorted(feature_cats, X[:, i])
            indices = np.clip(indices, 0, n_categories - 1)
            
            # Handle drop parameter
            if self.drop == 'first':
                # Shift indices to account for dropped first category
                indices[indices > 0] -= 1
                # Skip samples that would be encoded as the first category
                mask = indices >= 0
                sample_indices = np.arange(n_samples)[mask]
                feature_indices = feature_idx + indices[mask]
            else:
                sample_indices = np.arange(n_samples)
                feature_indices = feature_idx + indices
            
            # Set the corresponding values to 1
            if self.sparse:
                for sample_idx, feat_idx in zip(sample_indices, feature_indices):
                    X_out[sample_idx, feat_idx] = 1
            else:
                X_out[sample_indices, feature_indices] = 1
        
        return X_out