# Nvdya Kit - Regression Metrics Module
# Provides regression evaluation metrics with GPU acceleration

import numpy as np


def mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', gpu_enabled=False):
    """Mean squared error regression loss.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregating of multiple output values.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration if available.
        
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if gpu_enabled:
        try:
            from ..gpu import to_gpu, to_cpu
            y_true_gpu = to_gpu(y_true)
            y_pred_gpu = to_gpu(y_pred)
            
            # Compute squared error on GPU
            errors = (y_true_gpu - y_pred_gpu) ** 2
            
            if sample_weight is not None:
                sample_weight_gpu = to_gpu(np.asarray(sample_weight))
                errors = errors * sample_weight_gpu[:, np.newaxis if y_true.ndim > 1 else None]
                avg_errors = errors.sum(axis=0) / sample_weight_gpu.sum()
            else:
                avg_errors = errors.mean(axis=0)
            
            if multioutput == 'raw_values':
                return to_cpu(avg_errors)
            elif multioutput == 'uniform_average':
                return to_cpu(avg_errors.mean())
            else:
                raise ValueError(f"Unsupported multioutput parameter: {multioutput}")
        except (ImportError, RuntimeError):
            gpu_enabled = False
    
    # CPU implementation
    errors = (y_true - y_pred) ** 2
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        weights = sample_weight[:, np.newaxis if y_true.ndim > 1 else None]
        errors = errors * weights
        avg_errors = errors.sum(axis=0) / weights.sum()
    else:
        avg_errors = errors.mean(axis=0)
    
    if multioutput == 'raw_values':
        return avg_errors
    elif multioutput == 'uniform_average':
        return avg_errors.mean() if avg_errors.ndim > 0 else avg_errors
    else:
        raise ValueError(f"Unsupported multioutput parameter: {multioutput}")


def mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', gpu_enabled=False):
    """Mean absolute error regression loss.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregating of multiple output values.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration if available.
        
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if gpu_enabled:
        try:
            from ..gpu import to_gpu, to_cpu
            y_true_gpu = to_gpu(y_true)
            y_pred_gpu = to_gpu(y_pred)
            
            # Compute absolute error on GPU
            errors = np.abs(y_true_gpu - y_pred_gpu)
            
            if sample_weight is not None:
                sample_weight_gpu = to_gpu(np.asarray(sample_weight))
                errors = errors * sample_weight_gpu[:, np.newaxis if y_true.ndim > 1 else None]
                avg_errors = errors.sum(axis=0) / sample_weight_gpu.sum()
            else:
                avg_errors = errors.mean(axis=0)
            
            if multioutput == 'raw_values':
                return to_cpu(avg_errors)
            elif multioutput == 'uniform_average':
                return to_cpu(avg_errors.mean())
            else:
                raise ValueError(f"Unsupported multioutput parameter: {multioutput}")
        except (ImportError, RuntimeError):
            gpu_enabled = False
    
    # CPU implementation
    errors = np.abs(y_true - y_pred)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        weights = sample_weight[:, np.newaxis if y_true.ndim > 1 else None]
        errors = errors * weights
        avg_errors = errors.sum(axis=0) / weights.sum()
    else:
        avg_errors = errors.mean(axis=0)
    
    if multioutput == 'raw_values':
        return avg_errors
    elif multioutput == 'uniform_average':
        return avg_errors.mean() if avg_errors.ndim > 0 else avg_errors
    else:
        raise ValueError(f"Unsupported multioutput parameter: {multioutput}")


def r2_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average', gpu_enabled=False):
    """R^2 (coefficient of determination) regression score function.
    
    Best possible score is 1.0 and it can be negative (because the model can be
    arbitrarily worse). A constant model that always predicts the expected value
    of y, disregarding the input features, would get a R^2 score of 0.0.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average', 'variance_weighted'}, default='uniform_average'
        Defines aggregating of multiple output scores.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration if available.
        
    Returns
    -------
    z : float or ndarray of floats
        The R^2 score or ndarray of scores if 'multioutput' is 'raw_values'.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    
    if gpu_enabled:
        try:
            from ..gpu import to_gpu, to_cpu
            y_true_gpu = to_gpu(y_true)
            y_pred_gpu = to_gpu(y_pred)
            
            # GPU implementation would go here
            # For now, we'll fall back to CPU
            y_true = to_cpu(y_true_gpu)
            y_pred = to_cpu(y_pred_gpu)
            gpu_enabled = False
        except (ImportError, RuntimeError):
            gpu_enabled = False
    
    # CPU implementation
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1
    
    # Compute the weighted mean of y_true for each output
    if sample_weight is not None:
        y_true_mean = np.average(y_true, weights=sample_weight, axis=0)
        numerator = np.average((y_true - y_pred) ** 2, weights=sample_weight, axis=0)
        denominator = np.average((y_true - y_true_mean) ** 2, weights=sample_weight, axis=0)
    else:
        y_true_mean = np.mean(y_true, axis=0)
        numerator = np.mean((y_true - y_pred) ** 2, axis=0)
        denominator = np.mean((y_true - y_true_mean) ** 2, axis=0)
    
    # Avoid division by zero
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    
    # Initialize score array
    output_scores = np.ones(y_true.shape[1])
    
    # Compute R^2 score for each output
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0
    
    # Aggregate scores
    if multioutput == 'raw_values':
        return output_scores[0] if output_scores.shape[0] == 1 else output_scores
    elif multioutput == 'uniform_average':
        return np.mean(output_scores)
    elif multioutput == 'variance_weighted':
        weights = denominator
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            return np.sum(output_scores * weights) / weights_sum
        else:
            return np.mean(output_scores)
    else:
        raise ValueError(f"Unsupported multioutput parameter: {multioutput}")