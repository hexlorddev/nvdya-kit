# Nvdya Kit - Classification Metrics Module
# Provides classification evaluation metrics with GPU acceleration

import numpy as np


def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None, gpu_enabled=False):
    """Accuracy classification score.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels, as returned by a classifier.
    normalize : bool, default=True
        If False, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration if available.
        
    Returns
    -------
    score : float
        If normalize == True, return the fraction of correctly classified samples.
        Otherwise, return the number of correctly classified samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if gpu_enabled:
        try:
            from ..gpu import to_gpu, to_cpu
            y_true_gpu = to_gpu(y_true)
            y_pred_gpu = to_gpu(y_pred)
            
            # Compute accuracy on GPU
            correct = (y_true_gpu == y_pred_gpu)
            
            if sample_weight is not None:
                sample_weight_gpu = to_gpu(np.asarray(sample_weight))
                correct = correct * sample_weight_gpu
                score = correct.sum() / sample_weight_gpu.sum()
            else:
                score = correct.mean() if normalize else correct.sum()
            
            return to_cpu(score)
        except (ImportError, RuntimeError):
            gpu_enabled = False
    
    # CPU implementation
    correct = (y_true == y_pred)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        correct = correct * sample_weight
        return correct.sum() / sample_weight.sum() if normalize else correct.sum()
    
    return correct.mean() if normalize else correct.sum()


def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None, gpu_enabled=False):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. If None, the sorted unique labels
        in y_true and y_pred are used.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration if available.
        
    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    
    n_labels = len(labels)
    label_to_ind = {y: x for x, y in enumerate(labels)}
    
    # Map y_true and y_pred to their indices
    y_true_ind = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
    y_pred_ind = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    
    # Compute confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)
    
    # CPU implementation
    for i in range(len(y_true)):
        true_idx = y_true_ind[i]
        pred_idx = y_pred_ind[i]
        
        if true_idx < n_labels and pred_idx < n_labels:
            if sample_weight is not None:
                cm[true_idx, pred_idx] += sample_weight[i]
            else:
                cm[true_idx, pred_idx] += 1
    
    # Normalize if requested
    if normalize is not None:
        cm = cm.astype(np.float64)
        if normalize == 'true':
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm_sum = np.where(cm_sum == 0, 1, cm_sum)  # Avoid division by zero
            cm /= cm_sum
        elif normalize == 'pred':
            cm_sum = cm.sum(axis=0, keepdims=True)
            cm_sum = np.where(cm_sum == 0, 1, cm_sum)  # Avoid division by zero
            cm /= cm_sum
        elif normalize == 'all':
            cm_sum = cm.sum()
            cm_sum = 1 if cm_sum == 0 else cm_sum  # Avoid division by zero
            cm /= cm_sum
        else:
            raise ValueError(f"Unsupported normalize mode: {normalize}")
    
    return cm


def precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, gpu_enabled=False):
    """Compute the precision score.
    
    The precision is the ratio tp / (tp + fp) where tp is the number of true positives
    and fp the number of false positives. The precision is intuitively the ability
    of the classifier not to label as positive a sample that is negative.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    labels : array-like, default=None
        The set of labels to include when average != 'binary'.
    pos_label : str or int, default=1
        The class to report if average='binary' and the data is binary.
    average : {'micro', 'macro', 'weighted', 'binary', None}, default='binary'
        This parameter is required for multiclass/multilabel targets.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration if available.
        
    Returns
    -------
    precision : float (if average is not None) or array of float of shape (n_classes,)
        Precision score.
    """
    # For binary classification
    if average == 'binary':
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        true_pos = np.sum((y_true == pos_label) & (y_pred == pos_label))
        false_pos = np.sum((y_true != pos_label) & (y_pred == pos_label))
        
        if true_pos + false_pos == 0:
            return 0.0
        
        return true_pos / (true_pos + false_pos)
    
    # For multiclass classification
    cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    
    # Calculate precision for each class
    precision = np.zeros(cm.shape[0])
    for i in range(cm.shape[0]):
        if cm[:, i].sum() > 0:
            precision[i] = cm[i, i] / cm[:, i].sum()
    
    # Average the results
    if average == 'micro':
        return np.sum(np.diag(cm)) / np.sum(cm)
    elif average == 'macro':
        return np.mean(precision)
    elif average == 'weighted':
        weights = np.sum(cm, axis=1)
        return np.sum(precision * weights) / np.sum(weights)
    elif average is None:
        return precision
    else:
        raise ValueError(f"Unsupported average parameter: {average}")


def recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, gpu_enabled=False):
    """Compute the recall score.
    
    The recall is the ratio tp / (tp + fn) where tp is the number of true positives
    and fn the number of false negatives. The recall is intuitively the ability
    of the classifier to find all the positive samples.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    labels : array-like, default=None
        The set of labels to include when average != 'binary'.
    pos_label : str or int, default=1
        The class to report if average='binary' and the data is binary.
    average : {'micro', 'macro', 'weighted', 'binary', None}, default='binary'
        This parameter is required for multiclass/multilabel targets.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration if available.
        
    Returns
    -------
    recall : float (if average is not None) or array of float of shape (n_classes,)
        Recall score.
    """
    # For binary classification
    if average == 'binary':
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        true_pos = np.sum((y_true == pos_label) & (y_pred == pos_label))
        false_neg = np.sum((y_true == pos_label) & (y_pred != pos_label))
        
        if true_pos + false_neg == 0:
            return 0.0
        
        return true_pos / (true_pos + false_neg)
    
    # For multiclass classification
    cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    
    # Calculate recall for each class
    recall = np.zeros(cm.shape[0])
    for i in range(cm.shape[0]):
        if cm[i, :].sum() > 0:
            recall[i] = cm[i, i] / cm[i, :].sum()
    
    # Average the results
    if average == 'micro':
        return np.sum(np.diag(cm)) / np.sum(cm)
    elif average == 'macro':
        return np.mean(recall)
    elif average == 'weighted':
        weights = np.sum(cm, axis=1)
        return np.sum(recall * weights) / np.sum(weights)
    elif average is None:
        return recall
    else:
        raise ValueError(f"Unsupported average parameter: {average}")


def f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, gpu_enabled=False):
    """Compute the F1 score, also known as balanced F-score or F-measure.
    
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are equal.
    The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    labels : array-like, default=None
        The set of labels to include when average != 'binary'.
    pos_label : str or int, default=1
        The class to report if average='binary' and the data is binary.
    average : {'micro', 'macro', 'weighted', 'binary', None}, default='binary'
        This parameter is required for multiclass/multilabel targets.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration if available.
        
    Returns
    -------
    f1 : float (if average is not None) or array of float of shape (n_classes,)
        F1 score.
    """
    precision = precision_score(y_true, y_pred, labels=labels, pos_label=pos_label, 
                              average=average, sample_weight=sample_weight, 
                              gpu_enabled=gpu_enabled)
    
    recall = recall_score(y_true, y_pred, labels=labels, pos_label=pos_label, 
                        average=average, sample_weight=sample_weight, 
                        gpu_enabled=gpu_enabled)
    
    # Handle the case where precision and recall are arrays
    if isinstance(precision, np.ndarray) and isinstance(recall, np.ndarray):
        # Avoid division by zero
        denominator = precision + recall
        denominator = np.where(denominator == 0, 1, denominator)
        return 2 * precision * recall / denominator
    
    # Handle the case where precision and recall are scalars
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)