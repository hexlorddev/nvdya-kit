# Nvdya Kit - Visualization Plots Module
# Provides data visualization tools with GPU acceleration

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def scatter_plot(x, y, labels=None, title=None, xlabel=None, ylabel=None, figsize=(10, 6), 
                alpha=0.7, s=50, cmap='viridis', gpu_enabled=False):
    """Create a scatter plot with optional GPU acceleration for data preparation.
    
    Parameters
    ----------
    x : array-like of shape (n_samples,)
        The x coordinates of the data points.
    y : array-like of shape (n_samples,)
        The y coordinates of the data points.
    labels : array-like of shape (n_samples,), default=None
        The labels/classes of the data points for coloring.
    title : str, default=None
        The title of the plot.
    xlabel : str, default=None
        The label for the x-axis.
    ylabel : str, default=None
        The label for the y-axis.
    figsize : tuple, default=(10, 6)
        The figure size (width, height) in inches.
    alpha : float, default=0.7
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    s : int or array-like, default=50
        The marker size in points**2.
    cmap : str or Colormap, default='viridis'
        The colormap used for mapping labels to colors.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration for data preparation if available.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Use GPU for data preparation if enabled
    if gpu_enabled:
        try:
            from ..gpu import to_gpu, to_cpu
            x_gpu = to_gpu(x)
            y_gpu = to_gpu(y)
            
            # Any data preprocessing would happen here
            # For example, normalization or filtering outliers
            
            x = to_cpu(x_gpu)
            y = to_cpu(y_gpu)
            
            if labels is not None:
                labels = to_cpu(to_gpu(np.asarray(labels)))
        except (ImportError, RuntimeError):
            gpu_enabled = False
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        scatter = ax.scatter(x, y, c=labels, alpha=alpha, s=s, cmap=cmap)
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(x, y, alpha=alpha, s=s)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    
    return fig, ax


def line_plot(x, y, labels=None, title=None, xlabel=None, ylabel=None, figsize=(10, 6),
             alpha=0.7, linewidth=2, cmap='viridis', gpu_enabled=False):
    """Create a line plot with optional GPU acceleration for data preparation.
    
    Parameters
    ----------
    x : array-like of shape (n_samples,) or list of arrays
        The x coordinates of the data points. If a list is provided, each array
        corresponds to a different line.
    y : array-like of shape (n_samples,) or list of arrays
        The y coordinates of the data points. If a list is provided, each array
        corresponds to a different line.
    labels : list of str, default=None
        The labels for each line. Must have the same length as the number of lines.
    title : str, default=None
        The title of the plot.
    xlabel : str, default=None
        The label for the x-axis.
    ylabel : str, default=None
        The label for the y-axis.
    figsize : tuple, default=(10, 6)
        The figure size (width, height) in inches.
    alpha : float, default=0.7
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    linewidth : int, default=2
        The line width in points.
    cmap : str or Colormap, default='viridis'
        The colormap used for mapping labels to colors.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration for data preparation if available.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    # Check if x and y are lists of arrays
    if not isinstance(x, list):
        x = [np.asarray(x)]
        y = [np.asarray(y)]
        multiple_lines = False
    else:
        x = [np.asarray(xi) for xi in x]
        y = [np.asarray(yi) for yi in y]
        multiple_lines = True
    
    # Use GPU for data preparation if enabled
    if gpu_enabled:
        try:
            from ..gpu import to_gpu, to_cpu
            
            # Process each line's data
            for i in range(len(x)):
                x_gpu = to_gpu(x[i])
                y_gpu = to_gpu(y[i])
                
                # Any data preprocessing would happen here
                
                x[i] = to_cpu(x_gpu)
                y[i] = to_cpu(y_gpu)
        except (ImportError, RuntimeError):
            gpu_enabled = False
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors from colormap if multiple lines
    if multiple_lines:
        n_lines = len(x)
        cmap_obj = plt.cm.get_cmap(cmap, n_lines)
        colors = [cmap_obj(i) for i in range(n_lines)]
    else:
        colors = ['blue']
    
    # Plot each line
    for i in range(len(x)):
        if labels is not None and i < len(labels):
            ax.plot(x[i], y[i], color=colors[i], alpha=alpha, linewidth=linewidth, label=labels[i])
        else:
            ax.plot(x[i], y[i], color=colors[i], alpha=alpha, linewidth=linewidth)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    if labels is not None:
        ax.legend()
    
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    
    return fig, ax


def histogram(data, bins=10, title=None, xlabel=None, ylabel='Frequency', figsize=(10, 6),
             alpha=0.7, color='blue', density=False, gpu_enabled=False):
    """Create a histogram with optional GPU acceleration for data preparation.
    
    Parameters
    ----------
    data : array-like of shape (n_samples,)
        The data to plot.
    bins : int or sequence, default=10
        The number of bins or bin edges.
    title : str, default=None
        The title of the plot.
    xlabel : str, default=None
        The label for the x-axis.
    ylabel : str, default='Frequency'
        The label for the y-axis.
    figsize : tuple, default=(10, 6)
        The figure size (width, height) in inches.
    alpha : float, default=0.7
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    color : str, default='blue'
        The color of the histogram bars.
    density : bool, default=False
        If True, the result is the value of the probability density function at the bin,
        normalized such that the integral over the range is 1.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration for data preparation if available.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    data = np.asarray(data)
    
    # Use GPU for data preparation if enabled
    if gpu_enabled:
        try:
            from ..gpu import to_gpu, to_cpu
            data_gpu = to_gpu(data)
            
            # Any data preprocessing would happen here
            
            data = to_cpu(data_gpu)
        except (ImportError, RuntimeError):
            gpu_enabled = False
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(data, bins=bins, alpha=alpha, color=color, density=density)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    
    return fig, ax


def heatmap(data, row_labels=None, col_labels=None, title=None, xlabel=None, ylabel=None,
           figsize=(10, 8), cmap='viridis', annot=True, fmt='.2f', gpu_enabled=False):
    """Create a heatmap with optional GPU acceleration for data preparation.
    
    Parameters
    ----------
    data : array-like of shape (n_rows, n_cols)
        The data to plot.
    row_labels : array-like of shape (n_rows,), default=None
        The labels for the rows.
    col_labels : array-like of shape (n_cols,), default=None
        The labels for the columns.
    title : str, default=None
        The title of the plot.
    xlabel : str, default=None
        The label for the x-axis.
    ylabel : str, default=None
        The label for the y-axis.
    figsize : tuple, default=(10, 8)
        The figure size (width, height) in inches.
    cmap : str or Colormap, default='viridis'
        The colormap used for the heatmap.
    annot : bool, default=True
        If True, write the data value in each cell.
    fmt : str, default='.2f'
        String formatting code to use when adding annotations.
    gpu_enabled : bool, default=False
        Whether to use GPU acceleration for data preparation if available.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    data = np.asarray(data)
    
    # Use GPU for data preparation if enabled
    if gpu_enabled:
        try:
            from ..gpu import to_gpu, to_cpu
            data_gpu = to_gpu(data)
            
            # Any data preprocessing would happen here
            
            data = to_cpu(data_gpu)
        except (ImportError, RuntimeError):
            gpu_enabled = False
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the heatmap
    im = ax.imshow(data, cmap=cmap)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    
    # Set labels for rows and columns
    if row_labels is not None:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
    
    if col_labels is not None:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add annotations
    if annot:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = ax.text(j, i, format(data[i, j], fmt),
                              ha="center", va="center", color="white" if data[i, j] > data.max()/2 else "black")
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    fig.tight_layout()
    
    return fig, ax