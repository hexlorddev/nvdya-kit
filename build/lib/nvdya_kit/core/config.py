# Nvdya Kit - Configuration Module
# Manages global configuration settings

import threading

# Thread-local storage for configuration
_config_local = threading.local()

# Default configuration
_DEFAULT_CONFIG = {
    'gpu_enabled': False,           # Whether to use GPU acceleration by default
    'gpu_memory_fraction': 0.8,     # Fraction of GPU memory to use
    'n_jobs': -1,                   # Number of CPU cores to use (-1 means all)
    'random_state': None,           # Random seed for reproducibility
    'verbose': 1,                   # Verbosity level (0=silent, 1=normal, 2=detailed)
    'precision': 'float32',         # Default precision for computations
    'cache_dir': None,              # Directory for caching data
    'use_automl': False,            # Whether to use AutoML features by default
    'distributed_backend': None,    # Backend for distributed computing (None, 'dask', etc.)
}

def get_config():
    """Get the current configuration.
    
    Returns
    -------
    config : dict
        Current configuration settings.
    """
    if not hasattr(_config_local, 'config'):
        _config_local.config = _DEFAULT_CONFIG.copy()
    return _config_local.config

def set_config(**kwargs):
    """Set configuration parameters.
    
    Parameters
    ----------
    **kwargs : dict
        Configuration settings to update.
        
    Returns
    -------
    config : dict
        Updated configuration settings.
    """
    config = get_config()
    
    # Update configuration with provided values
    for key, value in kwargs.items():
        if key not in _DEFAULT_CONFIG:
            raise ValueError(f"Unknown configuration parameter: {key}")
        config[key] = value
    
    return config

def reset_config():
    """Reset configuration to default values.
    
    Returns
    -------
    config : dict
        Default configuration settings.
    """
    _config_local.config = _DEFAULT_CONFIG.copy()
    return _config_local.config