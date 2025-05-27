# Nvdya Kit - GPU Acceleration Module
# Provides CUDA-powered implementations for faster machine learning

import logging
import warnings

logger = logging.getLogger(__name__)

# Try to import GPU libraries
try:
    import cupy as cp
    import numba.cuda
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False
    logger.warning("GPU acceleration libraries (cupy, numba.cuda) not found. "
                 "GPU functionality will be disabled.")

def is_gpu_available():
    """Check if GPU acceleration is available.
    
    Returns
    -------
    bool
        True if GPU acceleration is available, False otherwise.
    """
    if not _HAS_GPU:
        return False
    
    try:
        # Check if CUDA is available
        if numba.cuda.is_available():
            # Get number of devices
            n_devices = numba.cuda.get_num_devices()
            if n_devices > 0:
                logger.info(f"Found {n_devices} CUDA-capable GPU device(s)")
                return True
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {str(e)}")
    
    return False

def get_gpu_memory_info():
    """Get information about GPU memory usage.
    
    Returns
    -------
    dict
        Dictionary containing memory information for each GPU.
    """
    if not is_gpu_available():
        return {}
    
    try:
        memory_info = {}
        for i in range(numba.cuda.get_num_devices()):
            with numba.cuda.gpus[i]:
                device = numba.cuda.get_current_device()
                free, total = numba.cuda.current_context().get_memory_info()
                memory_info[i] = {
                    'device_name': device.name,
                    'total_memory': total,
                    'free_memory': free,
                    'used_memory': total - free,
                    'memory_usage_percent': (total - free) / total * 100
                }
        return memory_info
    except Exception as e:
        logger.warning(f"Error getting GPU memory info: {str(e)}")
        return {}

def to_gpu(obj):
    """Transfer data to GPU memory.
    
    Parameters
    ----------
    obj : object
        Object to transfer to GPU. Can be a numpy array, a model,
        or any object with array attributes.
        
    Returns
    -------
    object
        Object with data on GPU.
    """
    if not is_gpu_available():
        warnings.warn("GPU not available. Data will remain on CPU.")
        return obj
    
    try:
        import numpy as np
        
        # If it's a numpy array, convert to cupy array
        if isinstance(obj, np.ndarray):
            return cp.asarray(obj)
        
        # If it's a model, transfer its array attributes to GPU
        elif hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                if isinstance(value, np.ndarray):
                    setattr(obj, key, cp.asarray(value))
                elif hasattr(value, '__dict__'):
                    setattr(obj, key, to_gpu(value))
            return obj
        
        # Return the object as is if we can't transfer it
        return obj
    
    except Exception as e:
        logger.error(f"Error transferring data to GPU: {str(e)}")
        raise RuntimeError(f"Failed to transfer data to GPU: {str(e)}")

def to_cpu(obj):
    """Transfer data from GPU to CPU memory.
    
    Parameters
    ----------
    obj : object
        Object to transfer to CPU. Can be a cupy array, a model,
        or any object with array attributes.
        
    Returns
    -------
    object
        Object with data on CPU.
    """
    try:
        # If it's a cupy array, convert to numpy array
        if _HAS_GPU and isinstance(obj, cp.ndarray):
            return obj.get()
        
        # If it's a model, transfer its array attributes to CPU
        elif hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                if _HAS_GPU and isinstance(value, cp.ndarray):
                    setattr(obj, key, value.get())
                elif hasattr(value, '__dict__'):
                    setattr(obj, key, to_cpu(value))
            return obj
        
        # Return the object as is if we can't transfer it
        return obj
    
    except Exception as e:
        logger.error(f"Error transferring data to CPU: {str(e)}")
        raise RuntimeError(f"Failed to transfer data to CPU: {str(e)}")

# Export public functions
__all__ = [
    'is_gpu_available',
    'get_gpu_memory_info',
    'to_gpu',
    'to_cpu',
]