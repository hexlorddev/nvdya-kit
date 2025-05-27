# Nvdya Kit
# A Comprehensive Machine Learning Library
# By Dneth Nethsara and DPC Media Unit

__version__ = "0.1.0"

# Import main modules for easy access
from . import models
from . import preprocessing
from . import metrics
from . import visualization
from . import gpu
from . import automl
from . import distributed
from . import utils

# Add new module imports for advanced features
from . import model_zoo
from . import deep_learning
from . import automl_plus
from . import explainability
from . import production
from . import big_data
from . import graph_ml
from . import rl_suite
from . import interactive_viz
from . import security
from . import docs

# Set up logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add handler to logger
logger.addHandler(ch)

# Check for GPU availability
try:
    from .gpu import is_gpu_available
    if is_gpu_available():
        logger.info("GPU acceleration is available and enabled")
    else:
        logger.info("GPU acceleration is not available. Using CPU only.")
except ImportError:
    logger.warning("GPU module could not be imported. Using CPU only.")

# Print welcome message
logger.info(f"Nvdya Kit v{__version__} loaded successfully")
