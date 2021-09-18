from pycuda import autoinit
import numpy as np
from skcuda import linalg

linalg.eye(3, dtype=np.float32)