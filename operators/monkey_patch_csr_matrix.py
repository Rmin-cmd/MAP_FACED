# --- monkey_patch_csr_array.py ---

# 1) Import the private module
import scipy.sparse.csr as _csr

# 2) Bring in the old csr_matrix
from scipy.sparse import csr_matrix

# 3) If scipy didn’t define csr_array yet, alias it to csr_matrix
if not hasattr(_csr, 'csr_array'):
    _csr.csr_array = csr_matrix
