"""VECTRA software reproduction: classical SfM + learned dense reconstruction."""
import os
# PyTorch and Open3D each bundle libomp; allow the duplicate to avoid OMP Error #15.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "6")
