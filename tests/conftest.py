"""Pytest configuration and global fixtures for ximinf tests."""

import sys
from pathlib import Path
import matplotlib

# Use a non-GUI backend for matplotlib during tests to prevent GUI popups or hangs in CI
matplotlib.use("Agg")

# Ensure the src directory is in sys.path so tests can import ximinf directly
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
