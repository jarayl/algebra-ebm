"""
Pytest configuration for algebra-ebm tests.

This module ensures the src directory is in the Python path when running tests.
"""

import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
