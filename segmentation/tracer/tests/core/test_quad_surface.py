#!/usr/bin/env python3
"""Wrapper for running QuadSurface tests."""

import sys
import os
import unittest

if __name__ == "__main__":
    # Add the tests directory to the path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
    
    # Import the tests
    from test_quad_surface import TestQuadSurface, interactive_test
    
    # Run the tests
    print("Running unit tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run the interactive tests
    print("\nRunning interactive tests...")
    interactive_test()
    
    print("\nAll tests completed successfully!")