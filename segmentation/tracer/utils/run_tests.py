#!/usr/bin/env python3
"""
Script to run tests for the tracer package.

This script provides a simple command-line interface for running tests
in the tracer package. It supports running tests for specific components
or all tests.

Examples:
    python run_tests.py  # Run all tests
    python run_tests.py cost_functions  # Run only cost function tests
    python run_tests.py core/test_grid.py  # Run a specific test module
"""

import sys
import unittest
import argparse
from pathlib import Path

def run_tests(test_path=None):
    """Run tests from the specified path."""
    if test_path is None:
        # Run all tests
        test_loader = unittest.defaultTestLoader
        test_suite = test_loader.discover('../tests')
    else:
        # Normalize path to handle both file and directory paths
        test_path = str(test_path)
        
        # Make sure path starts with tests/ for proper module importing
        if not test_path.startswith('tests'):
            test_path = 'tests/' + test_path.lstrip('./') if test_path else 'tests'
            
        # Check if this is a specific file or a directory
        path_obj = Path(test_path)
        if path_obj.is_file() or test_path.endswith('.py'):
            # Extract module name from path
            # For example, "tests/cost_functions/test_file.py" -> "tests.cost_functions.test_file"
            module_name = str(test_path).replace('/', '.').replace('.py', '')
            
            # Import the module and run its tests
            try:
                module = __import__(module_name, fromlist=['*'])
                test_suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            except ImportError as e:
                print(f"Error importing {module_name}: {e}")
                return unittest.TestResult()
        else:
            # It's a directory
            test_loader = unittest.defaultTestLoader
            test_suite = test_loader.discover(test_path)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(test_suite)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tracer tests')
    parser.add_argument('test_path', nargs='?', help='Path to specific test or directory')
    args = parser.parse_args()
    
    result = run_tests(args.test_path)
    sys.exit(not result.wasSuccessful())