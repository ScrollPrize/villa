"""Test utilities for enhanced test output and debugging."""

import sys
import unittest
import traceback


def run_tests_with_full_stack_traces(test_case_class):
    """Configure and run tests with full stack traces.
    
    Parameters
    ----------
    test_case_class : unittest.TestCase
        The test case class to run
        
    Returns
    -------
    unittest.TestResult
        The test result
    """
    # Set the default traceback display to show full stack traces
    sys.tracebacklimit = None
    
    # Create test suite with the specified test case
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(test_case_class))
    
    # Configure the test runner to show verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    return runner.run(test_suite)


class TestCaseWithFullStackTrace(unittest.TestCase):
    """Base TestCase class that provides full stack traces on errors.
    
    Inherit from this class instead of unittest.TestCase to get
    full stack traces automatically without having to use try/except
    in every test method.
    """
    
    def run(self, result=None):
        """Run the test with enhanced error reporting."""
        # Set unlimited stack trace depth
        old_tb_limit = getattr(sys, 'tracebacklimit', 1000)
        sys.tracebacklimit = None
        
        try:
            super().run(result)
        except Exception as e:
            # Print full stack trace
            traceback.print_exc()
            raise
        finally:
            # Restore previous traceback limit
            sys.tracebacklimit = old_tb_limit