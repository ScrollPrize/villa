# Cost Functions Refactoring Plan

## Implementation Status
We've successfully completed the initial refactoring of the cost functions codebase:

1. Created a new directory structure:
   - `cost_functions/base/` for regular cost function implementations
   - `cost_functions/autodiff/` for auto-differentiable implementations
   - `tracer/core/interpolation/` for core interpolation utilities

2. Moved cost function files to their appropriate directories.

3. Updated import statements in several key files:
   - `optimizer.py`
   - Test files like `test_theseus_integration.py`
   - Fixed import paths to maintain compatibility

4. The main integration tests are now passing, confirming that the refactored structure works properly.

We still have some remaining tasks to complete the refactoring, particularly updating imports in all test files and finalizing the documentation.

## Original Issues (✅ = Addressed)
- ✅ Manual and autodiff implementations are scattered across different files
  - Now organized into base/ and autodiff/ directories
- ❌ Tests are duplicated between regular and autodiff versions
  - Still need to create shared test base classes
- ✅ Naming conventions could be more consistent
  - Directory structure now more clearly reflects the purpose
- ✅ Trilinear interpolator is in cost_functions but is more of a core utility
  - Moved to tracer/core/interpolation/
- ✅ File organization doesn't reflect logical structure of the components
  - New structure better reflects the logical organization

## Remaining Issues
- Some import paths in tests still need to be updated
- The old trilinear_interpolator files still exist in cost_functions/
- Need to create unified test patterns for regular and autodiff versions
- Documentation needs to be updated to reflect the new structure

## Proposed Structure

```
cost_functions/
  │
  ├── base/           # Base implementations with manual Jacobians
  │   ├── __init__.py
  │   ├── anchor_loss.py
  │   ├── dist_loss.py
  │   └── ...
  │
  ├── autodiff/       # AutoDiff implementations
  │   ├── __init__.py
  │   ├── anchor_loss.py
  │   ├── dist_loss.py
  │   └── ...
  │
  ├── __init__.py     # Main init that imports from both subfolders
  │
  └── utils/          # Shared utilities for cost functions
      ├── __init__.py
      └── ...

tracer/
  │
  ├── core/           # Core functionality
  │   ├── __init__.py
  │   ├── interpolation/     # Interpolation utils
  │   │   ├── __init__.py
  │   │   ├── trilinear_interpolator.py
  │   │   └── trilinear_interpolator_autodiff.py
  │   └── ...
  │
  └── ...
```

## Refactoring Steps

1. **Create Folder Structure**
   - Create base, autodiff, and utils directories in cost_functions
   - Create core/interpolation directory in tracer
   - Set up proper __init__.py files

2. **Move Core Utilities**
   - Move trilinear_interpolator.py and trilinear_interpolator_autodiff.py to tracer/core/interpolation/
   - Update imports in all files that use these interpolators

3. **Move Cost Function Files**
   - Move regular implementations to base/
   - Move autodiff implementations to autodiff/
   - Rename files as needed for consistency
   
4. **Update Imports**
   - Update import statements in all files
   - Ensure main __init__.py properly exposes all cost functions
   - Ensure imports for trilinear interpolator are updated throughout codebase

5. **Refactor Tests**
   - Create test base classes that test both implementations
   - Refactor tests to reduce duplication
   - Add parametrization to run tests against both implementations
   - Move interpolator tests to appropriate location

6. **Documentation**
   - Add clear docstrings about the structure
   - Update any comments about autodiff vs. manual implementations
   - Document the interpolator's new location

## Implementation Timeline
1. ✅ Create folder structure 
2. ✅ Move core utilities (interpolator)
3. ✅ Move cost functions and update imports
4. ✅ Update imports in main files (optimizer.py, etc.)
5. ✅ Update imports in test files (test_theseus_integration.py, etc.)
6. 🔄 Verify all tests pass
   - ✅ Single function tests passing
   - ✅ Theseus integration tests passing 
   - ❌ Some remaining tests failing due to import paths
7. 🔄 Documentation updates
   - ❌ Update READMEs
   - ❌ Improve docstrings for the new structure

## Pending Tasks
1. Update imports in all remaining test files
2. Verify no files are importing from the wrong locations
3. Create a consistent test pattern that can be shared between regular and autodiff versions
4. Remove now-redundant trilinear_interpolator files from cost_functions/
5. Document the new file structure in README.md and other documentation