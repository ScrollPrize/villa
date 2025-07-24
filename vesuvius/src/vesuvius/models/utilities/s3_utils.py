import multiprocessing 

def detect_s3_paths(mgr):
    """
    Detect if any data paths are S3 URLs.
    
    Parameters
    ----------
    mgr : ConfigManager
        Configuration manager instance
        
    Returns
    -------
    bool
        True if S3 paths are detected, False otherwise
    """
    # Check data_paths in dataset_config
    if hasattr(mgr, 'dataset_config') and mgr.dataset_config:
        data_paths = mgr.dataset_config.get('data_paths', [])
        if data_paths:
            for path in data_paths:
                if isinstance(path, str) and path.startswith('s3://'):
                    return True
    
    # Check data_path
    if hasattr(mgr, 'data_path') and mgr.data_path:
        if str(mgr.data_path).startswith('s3://'):
            return True
            
    return False


def setup_multiprocessing_for_s3():

    multiprocessing.set_start_method('spawn', force=True)
    print(f"multprocessing start method is currently set to {multiprocessing.get_start_method()}")
