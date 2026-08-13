import os
import sys

def get_tqdm_kwargs():
    """
    Returns a dictionary of tqdm settings optimized for Kubernetes environments if TQDM_K8S is set.

    This configuration is designed to work well in environments where the console width
    may not be dynamic, such as Kubernetes pods, and ensures that progress bars are
    updated at reasonable intervals without overwhelming the output.

    Returns:
        dict: A dictionary of parameters for tqdm.
    """
    if os.getenv('TQDM_K8S') is not None:
        # Kubernetes environment settings
        return {
            'disable': False,
            'dynamic_ncols': False,  # Disable dynamic width
            'ncols': 80,              # Fixed width
            'leave': True,
            'file': sys.stdout,
            'mininterval': 5.0,       # Update every 5 seconds
            'maxinterval': 10.0,      # Force update at least every 10 seconds
            'miniters': 1,            # Update after at least 1 iteration
            'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n'
        }
    else:
        return {}
