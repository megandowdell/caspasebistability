# utils.py
import os
import pandas as pd

def ensure_dir(directory):
    """
    Ensure a directory exists. If it doesn't, create it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_csv(data, path, mode="w", include_header=True):
    """
    Write a list of dictionaries to a CSV file.

    Args:
        data (list[dict]): The data to write.
        path (str): Destination CSV file path.
        mode (str): Write mode, e.g., "w" or "a".
        include_header (bool): Whether to include the header row.
    """
    df = pd.DataFrame(data)
    df.to_csv(path, mode=mode, index=False, header=include_header)

def format_param_table(param_dict):
    """
    Returns a string with parameter values for use in plots.

    Args:
        param_dict (dict): Dictionary of parameter names and values.

    Returns:
        str: Multi-line formatted string for annotations.
    """
    return "\n".join([f"{str(k)} = {v:.2e}" for k, v in param_dict.items()])
