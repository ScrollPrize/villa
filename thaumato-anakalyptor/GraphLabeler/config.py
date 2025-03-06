import json, os

def load_config(config_file="config_labeling_gui.json"):
    """
    Load configuration from a JSON file. Returns a dictionary.
    """
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            return config
        except Exception as e:
            print("Error loading config:", e)
            return {}
    else:
        print("Config file not found.")
        return {}

def save_config(config, config_file="config_labeling_gui.json"):
    """
    Save the configuration dictionary to a JSON file.
    """
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        print("Config saved.")
    except Exception as e:
        print("Error saving config:", e)
