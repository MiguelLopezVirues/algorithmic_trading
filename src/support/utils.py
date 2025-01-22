import yaml
import os


def load_tickers_to_include():
    base_dir = os.path.dirname(__file__)
    yaml_file_path = os.path.join(base_dir, "config", "tickers_to_include.yaml")
    with open(yaml_file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_region_config():
    base_dir = os.path.dirname(__file__)
    yaml_file_path = os.path.join(base_dir, "config", "countries_by_region.yaml")
    with open(yaml_file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_region(country: str, config=None) -> str:
    if config is None:
        config = load_region_config()

    c = country.strip().lower()

    for region_key, countries_list in config.items():
        if any(c == item.lower() for item in countries_list):
            return region_key

    return "OTHERS"



