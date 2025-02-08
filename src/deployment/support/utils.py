import yaml
import os


def load_config(file_name):
    base_dir = os.path.dirname(__file__)
    yaml_file_path = os.path.join(base_dir, "config", file_name)
    with open(yaml_file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_region(country: str, config=None) -> str:
    if config is None:
        config = load_config("countries_by_region.yaml")

    c = country.strip().lower()

    for region_key, countries_list in config.items():
        if any(c == item.lower() for item in countries_list):
            return region_key

    return "OTHERS"

def get_index_country(symbol: str, config=None) -> str:
    if config is None:
        config = load_config("indices_by_country.yaml")

    for country_key, symbols_list in config.items():
        if symbol in symbols_list:
            return country_key

    return "OTHERS"



