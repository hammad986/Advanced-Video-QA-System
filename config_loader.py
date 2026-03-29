import yaml
from pathlib import Path

def running_on_streamlit_cloud():
    try:
        import streamlit as st
        return st.secrets.get("CLOUD", False)
    except Exception:
        return False


def load_config():
    if running_on_streamlit_cloud():
        config_path = Path("config_cloud.yaml")
    else:
        config_path = Path("config_local.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
