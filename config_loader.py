import yaml
from pathlib import Path
try:
    import streamlit as st
except ImportError:
    st = None

def running_on_streamlit_cloud():
    if st is not None:
        try:
            return st.secrets.get("CLOUD", False)
        except Exception:
            return False
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
