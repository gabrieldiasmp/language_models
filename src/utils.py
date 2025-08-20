import os
from pathlib import Path

def load_env_file(env_path=".env"):
    env_file = Path(env_path)
    if not env_file.exists():
        raise FileNotFoundError(f"{env_path} does not exist.")

    with env_file.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value
