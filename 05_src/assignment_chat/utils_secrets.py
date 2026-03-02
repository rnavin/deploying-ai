import os

def apply_secrets_to_env(secrets_path: str) -> None:
    """
    Reads KEY=VALUE lines from a .secrets file and sets os.environ.
    Ignores empty lines and comments (#).
    """
    if not os.path.exists(secrets_path):
        raise FileNotFoundError(f".secrets file not found at: {secrets_path}")

    with open(secrets_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()