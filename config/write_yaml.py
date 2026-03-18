import tomllib
import yaml

# Load TOML
with open("config.toml", "rb") as f:
    data = tomllib.load(f)

conda_packages = data["dependencies"]["conda"]
pip_packages = data["dependencies"]["pip"]

# Build YAML structure
env = {
    "name": "nsa_env",
    "channels": ["conda-forge", "defaults"],
    "dependencies": conda_packages + [
        {"pip": pip_packages}
    ]
}

# Write environment.yml
with open("environment.yml", "w") as f:
    yaml.dump(env, f, sort_keys=False)

