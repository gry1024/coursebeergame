"""Load and merge YAML configuration files."""

import os
import yaml


def load_config(default_path="configs/default.yaml", algo_path=None):
    """Load the default config and overlay an algorithm-specific YAML on top.

    The algorithm YAML takes precedence over the default, so command-line
    arguments can in turn override the merged result at the caller level.
    """
    with open(default_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if algo_path is not None and os.path.isfile(algo_path):
        with open(algo_path, "r", encoding="utf-8") as f:
            algo_config = yaml.safe_load(f)
        config.update(algo_config)
    return config