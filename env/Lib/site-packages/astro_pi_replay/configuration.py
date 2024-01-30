import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from astro_pi_replay import PROGRAM_NAME

logger = logging.getLogger(__name__)

CONFIG_FILE_ENV_VAR: str = f"{PROGRAM_NAME.upper()}_CONFIG_FILE"
CONFIG_FILE: Path = Path.home() / f".{PROGRAM_NAME}" / "config.json"


def get_config_file_path() -> Path:
    config_file: Optional[str] = os.environ.get(CONFIG_FILE_ENV_VAR)
    if config_file is not None:
        return Path(config_file)
    return CONFIG_FILE


@dataclass
class Configuration:
    """
    Persistent configuration stored in the home directory.
    """

    no_wait_images: bool
    interpolate_sense_hat: bool
    debug: bool
    sequence: Optional[str]

    @staticmethod
    def _from_json(jstr: str) -> "Configuration":
        return Configuration(**json.loads(jstr))

    @staticmethod
    def from_args(args: argparse.Namespace) -> "Configuration":
        return Configuration(
            args.no_match_original_photo_intervals,
            args.interpolate_sense_hat,
            args.debug,
            args.sequence,
        )

    @staticmethod
    def load() -> "Configuration":
        """
        Loads the current configuration from the file
        """
        with get_config_file_path().open() as f:
            return Configuration._from_json(f.read())

    # instance methods
    def _to_json(self) -> str:
        # filter out hidden attributes
        return json.dumps(
            dict(
                {
                    (key, value)
                    for key, value in self.__dict__.items()
                    if not key.startswith("_")
                }
            )
        )

    def save(self) -> None:
        """Writes out the configuration to config.json"""
        config_file = get_config_file_path()
        config_file.parent.mkdir(exist_ok=True)
        if config_file.exists():
            logger.debug("Overwriting config.json file")
        with config_file.open("w") as f:
            f.write(self._to_json())
