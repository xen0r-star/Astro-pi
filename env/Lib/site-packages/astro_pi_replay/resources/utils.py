import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from astro_pi_replay import PROGRAM_NAME
from astro_pi_replay.configuration import Configuration

RESOURCE_DIR: Path = Path(__file__).parent
EXPECTED_DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S.%f"
REPLAY_SEQUENCE_ENV_VAR: str = f"{PROGRAM_NAME.upper()}_REPLAY_SEQUENCE"
SENSE_HAT_CSV_FILE: Path = Path("data") / "data.csv"


def get_resource(path_relative_to_resources_dir: Union[str, Path]) -> Path:
    """
    Finds the given resource in the resource dir.
    """

    path = RESOURCE_DIR / path_relative_to_resources_dir
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path_relative_to_resources_dir}" + f" in '{RESOURCE_DIR}'"
        )
    return path


def get_replay_dir() -> Path:
    return get_resource("replay")


def get_replay_sequence_dir() -> Path:
    replay_dir: Path = get_replay_dir()

    # precedence is ENV VARS (so that testing works)
    # and then cli (hence config)
    try:
        config = Configuration.load()
        if config.sequence is not None:
            for photography_type in (
                f for f in os.listdir(replay_dir) if not f.startswith(".")
            ):
                if config.sequence in (
                    f
                    for f in os.listdir(replay_dir / photography_type)
                    if not f.startswith(".")
                ):
                    return replay_dir / photography_type / config.sequence
    except FileNotFoundError:
        pass

    replay_sequence: Optional[str] = os.environ.get(REPLAY_SEQUENCE_ENV_VAR)
    if replay_sequence is not None:
        return replay_dir / Path(replay_sequence)
    raise FileNotFoundError(f"Could not find the sequence {replay_sequence} to replay.")


def get_metadata(key: str) -> Any:
    """
    Loads the photo album metadata
    """
    # TODO load the file once
    with (get_replay_sequence_dir() / "metadata.json").open() as f:
        metadata: dict[str, Any] = json.loads(f.read())
        return metadata[key]


def get_start_time() -> datetime:
    return datetime.strptime(get_metadata("start"), EXPECTED_DATETIME_FORMAT)
