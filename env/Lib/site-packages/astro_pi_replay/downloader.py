import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Optional, TypeVar

import requests
from tqdm import tqdm

from astro_pi_replay import PROGRAM_NAME, __version__
from astro_pi_replay.exception import AstroPiReplayException
from astro_pi_replay.resources import REPLAY_SEQUENCE_ENV_VAR, get_replay_dir

logger = logging.getLogger(__name__)

GPG_EMAIL = "enquiries@astro-pi.org"
URL_BASE: str = "https://static.raspberrypi.org/files/astro-pi"
GPG_KEY_URL = f"{URL_BASE}/astro-pi.gpg"  # TODO add key-rotation
url_prefix: str = f"{URL_BASE}/{PROGRAM_NAME}/{__version__}"

T = TypeVar("T")

ONE_HOUR: int = 60 * 60


class Downloader:
    TEST_ASSETS = "test_data"

    def __init__(self) -> None:
        tempdir: Path = Path(tempfile.gettempdir())
        tempdir /= str(uuid.uuid4())
        tempdir.mkdir()
        self.tempdir = tempdir

    def _check_sha256(self, sha256_file: Path) -> Optional[bool]:
        with sha256_file.open("r") as f:
            lines: list[str] = f.read().strip().split("\n")
        for line in lines:
            split_line: list[str] = re.split(r"\s+", line)
            if len(split_line) != 2:
                raise ValueError(f"File {sha256_file.name} has an invalid format")
            expected_sha256, filename = split_line[0], Path(split_line[1])
            with (sha256_file.parent / filename).open("rb") as f:
                actual_sha256 = hashlib.sha256(f.read()).hexdigest()
            if expected_sha256 != actual_sha256:
                return False
        return True

    def _check_gpg_signature(self, gpg_sig_file: Path) -> Optional[bool]:
        logger.debug("Checking if gpg is installed...")
        gpg_path = shutil.which("gpg")

        if gpg_path is not None and Path(gpg_path).exists():
            logger.debug("Checking if the astro pi GPG key has been imported...")
            command_args = ["gpg", "--list-public-keys", f"<{GPG_EMAIL}>"]
            logger.debug(" ".join(command_args))
            proc = subprocess.run(  # nosec B603
                command_args,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if proc.returncode != 0:
                logger.info(
                    f"GPG public key for {GPG_EMAIL} not found. "
                    + "Skipping integrity check"
                )
                commands: str = os.linesep.join(
                    [
                        f"wget {GPG_KEY_URL}",
                        f"gpg --import {GPG_KEY_URL.split('/')[-1]}",
                    ]
                )
                "\n"
                logger.info(f"You may import a key with: {commands}")
                return None

            logger.debug("Verifying the integrity of the download")
            command_args = [
                "gpg",
                "--verify",
                str(gpg_sig_file),
                str(gpg_sig_file).replace(gpg_sig_file.suffix, ""),
            ]
            logger.debug(" ".join(command_args))
            proc = subprocess.run(  # nosec B603
                command_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return True if proc.returncode == 0 else False
        return None

    def _unzip(self, zip_file: Path) -> Path:
        with zipfile.ZipFile(str(zip_file)) as z:
            for member in tqdm(z.infolist(), unit="iB"):
                logger.debug(member)
                try:
                    z.extract(member, self.tempdir)
                except zipfile.error as e:
                    logger.error(e)

        os.remove(zip_file)
        return self.tempdir

    def download_file(self, url: str, destination_dir: Path) -> Path:
        local_filename: str = url.split("/")[-1]
        with requests.get(url, stream=True, timeout=ONE_HOUR) as r:
            r.raise_for_status()
            total_length = int(r.headers.get("content-length", 0))
            chunk_size = 5 * 1024
            prog_bar = tqdm(total=total_length, unit="iB", unit_scale=True)
            with (self.tempdir / local_filename).open("wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    prog_bar.update(len(chunk))
                    f.write(chunk)
            if self.tempdir != destination_dir:
                shutil.copy2(
                    self.tempdir / local_filename, destination_dir / local_filename
                )

        logger.debug(f"Download to {destination_dir / local_filename}")
        return destination_dir / local_filename

    def download(self, asset_name: str) -> None:
        downloaded: list[Path] = []
        asset_name += ".zip"
        for file in [f"{asset_name}.sha256", f"{asset_name}.sig", f"{asset_name}"]:
            logger.info(f"Downloading {file}...")
            url = f"{url_prefix}/{file}"
            downloaded.append(self.download_file(url, self.tempdir))

        logger.debug(f"Tempdir {self.tempdir} contains: {os.listdir(self.tempdir)}")
        logger.info("Checking the integrity of the downloaded data...")
        if not self._check_sha256(downloaded[0]):
            raise AstroPiReplayException(
                "Downloaded file failed integrity check. Try again."
            )
        result: Optional[bool] = self._check_gpg_signature(downloaded[1])

        if result is not None and result is False:
            raise AstroPiReplayException("Downloaded file failed security check.")
        else:
            os.remove(downloaded[0])
            os.remove(downloaded[1])

    def has_downloaded(self, asset_name: str) -> bool:
        return f"{asset_name}" in os.listdir(self.tempdir)

    def has_installed(
        self,
        resolution: tuple[int, int],
        photography_type: str,
        sequence_name: Optional[str] = None,
    ) -> bool:
        sequence: str
        if sequence_name is None:
            sequence = os.environ.get(
                REPLAY_SEQUENCE_ENV_VAR,
                self.search_for_sequence(resolution, photography_type),
            )
        else:
            sequence = sequence_name

        try:
            return (get_replay_dir() / f"{photography_type}/{sequence}").exists()
        except FileNotFoundError:
            return False

    def search_for_sequence(
        self, resolution: tuple[int, int], photography_type: str
    ) -> str:
        sequence: str
        if photography_type == "IR":
            raise AstroPiReplayException("No IR data is currently available")
        if resolution == (4056, 3040):
            # download AstroX
            sequence = "AstroX"
        elif resolution == (1280, 720):
            sequence = "OrbitAz"
        else:
            raise AstroPiReplayException(
                f"No photos with resolution {resolution} " + "are available"
            )
        return sequence

    def install(
        self,
        resolution: tuple[int, int],
        photography_type: str,
        sequence: Optional[str],
        test_assets_only: bool = False,
        with_video: bool = False,
    ) -> None:
        sequence_id: str
        if test_assets_only:
            sequence_id = "test_data"
        elif sequence is not None:
            sequence_id = sequence
        else:
            sequence_id = self.search_for_sequence(resolution, photography_type)

        sequences_to_install: list[str] = [sequence_id]
        if with_video:
            sequences_to_install.append(sequence_id + "_videos")

        for seq in sequences_to_install:
            logger.debug(f"Request to install {seq}")

            if self.has_installed(resolution, photography_type, seq):
                logger.debug(f"{seq} already installed")
                continue
            if not self.has_downloaded(seq):
                self.download(seq)

            downloaded_file: Path = self.tempdir / (seq + ".zip")
            unzipped_dir: Path = self._unzip(downloaded_file)
            destination_dir: Path = get_replay_dir() / photography_type

            if unzipped_dir != destination_dir:
                shutil.copytree(unzipped_dir, destination_dir, dirs_exist_ok=True)
