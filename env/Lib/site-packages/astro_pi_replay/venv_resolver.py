import dataclasses
import logging
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Optional, Union

from astro_pi_replay import PROGRAM_NAME
from astro_pi_replay.exception import AstroPiReplayRuntimeError

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class VenvInfo:
    activate: Path
    deactivate: Optional[Path]
    executor: Optional[Path]
    pip: Path
    python: Path
    script_dir: Path
    site_packages_dir: Path


class VenvResolver:
    """
    Wrapper class used to interact with a Python venv
    """

    # TODO: move these to __init__ since this is package wide
    SUPPORTED_PLATFORMS: list[str] = ["linux", "win32", "darwin"]
    ALLOW_UNSUPPORTED_PLATFORM: bool = False

    NOT_FOUND = f"{PROGRAM_NAME} not found"

    def __init__(
        self, venv_dir: Optional[Union[str, Path]] = None, init_venv: bool = True
    ) -> None:
        self.venv_dir: Path
        if venv_dir is None and init_venv:
            logger.debug("Initialising venv...")
            self.venv_dir = self._init_venv()
        elif venv_dir is None:
            raise AstroPiReplayRuntimeError(
                "venv_dir is None " + "but init_venv is False. Aborting"
            )
        else:
            self.venv_dir = Path(venv_dir)
        if not self.venv_dir.exists() and init_venv:
            logger.debug(f"Venv {self.venv_dir} does not exist, initialising...")
            self.venv_dir = self._init_venv(self.venv_dir)
        self.platform: str = self._verify_platform()
        self.venv_info: VenvInfo = self._resolve_venv_dirs()

    def install(
        self,
        name: str,
        workdir: Optional[Path] = None,
        flags: Optional[list[str]] = None,
        editable: bool = False,
    ) -> None:
        """Executes pip install with the given args using the resolved pip"""
        before_directory: str = os.getcwd()
        chdir: bool = False
        try:
            logger.debug(f"Initial working directory: {str(os.getcwd())}")
            if workdir is not None:
                os.chdir(workdir)
                chdir = True
            print_name: str = name
            if name == os.curdir and workdir is not None:
                print_name = workdir.name
            logger.debug(f"Installing {print_name} into venv...")
            args: list[str] = [str(self.venv_info.pip), "install", name]
            if editable:
                args.insert(2, "--editable")
            if flags is not None:
                args = [str(self.venv_info.pip), "install"] + flags + [name]
            logger.debug(" ".join(args))
            subprocess.run(args, check=True)  # nosec B603: no user input
        finally:
            if chdir:
                os.chdir(before_directory)

    def is_package_installed(self, name: str) -> Optional[str]:
        """
        Runs a program using the venv Python to check if
        the given package is installed.
        """
        dynamic_program: str = "; ".join(
            [
                "import importlib.util",
                "from pathlib import Path",
                "module = importlib.util.find_spec(" + f"'{name}')",
                f"to_print = '{VenvResolver.NOT_FOUND}' if module is None "
                + "else Path(module.origin).parent",
                "print(to_print)",
            ]
        )

        args: list[str] = [str(self.venv_info.python), "-c", dynamic_program]
        logger.debug(" ".join(args))
        out = subprocess.run(  # nosec B603: no user input
            args,
            check=True,
            capture_output=True,
            text=True,
        )

        stripped: str = out.stdout.strip()
        found: bool = stripped != VenvResolver.NOT_FOUND
        if found:
            logger.debug(f"Found {PROGRAM_NAME} installed at {stripped}")
            return stripped
        return None

    def is_in_venv(self) -> bool:
        return sys.prefix != sys.base_prefix

    def list_dependencies(self) -> str:
        """Executes pip freeze using the resolved pip"""
        args: list[str] = [str(self.venv_info.python), "-m", "pip", "freeze"]
        logger.debug(" ".join(args))
        out = subprocess.run(
            args, text=True, check=True, capture_output=True, shell=False
        )  # nosec B603
        logger.debug(out)
        return out.stdout

    def _resolve_venv_dirs(self) -> VenvInfo:
        activate: Path
        deactivate: Optional[Path]
        executor: Optional[Path]
        pip: Path
        python: Path
        script_dir: Path
        site_packages_dir: Path

        script_dir = self.venv_dir / "bin"
        site_packages_dir = self.venv_dir / "lib"
        site_packages_dir /= f"python{sys.version_info.major}.{sys.version_info.minor}"
        site_packages_dir /= "site-packages"
        activate = script_dir / "activate"
        deactivate = script_dir / "deactivate"
        executor = script_dir / PROGRAM_NAME
        pip = script_dir / "pip"
        python = script_dir / "python"

        if self.platform in ["win32"]:
            script_dir = self.venv_dir / "Scripts"
            site_packages_dir = self.venv_dir / "Lib" / "site-packages"
            activate = script_dir / (activate.name + ".bat")
            deactivate = script_dir / (deactivate.name + ".bat")
            executor = script_dir / (executor.name + ".exe")
            pip = script_dir / (pip.name + ".exe")
            python = script_dir / (python.name + ".exe")

        mandatory_paths: list[Path] = [
            script_dir,
            site_packages_dir,
            activate,
            pip,
            python,
        ]
        for path in mandatory_paths:
            if not path.exists():
                error_message: str = (
                    f"Couldn't resolve {path} in {self.venv_dir}. "
                    + "Try deleting the venv and recreating. If this issue persists, "
                    + f"please log an issue on github.com/{PROGRAM_NAME}."
                )
                logger.error(error_message)
                raise AstroPiReplayRuntimeError(error_message)

        return VenvInfo(
            activate, deactivate, executor, pip, python, script_dir, site_packages_dir
        )

    def _init_venv(self, venv_dir: Path = Path("venv")) -> Path:
        if self.is_in_venv():
            logger.debug(
                "Detected that you running in a venv:"
                + f"\n\t{sys.prefix}.\n"
                + "However, running in replay mode will use a "
                + "separate copied (modified) venv."
            )
            logger.info("Preparing environment (this may take a few moments)...")
            shutil.copytree(sys.prefix, venv_dir, symlinks=True)
        else:
            venv.create(
                venv_dir, symlinks=True, system_site_packages=True, with_pip=True
            )

        return venv_dir

    def _verify_platform(self) -> str:
        """Verifies that the curent platform is supported. Returns
        True if the current platform is supported, otherwise raises
        an AstroPiExecutorRuntimeError.
        """
        if (
            sys.platform not in VenvResolver.SUPPORTED_PLATFORMS
            and not VenvResolver.ALLOW_UNSUPPORTED_PLATFORM
        ):
            message: str = os.linesep.join(
                [
                    f"{sys.platform} is not supported. The currently "
                    + "supported platforms are: "
                    + f"{VenvResolver.SUPPORTED_PLATFORMS}. To override this "
                    + "use the --allow-unsupported-platform option."
                ]
            )
            logger.error(message)
            raise AstroPiReplayRuntimeError(message)
        return sys.platform


"""
Set-ExecutionPolicy Bypass -Scope Process -Force
# and then, for example:
venv\\Scripts\\Activate.ps1
"""
