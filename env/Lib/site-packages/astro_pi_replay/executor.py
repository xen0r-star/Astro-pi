import collections
import functools
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timedelta
from enum import Enum
from functools import partial, wraps
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import scipy as sp
from scipy.interpolate._interpolate import interp1d as Interpolator

from astro_pi_replay import PROGRAM_CMD_NAME, PROGRAM_NAME
from astro_pi_replay.configuration import Configuration
from astro_pi_replay.custom_types import ExecutionMode
from astro_pi_replay.exception import AstroPiReplayException
from astro_pi_replay.resources import (
    SENSE_HAT_CSV_FILE,
    get_replay_sequence_dir,
    get_start_time,
)
from astro_pi_replay.venv_resolver import VenvResolver

logger = logging.getLogger(__name__)


class Lifecycle(Enum):
    BEFORE = "BEFORE"
    AFTER = "AFTER"


class AstroPiExecutorState:
    """
    Wrapper class for the executor instance's shared, mutable state.
    """

    def __init__(self) -> None:
        self._last_sense_hat_row_index: int = 0
        self._last_picamera_photo_index: int = 0
        self._start_time: datetime = datetime.now()


class AstroPiExecutor:
    """
    Class containing the replaying logic (as instance methods)
    + the CLI main methods (as static methods).

    This class is instantiated (by the API adapter classes) only
    when ExecutionMode is REPLAY, in order control the replaying
    of data. Otherwise, its static methods are used to setup a
    venv and run main.py files.

    The class is a singleton
    """

    MODULES_TO_STUB: list[str] = ["sense_hat", "picamera", "orbit"]
    NOT_FOUND = f"{PROGRAM_CMD_NAME} not found"

    """
    Checks whether the current interpreter is running in a venv,
    as defined here in https://docs.python.org/3/library/venv.html#how-venvs-work
    """
    _instance: Optional["AstroPiExecutor"] = None  # singleton instance
    _callbacks: dict[Lifecycle, list[Callable]] = collections.defaultdict(list)

    def __new__(
        cls,
        datetime_col: str = "datetime",
        datetime_format: str = "%Y-%m-%d %H:%M:%S.%f",
        replay_mode: bool = True,
        state: Optional[AstroPiExecutorState] = None,
        configuration: Optional[Configuration] = None,
    ) -> "AstroPiExecutor":
        """
        datetime_format example: 2022-01-31 12:21:15.123456
        """
        if cls._instance is None:
            logger.debug("Creating new instance")
            cls._instance = super(AstroPiExecutor, cls).__new__(cls)

            cls.datetime_col: str = datetime_col
            cls.datetime_format: str = datetime_format
            cls.replay_mode: bool = replay_mode
            if state is None:
                state = AstroPiExecutorState()
            cls.interpolators: dict[str, Interpolator] = {}
            cls._state: AstroPiExecutorState = state

            cls.configuration = (
                configuration if configuration is not None else Configuration.load()
            )

            # TODO add option to be a bit like easyrandom / haskell type testing
            # random_mode = False # whether or not to randomly generate data
            # mode: ir or vis

        return cls._instance

    def picamera_replay(self) -> Callable:
        """
        Decorator used to conditionally replay photos from file for the PiCamera
        """
        return lambda: 1

    def sense_hat_replay(self, *args, **kwargs) -> Callable:
        """
        Decorator used to conditionally replay data from file for the SenseHat.
        """
        filename: str = str(get_replay_sequence_dir() / SENSE_HAT_CSV_FILE)

        if "filename" not in kwargs:
            kwargs["filename"] = filename
        return self.replay(*args, **kwargs)

    def replay(
        self,
        reducer: Callable[[pd.DataFrame], object] = lambda df: df.iloc[0],
        filename: Optional[str] = None,
        col_names: Optional[list[str]] = None,
        *args,
        **kwargs,
    ) -> Callable:
        """
        Decorator used to replay data from files, conditionally.
        """

        # TODO check args and kwargs for unexpected inputs (it should
        # only be the func to be decorated)

        # This is the actual decorator
        def decorator(func: Callable):
            # Activity here is processed at load-time
            logger.debug(f"Decorating function '{func.__name__}'")

            # This defines the functionality the decorator should do
            @wraps(func)
            def _replay(*_args, **_kwargs):
                if self.replay_mode:
                    nonlocal filename, col_names, reducer
                    if filename is None:
                        raise AstroPiReplayException("Cannot have empty filename")
                    if col_names is None:
                        col_names = [func.__name__]

                    return self._replay_next(
                        filename, self.datetime_col, col_names, reducer
                    )
                else:
                    return func(*_args, **_kwargs)

            return _replay

        # This deals with the standard decorator case (no brackets)
        # whereby: no kwargs + standard func positional arg
        # is given.
        if len(kwargs) == 0 and len(args) > 0 and callable(args[0]):
            logger.debug("Standard decorator")
            return partial(decorator, args[0])
        # Otherwise, return the actual decorator function
        else:
            logger.debug("Returning actual decorator")
        return decorator

    def _find_next_datum(self, df: pd.DataFrame) -> int:
        """
        Finds the next row in the given dataframe indexed by
        datetime, based on the elapsed time.
        """
        start_time: datetime = self._state._start_time
        logger.debug(f"Start_time: {start_time}")
        now: datetime = datetime.now()
        # TODO manually code the first call to return index 0 to not
        # skip the first index...not rounding
        # as that would make it get stuck
        delta_in_seconds: pd.Timedelta = pd.Timedelta(
            (now - start_time).total_seconds(), "seconds"
        )
        first_time: pd.Timestamp = df.iloc[0].name
        proposed_time: pd.Timestamp = first_time + delta_in_seconds

        # Find the nearest time using the proposed time
        nearest_i = df.index.get_indexer(pd.Index([proposed_time]), method="backfill")[
            0
        ]
        logging.debug(f"Nearest i: {nearest_i}")
        logger.debug(f"now: {now}")
        logger.debug(f"delta_tmp: {(now - start_time).total_seconds()}")
        logger.debug(f"delta_in_seconds: {delta_in_seconds}")
        logger.debug(f"first_time: {first_time}")
        logger.debug(f"proposed_time: {proposed_time}")
        self._state._last_sense_hat_row_index = nearest_i

        if not self.configuration.no_wait_images:
            actual_time = df.iloc[nearest_i].name
            logger.debug(f"Actual time: {actual_time}")
            actual_delta: int = (actual_time - first_time).total_seconds()
            logger.debug(f"Actual delta: {actual_delta}")
            cutoff: datetime = self._state._start_time + timedelta(seconds=actual_delta)
            logger.debug(f"Cutoff: {cutoff}")
            delta = (cutoff - datetime.now()).total_seconds()
            logger.debug(f"Replay delta: {delta}")
            if delta > 0:
                logger.debug("Sleeping until delta has passed")
                time.sleep(delta)

        return nearest_i

    @functools.cache
    def _df_from_replay_file(self, filename: str, datetime_col: str) -> pd.DataFrame:
        # Detect file type
        suffix = filename.split(".")[-1]
        if suffix == "csv":
            df = pd.read_csv(filename, parse_dates=[datetime_col])
        elif suffix == "tsv":
            df = pd.read_csv(filename, sep="\t", parse_dates=[datetime_col])
        elif suffix == "parquet":
            df = pd.read_parquet(
                filename,
            )
        else:
            raise AstroPiReplayException(f"Unsupported filetype '{suffix}'.")
        df = df.set_index(datetime_col)
        return df

    def _interpolate(
        self, datetime_col: str, col_names: list[str], df: pd.DataFrame
    ) -> pd.DataFrame:
        d: datetime = datetime.now()
        sub_df_dict: dict[str, list[float | int | datetime]] = {
            datetime_col: [d.timestamp()]
        }
        for col_name in col_names:
            if col_name not in self.interpolators:
                self.interpolators[col_name] = sp.interpolate.interp1d(
                    df.index.map(datetime.timestamp), df[col_name].to_numpy()
                )
            interpolator = self.interpolators[col_name]
            try:
                value = interpolator(d.timestamp())
            except ValueError:
                if d.timestamp() < interpolator.x[0]:
                    value = interpolator.y[0]
                else:
                    value = interpolator.y[-1]
            sub_df_dict[col_name] = value
        sub_df: pd.DataFrame = pd.DataFrame.from_dict(sub_df_dict)
        sub_df = sub_df.set_index(datetime_col)
        return sub_df.iloc[0]

    def _replay_next(
        self,
        filename: str,
        datetime_col: str,
        col_names: list[str],
        reducer: Callable[[pd.DataFrame], object] = lambda s: s.iloc[0],
        allow_interpolation: bool = True,
    ) -> object:
        """Internal method that opens the given filename and
        returns the given col names, using the reducer. In effect,
        this replays the data.

        allow_interpolation: Whether to respect the interpolate_sense_hat
        variable.
        """

        df = self._df_from_replay_file(filename, datetime_col)

        for col_name in col_names:
            if col_name not in df.columns:
                raise AstroPiReplayException(
                    f"Column '{col_name}' not found "
                    + f"in file '{filename}'.\n\n"
                    + "Detected columns: \n\t"
                    + ", ".join(df.columns)
                )

        if allow_interpolation and self.configuration.interpolate_sense_hat:
            return reducer(self._interpolate(datetime_col, col_names, df))

        nearest_i = self._find_next_datum(df)

        return reducer(df[col_names].iloc[nearest_i])

    @staticmethod
    def _check_package_installed(venv_python3) -> subprocess.CompletedProcess[str]:
        """
        Runs a program using the venv Python to check if
        the current package is installed.
        """
        dynamic_program: str = "; ".join(
            [
                "import importlib.util",
                "from pathlib import Path",
                "module = importlib.util.find_spec(" + f"'{PROGRAM_NAME}')",
                f"to_print = '{AstroPiExecutor.NOT_FOUND}' if module is None "
                + "else Path(module.origin).parent",
                "print(to_print)",
            ]
        )

        out = subprocess.run(  # nosec B603: no user input
            [venv_python3, "-c", dynamic_program],
            check=True,
            capture_output=True,
            text=True,
        )
        return out

    def time_since_start(self) -> datetime:
        """Time relative to the original start time, as specified
        in the metadata.json file"""
        execution_start_time: datetime = self._state._start_time
        now: datetime = datetime.now()
        delta: timedelta = now - execution_start_time

        original_start_time: datetime = get_start_time()
        return original_start_time + delta

    @staticmethod
    def _detect_execution_mode() -> ExecutionMode:
        return (
            ExecutionMode.LIVE
            if all(
                importlib.util.find_spec(module) is not None
                for module in AstroPiExecutor.MODULES_TO_STUB
            )
            else ExecutionMode.REPLAY
        )

    @staticmethod
    def add_name_is_main_guard(main: Path) -> Path:
        """
        Checks if the file at the given path includes an if name == "__main__"
        expression. If it does, returns the same file.
        Otherwise, returns a modified copy of the file with the original contents
        inside the if expression.
        """
        substrings: list[str] = [
            'if __name__ == "__main__":',
            "if __name__ == '__main__':",
        ]
        with main.open("r") as f:
            contents = f.read().strip()

        includes_guard = False
        for substring in substrings:
            if substring in contents:
                includes_guard = True
        logger.debug(f"Main file includes guard: {includes_guard}")

        if not includes_guard:
            # 1. Copy the original file to a temp folder
            tempdir = Path(tempfile.gettempdir())
            executor_temp_dir: Path = tempdir / PROGRAM_NAME
            executor_temp_dir.mkdir(exist_ok=True)
            original_main: Path = executor_temp_dir / "original_main.py"
            logger.debug(f"Copying original main to {original_main}")
            shutil.copy2(main, original_main)

            # 2. Write modifications to a temp file
            tabbed = os.linesep.join([f"    {line}" for line in contents.splitlines()])
            if len(tabbed) == 0:
                tabbed = "    pass"
            main_copy: Path = tempdir / "main.py"
            with main_copy.open("w") as f:
                f.write(substrings[0] + os.linesep)
                f.write(tabbed)

            # 3. Temporarily overwrite the main.py in the original location
            # with the modified version
            logger.debug(f"Overwriting {main} with {main_copy}")
            shutil.copy2(main_copy, main)

            # 4. Add teardown to replace the modified main.py with the
            # original after execution
            AstroPiExecutor._register_callback(
                Lifecycle.AFTER, lambda: shutil.copy2(original_main, main)
            )

        return main

    @staticmethod
    def add_debug_logging_config(main: Path) -> Path:
        logger.debug("Setting log level inside main")
        tempdir: Path = Path(tempfile.gettempdir()) / PROGRAM_NAME
        main_copy: Path = tempdir / "main_debug_copy.py"
        with main.open() as f:
            contents = f.read()
        with main_copy.open("w") as f:
            f.write("import logging" + os.linesep)
            f.write("logging.basicConfig(level=logging.DEBUG)" + os.linesep)
            f.write(contents)

        shutil.copy2(main_copy, main)
        # Teardown is taken care of in add_name_is_main_guard
        return main

    @staticmethod
    def _register_callback(lifecycle: Lifecycle, callback: Callable) -> None:
        """
        Registers a callback to be executed at a specific point in
        the executor lifecycle
        """
        AstroPiExecutor._callbacks[lifecycle].append(callback)

    @staticmethod
    def _run_callbacks(lifecycle: Lifecycle) -> None:
        """
        Runs the registered callbacks for the specified point in the executor
        lifecycle
        """
        for callback in AstroPiExecutor._callbacks[lifecycle]:
            callback()

    @staticmethod
    def _setup_venv(venv_dirname: Path, name: str = "venv") -> VenvResolver:
        venv_dir: Path = venv_dirname / name

        def list_dependencies(python: Optional[Path] = None) -> str:
            if python is None:
                executable_name: str = (
                    "python.exe" if sys.platform == "win32" else "python"
                )
                resolved: Optional[str] = shutil.which(executable_name)
                if resolved is None:
                    # try python3
                    executable_name = executable_name.replace("python", "python3")
                    resolved = shutil.which(executable_name)
                if resolved is None:
                    raise Exception(f"Cannot find {executable_name}. Is it installed?")
                else:
                    python = Path(resolved)
            args: list[str] = [rf"{str(python)}", "-m", "pip", "freeze"]
            logger.debug(" ".join(args))
            out = subprocess.run(
                args, text=True, check=True, capture_output=True, shell=False
            )  # nosec B603
            logger.debug(out)
            return out.stdout

        # 1. Compare the current environment and the venv for changes.
        # Deletes the venv if there is a change so that the venv
        # dependencies remain up to date.
        if venv_dir.exists():
            current_deps: str = list_dependencies()  # current venv
            logger.debug(f"current_deps: {current_deps}")
            logger.debug("")

            existing_venv: VenvResolver = VenvResolver(venv_dir)
            # The problem is here
            executor_venv_deps = existing_venv.list_dependencies()
            logger.debug(f"executor_venv_deps: {executor_venv_deps}")
            if current_deps == executor_venv_deps:
                logger.debug("venv already created - skipping")
                return existing_venv
            else:
                logger.debug(
                    "Dependencies have changed - deleting "
                    + f"old venv at {venv_dir} and recreating..."
                )
            shutil.rmtree(venv_dir)

        # 2. Copy or creates the venv to the venv_dir, depending on if we're
        # already in one
        venv_resolver: VenvResolver = VenvResolver(venv_dir)

        # 3. Install the executor package as required
        logger.debug("Installing stubbed modules in the venv...")

        executor_install_path: Optional[str] = venv_resolver.is_package_installed(
            PROGRAM_NAME
        )
        if executor_install_path is None:
            logger.debug(f"Installing {PROGRAM_CMD_NAME} into venv...")
            # editable to access the resources already installed...
            # TODO copy the resources explicitly rather than depending
            # on a pip quirk.
            # TODO do not rely on curdir since people will not be running from the root!
            venv_resolver.install(os.curdir, editable=True)

            executor_install_path = venv_resolver.is_package_installed(PROGRAM_NAME)
            if executor_install_path is None:
                raise AstroPiReplayException(
                    f"Could not set up {PROGRAM_CMD_NAME} environment"
                )

        # 4. Install stubs into the venv
        logger.debug("Installing stubbed modules in the venv...")

        for module in AstroPiExecutor.MODULES_TO_STUB:
            logger.debug(f"Installing {module}")
            shutil.copytree(
                Path(executor_install_path) / module,
                venv_resolver.venv_info.site_packages_dir / module,
            )

        return venv_resolver

    @staticmethod
    def run(
        execution_mode: Optional[ExecutionMode],
        venv_dirname: Optional[Path],
        main: Path,
        debug: bool = False,
    ) -> None:
        """
        This method runs the given main file using the given execution mode.
        If the passed in execution mode is Replay mode, then creates a venv
        in the venv_dirname if it does not already exist.

        execution_mode: Whether to replay data or capture live data.
        venv_dirname: The directory to create the venv in replay mode, if it does not
        exist.
        main: The filename to execute - generally called main.py.
        """

        if execution_mode is None:
            execution_mode = AstroPiExecutor._detect_execution_mode()
            logging.debug(f"Detected execution mode: {execution_mode}")
        if not main.exists() or not main.is_file():
            raise AstroPiReplayException(f"File {main} is not a regular file")

        env: Optional[dict[str, str]]
        python: str

        # Conditionally create the venv
        # TODO create spinner/progress bar for this
        if execution_mode == ExecutionMode.REPLAY:
            if venv_dirname is None:
                logging.debug("venv_dirname is None - fetching value from env")
                venv_dirname = (
                    # TODO extract this into astro_pi_replay.config
                    Path(os.environ.get("HOME", tempfile.gettempdir()))
                    / f".{PROGRAM_NAME}"
                )
                logging.debug(f"Found {venv_dirname}")

            venv: VenvResolver = AstroPiExecutor._setup_venv(venv_dirname)

            # Prepare the environment to be used in the subprocess.
            env = os.environ.copy()
            env["PATH"] = os.path.pathsep.join(
                [str(venv.venv_info.script_dir), env["PATH"]]
            )
            env["VIRTUAL_ENV"] = str(venv.venv_dir)

            python = str(venv.venv_info.python)
        else:
            logging.debug("Running in live mode")
            env = None
            python = "python.exe" if sys.platform == "win32" else "python"
            resolved_python: Optional[str] = shutil.which(python)
            if resolved_python is not None:
                python = resolved_python
            else:
                raise Exception(f"Could not find {python}. Is it installed?")

        # Add if __name__ == "__main__" guard as needed
        # (required by multiprocessing in CameraPreview currently FIXME)
        main = AstroPiExecutor.add_name_is_main_guard(main)

        if debug:
            AstroPiExecutor.add_debug_logging_config(main)

        try:
            # Run the program that was passed in
            # TODO this is checked already
            if platform.system() in ["Linux", "Darwin", "Windows"]:
                # -u is for unbuffered Python, which is what is used on the
                # Astro Pis on the ISS.
                args: list[str] = [rf"{python}", "-u", str(main.resolve())]
                logging.debug(f"Executing '{' '.join(args)}' in subprocess")

                def custom_excepthook(type, value, tb):
                    """Hides the internals of the lib
                    from the stack trace"""
                    size = len(list(traceback.walk_tb(tb)))
                    traceback.print_tb(tb, size - 5)

                sys.excepthook = custom_excepthook
                subprocess.run(
                    args, env=env if env is not None else env, check=True
                )  # nosec B603: runs main as intended

            else:
                raise OSError(f"Unsupported system {os}")
        finally:
            AstroPiExecutor._run_callbacks(Lifecycle.AFTER)  # teardown


# Integration tests:
# - using qemu?
# - On Windows, Linux, Darwin ensure that the adapter can be installed
# - On RP4, ensure it calls the real lib (integration test) - I should test this now...!
# perhaps an i2c bus can be emulated...
