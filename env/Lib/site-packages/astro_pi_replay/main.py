import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from astro_pi_replay import PROGRAM_CMD_NAME, PROGRAM_NAME
from astro_pi_replay.configuration import Configuration
from astro_pi_replay.custom_types import ExecutionMode
from astro_pi_replay.downloader import Downloader
from astro_pi_replay.executor import AstroPiExecutor
from astro_pi_replay.resources import get_resource

logger = logging.getLogger(__name__)

RUN_CMD: str = "run"
DOWNLOAD_CMD: str = "download"


def get_argument_parser() -> ArgumentParser:
    arg_parser = ArgumentParser(prog=PROGRAM_CMD_NAME, description="")
    arg_parser.add_argument("--debug", action="store_true", help="Emit debug messages")
    subparsers = arg_parser.add_subparsers(help="sub-command help")

    download_parser = subparsers.add_parser(
        DOWNLOAD_CMD, help="Download the photos to use during run (required)"
    )
    download_parser.set_defaults(cmd="download")
    download_parser.add_argument(
        "--test-assets-only",
        action="store_true",
        default=False,
        help="Downloads only the files required " + "for the automated tests to pass.",
    )
    download_parser.add_argument(
        "--with-video",
        action="store_true",
        default=False,
        help="Whether to download the video assets",
    )
    download_parser.add_argument(
        "--resolution",
        default=(4056, 3040),
        choices=((4056, 3040), (1280, 720)),
        help="The resolution of images to playback. Default is (4056, 3040)",
    )
    download_parser.add_argument(
        "--photography-type",
        default="VIS",
        choices=(("VIS", "IR")),
        help="Whether to playback visible light photos "
        + "(VIS) or infrared light (IR). Default is VIS.",
    )
    download_parser.add_argument(
        "--sequence", default=None, help="The sequence id to use in replays."
    )

    run_parser = subparsers.add_parser(RUN_CMD, help="Run a main.py program")
    run_parser.add_argument("main", type=Path, help="Path to the main.py file to run")
    run_parser.add_argument(
        "--interpolate-sense-hat-values",
        action="store_true",
        default=True,
        dest="interpolate_sense_hat",
        help="Whether to interpolate measurements from the " + "sense hat.",
    )
    run_parser.add_argument(
        "--no-match-original-photo-intervals",
        action="store_true",
        default=False,
        help="Disable this mode to stop sleeping in between successive captures to "
        + "try and match the timestamps of the original photos.",
    )
    run_parser.add_argument(
        "--mode",
        type=ExecutionMode,
        required=False,
        help="Whether to replay data (REPLAY) or fetch" + "live data (LIVE)",
    )
    run_parser.add_argument(
        "--venv_dir",
        type=Path,
        required=False,
        help=f"Path to venv (if not using ~/.{PROGRAM_NAME})",
    )
    run_parser.add_argument(
        "--resolution",
        default=(4056, 3040),
        choices=((4056, 3040), (1280, 720)),
        help="The resolution of images to playback. Default is (4056, 3040)",
    )
    run_parser.add_argument(
        "--photography-type",
        default="VIS",
        choices=(("VIS", "IR")),
        help="Whether to playback visible light photos "
        + "(VIS) or infrared light (IR). Default is VIS.",
    )
    run_parser.add_argument(
        "--sequence", default=None, help="The sequence id to use in replays."
    )
    run_parser.set_defaults(cmd="run")

    return arg_parser


def _main(args: Namespace) -> None:
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    logger.debug(args)
    if hasattr(args, "cmd"):
        downloader = Downloader()
        if args.cmd == "run":
            if args.sequence is None:
                args.sequence = downloader.search_for_sequence(
                    args.resolution, args.photography_type
                )
                logger.debug(f"Selected {args.sequence}")

            if not downloader.has_installed(
                args.resolution, args.photography_type, args.sequence
            ):
                downloader.install(
                    args.resolution, args.photography_type, args.sequence
                )

            with get_resource("motd").open("r") as f:
                sys.stdout.write(f.read())

            Configuration.from_args(args).save()
            AstroPiExecutor.run(args.mode, args.venv_dir, args.main, args.debug)
        elif args.cmd == "download":
            downloader.install(
                args.resolution,
                args.photography_type,
                args.sequence,
                args.test_assets_only,
                args.with_video,
            )
        else:
            get_argument_parser().print_usage()
            sys.exit(1)


def main() -> None:
    arg_parser = get_argument_parser()
    args: Namespace = arg_parser.parse_args(sys.argv[1:])
    _main(args)
