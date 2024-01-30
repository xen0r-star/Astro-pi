import datetime
import itertools
import logging
import multiprocessing
import re
import subprocess
import threading
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Optional, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import astro_pi_replay.picamera.mmalobj as mo
from astro_pi_replay.custom_types import IO_TYPE, RGB
from astro_pi_replay.exception import (
    AstroPiReplayException,
    AstroPiReplayRuntimeError,
)
from astro_pi_replay.executor import AstroPiExecutor
from astro_pi_replay.picamera.abstract_camera import PiCamera
from astro_pi_replay.picamera.encoders import PiEncoder, PiVideoEncoder
from astro_pi_replay.picamera.exc import (
    PiCameraAlreadyRecording,
    PiCameraError,
    PiCameraNotRecording,
    PiCameraRuntimeError,
    PiCameraValueError,
)
from astro_pi_replay.picamera.exif import modify_exif_tags
from astro_pi_replay.picamera.frames import PiVideoFrame, PiVideoFrameType
from astro_pi_replay.picamera.preview import CameraPreview
from astro_pi_replay.picamera.renderers import PiOverlayRenderer, PiRenderer
from astro_pi_replay.resources import get_replay_sequence_dir, get_resource

logger = logging.getLogger(__name__)

photo_formats = [
    "jpg",
    "jpeg",
    "png",
    "gif",
    "bmp",
    "yuv",
    "rgb",
    "rgba",
    "bgr",
    "bgra",
]
video_formats = ["h264", "mjpeg", "yuv", "rgb", "rgba", "bgr", "bgra"]


def PiCameraAdapter(maybe_executor: Optional[AstroPiExecutor] = None) -> PiCamera:
    executor: AstroPiExecutor
    if maybe_executor is None:
        executor = AstroPiExecutor()
    else:
        executor = maybe_executor

    class _PiCameraAdapter(PiCamera):
        # TODO make these instance attribtues
        _preview_proc: Optional[multiprocessing.Process] = None
        _recording_proc: Optional[subprocess.Popen[bytes]] = None
        _recording_fmt: Optional[str] = None
        _recording_consumer: Optional[ProcStdoutConsumer] = None
        _frame_counter: Iterator[int] = itertools.count(0)
        _preview: Optional[PiRenderer] = None
        _encoders: dict[int, PiEncoder] = {}

        # TODO
        def __enter__(self):
            pass

        def __exit__(self):
            # TODO remove zombie processes
            self.close()

        # TODO make this more elegant... perhaps use the close method instead :)
        def _teardown(self):
            """
            Close any lingering processes
            """
            for process in [self._preview_proc, self._recording_proc]:
                if process is not None and process.poll() is None:
                    process.terminate()

        def _annotatate_text_in_image(self, img: Image.Image, frame_num) -> None:
            """
            Annotates the image with the annotation text inplace.
            """
            font_path: Path = get_resource("Share_Tech_Mono/ShareTechMono-Regular.ttf")
            font = ImageFont.truetype(str(font_path), self.annotate_text_size)
            draw_context: ImageDraw.ImageDraw = ImageDraw.Draw(img)

            text = self.annotate_text
            if self.annotate_frame_num:
                text += "\n" + str(frame_num)

            bbox = draw_context.multiline_textbbox((0, 0), text, font=font)
            _, _, box_width, box_height = bbox

            img_width, _ = img.size
            coordinate: tuple[int, int] = (round((img_width / 2) - (box_width / 2)), 20)

            if self.annotate_background is not None:
                coords_box = (
                    coordinate[0],
                    coordinate[1],
                    coordinate[0] + box_width,
                    coordinate[1] + box_height,
                )
                draw_context.rectangle(
                    coords_box, fill=cast(RGB, self.annotate_background.rgb_bytes)
                )

            draw_context.multiline_text(
                coordinate,
                text,
                font=font,
                fill=cast(RGB, self.annotate_foreground.rgb_bytes),
            )

        def add_overlay(
            self,
            source: BinaryIO,  # TODO is this definitely the right type?
            size: Optional[tuple[int, int]] = None,
            format: Optional[str] = None,
            **options,
        ) -> PiOverlayRenderer:
            overlay = PiOverlayRenderer(self, source, size, format, **options)
            self.overlays.append(overlay)
            return overlay

        def _detect_format(
            self,
            output: IO_TYPE,
            format: Optional[str],
            allowed_formats: list[str] = photo_formats,
        ) -> tuple[IO_TYPE, str]:
            final_output: IO_TYPE
            final_format: str

            if format is None and isinstance(output, str):
                split: list[str] = output.split(".")
                if len(split) < 2 or split[-1] not in allowed_formats:
                    raise PiCameraValueError(
                        f"Couldn't detect a valid format in {output}"
                    )
                final_output = ".".join(split[:-1])
                final_format = split[-1]
            elif format not in allowed_formats and isinstance(output, str):
                raise PiCameraValueError("Format not allowed")
            elif format is not None and isinstance(output, str) and format == "jpeg":
                # change format so it appears to not overwrite the suffix
                # given in the filename
                if output.endswith(".jpg"):
                    final_format = "jpg"
                else:
                    final_format = format

                final_output = re.sub(r"\.jpg$", "", output)
                final_output = re.sub(r"\.jpeg$", "", final_output)
            elif format is not None and isinstance(output, str):
                final_output = re.sub(r"\." + format + r"$", "", output)
                final_format = format
            else:
                final_output = output
                if format is None:
                    raise PiCameraValueError("Must specify a format")
                final_format = format

            return final_output, final_format

        def capture(
            self,
            output: IO_TYPE,
            format: Optional[str] = None,
            use_video_port: bool = False,
            resize: Optional[tuple[int, int]] = None,
            splitter_port: int = 0,
            bayer: bool = False,
            **options,
        ) -> None:
            final_output, final_format = self._detect_format(output, format)

            name: str = str(
                executor._replay_next(
                    str(get_replay_sequence_dir() / "photos" / "photo_index.csv"),
                    "datetime",
                    ["name"],
                    allow_interpolation=False,
                )
            )

            image_path: Path = get_replay_sequence_dir() / "photos" / name
            im = Image.open(image_path)

            # Conditionally add text annotation
            if len(self.annotate_text) > 0:
                matches = re.search(r"\d+", name)
                frame_num: int = (
                    next(self._frame_counter)
                    if matches is None
                    else int(matches.group())
                )
                self._annotatate_text_in_image(im, frame_num)

            if resize is not None:
                im = im.resize(resize)

            if isinstance(final_output, str):
                stream, opened = mo.open_stream(f"{final_output}.{final_format}")
            else:
                stream, opened = mo.open_stream(final_output)

            # raw image
            if final_format is not None and final_format in [
                "rgb",
                "rgba",
                "bgr",
                "bgra",
            ]:
                np_image = np.array(im)
                if final_format.startswith("bgr"):
                    np_image[:, :, [0, 1, 2]] = np_image[
                        :, :, [2, 1, 0]
                    ]  # type: ignore
                stream.write(np_image.tobytes())
            else:
                exif = None
                if final_format == "yuv":
                    im = im.convert("YCbCr")
                elif final_format in ["jpeg", "jpg"] and len(self.exif_tags.keys()) > 0:
                    # exif tags are only supported for jpeg in the original picamera
                    exif = modify_exif_tags(im.getexif(), self.exif_tags)
                im.save(stream, format=None, exif=exif)
            mo.close_stream(stream, opened)

        def capture_continuous(
            self,
            output: IO_TYPE,
            format: Optional[str] = None,
            use_video_port: bool = False,
            resize: Optional[tuple[int, int]] = None,
            splitter_port: int = 0,
            burst: bool = False,
            bayer: bool = False,
            **options,
        ) -> Iterable:
            final_output, final_format = self._detect_format(output, format)
            counter: int = 1
            while True:
                if isinstance(final_output, str):
                    filename: str = final_output.format(
                        counter=counter, timestamp=datetime.datetime.now()
                    )
                    self.capture(
                        filename,
                        final_format,
                        use_video_port,
                        resize,
                        splitter_port,
                        bayer,
                        **options,
                    )
                    yield f"{filename}.{final_format}"
                else:
                    self.capture(
                        output,
                        format,
                        use_video_port,
                        resize,
                        splitter_port,
                        bayer,
                        **options,
                    )
                    yield output
                counter += 1

        def capture_sequence(
            self,
            outputs: Iterable[IO_TYPE],
            format: str = "jpeg",
            use_video_port: bool = False,
            resize: Optional[tuple[int, int]] = None,
            splitter_port: int = 0,
            burst: bool = False,
            bayer: bool = False,
            **options,
        ) -> None:
            for output in outputs:
                self.capture(
                    output, format, use_video_port, resize, splitter_port, bayer
                )

        # TODO
        @property
        def frame(self) -> Optional[PiVideoFrame]:
            if not self.recording:
                raise PiCameraRuntimeError(
                    "Cannot query frame information " + "when camera is not recording"
                )
            return None

        def record_sequence(
            self,
            outputs: Iterable[IO_TYPE],
            format: str = "h264",
            resize: Optional[tuple[int, int]] = None,
            splitter_port: int = 1,
            **options,
        ) -> Iterable[IO_TYPE]:
            for i, output in enumerate(outputs):
                if i == 0:
                    self.start_recording(
                        output, format, resize, splitter_port, **options
                    )
                else:
                    self.split_recording(output, splitter_port, **options)
                yield output

        # TODO
        def remove_overlay(self, overlay: PiOverlayRenderer) -> None:
            return super().remove_overlay(overlay)

        def split_recording(
            self,
            output: IO_TYPE,
            splitter_port: int = 1,
            **options,
        ) -> Optional[PiVideoFrame]:
            # TODO the _recording_fmt should be stored in an encoder object
            # as per the real implementation...
            if self._recording_fmt is None:
                raise PiCameraNotRecording(
                    "There is no recording in progress on " f"port {str(splitter_port)}"
                )
            else:
                format: str = self._recording_fmt
            self.stop_recording()

            self.start_recording(output, format=format)
            return None

        def start_preview(self, **options) -> PiRenderer:
            if self._preview_proc is None:
                preview: CameraPreview = CameraPreview(
                    str(get_replay_sequence_dir() / "videos" / "OrbitAz.mp4")
                )
                self._preview_proc = preview
                preview.start()
                renderer = PiRenderer(self)
                self._preview = renderer
                return renderer
            elif self.preview is not None:
                return self.preview
            else:
                raise AstroPiReplayRuntimeError("Invalid State")

        def start_recording(
            self,
            output: IO_TYPE,
            format: Optional[str] = None,
            resize: Optional[tuple[int, int]] = None,
            splitter_port: int = 1,
            **options,
        ):
            if self._recording_proc is not None:
                raise PiCameraError("Recording already started")

            # Determine the format
            final_output, final_format = self._detect_format(
                output, format, allowed_formats=video_formats
            )

            if not self._has_ffmpeg:
                raise AstroPiReplayException("Please install ffmpeg")

            video: Path = get_replay_sequence_dir() / "videos" / "OrbitAz.mp4"

            # TODO add annotations
            # TODO resize

            # Start streaming to the output in real-time

            delta: datetime.timedelta = (
                datetime.datetime.now() - executor._state._start_time
            )
            vcodec = self.__get_vcodec(final_format)

            # See http://trac.ffmpeg.org/wiki/StreamingGuide
            command_args: list[str] = [
                "ffmpeg",
                # real-time reads
                "-re",
                # seek to time
                "-ss",
                str(delta.total_seconds()),
                # input
                "-i",
                str(video),
                # output codec
                "-vcodec",
                vcodec,
            ]
            if vcodec == "rawvideo":
                pix_fmt = self.__get_pix_fmt(final_format)
                command_args.extend(
                    ["-f", "rawvideo", "-vf", f"format={pix_fmt}", "-pix_fmt", pix_fmt]
                )
            else:
                command_args.extend(["-f", final_format])

            with self._encoders_lock:
                camera_port, output_port = self._get_ports(True, splitter_port)
                encoder = PiVideoEncoder(
                    self, camera_port, output_port, final_format, resize, **options
                )
                self._encoders[splitter_port] = encoder
            if isinstance(final_output, str):
                command_args.append(f"{final_output}.{final_format}")

                logger.debug(" ".join(command_args))
                # non-blocking.
                self._recording_proc = subprocess.Popen(  # nosec B603
                    command_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            else:
                encoder.start(output)

                command_args.append("pipe:1")
                logger.debug(" ".join(command_args))
                self._recording_proc = subprocess.Popen(  # nosec B603
                    command_args, stdout=subprocess.PIPE
                )

                self._recording_consumer = ProcStdoutConsumer(
                    self.resolution, final_format, encoder, self._recording_proc
                )
                self._recording_consumer.start()
                # TODO make sure this is closed properly on closure

            # TODO make sure this is consistent with the _recording_proc
            self._recording_fmt = final_format  # TODO move to an encoder object.

            # TODO add proper error handling to the Popen bits to ensure that
            # TODO add tear-down method (probably on the executor obj)
            # to avoid zombie processes
            # possible using atexit module

        # TODO
        @property
        def preview(self) -> Optional[PiRenderer]:
            # # TODO add overlays
            return self._preview if self.previewing else None

        def stop_preview(self):
            if self._preview_proc is not None:
                self._preview_proc.terminate()
                self._preview_proc = None
                self._preview = None

        def stop_recording(self, splitter_port: int = 1) -> None:
            super().stop_recording(splitter_port)

            if self._recording_proc is None or splitter_port not in self._encoders:
                raise PiCameraNotRecording(
                    "There is no recording in progress on " f"port {str(splitter_port)}"
                )
            else:
                self._recording_proc.terminate()
                self._recording_proc.wait(timeout=30)
                if self._recording_consumer is not None:
                    self._recording_consumer.join()
                self._recording_proc = None
                self._recording_fmt = None
                self._recording_consumer = None
                self._encoders[splitter_port].close()
                with self._encoders_lock:
                    del self._encoders[splitter_port]

        @property
        def _has_ffmpeg(self) -> bool:
            try:
                subprocess.run(  # nosec B603, B607
                    ["ffmpeg", "-version"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except FileNotFoundError:
                logger.error("ffmpeg not found. Please install it.")
                return False

        @property
        def _has_ffprobe(self) -> bool:
            try:
                subprocess.run(  # nosec B603, B607
                    ["ffprobe", "-version"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except FileNotFoundError:
                logger.error("ffprobe not found. Please install it.")
                return False

        def _get_ports(
            self, from_video_port: bool, splitter_port: int
        ) -> tuple[mo.MMALVideoPort, mo.MMALVideoPort]:
            """
            Determine the camera and output ports for given capture options.

            See :ref:`camera_hardware` for more information on picamera's usage of
            camera, splitter, and encoder ports. The general idea here is that the
            capture (still) port operates on its own, while the video port is
            always connected to a splitter component, so requests for a video port
            also have to specify which splitter port they want to use.
            """
            # self._check_camera_open()
            if from_video_port and (splitter_port in self._encoders):
                raise PiCameraAlreadyRecording(
                    "The camera is already using port %d " % splitter_port
                )

            camera_port: int = (
                self.CAMERA_VIDEO_PORT if from_video_port else self.CAMERA_CAPTURE_PORT
            )
            output_port: int = splitter_port if from_video_port else camera_port
            return (mo.MMALVideoPort(camera_port), mo.MMALVideoPort(output_port))

        def __get_vcodec(self, format: Optional[str]) -> str:
            if format == "h264":
                return "copy"
            elif format == "mjpeg":
                return "mjpeg"
            else:
                return "rawvideo"

        def __get_pix_fmt(self, format: Optional[str]) -> str:
            if format == "yuv":
                return "yuv420p"
            elif format == "rgb":
                return "rgb24"
            elif format == "bgr":
                return "bgr24"
            elif format == "rgba":
                return "rgba"
            else:
                return "bgra"

    return _PiCameraAdapter()


# TODO move me
class ProcStdoutConsumer(threading.Thread):
    def __init__(
        self,
        resolution: mo.PiResolution,
        final_format: str,
        encoder: PiEncoder,
        proc: subprocess.Popen[bytes],
    ):
        super().__init__()
        self.resolution = resolution
        self.final_format = final_format
        self.encoder = encoder
        self.proc = proc

    def run(self) -> None:
        """Consumes the ffmpeg subprocess"""

        is_raw: bool = (
            True
            if self.final_format in ["rgb", "rgba", "bgr", "bgra", "yuv"]
            else False
        )

        frame_size: int
        if is_raw:
            # stream frame by frame when raw
            frame_size = (
                self.resolution.width * self.resolution.height * len(self.final_format)
            )

            if self.final_format == "yuv":
                # YUV has a byte ratio of 4:6 hence multiply by 2/3
                frame_size = round(frame_size * 2 / 3)
        else:
            # When not raw, just copy 100kb at a time
            # FIXME recalculate to get desired bitrate
            frame_size = 100 * 1000

        contents: bytes
        while self.proc.poll() is None:
            if self.proc.stdout is None:
                break
            contents = self.proc.stdout.read(frame_size)
            self.encoder.outputs[PiVideoFrameType.frame][0].write(contents)
        if self.proc.stdout is not None:
            contents = self.proc.stdout.read()
            self.encoder.outputs[PiVideoFrameType.frame][0].write(contents)
