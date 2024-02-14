# TODO just copy everything from the original.
class PiCameraError(Exception):
    pass


class PiCameraRuntimeError(RuntimeError):
    pass


class PiCameraValueError(PiCameraError, ValueError):
    pass


class PiCameraWarning(Warning):
    """
    Base class for PiCamera warnings.
    """


class PiCameraDeprecated(PiCameraWarning, DeprecationWarning):
    """
    Raised when deprecated functionality in picamera is used.
    """


class PiCameraFallback(PiCameraWarning, RuntimeWarning):
    """
    Raised when picamera has to fallback on old functionality.
    """


class PiCameraNotRecording(PiCameraRuntimeError):
    """
    Raised when :meth:`~PiCamera.stop_recording` or
    :meth:`~PiCamera.split_recording` are called against a port which has no
    recording active.
    """


class PiCameraAlreadyRecording(PiCameraRuntimeError):
    """
    Raised when :meth:`~PiCamera.start_recording` or
    :meth:`~PiCamera.record_sequence` are called against a port which already
    has an active recording.
    """
