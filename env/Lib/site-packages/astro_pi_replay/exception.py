class AstroPiReplayRuntimeError(RuntimeError):
    pass


class AstroPiReplayException(Exception):
    def __init__(self, message: str) -> None:
        self.message: str = message
        super().__init__(message)

    def __repr__(self) -> str:
        return self.message

    def __str__(self) -> str:
        return self.message
