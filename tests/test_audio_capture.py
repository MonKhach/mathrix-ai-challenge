"""AudioCapture.raise_if_failed: a lost device must never fail silently (TESTS.md E3)."""
import pytest

from app.audio_capture import AudioCapture, AudioDeviceLost


class FakeStream:
    def __init__(self, active=True, stop_fails=False):
        self.active, self.stop_fails, self.closed = active, stop_fails, False

    def is_active(self):
        return self.active

    def stop_stream(self):
        if self.stop_fails:
            raise OSError("Unanticipated host error")

    def close(self):
        self.closed = True


def capture_with(stream):
    capture = AudioCapture.__new__(AudioCapture)  # skip opening a real device
    capture.name, capture._error, capture._stream = "Headphones [Loopback]", None, stream
    return capture


def test_running_stream_is_fine():
    capture_with(FakeStream(active=True)).raise_if_failed()


def test_stopped_stream_raises_clear_error():
    with pytest.raises(AudioDeviceLost, match="Audio device stopped: Headphones"):
        capture_with(FakeStream(active=False)).raise_if_failed()


def test_callback_error_raises_device_lost():
    capture = capture_with(FakeStream())
    capture._error = ValueError("bad chunk")
    with pytest.raises(AudioDeviceLost, match="bad chunk"):
        capture.raise_if_failed()


def test_close_works_even_if_device_is_gone():
    stream = FakeStream(stop_fails=True)
    capture_with(stream).close()
    assert stream.closed
