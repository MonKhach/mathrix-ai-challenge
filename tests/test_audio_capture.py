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


class FakePyAudio:
    """Only what AudioCapture reads before opening a stream (TESTS.md A2)."""
    def __init__(self, info):
        self.info = info

    def get_device_info_by_index(self, index):
        return self.info


MIC = {"name": "Microphone (Realtek(R) Audio)", "maxInputChannels": 2, "isLoopbackDevice": False}
LOOPBACK = {"name": "Headphones [Loopback]", "maxInputChannels": 2, "isLoopbackDevice": True}
SPEAKERS = {"name": "Headphones (output only)", "maxInputChannels": 0, "isLoopbackDevice": False}


def test_microphone_given_as_system_fails_clearly():
    with pytest.raises(ValueError, match="is not a loopback device.*--list-devices"):
        AudioCapture(FakePyAudio(MIC), 12, "system", None)


def test_loopback_given_as_mic_fails_clearly():
    with pytest.raises(ValueError, match="is not a microphone.*--list-devices"):
        AudioCapture(FakePyAudio(LOOPBACK), 13, "mic", None)


def test_output_only_device_fails_clearly():
    with pytest.raises(ValueError, match="is not a microphone"):
        AudioCapture(FakePyAudio(SPEAKERS), 5, "mic", None)
