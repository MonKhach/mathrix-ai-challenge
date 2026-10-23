from app.output import format_segment, format_timestamp


def test_timestamp_format():
    assert format_timestamp(72.9) == "00:01:12"
    assert format_timestamp(3725) == "01:02:05"


def test_segment_has_time_speaker_armenian_english():
    text = format_segment(72, "Speaker 1", "Բարև, ինչպե՞ս ես։", "Hi, how are you?")
    assert text == "[00:01:12] Speaker 1\nHY: Բարև, ինչպե՞ս ես։\nEN: Hi, how are you?\n"


def test_latency_is_shown_when_given():
    assert "(latency 2.3 s)" in format_segment(0, "Me", "ա", "a", latency_s=2.26)
