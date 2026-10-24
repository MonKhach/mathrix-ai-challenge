from app.output import format_segment, format_timestamp


def test_timestamp_format():
    assert format_timestamp(72.9) == "00:01:12"
    assert format_timestamp(3725) == "01:02:05"


def test_segment_has_time_speaker_armenian_english():
    text = format_segment(72, "Speaker 1", "Բարև, ինչպե՞ս ես։", "Hi, how are you?")
    assert text == "[00:01:12] Speaker 1\nHY: Բարև, ինչպե՞ս ես։\nEN: Hi, how are you?\n"


def test_latency_is_shown_when_given():
    assert "(latency 2.3 s)" in format_segment(0, "Me", "ա", "a", latency_s=2.26)


def test_similarity_is_shown_when_given():
    assert "(similarity 0.73)" in format_segment(0, "Speaker 1", "ա", "a", similarity=0.731)


def test_confidence_is_shown_when_given():
    assert "(confidence 0.42)" in format_segment(0, "Me", "ա", "a", confidence=0.421)


def test_timing_is_shown_in_order_when_given():
    text = format_segment(0, "Me", "ա", "a", timing={"phrase": 4.24, "wait": 0.7, "asr": 0.21})
    assert "(phrase 4.2 s, wait 0.7 s, asr 0.2 s)" in text
