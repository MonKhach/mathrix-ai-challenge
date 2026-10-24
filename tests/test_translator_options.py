"""decoding_options in app/translator.py: settings against translation loops (M5)."""
from app.translator import LENGTH_MARGIN, decoding_options


def test_defaults_are_length_cap_and_no_repeat_4_d33():
    assert decoding_options(20) == {"max_decoding_length": 40 + LENGTH_MARGIN,
                                    "no_repeat_ngram_size": 4}


def test_all_off_gives_no_options():
    assert decoding_options(20, max_length_ratio=None, no_repeat_ngram_size=0,
                            repetition_penalty=1.0) == {}


def test_length_cap_grows_with_input():
    assert decoding_options(20, max_length_ratio=2.0, no_repeat_ngram_size=0,
                            repetition_penalty=1.0) == {"max_decoding_length": 40 + LENGTH_MARGIN}


def test_repetition_settings_are_passed_when_on():
    options = decoding_options(20, max_length_ratio=None, no_repeat_ngram_size=4,
                               repetition_penalty=1.2)
    assert options == {"no_repeat_ngram_size": 4, "repetition_penalty": 1.2}
