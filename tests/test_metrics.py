"""Automated version of TESTS.md section C (normalization and scoring, D11)."""
import pytest

from evaluation.metrics import normalize, score


# --- C1-C6: normalization rules ---------------------------------------------------------

def test_c1_armenian_question_mark_inside_word_and_full_stop():
    assert normalize("քեզ դուր եկա՞վ։") == "քեզ դուր եկավ"


def test_c2_comma_and_capital_letter():
    assert normalize("Ողջույն, ինչ ուրախ եմ։") == "ողջույն ինչ ուրախ եմ"


def test_c3_ev_ligature():
    assert normalize("սիրելի ռադիունկնդիրներ և դիտողներ") == "սիրելի ռադիունկնդիրներ եւ դիտողներ"


def test_c4_extra_spaces():
    assert normalize("ենք   ու") == "ենք ու"


def test_c5_colloquial_forms_unchanged():
    assert normalize("թանկացել ա") == "թանկացել ա"


def test_c6_digits_unchanged():
    assert normalize("30 տարեկան") == "30 տարեկան"


# --- extra normalization checks ---------------------------------------------------------

def test_hyphen_keeps_one_word():
    assert normalize("մարտի 30-ին") == "մարտի 30ին"


def test_armenian_hyphen_behaves_like_normal_hyphen():
    assert normalize("Էյ այ\u058Aը") == normalize("Էյ այ-ը") == "էյ այը"


def test_punctuation_without_space_still_separates_words():
    assert normalize("ձեզ,հետ") == "ձեզ հետ"


def test_ascii_colon_used_as_armenian_full_stop():
    assert normalize("եկավ:") == "եկավ"


def test_armenian_comma_and_exclamation():
    assert normalize("քննարկեն՝ ծառեր, հա՜") == "քննարկեն ծառեր հա"


# --- C7: scoring ------------------------------------------------------------------------

def test_c7_identical_text_scores_zero():
    result = score("քեզ դուր եկա՞վ։", "Քեզ դուր եկավ")
    assert result["wer"] == 0
    assert result["cer"] == 0
    assert result["raw_wer"] > 0  # raw score still shows the punctuation/case difference


def test_c7_wrong_words_are_not_hidden():
    result = score("քեզ դուր եկա՞վ։", "քեզ դուր էր եկել")
    assert result["wer"] > 0
    assert result["substitutions"] + result["insertions"] + result["deletions"] > 0


def test_missing_word_counts_as_deletion():
    result = score("քեզ դուր եկա՞վ։", "դուր եկավ")
    assert result["deletions"] == 1
    assert result["wer"] == pytest.approx(1 / 3)


def test_empty_model_output_is_100_percent_wrong():
    assert score("քեզ դուր եկավ", "")["wer"] == 1.0


def test_empty_reference_fails_loudly():
    with pytest.raises(ValueError):
        score("։ ,", "որևէ բան")
