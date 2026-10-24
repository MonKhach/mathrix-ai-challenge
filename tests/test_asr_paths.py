from evaluation.asr_paths import doubled_words


def test_doubled_words_found_when_not_in_reference():
    assert doubled_words("քեզ դուրր եկավ", "քեզ դուր եկավ") == ["դուրր"]
    assert doubled_words("Շատատ ահավոր", "Շատ ահավոր") == ["շատատ"]


def test_real_double_letters_in_reference_are_not_counted():
    assert doubled_words("աննորմալ գինը", "աննորմալ գինը") == []
