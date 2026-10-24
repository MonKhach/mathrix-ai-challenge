from evaluation.translation_settings import looks_like_loop, repetition_kept


def test_loop_is_detected():
    assert looks_like_loop("you want to go, you want to go, you want to go, you want to go")


def test_normal_sentence_and_short_repetition_are_not_loops():
    assert not looks_like_loop("By the way, by the way, I forgot to tell you.")
    assert not looks_like_loop("No, no, no, I didn't say that.")


def test_repetition_kept_with_alternatives():
    assert repetition_kept("Yes, yes, of course.", "yes")
    assert repetition_kept("Good, good, I agree.", "okay|good")
    assert not repetition_kept("Yes, of course.", "yes")
    assert repetition_kept("Let's go, let's go, we're late.", "let's go")


def test_repetition_kept_counts_whole_words_only():
    assert not repetition_kept("Nobody said no.", "no")
    assert repetition_kept("No, no, no, I didn't say that.", "no")
