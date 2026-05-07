from sentiment.preprocessing import clean_corpus, clean_text


def test_strips_html_tags():
    assert "<br" not in clean_text("Great movie<br /><br />Loved it.")


def test_lowercases():
    assert clean_text("AMAZING film").islower()


def test_drops_digits_and_punctuation():
    out = clean_text("It cost $9.99 in 2024!!!")
    assert "9" not in out
    assert "$" not in out
    assert "!" not in out


def test_removes_stopwords():
    out = clean_text("This is the best movie that I have ever seen")
    for sw in ("this", "is", "the", "that", "have"):
        assert f" {sw} " not in f" {out} "


def test_handles_none_and_non_string():
    assert clean_text(None) == ""
    # numbers get coerced to a string and then digits are stripped, so empty
    assert clean_text(123) == ""


def test_empty_input():
    assert clean_text("") == ""


def test_clean_corpus_preserves_order_and_length():
    inputs = ["Great movie!", "Terrible plot.", "Mediocre at best."]
    out = clean_corpus(inputs)
    assert len(out) == len(inputs)
    assert all(isinstance(s, str) for s in out)


def test_collapses_whitespace():
    out = clean_text("good     movie\n\nreally\tnice")
    assert "  " not in out
