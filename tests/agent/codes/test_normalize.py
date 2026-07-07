from agent.codes.normalize import norm_code, norm_term


def test_norm_code_strips_dots_and_uppercases():
    assert norm_code("e11.9") == "E119"
    assert norm_code(" I10 ") == "I10"
    assert norm_code("250.00") == "25000"


def test_norm_term_lowercases_and_collapses_whitespace():
    assert norm_term("  Type 2   Diabetes ") == "type 2 diabetes"
