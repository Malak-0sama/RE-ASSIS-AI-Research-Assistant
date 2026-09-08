from prompts.qa_prompt import FALLBACK_ANSWER, build_qa_prompt


def test_prompt_contains_context_and_question():
    prompt = build_qa_prompt(context="Paper says X.", question="What does the paper say?")
    assert "Paper says X." in prompt
    assert "What does the paper say?" in prompt
    assert FALLBACK_ANSWER in prompt
