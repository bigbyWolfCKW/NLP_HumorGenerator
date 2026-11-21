"""Final-answer handling for the DeepSeek R1 candidate generator only."""

DEEPSEEK_MODELS = {"deepseek", "deepseek14b"}


def extract_ad(text):
    """Require a completed reasoning block; never treat unfinished thoughts as an ad."""
    before, separator, answer = text.partition("</think>")
    if not separator:
        return "", ["missing_final_answer"]
    answer = answer.strip()
    if "<think>" in answer or "</think>" in answer:
        return "", ["invalid_final_answer"]
    if not answer:
        return "", ["missing_final_answer"]
    return answer, []
