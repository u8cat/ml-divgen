WRITING_PROMPT_USER_INSTRUCTION = "Write a creative story based on the following prompt:\n"


def apply_template(text, model_name_or_path):
    """
    Apply a template to the text according to the model name or path.

    Args:
        text (str): The text to be processed.
        model_name_or_path (str): The model name or path.

    Returns:
        str: The processed text.
    """

    if "archangel" in model_name_or_path.lower():
        text = "<|user|>\n" + WRITING_PROMPT_USER_INSTRUCTION + text + "\n<|assistant|>\n"
    else: # TODO if more models are added please add cases here
        text = WRITING_PROMPT_USER_INSTRUCTION + text + "\n"

    return text
