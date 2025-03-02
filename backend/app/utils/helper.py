import re

def extract_python_code(raw_code: str) -> str:
    """
    Extracts clean Python code from a string that may contain markdown formatting or code fences.

    Args:
        raw_code (str): The raw string containing Python code.

    Returns:
        str: Cleaned Python code ready for execution.
    """
    # Use regex to extract code between markdown code fences
    code_match = re.search(r"```python\s*(.*?)\s*```", raw_code, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # If no python-specific fence, try generic markdown fence
        code_match = re.search(r"```\s*(.*?)\s*```", raw_code, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # If no fences, assume the entire string is code
            code = raw_code.strip()

    # Remove unnecessary imports (like pandas) if already imported
    code_lines = code.splitlines()
    cleaned_lines = [line for line in code_lines if not line.strip().startswith("import pandas")]
    cleaned_code = "\n".join(cleaned_lines).strip()

    return cleaned_code