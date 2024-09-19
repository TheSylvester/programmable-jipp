from pathlib import Path
import inspect


def load_md_content(filename: str) -> str:
    """
    Load the content of a markdown file.

    Args:
        filename (str): The name of the markdown file to load.

    Returns:
        str: The content of the markdown file.

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        IOError: If there's an error reading the file.
    """
    current_dir = Path(__file__).parent
    file_path = current_dir / filename

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} was not found in {current_dir}")
    except IOError as e:
        raise IOError(f"Error reading the file {filename}: {str(e)}")


def load_prompts() -> tuple[str, str]:
    """
    Load the contents of system.md and user.md from the same relative directory
    as the calling script, per our project structure.

    Returns:
        tuple[str, str]: A tuple containing the contents of system.md and user.md.

    Raises:
        FileNotFoundError: If either system.md or user.md is not found.
        IOError: If there's an error reading either file.
    """
    # Get the directory of the calling script
    caller_frame = inspect.stack()[1]
    caller_filename = caller_frame.filename
    caller_dir = Path(caller_filename).parent

    system_path = caller_dir / "system.md"
    user_path = caller_dir / "user.md"

    try:
        with open(system_path, "r", encoding="utf-8") as system_file:
            system_content = system_file.read()

        with open(user_path, "r", encoding="utf-8") as user_file:
            user_content = user_file.read()

        return system_content, user_content
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {e.filename}")
    except IOError as e:
        raise IOError(f"Error reading prompt files: {str(e)}")
