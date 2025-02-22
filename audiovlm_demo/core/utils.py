import base64
import mimetypes
from pathlib import Path


def resolve_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def encode_file_to_data_url(filename, file_contents) -> str:
    """Converts image file to data url to display in browser."""
    base64_encoded = base64.b64encode(file_contents)
    filename_ = filename.lower()
    mime_type = mimetypes.guess_type(filename_)[0]
    if mime_type is None:
        raise ValueError(f"Could not determine mimetype of file {filename_}")
    data_url = f"data:{mime_type};base64,{base64_encoded.decode('utf-8')}"
    return data_url
