"""
Utility functions for path handling in LiveRAG.
"""
import os
from typing import NamedTuple
from urllib.parse import urlparse

import urllib.parse

from tools.logging_utils import get_logger

log = get_logger("path_utils")


def get_project_root():
    """
    Returns the absolute path to the project root directory.
    Uses the module file location to determine the project root.

    Returns:
        str: Absolute path to the project root directory
    """
    # Get the directory of the current module
    current_module_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up to the project root (src/utils -> src -> project_root)
    project_root = os.path.dirname(os.path.dirname(current_module_dir))

    log.debug(f"Project root determined as: {project_root}")
    return project_root


def get_data_dir():
    """
    Returns the absolute path to the data directory.

    Returns:
        str: Absolute path to the data directory
    """
    data_dir = os.path.join(get_project_root(), 'data')
    log.debug(f"Data directory path: {data_dir}")
    return data_dir


def ensure_dir(file_path, create_if_not=True):
    """
    The function ensures the dir exists,
    if it doesn't it creates it and returns the path or raises FileNotFoundError
    In case file_path is an existing file, returns the path of the parent directory
    """
    # tilde expansion
    file_path = os.path.normpath(os.path.expanduser(file_path))
    if os.path.isfile(file_path):
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
    if not os.path.exists(directory):
        if create_if_not:
            try:
                os.makedirs(directory)
            except FileExistsError:
                # This exception was added for multiprocessing, in case multiple processes try to create the directory
                pass
        else:
            raise FileNotFoundError(
                f"The directory {directory} doesn't exist, create it or pass create_if_not=True"
            )
    return directory


def to_icon_url(url: str | None) -> str:
    """
    Convert a URL to its favicon URL using Google's favicon service.

    Args:
        url (str): The original URL

    Returns:
        str: The favicon URL
    """
    if not url:
        # Google favicon service works with any domain, this is a placeholder
        url = "https://example.com"
    return f"https://www.google.com/s2/favicons?domain={urlparse(url).netloc}&sz=64"


class ParsedURL(NamedTuple):
    url: str
    """https://domain.com:port/v1"""
    host: str
    """https://domain.com:port"""
    port: int


def parse_url(url: str):
    parsed = urllib.parse.urlparse(url)
    port = parsed.port
    host = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        host += f":{parsed.port}"

    if not port:
        port = 443 if parsed.scheme == 'https' else 80

    return ParsedURL(url=url, host=host, port=port)
