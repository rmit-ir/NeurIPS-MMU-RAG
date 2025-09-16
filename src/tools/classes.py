import importlib
import sys
import os
import re
from typing import Dict, List

from systems.rag_interface import RAGInterface
from tools.logging_utils import get_logger

logger = get_logger('classes')


def find_system_class_paths(class_name: str) -> List[str]:
    """
    Search recursively for Python files that contain a class with the given name
    and return the full module paths.
    """
    matches = []
    systems_dir = "src/systems"

    # Regular expression to find class definitions
    class_pattern = re.compile(rf"class\s+{class_name}\s*\(\s*.*RAGInterface")

    # Walk through all directories under systems
    for root, _, files in os.walk(systems_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if class_pattern.search(content):
                            # Convert file path to module path
                            rel_path = os.path.relpath(file_path, "src")
                            module_path = rel_path.replace(
                                os.path.sep, '.').replace('.py', '')
                            # Create full module path with class name
                            full_path = f"{module_path}.{class_name}"
                            matches.append(full_path)
                except Exception:
                    # Skip files that can't be read
                    continue

    return matches


def load_system_class(class_name: str, kwargs_dict: dict) -> RAGInterface:
    """
    Load RAG system class from class name with optional parameters.

    Args:
        class_name: Class name (e.g., 'AzureO3ResearchRAG')
        **kwargs: Additional parameters to pass to the constructor

    Returns:
        Instantiated RAG system
    """
    matching_paths = find_system_class_paths(class_name)

    if not matching_paths:
        logger.error("No system class found", class_name=class_name)
        sys.exit(1)
    elif len(matching_paths) == 1:
        # Get the full module path
        full_path = matching_paths[0]
        logger.info("Found matching system class",
                    class_name=class_name, full_path=full_path)

        try:
            # Split module path and class name
            module_path, _ = full_path.rsplit('.', 1)

            # Import the module
            module = importlib.import_module(module_path)

            # Get the class
            system_class = getattr(module, class_name)

            logger.info("System class loaded",
                        class_name=class_name, module_path=module_path)
            if kwargs_dict:
                logger.info("Passing additional parameters to constructor",
                            class_name=class_name, kwargs=kwargs_dict)
            return system_class(**kwargs_dict)

        except (ImportError, AttributeError) as e:
            logger.error("Error loading system class",
                         class_name=class_name, error=str(e))
            sys.exit(1)
    else:
        # Multiple matches found
        logger.error("Multiple system classes found",
                     class_name=class_name, matching_paths=matching_paths)
        sys.exit(1)


def unknown_args_to_dict(unknown_args: List[str]) -> Dict[str, str]:
    """Convert unknown args list to dictionary."""
    result = {}
    key = None

    for arg in unknown_args:
        if arg.startswith('--'):
            if key:  # Previous key was a flag
                result[key] = True
            # normal `--key value`
            key = arg[2:]  # Remove '--'
        else:
            if key:
                result[key] = arg
                key = None
            else:
                logger.warning("Positional argument not supported", arg=arg)

    if key:  # Last argument was a flag
        result[key] = True

    logger.info("Converted unknown args to dict", args=result)

    return result
