#!/usr/bin/env python3
"""
Script to add Q&A entries to self-donated JSONL files.
Usage: python scripts/add_qa_entry.py
"""

import json
import os
import sys
from pathlib import Path


def get_multiline_input(prompt):
    """
    Get multiline input from user until they type 'DONE' on a new line.
    """
    print(f"\n{prompt}")
    print("(Type 'DONE' on a new line when finished)")
    print("-" * 50)

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "DONE":
                break
            lines.append(line)
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(1)
        except EOFError:
            break

    # Join lines and clean up formatting
    content = "\n".join(lines).strip()
    return content


def get_next_id(file_path):
    """
    Get the next available ID by reading the existing file.
    """
    if not os.path.exists(file_path):
        return "1"

    max_id = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "id" in data:
                            try:
                                current_id = int(data["id"])
                                max_id = max(max_id, current_id)
                            except ValueError:
                                # If ID is not a number, skip it
                                pass
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
    except Exception as e:
        print(f"Warning: Could not read existing file: {e}")

    return str(max_id + 1)


def find_project_root():
    """
    Find the project root by looking for characteristic files/directories.
    This ensures the script works from any location on any laptop.
    """
    current_path = Path(__file__).resolve()

    # Look for project-specific markers (files that should exist at project root)
    project_markers = ["pyproject.toml", "README.md", "src", "data"]

    # Start from script location and go up until we find the project root
    for parent in [current_path] + list(current_path.parents):
        # Check if this directory contains all our project markers
        if all((parent / marker).exists() for marker in project_markers):
            return parent

    # If we can't find the project root, raise an error with helpful message
    raise FileNotFoundError(
        f"Could not find project root. Please ensure you're running this script "
        f"from within the NeurIPS-MMU-RAG project directory. Looking for: {project_markers}"
    )


def main():
    # Get the project root dynamically
    try:
        project_root = find_project_root()
        data_dir = project_root / "data" / "self-donated-topics"
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    print("=== Self-Donated Q&A Entry Tool ===")
    print(f"📁 Project root: {project_root}")
    print(f"💾 Data directory: {data_dir}")
    print()

    # Get filename
    while True:
        filename = input(
            "Enter the filename (without extensions, will create as {filename}_self-donated_qa.jsonl): "
        ).strip()
        if filename:
            break
        print("Please enter a valid filename.")

    # Construct file path with self-donated qa label
    file_path = data_dir / f"{filename}_self-donated_qa.jsonl"

    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get the next ID
    next_id = get_next_id(file_path)
    print(f"\nNext entry ID will be: {next_id}")

    # Get question
    question = get_multiline_input("Enter your question:")
    if not question.strip():
        print("Question cannot be empty. Exiting.")
        sys.exit(1)

    # Get answer
    answer = get_multiline_input("Enter your answer:")
    if not answer.strip():
        print("Answer cannot be empty. Exiting.")
        sys.exit(1)

    # Get source
    print("\nEnter the source:")
    source = input().strip()
    if not source:
        print("Source cannot be empty. Exiting.")
        sys.exit(1)

    # Create the entry
    entry = {"id": next_id, "question": question, "answer": answer, "source": source}

    # Convert to JSON string
    json_line = json.dumps(entry, ensure_ascii=False)

    # Show preview
    print("\n" + "=" * 60)
    print("PREVIEW OF ENTRY TO BE ADDED:")
    print("=" * 60)
    print(f"ID: {entry['id']}")
    print(
        f"Question: {entry['question'][:100]}{'...' if len(entry['question']) > 100 else ''}"
    )
    print(
        f"Answer: {entry['answer'][:100]}{'...' if len(entry['answer']) > 100 else ''}"
    )
    print(f"Source: {entry['source']}")
    print("=" * 60)

    # Confirm
    while True:
        confirm = input("\nAdd this entry? (y/n): ").strip().lower()
        if confirm in ["y", "yes"]:
            break
        elif confirm in ["n", "no"]:
            print("Entry cancelled.")
            sys.exit(0)
        else:
            print("Please enter 'y' or 'n'.")

    # Write to file
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")

        print(f"\n✅ Entry successfully added to {file_path}")
        print(f"Entry ID: {next_id}")

    except Exception as e:
        print(f"\n❌ Error writing to file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
