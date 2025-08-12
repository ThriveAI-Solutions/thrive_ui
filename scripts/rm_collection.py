import argparse
import os
import sys

import chromadb


def remove_collections(db_path: str, collection_names: list[str]) -> int:
    """Remove one or more collections from a ChromaDB persistent client.

    Returns process exit code (0 on success, non-zero if any deletion failed).
    """
    client = chromadb.PersistentClient(path=db_path)

    exit_code = 0
    for name in collection_names:
        try:
            print(f"Deleting collection: '{name}' from '{db_path}' ...")
            client.delete_collection(name=name)
            print(f"Deleted: '{name}'")
        except Exception as exc:  # Chroma may raise different exceptions across versions
            # Do not fail the whole run; report and continue
            print(f"Failed to delete collection '{name}': {exc}")
            exit_code = 1

    return exit_code


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove one or more collections from a ChromaDB persistent store.\n\n"
            "Examples:\n"
            "  uv run scripts/rm_collection.py --collection ddl\n"
            "  uv run scripts/rm_collection.py --path ./chromadb --collection ddl sql\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--path",
        default="./chromadb",
        help="Path to the ChromaDB persistent directory (default: ./chromadb)",
    )
    parser.add_argument(
        "-c",
        "--collection",
        dest="collections",
        nargs="+",
        required=True,
        help="One or more collection names to remove",
    )

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    db_path = args.path
    if not os.path.isdir(db_path):
        print(f"Error: ChromaDB path not found at '{db_path}'")
        print("Ensure you are running from the project root or provide a valid --path.")
        return 2

    collections: list[str] = list(args.collections)
    if not collections:
        # argparse enforces required=True, but guard anyway
        print("Error: At least one --collection must be specified.")
        return 2

    return remove_collections(db_path=db_path, collection_names=collections)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


