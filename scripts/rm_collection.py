import argparse
import os
import sys

import chromadb


def remove_collections(db_path: str, collection_names: list[str]) -> int:
    """Clear all items from one or more collections in a ChromaDB persistent client.

    Leaves the collections themselves intact. Returns process exit code (0 on success,
    non-zero if any collection could not be cleared or does not exist).
    """
    client = chromadb.PersistentClient(path=db_path)

    exit_code = 0
    for name in collection_names:
        print(f"Clearing collection: '{name}' in '{db_path}' ...")
        try:
            collection = client.get_collection(name=name)
        except Exception as exc:
            print(f"Failed to access collection '{name}': {exc}")
            exit_code = 1
            continue

        try:
            cleared_total = 0
            batch_size = 100
            while True:
                batch = collection.get(limit=batch_size)
                ids = batch.get("ids", [])
                if not ids:
                    break
                collection.delete(ids=ids)
                cleared_total += len(ids)
            print(f"Cleared {cleared_total} items from collection '{name}'.")
        except Exception as exc:  # Chroma may raise different exceptions across versions
            print(f"Failed to clear collection '{name}': {exc}")
            exit_code = 1

    return exit_code


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove all documents from one or more ChromaDB collections (leave collections intact).\n\n"
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
        help="One or more collection names to clear",
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


