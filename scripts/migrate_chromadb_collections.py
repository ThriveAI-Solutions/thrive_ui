import os
import shutil

import chromadb


def backup_and_migrate_collection(client, collection_name: str):
    """
    Backs up an existing ChromaDB collection and migrates it to use the 'cosine' distance metric.

    Args:
        client: The ChromaDB client.
        collection_name (str): The name of the collection to migrate.
    """
    print(f"Starting migration for collection: {collection_name}")

    # 1. Get the existing collection
    try:
        original_collection = client.get_collection(collection_name)
        print(f"Found existing collection: '{collection_name}'")
    except Exception as e:
        print(f"Could not get collection '{collection_name}'. It might not exist. Error: {e}")
        print("Skipping migration for this collection.")
        return

    # Determine the distance metric of the original collection
    original_metadata = original_collection.metadata
    space = "l2"  # Default distance metric in ChromaDB if not specified
    if original_metadata and "hnsw:space" in original_metadata:
        space = original_metadata["hnsw:space"]

    # Check if the collection is already cosine
    if space == "cosine":
        print(f"Collection '{collection_name}' is already using 'cosine' distance. No migration needed.")
        print("-" * 20)
        return

    # Ensure the collection is l2 before proceeding
    if space != "l2":
        print(f"Collection '{collection_name}' is using '{space}', not 'l2'. Skipping migration.")
        print("-" * 20)
        return
        
    print(f"Collection '{collection_name}' is using '{space}'. Proceeding with migration to 'cosine'.")

    # 2. Backup collection with a dynamic suffix
    backup_collection_name = f"{collection_name}_{space}"
    print(f"Creating backup collection: '{backup_collection_name}'")

    try:
        # If a backup from a previous run exists, delete it.
        client.delete_collection(name=backup_collection_name)
        print(f"Deleted existing backup collection: '{backup_collection_name}'")
    except Exception:
        # It's ok if it doesn't exist
        pass
    
    backup_collection = client.create_collection(
        name=backup_collection_name, 
        metadata=original_metadata # Preserve original metadata
    )
    
    existing_count = original_collection.count()
    if existing_count > 0:
        print(f"Copying {existing_count} items to backup collection '{backup_collection_name}'...")
        batch_size = 100
        for i in range(0, existing_count, batch_size):
            batch = original_collection.get(
                include=["metadatas", "documents", "embeddings"],
                limit=batch_size,
                offset=i
            )
            backup_collection.add(
                ids=batch["ids"],
                documents=batch["documents"],
                metadatas=batch["metadatas"],
                embeddings=batch["embeddings"]
            )
        print("Backup copy complete.")
    else:
        print(f"Original collection '{collection_name}' is empty. Nothing to backup.")


    # 3. Delete original collection
    print(f"Deleting original collection: '{collection_name}'")
    client.delete_collection(name=collection_name)

    # 4. Create new collection with cosine metric
    print(f"Creating new collection '{collection_name}' with 'cosine' distance metric.")
    new_collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # 5. Copy from backup to new collection
    backup_count = backup_collection.count()
    if backup_count > 0:
        print(f"Copying {backup_count} items from backup to new '{collection_name}' collection...")
        batch_size = 100
        for i in range(0, backup_count, batch_size):
            batch = backup_collection.get(
                include=["metadatas", "documents", "embeddings"],
                limit=batch_size,
                offset=i
            )
            new_collection.add(
                ids=batch["ids"],
                documents=batch["documents"],
                metadatas=batch["metadatas"],
                embeddings=batch["embeddings"]
            )
        print("Data copy to new collection complete.")

    # 6. Verification
    new_count = new_collection.count()
    print(f"New collection '{collection_name}' has {new_count} items.")
    if backup_count != new_count:
        print(f"WARNING: Item count mismatch! Backup: {backup_count}, New: {new_count}")

    new_metadata = new_collection.metadata
    print(f"New collection metadata: {new_metadata}")
    if new_metadata and new_metadata.get("hnsw:space") == "cosine":
        print(f"Successfully verified that '{collection_name}' is using 'cosine' distance.")
    else:
        print(f"ERROR: Verification failed! '{collection_name}' is not using 'cosine' distance.")
    
    print("-" * 20)


def main():
    """
    Main function to run the migration script.
    """
    db_path = "./chromadb"
    if not os.path.isdir(db_path):
        print(f"Error: ChromaDB path not found at '{db_path}'")
        print("Please ensure you are running this script from the root of the project.")
        return

    client = chromadb.PersistentClient(path=db_path)

    collections_to_migrate = ["documentation", "ddl", "sql"]

    for collection_name in collections_to_migrate:
        backup_and_migrate_collection(client, collection_name)

    print("Migration script finished.")


if __name__ == "__main__":
    main() 