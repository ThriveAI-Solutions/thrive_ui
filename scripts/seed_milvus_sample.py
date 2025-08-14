import argparse
from pathlib import Path

from utils.milvus_vector import ThriveAI_Milvus


def main():
    parser = argparse.ArgumentParser(description="Seed a Milvus Lite DB with a small sample corpus.")
    parser.add_argument("--path", type=str, default="./milvus_demo.db", help="Path for Milvus Lite persistence file")
    parser.add_argument("--prefix", type=str, default="sample", help="Collection name prefix")
    parser.add_argument("--role", type=int, default=1, help="User role to assign to seeded docs")
    args = parser.parse_args()

    store = ThriveAI_Milvus(
        user_role=args.role,
        config={
            "mode": "lite",
            "persist_path": str(Path(args.path).expanduser().resolve()),
            "text_dim": 128,
            "collection_prefix": args.prefix,
        },
    )

    docs = [
        "Discharge steps include UNIQUE_A",
        "Follow-up schedule contains UNIQUE_B",
        "Medication notes mention UNIQUE_C",
        "General health advice",
        "Another general document",
    ]

    for d in docs:
        store.add_documentation(d)

    store.add_ddl("CREATE TABLE patients (id INT, name TEXT);")
    store.add_question_sql("How many patients?", "SELECT COUNT(*) FROM patients;")

    print("Seeded Milvus Lite at:", args.path)


if __name__ == "__main__":
    main()


