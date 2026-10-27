import os
from pathlib import Path
import chromadb
import pandas as pd

CHROMA_PATH = str(Path(__file__).parent / "data" / "chroma_db")
COLLECTION_NAME = "chat_messages__intfloat_multilingual_e5_base"
OUTPUT_DIR = str(Path(__file__).parent / "projector_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

data = collection.get(include=["embeddings", "documents", "metadatas"])

# vectors.tsv — числа, разделённые табами, по вектору в строке
vectors_path = os.path.join(OUTPUT_DIR, "vectors.tsv")
with open(vectors_path, "w", encoding="utf-8") as f:
    for vec in data["embeddings"]:
        f.write("\t".join(f"{x:.6f}" for x in vec) + "\n")

# metadata.tsv — то, что будет видно при клике на точку
meta_path = os.path.join(OUTPUT_DIR, "metadata.tsv")
with open(meta_path, "w", encoding="utf-8") as f:
    f.write("id\ttext\n")  # заголовок обязателен, если колонок >1
    for doc_id, doc in zip(data["ids"], data["documents"]):
        # обрежем длинные тексты и почистим переводы строк/табы
        clean_text = (doc or "").replace("\n", " ").replace("\t", " ")[:200]
        f.write(f"{doc_id}\t{clean_text}\n")

print(f"Готово!")
print(f"  Векторы: {vectors_path}")
print(f"  Метаданные: {meta_path}")
print(f"  Записей: {len(data['ids'])}")
print()
print("Дальше:")
print("  1. Открой https://projector.tensorflow.org/")
print("  2. Слева кнопка 'Load' → загрузи vectors.tsv и metadata.tsv")
print("  3. Справа выбери UMAP (или t-SNE), увидишь облако")