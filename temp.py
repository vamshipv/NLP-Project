import numpy as np
import faiss

def index_chunks(self):
    if not self.chunked_reviews:
        raise ValueError("No chunked reviews to index. Run chunk_reviews first.")

    # Index general (non-aspect) chunks
    general_chunks = [c for c in self.chunked_reviews if "aspect" not in c]
    general_texts = [f"passage: {c['text']}" for c in general_chunks]
    embeddings = self.model.encode(general_texts, convert_to_numpy=True, normalize_embeddings=True)
    dimension = embeddings.shape[1]

    # Use IndexIDMap for general chunks
    general_index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    ids = np.arange(len(general_chunks))
    general_index.add_with_ids(embeddings, ids)

    faiss.write_index(general_index, index_path)

    # Save metadata
    with open(os.path.join(output_dir, "general_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(general_chunks, f, indent=2)

    logging.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "event": "index_chunks",
        "num_chunks": len(general_chunks),
        "index_file": index_path
    }))

    # Index each aspect separately
    for aspect in self.aspect_keywords:
        aspect_chunks = [c for c in self.chunked_reviews if c.get("aspect") == aspect]
        if not aspect_chunks:
            continue

        aspect_dir = os.path.join(aspect_index_dir, aspect)
        os.makedirs(aspect_dir, exist_ok=True)

        with open(os.path.join(aspect_dir, f"{aspect}_chunks.json"), "w", encoding="utf-8") as f:
            json.dump(aspect_chunks, f, indent=2)

        texts = [f"passage: {c['text']}" for c in aspect_chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]

        # ✅ Use IndexIDMap for aspect chunks
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        ids = np.arange(len(aspect_chunks))
        index.add_with_ids(embeddings, ids)

        faiss.write_index(index, os.path.join(aspect_dir, f"{aspect}.index"))

        logging.info(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "event": f"index_{aspect}",
            "num_chunks": len(aspect_chunks),
            "index_file": f"{aspect}.index"
        }))
