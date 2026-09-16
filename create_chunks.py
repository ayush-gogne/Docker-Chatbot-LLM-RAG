# Importing all the neccasary libraries
# The purpose of this file is to break the documents down into smaller chunks for processing
import json
import os
import random
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

set_seed(42)

# Loading the collected documents
documents = json.load(
    open(
        "data/metadata.json",
        encoding="utf-8"
    )
)
# Storing all the document chunks
chunks = []
# Give each chunk its own ID
chunkId = 0
# Go through every document
for doc in documents:
    # Split the document into words
    words = doc["text"].split()

    # Split the document into chunks of 200 words
    for i in range(
        0,
        len(words),
        200
    ):
        chunkText = " ".join(
            words[i:i+200]
        )
        # Save each chunk
        chunks.append(
            {
                "chunk_id": chunkId,
                "document_id": doc["id"],
                "source": doc["source"],
                "text": chunkText
            }
        )
        # Move to the next chunk ID
        chunkId += 1
# Save all chunks into a JSON file
with open(
    "data/chunks.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        chunks,
        f,
        indent=4
    )
# Show how many chunks were created
print(
    "Created:",
    len(chunks),
    "chunks"
)
