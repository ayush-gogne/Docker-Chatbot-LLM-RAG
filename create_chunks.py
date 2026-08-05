#import all th neccasary libraries
import json

#this file is to break the documents down into smaller pieces and chunks

# Load the documents we collected earlier
documents = json.load(
    open(
        "data/metadata.json",
        encoding="utf-8"
    )
)
# Store all document chunks
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
    " we created this many: ",
    len(chunks),
    "chunks"
)