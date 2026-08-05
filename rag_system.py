#first we have to import all of the necassary libraries
import os
#downloaded ollama so now we can use it
import ollama
#Used for keyword searching
from rank_bm25 import BM25Okapi
# Used to convert text into numbers
from sentence_transformers import SentenceTransformer
# Used to compare similarity, use for dense retrieval
from sklearn.metrics.pairwise import cosine_similarity
#frist Loading all text files from  the documents folder :)
#this is our main file for running the chatbout, it collexts the dcuments, searches them for relavent info, and uses llama to create an answer
def loadDocs():
    docs = []
    #Going through every document file
    for file in os.listdir("data/documents"):
        #Read the document text
        with open(
            "data/documents/" + file,
            encoding="utf-8"
        ) as f:
            docs.append(
                f.read()
            )
    return docs
# Split documents into smaller chunks
def splitChunks(docs):
    chunks = []
    # Go through each document
    for docNum, doc in enumerate(docs):
        # Splitting document into words
        words = doc.split()
        # Creating 250 word chunks
        for i in range(0, len(words), 250):
            chunkText = " ".join(
                words[i:i+250]
            )
            # Saving chunk information
            chunks.append({
                "chunkId": len(chunks),
                "docId": docNum,
                "source": "Document " + str(docNum),
                "text": chunkText
            })
    return chunks
# Create BM25 search model
def makeBM25(chunks):
    allWords = []
    #Converting chunks into lists of words
    for chunk in chunks:
        allWords.append(
            chunk["text"].split()
        )
    return BM25Okapi(allWords)
#Searching for the best chunk using BM25 moodel
def searchBM25(model, chunks, question):
    #Find matching scores
    scores = model.get_scores(
        question.split()
    )
    # Get highest scoring chunk
    best = scores.argmax()
    return chunks[best]
# Create embedding model
encoder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
#Converting chunks into vectors
def makeEmbeddings(chunks):
    texts = []
    #Getting text from every chunk
    for chunk in chunks:
        texts.append(
            chunk["text"]
        )
    #Converting the text into numbers now
    vectors = encoder.encode(
        texts
    )
    return vectors
#Searching for the most similar chunk using embeddings
def searchDense(chunks, vectors, question):
    # Convert question into a vector
    questionVector = encoder.encode(
        [question]
    )
    #Compare question with all chunks
    scores = cosine_similarity(
        questionVector,
        vectors
    )[0]
    #Finding the highest similarity
    best = scores.argmax()
    return chunks[best]
#Generating the answer using Llama now
def getAnswer(question, chunk):
    #Giving Llama the context and question
    #the default prommpts below are to tell Llama to only use the documents we provide and also what to answer when it doesnt no an answer which is outside its knowledge base
    prompt = f"""
Answer only using the context.
If the answer is not available,
say "I don't know".
the Context:
{chunk["text"]}
the Question:
{question}
"""
    #Asking Llama for answer
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    answer = response["message"]["content"]

    return answer

#we must load and prepare the RAG system
docs = loadDocs()
#splitting to chunks
chunks = splitChunks(docs)
#Create BM25 search
bm25 = makeBM25(chunks)
# Create vectors for dense search
vectors = makeEmbeddings(chunks)
def askQuestion(question, type="dense"):
    if type == "bm25":
        chunk = searchBM25(
            bm25,
            chunks,
            question
        )
    else:
        chunk = searchDense(
            chunks,
            vectors,
            question
        )
    answer = getAnswer(
        question,
        chunk
    )
    return answer
#This part starts our RAG system as a chatbot
if __name__ == "__main__":
    #Keep asking questions until the user wants to stop
    while True:
        #et a question from the user :)
        question = input(
            "\nQuestion: "
        )
        #   if the user types exit then stop the program
        if question == "exit":
            break
        # Send the question to the RAG system
        # and print the generated answer
        print(
            askQuestion(question)
        )