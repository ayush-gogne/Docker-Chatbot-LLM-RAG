# This file collects and prepares Docker documentation for use in the RAG system.
# It downloads relevant documentation pages, extracts useful content, and prepares the data for retrieval.

# Importing the necassary libraries
import json
import os
import random
import numpy as np
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

try:
    import torch
except ImportError: 
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

# Website where the Docker documention files are collected from
startPage = "https://docs.docker.com/"

# Number of documents to collect for the knowledge base.
# Corpus size of 1000 documents was hardcoded, typically a learning curve analysis is done to find the correct amount of data that balances overfitting and underfitting.
maxDocs = 200

# docs list to store pages we already visited
docs = []
seenPages = set()

# code to extract the webpage, disassemble and keeping only the useful text
def cleanText(html):

    soup = BeautifulSoup(html, "html.parser")

    # Remove sections of the page that are not useful for this purpose
    for item in soup(["script", "style", "nav", "footer"]):
        item.decompose()

    text = soup.get_text("\n")

    cleanLines = []

    # Removing empty lines
    for line in text.split("\n"):
        line = line.strip()

        if line:
            cleanLines.append(line)

    return "\n".join(cleanLines)


# Get more Docker pages from a single webpage through hyperlinks
def getLinks(url):
    # Download the webpage
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    # List to store the links
    links = []
    # Find all the links on the page
    for a in soup.find_all("a", href=True):
        link = urljoin(url, a["href"])
        # Only keep Docker documentation links
        # also checking to make sure we havent visited that link already, if not then add to the list
        if link.startswith("https://docs.docker.com") and link not in seenPages:
            links.append(link)

    return links


# Pages we still need to collect
# Put the first docker page that we will visit, initialization
pages = [startPage]


# Keep collecting until we have enough documents
while pages and len(docs) < maxDocs:
    #extract the first page from thre list
    currentPage = pages.pop(0)
    # Skip already collected pages
    if currentPage in seenPages:
        continue
    seenPages.add(currentPage)
    try:
        # Download the webpage
        html = requests.get(currentPage).text
        # Get the useful text from the page
        text = cleanText(html)
        # Save pages that have enough information
        # hardcoded document length to 400 to ensure useful documents are selected
        if len(text) > 400:
            docs.append({
                "id": len(docs),
                "source": currentPage,
                "text": text
            })
        # Add new pages to collect later
        pages.extend(getLinks(currentPage))


    except:
        #skip pages with errors
        pass



# Create a folder for documents, if it already exists then don't do anything
os.makedirs("data/documents", exist_ok=True)

# Save each document as a text file
for doc in docs:
    fileName = "data/documents/" + str(doc["id"]) + ".txt"
    with open(fileName, "w", encoding="utf-8") as file:
        file.write(doc["text"])
# Save the document information
with open("data/metadata.json", "w") as file:
    #put everything in json
    json.dump(
        docs,
        file,
        indent=4
    )

print("Collected:", len(docs), "documents")
