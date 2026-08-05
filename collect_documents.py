#By: Ayush gogne and Ryan Chisholm

#This file is to collect the docker documents and prepares the data :)
# For our RAG system it downloads the pages and retrieves useful  content

#the first step is we have to import the necassary libraries
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import json

# Website where we start collecting Docker documents
startPage = "https://docs.docker.com/"

# Number of documents we want to collect
# 200-2000 recommended but we decided to do 200 for simplicity, perhaps the model would learn and perform better if we used more documents for its knowledgebase
maxDocs = 200

# Store documents and pages we already visited
docs = []
seenPages = set()

# extract webpage and disassemble and keep only the useful text
def cleanText(html):

    soup = BeautifulSoup(html, "html.parser")

    # now we are going to Remove parts of the website we do not need for simplicity
    for item in soup(["script", "style", "nav", "footer"]):
        item.decompose()

    text = soup.get_text("\n")

    cleanLines = []

    # Remove empty lines
    for line in text.split("\n"):
        line = line.strip()

        if line:
            cleanLines.append(line)

    return "\n".join(cleanLines)


# Get more Docker pages from a single webpage for more knowledge
def getLinks(url):
    #downloading the webpage
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    #list to store the links
    links = []
    # Find all links on the page
    for a in soup.find_all("a", href=True):
        link = urljoin(url, a["href"])
        # Only keep Docker documentation links
        # also check to make sure we havent visited that link already, if not then add to the list
        if link.startswith("https://docs.docker.com") and link not in seenPages:
            links.append(link)

    return links


# Pages we still need to collect
# Put the first docker page that we will visit, initializing
pages = [startPage]


# Keep collecting until we have enough documents, max 200
while pages and len(docs) < maxDocs:
    #extract the first page from thre list
    currentPage = pages.pop(0)
    # Skip pages we already collected
    if currentPage in seenPages:
        continue
    seenPages.add(currentPage)
    try:
        # Download the webpage
        html = requests.get(currentPage).text
        # Get the useful text from the page
        text = cleanText(html)
        # Save pages that have enough information
        # we hardcoded 400 because we dont want a whole bunch of small documents with little information and we don't want it to be too high, otherwise it will miss documents of reasonable length that gets to the point
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



# Create folder for documents, if it already exists then dont do anything
os.makedirs("data/documents", exist_ok=True)

# Save each document as a text file
for doc in docs:
    fileName = "data/documents/" + str(doc["id"]) + ".txt"
    with open(fileName, "w", encoding="utf-8") as file:
        file.write(doc["text"])
# Save document information
with open("data/metadata.json", "w") as file:
    #put everything in json
    json.dump(
        docs,
        file,
        indent=4
    )

print("Finished collecting!! :", len(docs), "documents")