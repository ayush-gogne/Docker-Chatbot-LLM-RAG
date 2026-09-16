This project is a chatbot interface that allows users to ask questions about Docker documentation. Behind the scenes, a Llama-based large language model uses Retrieval-Augmented Generation (RAG) to search a collection of Docker-related documentation and retrieve relevant information. The model then uses that context to generate accurate, relevant responses to the user’s query.

Steps to run the chatbot:
1. Ensure you have Python 3 or higher installed<br>
2. Install all the required packages by running the command "pip install -r requirements.txt." This will install all the neccasary libraries listed in the requirements text file.<br>
3. Install Ollama<br>
4. Download the Llama 3.2 model by running the command "ollama pull llama3.2"<br>
5. Collect the docker documentation by running the command "python collect_documents.py" in the terminal<br>
6. Start the chatbot by running the command "rag_system.py"<br>
7. Now that the program is running, you can ask the chatbot questions and it will answer. Type "exit" and hit enter to end the program<br>
8. To evaluate the chatbot's responses to the expected answer, Run the command "python evaluate.py"<br>

Note:
You can also use this single commmand to run all experimental results from the report: cd your dir/Docker-RAG-Project && python3 -m pip install -r requirements.txt && ollama pull llama3.2 && python3 collect_documents.py && python3 create_chunks.py && yes y | head -n 20 | python3 evaluate.py
