Final Project
By: Ayush Gogne & Ryan Chisholm :)

To run our project: 

-first make sure you have Python 3 or higher installed<br>
-install all the required packages by running pip install -r requirements.txt because we put all the necassary libraries you need to download in the text file so its easier<br>
-install Ollama<br>
-Download the Llama 3.2 model by running ollama pull llama3.2<br>
-collect the docker documentation by running python collect_documents.py in the terminal<br>
-start the chatbot by running rag_system.py<br>
-You can then ask questions and the chatbot will answer. type exit to end it<br>
-To evaluate the chatbots responses vs what it is supposed to be run the command python evaluate.py<br>

Or use this single commmand to run all experimental results from the report: python3 -m pip install -r requirements.txt && ollama pull llama3.2 && python3 collect_documents.py && python3 create_chunks.py && yes y | head -n 20 | python3 evaluate.py
