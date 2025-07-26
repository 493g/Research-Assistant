# Research-Assistant
This is a Research Assistant that can help you with your research work.
For this project I have used Llama3-70B-8192 as LLM and Qwen/Qwen3-Embedding-8B embedding model from langchain.
ChromaDB in memory vectorestore has been used in this project.

## Features
* Pdf Analysis : Upload your research papers (PDFs) and ask questions directly about their content. The AI will provide answers based on the provided document.
* Wikipedia Integration: Get quick facts and information from Wikipedia for general knowledge queries.
* ArXiv Search: Seamlessly query the ArXiv scientific pre-print archive for research papers and summaries.
* Fast Responses: Powered by Groq's high-speed inference engine for Llama3-70B-8192
## Setup- installation
## 1. clone repository:</br>
git clone https //github.com/493g/Research-Assistant.git \
cd Research-Assistant

## 2. Create a virtual environment :</br>
python -m venv venv \
.\venv\Scripts\activate
## 3. Install dependencies/
pip install -r requirements.txt
## 4. This project requires Groq API key and HuggingFace token:
* Go to GroqConsole
* Sign up or login.
* Navigate to "API Keys" section
* generate a new API Key 
   
For HuggingFace Token:
* Visit HuggingFace.com
* Sign up or Login
* Click on profile and choose Access Tokens
* Click on Create new token
* Select Read option, give name to you token and copy it\
* Open project terminal, type: \
HuggingFace-cli login \
* Paste the token press enter , type y after this step you will be loggedin to you HuggingFace account.

## 5. Create a new .env file 
Paste your Groq api token here \
GROQ_API_KEY="your_groq_api_key_here"

## Run the Applicatiom 
streamlit run research_assistant.py

   
