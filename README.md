# Its my toddler steps to learn and understanding the LLM/AI/ML topics by doing simple code.

* Trying to create very simple RAG system that will query the local pdf file(s).
* I used the tiny LLM phi3:latest and embedding mode all-MiniLM-L6-v2.
## Steps followed on my local machine
1. Installed Ollama and pulled the phi3:latest llm
2. Ollama CLI : ollama serve and the ollama run phi3:latest
3. Python code
    1. Load text file
    2. Chunked and embedding the data
    3. Storing the embeddings in the chromdb
    4. Hitting the Ollama API endpoint --> localhost:11434/api/generate with required "json" payload