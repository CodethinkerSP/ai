from sentence_transformers import SentenceTransformer
import chromadb,requests,jsom
dataset = []
# data loading
with open("data.txt", 'r', encoding='utf-8') as f:
  dataset = f.readlines()
# variables
VECTOR_DB = []
EMBEDDING_MODEL =  'all-MiniLM-L6-v2'
LANGUAGE_MODEL = 'phi3:latest'

# initialize vector db
chroma_client = chromadb.PersistentClient(path="./chroma_db3")
# create collection
collection = chroma_client.get_or_create_collection(name="mydataset")

#embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
for data in dataset:
  data = data.strip()
  if data:
    embedding = model.encode(data).tolist()
    # storing in vector db
    VECTOR_DB.append((data, embedding))
print(f"Inserted {len(VECTOR_DB)} records into the vector database.")

# Local ollama query
def query_ollama(prompt, model="phi3:latest"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
             "options": {
            "temperature": 0.1,  # Lower temperature for more deterministic responses
            "top_p": 0.9,
            "num_ctx": 4096
            }   
        }
    )
    return response.json()["response"]

def generate_answer(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.Follow these rules strictly: 
     1. Only use information from the provided context 
     2. If the context doesn't contain the answer, say so explicitly 
     3. Do not add any information from your training data
    Context:
    {context}
    Question: {question}
    Answer:"""
    return query_ollama(prompt)
# call the function to generate answer
generate_answer("what is the context of this content ?", [record[0] for record in VECTOR_DB[:2]])