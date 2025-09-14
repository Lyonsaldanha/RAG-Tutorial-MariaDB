import mariadb
import json
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import requests
import time

# Load BGE embedding model once (outside function so it doesn't reload each call)
model_name = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed(text: str):
    """Return a normalized embedding vector for given text using BGE-small-en-v1.5."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0]  # CLS token
    return F.normalize(embeddings, p=2, dim=1)[0].tolist()  # convert tensor → list

# Database connection
conn = mariadb.connect(
       host="127.0.0.1",
       port=3306,
       user="root",
       password="example"
   )
cur = conn.cursor()

def prepare_database():
    print("Create database and table")
    cur.execute("""
        CREATE DATABASE IF NOT EXISTS kb_rag;
        """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kb_rag.content (
            title VARCHAR(255) NOT NULL,
            url VARCHAR(255) NOT NULL,
            content LONGTEXT NOT NULL,
            embedding VECTOR(384) NOT NULL,
            VECTOR INDEX (embedding)
        );
        """)
    
prepare_database()

def read_kb_from_file(filename):
    with open(filename, "r") as file:
        return [json.loads(line) for line in file]

# chunkify by paragraphs, headers, etc.
def chunkify(content, min_chars=1000, max_chars=10000):
    lines = content.split('\n')
    chunks, chunk, length, start = [], [], 0, 0
    for i, line in enumerate(lines + [""]):  # Add sentinel line for final chunk
        if (chunk and (line.lstrip().startswith('#') or not line.strip() or length + len(line) > max_chars)
                and length >= min_chars):
            chunks.append({'content': '\n'.join(chunk).strip(), 'start_line': start, 'end_line': i - 1})
            chunk, length, start = [], 0, i
        chunk.append(line)
        length += len(line) + 1
    return chunks

def insert_kb_into_db():
    kb_pages = read_kb_from_file("kb_scraped_md_full.jsonl") # change to _full.jsonl for 6000+ KB pages
    for p in kb_pages:
        chunks = chunkify(p["content"])
        for index, chunk in enumerate(chunks):
            embedding = embed(chunk["content"])
            cur.execute("""INSERT INTO kb_rag.content (title, url, content, embedding)
                        VALUES (%s, %s, %s, VEC_FromText(%s))""",
                    (p["title"], p["url"], chunk["content"], json.dumps(embedding)))
        conn.commit()

def search_for_closest_content(text, n=5):
    """Search for the closest content using vector similarity."""
    embedding = embed(text)  # using same embedding model as in preparations
    cur.execute("""
        SELECT title, url, content,
               VEC_DISTANCE_EUCLIDEAN(embedding, VEC_FromText(%s)) AS distance
        FROM kb_rag.content
        ORDER BY distance ASC
        LIMIT %s;
    """, (json.dumps(embedding), n))

    closest_content = [
        {"title": title, "url": url, "content": content, "distance": distance}
        for title, url, content, distance in cur
    ]
    return closest_content

def call_mistral_llm(prompt, max_tokens=1000, temperature=0.7):
    """Call local Mistral model via Ollama API."""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Mistral: {e}")
        return "Error: Could not connect to Mistral model. Make sure Ollama is running with 'ollama serve' and the mistral model is installed."

def create_rag_prompt(user_question, context_chunks):
    """Create a RAG prompt with context and user question."""
    context = "\n\n---\n\n".join([
        f"**{chunk['title']}**\n{chunk['content']}" 
        for chunk in context_chunks
    ])
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided context. Use the context below to answer the user's question accurately and concisely.

Context:
{context}

Question: {user_question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite relevant sections when possible
- Be concise but thorough in your response

Answer:"""
    
    return prompt

def query_with_rag(user_question, num_chunks=5, temperature=0.7):
    """Execute a complete RAG query: retrieve relevant chunks and generate answer."""
    print(f"🔍 Searching for relevant content...")
    
    # Retrieve relevant chunks
    relevant_chunks = search_for_closest_content(user_question, num_chunks)
    
    if not relevant_chunks:
        return "No relevant content found in the knowledge base."
    
    print(f"📚 Found {len(relevant_chunks)} relevant chunks")
    
    # Create RAG prompt
    rag_prompt = create_rag_prompt(user_question, relevant_chunks)
    
    print("🤖 Generating answer with Mistral...")
    
    # Generate answer using Mistral
    answer = call_mistral_llm(rag_prompt, temperature=temperature)
    
    return {
        "question": user_question,
        "answer": answer,
        "sources": [{"title": chunk["title"], "url": chunk["url"], "distance": chunk["distance"]} 
                   for chunk in relevant_chunks]
    }

def query_direct_mistral(user_question, temperature=0.7, max_tokens=1000):
    """Query Mistral directly without RAG context."""
    print("🤖 Querying Mistral directly...")
    
    prompt = f"""Please answer the following question concisely and accurately:

Question: {user_question}

Answer:"""
    
    answer = call_mistral_llm(prompt, max_tokens=max_tokens, temperature=temperature)
    
    return {
        "question": user_question,
        "answer": answer,
        "method": "direct_mistral"
    }

def interactive_query_session():
    """Interactive session for querying the RAG system."""
    print("🚀 RAG Query System Ready!")
    print("Commands:")
    print("  'rag <question>' - Query with RAG")
    print("  'direct <question>' - Query Mistral directly")
    print("  'search <query>' - Just search without LLM")
    print("  'quit' - Exit")
    print()
    
    while True:
        user_input = input("Query> ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.startswith('rag '):
            question = user_input[4:].strip()
            if question:
                result = query_with_rag(question)
                print(f"\n📝 Answer: {result['answer']}")
                print(f"\n📚 Sources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['title']} (distance: {source['distance']:.3f})")
                    print(f"     {source['url']}")
                print()
        
        elif user_input.startswith('direct '):
            question = user_input[7:].strip()
            if question:
                result = query_direct_mistral(question)
                print(f"\n📝 Answer: {result['answer']}\n")
        
        elif user_input.startswith('search '):
            query = user_input[7:].strip()
            if query:
                chunks = search_for_closest_content(query, 3)
                print(f"\n🔍 Search Results:")
                for i, chunk in enumerate(chunks, 1):
                    print(f"{i}. {chunk['title']} (distance: {chunk['distance']:.3f})")
                    print(f"   {chunk['content'][:200]}...")
                    print(f"   {chunk['url']}\n")
        
        else:
            print("Invalid command. Use 'rag <question>', 'direct <question>', 'search <query>', or 'quit'")

# Example usage functions
def example_queries():
    """Run some example queries to demonstrate the system."""
    examples = [
        "Can MariaDB be used instead of an Oracle database?",
        "What are the main features of the system?",
        "How do I configure the database connection?"
    ]
    
    print("🧪 Running example queries...\n")
    
    for question in examples:
        print(f"Question: {question}")
        print("="*50)
        
        # RAG query
        rag_result = query_with_rag(question, num_chunks=3)
        print(f"RAG Answer: {rag_result['answer']}\n")
        
        # Direct query for comparison
        direct_result = query_direct_mistral(question)
        print(f"Direct Answer: {direct_result['answer']}\n")
        
        print("-"*50)
        time.sleep(1)  # Small delay between queries

if __name__ == "__main__":
    # Test basic functionality
    print("🔧 Testing system components...")
    
    # Test embedding
    test_embedding = embed("test text")
    print(f"✅ Embedding model working (dimension: {len(test_embedding)})")
    
    # Test database connection
    cur.execute("SELECT 1")
    print("✅ Database connection working")
    
    # Test Mistral connection
    test_response = call_mistral_llm("Say 'Hello, RAG system is working!'", max_tokens=50)
    if "Error:" not in test_response:
        print("✅ Mistral LLM connection working")
    else:
        print(f"⚠️  Mistral LLM issue: {test_response}")
    
    print("\n" + "="*50)
    
    # Uncomment to run example queries
    # example_queries()
    
    # Start interactive session
    interactive_query_session()