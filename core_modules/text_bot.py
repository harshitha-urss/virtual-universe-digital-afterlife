import glob
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

# 1. Load transcript files:
knowledge_texts = []
for file in glob.glob('transcripts/*.txt'):
    with open(file, 'r', encoding='utf-8') as f:
        knowledge_texts.append(f.read())

# 2. Chunking:
def chunk_text(text, max_tokens=150):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, token_count = [], [], 0
    for sent in sentences:
        token_count += len(sent.split())
        current.append(sent)
        if token_count >= max_tokens:
            chunks.append(' '.join(current))
            current, token_count = [], 0
    if current:
        chunks.append(' '.join(current))
    return chunks

all_chunks = []
for doc in knowledge_texts:
    all_chunks.extend(chunk_text(doc))

# 3. Embeddings & FAISS index:
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(all_chunks)
np_emb = np.array(chunk_embeddings).astype('float32')
index = faiss.IndexFlatL2(np_emb.shape[1])
index.add(np_emb)

# 4. Load GPT4All model:
model_path = r'C:\Users\hp\AppData\Local\nomic.ai\GPT4All\qwen2-1_5b-instruct-q4_0.gguf'
llm = GPT4All(model_path, allow_download=False)

print("\n=== Welcome to the Ratan Tata Knowledge Bot (Powered by GPT4All) ===\n \nPlease Include Question mark at the end of the question(?)\n")
print("\nType your question (must mention Ratan Tata) and press Enter. Type 'exit' to quit.\n")

# 5. Chat loop with Ratan Tata relevance check:
while True:
    user_query = input("Ask a question (or 'exit'): ").strip()
    if user_query.lower() == 'exit':
        print("Goodbye!")
        break
    
    if user_query.lower() in ["who are you", "what are you", "what is your name", "who is this", "who are you?", "what are you?", "what is your name?", "who is this?"]:
        print("Hello! I am the Ratan Tata Knowledge Bot—your expert assistant here to answer questions about Ratan Tata, his legacy, achievements, and work. Feel free to ask me anything related to Ratan Tata, and I'll do my best to help you.")
        continue

    # Require user's question to actually mention 'ratan tata'
    if not (('ratan tata' in user_query.lower()) or ('tata group' in user_query.lower()) or ('tata sons' in user_query.lower()) or ('tata' in user_query.lower())):
        print("This is not relevant to Ratan Tata. Please ask a question about Ratan Tata.")
        continue
    query_emb = model.encode([user_query]).astype('float32')
    k = 2
    D, I = index.search(query_emb, k)
    context = "\n".join([all_chunks[idx] for idx in I[0]])
    prompt = (
        "\nYou are an expert on Ratan Tata. Using the context below if available, answer the following question in one concise and complete sentence. Do not use disclaimers or list items. Be direct and factual.\n"
        f"Context:\n{context}\n"
        f"Question: {user_query}\n"
        f"Answer:"
    )
    with llm.chat_session():
        response = llm.generate(prompt, max_tokens=36)
    print("\nBot:", response.strip(), "\n---")
