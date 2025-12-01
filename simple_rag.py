import torch
import ollama
import os

print("=" * 60)
print("   MAKERSPACE RAG - Høgskolen i Østfold")
print("   3D Printing | Laser Cutting | Electronics | Making")
print("=" * 60)

# Load vault
print("\nLoading knowledge base...")
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as f:
        vault_content = [line.strip() for line in f.readlines() if line.strip()]
print(f"✓ Loaded {len(vault_content)} knowledge chunks")

# Generate embeddings
print("Generating embeddings (this may take a moment)...")
vault_embeddings = []
for i, content in enumerate(vault_content):
    response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
    vault_embeddings.append(response["embedding"])
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{len(vault_content)} chunks embedded")

vault_embeddings_tensor = torch.tensor(vault_embeddings)
print(f"✓ Embeddings ready!")
print("=" * 60)

def search_vault(query, top_k=3):
    """Find most relevant chunks for a query"""
    query_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=query)["embedding"]
    cos_scores = torch.cosine_similarity(torch.tensor(query_embedding).unsqueeze(0), vault_embeddings_tensor)
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    return [vault_content[idx] for idx in top_indices]

def ask_llm(query, context):
    """Send query + context to llama3"""
    prompt = f"""You are a helpful assistant for the Makerspace at Høgskolen i Østfold.
You help students and staff with questions about 3D printing, laser cutting, electronics (Arduino/Raspberry Pi), and maker projects.

Based on the following context from our knowledge base, answer the question accurately and helpfully.

Context:
{context}

Question: {query}

Answer:"""
    
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content']

# Main loop
print("\n🔧 Ask questions about:")
print("   • 3D Printing (Prusa MK4S, Mini, filaments, settings)")
print("   • Laser Cutting (materials, settings, safety)")
print("   • Electronics (Arduino, Raspberry Pi, sensors)")
print("   • 3D Modeling (software, file formats, design tips)")
print("   • Maker Movement & Makerspace resources")
print("\nType 'quit' to exit.\n")

while True:
    try:
        query = input("You: ").strip()
        if query.lower() == 'quit':
            print("\n👋 Goodbye! Happy making!")
            break
        if not query:
            continue
            
        print("\n🔍 Searching knowledge base...")
        relevant_chunks = search_vault(query)
        context = "\n\n".join(relevant_chunks)
        
        print("📚 Found relevant information:")
        for i, chunk in enumerate(relevant_chunks, 1):
            preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
            print(f"   {i}. {preview}")
        
        print("\n💭 Generating response...\n")
        response = ask_llm(query, context)
        
        print(f"Assistant: {response}\n")
        print("-" * 60)
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Happy making!")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        continue
