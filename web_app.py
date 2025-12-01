from flask import Flask, render_template, request, jsonify
import torch
import ollama
import os
import re

app = Flask(__name__)

# Global variables for embeddings
vault_content = []
vault_embeddings_tensor = None

def load_vault():
    """Load vault content and generate embeddings"""
    global vault_content, vault_embeddings_tensor
    
    print("Loading knowledge base...")
    vault_content = []
    if os.path.exists("vault.txt"):
        with open("vault.txt", "r", encoding='utf-8') as f:
            vault_content = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(vault_content)} knowledge chunks")
    
    print("Generating embeddings (this may take a few minutes)...")
    vault_embeddings = []
    for i, content in enumerate(vault_content):
        response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
        vault_embeddings.append(response["embedding"])
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(vault_content)}")
    
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    print("Embeddings ready!")

def search_vault(query, top_k=5):
    """Find most relevant chunks for a query"""
    if vault_embeddings_tensor is None or len(vault_content) == 0:
        return []
    
    # Remove level commands from search query for better matching
    clean_query = re.sub(r'/(beginner|new|intermediate|advanced|expert)\s*', '', query)
    
    query_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=clean_query)["embedding"]
    cos_scores = torch.cosine_similarity(torch.tensor(query_embedding).unsqueeze(0), vault_embeddings_tensor)
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    return [vault_content[idx] for idx in top_indices]

def detect_level(query):
    """Detect explanation level from query"""
    query_lower = query.lower()
    if '/beginner' in query_lower:
        return 'beginner', "Explain like I know absolutely nothing. Use simple words, no jargon, step-by-step with examples. Be encouraging."
    elif '/new' in query_lower:
        return 'new', "Explain assuming basic familiarity but still learning. Define technical terms when first used. Include beginner tips."
    elif '/advanced' in query_lower:
        return 'advanced', "Assume significant experience. Go into technical depth, edge cases, optimization. Skip basic explanations."
    elif '/expert' in query_lower:
        return 'expert', "Assume deep expertise. Discuss at professional level with theory, advanced troubleshooting, and technical precision."
    else:
        # Default to intermediate
        return 'intermediate', "Assume working knowledge. Use standard terminology, focus on practical details and tips."

def ask_llm(query, context):
    """Send query + context to llama3"""
    level, level_instruction = detect_level(query)
    
    # Clean query for display (remove level commands)
    clean_query = re.sub(r'/(beginner|new|intermediate|advanced|expert)\s*', '', query).strip()
    
    prompt = f"""You are a helpful assistant for the Makerspace at Høgskolen i Østfold (HiØF).
You help students and staff with questions about 3D printing, laser cutting, electronics, vinyl cutting, textiles, and maker projects.

IMPORTANT - Explanation Level: {level.upper()}
{level_instruction}

Use the following knowledge base context to answer accurately. If the context doesn't contain relevant information, use your general knowledge but mention that.

=== KNOWLEDGE BASE CONTEXT ===
{context}
=== END CONTEXT ===

Question: {clean_query}

Respond in the same language as the question (Norwegian or English). Be practical and helpful."""
    
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message', '')
    
    if not query:
        return jsonify({'error': 'No message provided'}), 400
    
    # Search for relevant context
    relevant_chunks = search_vault(query, top_k=5)
    context = "\n\n".join(relevant_chunks)
    
    # Debug: print what context was found
    print(f"\n--- Query: {query[:50]}...")
    print(f"--- Found {len(relevant_chunks)} relevant chunks")
    
    # Get response from LLM
    response = ask_llm(query, context)
    
    return jsonify({
        'response': response,
        'sources': len(relevant_chunks)
    })

@app.route('/status')
def status():
    return jsonify({
        'loaded': vault_embeddings_tensor is not None,
        'chunks': len(vault_content)
    })

if __name__ == '__main__':
    load_vault()
    print("\n" + "="*50)
    print("Makerspace RAG Web Interface")
    print("Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
