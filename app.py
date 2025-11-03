import hashlib
from dotenv import load_dotenv
import time
import os


load_dotenv()

# ============================================================================
# 1. Load a PDF document
# ============================================================================
from langchain_community.document_loaders import PyPDFLoader

file_path = "data/10. Payment Services Act.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

print(f"Number of pages in the document: {len(docs)}\n")
print(f"The string content of the page:\n {docs[0].page_content[:200]}\n")

from pprint import pprint
pprint(f"Metadata containing the file name and page number:\n {docs[0].metadata}\n")

# ============================================================================
# 2. Split the document into smaller chunks
# ============================================================================
from langchain_text_splitters import RecursiveCharacterTextSplitter

japanese_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        "。",
        "、",
        "，",
        "．",
        ".",
        ",",
        "\u200b",
        " ",
        "",
    ],
    chunk_size=512,
    chunk_overlap=50,
    add_start_index=True,
)
all_splits = japanese_splitter.split_documents(docs)

print(f"Number of chunks created: {len(all_splits)}\n")

# ============================================================================
# 3. Generate embeddings using Ollama's all-minilm model
# ============================================================================
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="all-minilm")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(f"First 10 dimensions of first vector: {vector_1[:10]}\n")

# ============================================================================
# 4. Add unique IDs based on content hash (prevent duplicates)
# ============================================================================
def generate_doc_id(doc) -> str:
    """Generate a unique ID based on document content and metadata"""
    content_str = f"{doc.page_content}{doc.metadata.get('source', '')}{doc.metadata.get('page', '')}"
    content_hash = hashlib.md5(content_str.encode()).hexdigest()
    return content_hash

# Assign IDs to all splits
for doc in all_splits:
    doc.id = generate_doc_id(doc)

print(f"Assigned unique IDs to all {len(all_splits)} chunks\n")

# ============================================================================
# 5. Create Pinecone index
# ============================================================================
from pinecone import Pinecone
from pinecone import ServerlessSpec

pc = Pinecone()
index_name = "japanese-payment-services-act"

print("Waiting for Pinecone to sync...")
time.sleep(2)

try:
    print(f"Creating index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=len(vector_1),
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled"
    )
    print(f"Index {index_name} created successfully\n")
    time.sleep(3)
    
except Exception as e:
    if "already exists" in str(e):
        print(f"Index {index_name} already exists\n")
    else:
        raise

# ============================================================================
# 6. Connect to Pinecone and store embeddings in vector store
# ============================================================================
from langchain_pinecone import PineconeVectorStore

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    embedding=embeddings,
    index=index,
    namespace="japanese-legal"
)

# Check if documents already exist by checking namespace stats
print("Checking for existing documents in Pinecone...")
try:
    stats = index.describe_index_stats()
    namespace_stats = stats.namespaces.get('japanese-legal', {})
    existing_count = namespace_stats.vector_count if hasattr(namespace_stats, 'vector_count') else 0
    
    if existing_count >= len(all_splits):
        print(f"Found {existing_count} existing vectors in namespace (>= {len(all_splits)} chunks).")
        print("All documents already uploaded. Skipping re-upload.\n")
        print(f"✓ All documents already in Pinecone (skipped upload)")
        print(f"Time elapsed: 0.00 seconds (0.00 minutes)\n")
    else:
        print(f"Found {existing_count} existing vectors. Proceeding with upload...\n")
        batch_size = 50
        total_ids = []
        start_time = time.time()

        # Embed and upload directly using Pinecone API
        vectors_to_upsert = []
        for doc in all_splits:
            vector = embeddings.embed_query(doc.page_content)
            vectors_to_upsert.append({
                "id": doc.id,
                "values": vector,
                "metadata": {
                    "text": doc.page_content[:1000],
                    "source": doc.metadata.get("source", ""),
                    "page": str(doc.metadata.get("page", ""))
                }
            })

        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i+batch_size]
            index.upsert(vectors=batch, namespace="japanese-legal")
            total_ids.extend([v["id"] for v in batch])
            print(f"Adding batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1} ({len(batch)} chunks)...")
            print(f"  ✓ Added {len(batch)} documents")

        elapsed_time = time.time() - start_time
        print(f"\n✓ Successfully added/updated {len(total_ids)} total documents to Pinecone")
        print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
        
except Exception as e:
    print(f"Error checking existing docs: {e}")
    print("Proceeding with upload anyway...\n")
    
    batch_size = 50
    total_ids = []
    start_time = time.time()

    # Embed and upload directly using Pinecone API
    vectors_to_upsert = []
    for doc in all_splits:
        vector = embeddings.embed_query(doc.page_content)
        vectors_to_upsert.append({
            "id": doc.id,
            "values": vector,
            "metadata": {
                "text": doc.page_content[:1000],
                "source": doc.metadata.get("source", ""),
                "page": str(doc.metadata.get("page", ""))
            }
        })

    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i+batch_size]
        index.upsert(vectors=batch, namespace="japanese-legal")
        total_ids.extend([v["id"] for v in batch])
        print(f"Adding batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1} ({len(batch)} chunks)...")
        print(f"  ✓ Added {len(batch)} documents")

    elapsed_time = time.time() - start_time
    print(f"\n✓ Successfully added/updated {len(total_ids)} total documents to Pinecone")
    print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")

# ============================================================================
# 7. Sample query to verify embeddings are searchable
# ============================================================================

from langchain.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
import json

def detect_response_language(response):
    """
    Detect if response is primarily in English or other languages.
    Returns: {"language": "english" or "other", "english_ratio": 0.0-1.0}
    """
    # Count ASCII letters (typically English)
    ascii_letters = sum(1 for c in response if ord(c) < 128 and c.isalpha())
    
    # Count non-ASCII letters (Chinese, Japanese, etc.)
    non_ascii_letters = sum(1 for c in response if ord(c) >= 128 and c.isalpha())
    
    total_letters = ascii_letters + non_ascii_letters
    
    if total_letters == 0:
        return {"language": "unknown", "english_ratio": 0.0}
    
    english_ratio = ascii_letters / total_letters
    
    # If >70% English letters, it's English; otherwise flag as mixed/other
    language = "english" if english_ratio > 0.7 else "other"
    
    return {"language": language, "english_ratio": round(english_ratio, 2)}


def calculate_context_overlap(response, retrieved_text):
    """Calculate % of response vocabulary from retrieved context"""
    if not response or not retrieved_text:
        return 0.0
    
    context_words = set(retrieved_text.lower().split())
    response_words = set(response.lower().split())
    
    overlap = len(context_words & response_words) / len(response_words) if response_words else 0
    return round(overlap, 2)


def run_rag_pipeline(user_query, config):
    """RAG pipeline with configurable temperature and observability"""
    
    query_vector = embeddings.embed_query(user_query)
    retrieval_results = index.query(
        vector=query_vector,
        top_k=config["top_k"],
        namespace="japanese-legal",
        include_metadata=True
    )
    
    retrieved_text = "\n\n".join(
        [m["metadata"]["text"] for m in retrieval_results["matches"]]
    )
    
    # Create LLM with config temperature
    llm = ChatOllama(
        model="llama3.2:3b", 
        temperature=config["temperature"]
    )
    
    messages = [
        SystemMessage(
            content="Do NOT reply me other languages except English. You are a legal assistant. Use only the provided context."
        ),
        HumanMessage(
            content=f"Context:\n\n{retrieved_text}\n\nQuestion: {user_query}"
        )
    ]
    
    response = llm.invoke(messages)
    answer = response.content
    
    # Calculate context overlap
    context_overlap = calculate_context_overlap(answer, retrieved_text)

    # Detect language
    language_info = detect_response_language(answer)

    # Log the run
    log_entry = {
        "query": user_query,
        "config": config,
        "retrieval": {
            "num_results": len(retrieval_results["matches"]),
            "scores": [m["score"] for m in retrieval_results["matches"]],
            "min_score": min([m["score"] for m in retrieval_results["matches"]]),
            "max_score": max([m["score"] for m in retrieval_results["matches"]]),
            "avg_score": sum(m["score"] for m in retrieval_results["matches"]) / len(retrieval_results["matches"]),
        },
        "generation": {
            "temperature": config["temperature"],
            "response_length": len(answer),
            "context_overlap": context_overlap,
            "response_text": answer,
            "language": language_info["language"],  # ADD THIS
            "english_ratio": language_info["english_ratio"],  # ADD THIS
        }
    }

    return answer, log_entry



# ============================================================================
# Experiment 1: Vary top_k (retrieval depth) - FIXED temperature
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 1: Testing top_k impact (retrieval depth)")
print("="*80 + "\n")

configs_exp1 = [
    {"top_k": 1, "temperature": 0.1},
    {"top_k": 3, "temperature": 0.1},
    {"top_k": 5, "temperature": 0.1},
    {"top_k": 10, "temperature": 0.1},
]

for config in configs_exp1:
    query = "What are the payment laws?"
    answer, log = run_rag_pipeline(query, config)
    
    with open("rag_logs.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")
    
    print(f"Config: {config}")
    print(f"Scores - Min: {log['retrieval']['min_score']:.2f}, Max: {log['retrieval']['max_score']:.2f}, Avg: {log['retrieval']['avg_score']:.2f}")
    print(f"Response length: {log['generation']['response_length']}")
    print(f"Language: {log['generation']['language']} (English ratio: {log['generation']['english_ratio']})")
    print(f"\nResponse:\n{answer}")
    print("-" * 80 + "\n")


# ============================================================================
# Experiment 2: Vary temperature (generation randomness) - FIXED top_k
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 2: Testing temperature impact (generation randomness)")
print("="*80 + "\n")

configs_exp2 = [
    {"top_k": 3, "temperature": 0.0},
    {"top_k": 3, "temperature": 0.1},
    {"top_k": 3, "temperature": 0.3},
    {"top_k": 3, "temperature": 0.5},
    {"top_k": 3, "temperature": 1.0},
]

for config in configs_exp2:
    query = "What are the payment laws?"
    answer, log = run_rag_pipeline(query, config)
    
    with open("rag_logs.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")
    
    print(f"Config: {config}")
    print(f"Scores - Min: {log['retrieval']['min_score']:.2f}, Max: {log['retrieval']['max_score']:.2f}, Avg: {log['retrieval']['avg_score']:.2f}")
    print(f"Response length: {log['generation']['response_length']}")
    print(f"Language: {log['generation']['language']} (English ratio: {log['generation']['english_ratio']})")
    print(f"\nResponse:\n{answer}")
    print("-" * 80 + "\n")
