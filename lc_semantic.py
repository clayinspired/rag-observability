import hashlib
from dotenv import load_dotenv
import time

load_dotenv()

# ============================================================================
# 1. Load a PDF document
# ============================================================================
from langchain_community.document_loaders import PyPDFLoader

file_path = r"C:\Users\monster\Desktop\Coding\shipit\data\10. Payment Services Act.pdf"
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

# https://www.pinecone.io/learn/vector-similarity/

query_1 = "What are the payment laws?"
query_vector = embeddings.embed_query(query_1)

results = index.query(
    vector=query_vector,
    top_k=3,
    namespace="japanese-legal",
    include_metadata=True
)

print(results)

for match in results["matches"]:
    print(f"Score: {match['score']}")
#query_1 = "What are the relevant laws related to overseas tourists using Paypay but using overseas app to pay in Japan?"
results = vector_store.similarity_search(query_1, k=3)

print(f"Query: {query_1}")
if results:
    print(f"Answer: {results[0].page_content}\n")
else:
    print("No results found.\n")

# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.

query_2 = "What are the payment laws?"
#query_2 = "What are the relevant laws related to overseas tourists using Paypay but using overseas app to pay in Japan?"
results = vector_store.similarity_search_with_score(query_2, k=3)

print(f"Query: {query_2}")
if results:
    doc, score = results[0]
    print(f"Score: {score}")
    print(f"Answer: {doc.page_content}\n")
else:
    print("No results found.\n")

# return documents based on similarity to an embedded query
query_3 = "What are the relevant laws related to overseas tourists using Paypay but using overseas app to pay in Japan?"
embedding = embeddings.embed_query(query_3)

results = vector_store.similarity_search_by_vector(embedding)
print(f"Query: {query_3}")
if results:
    print(f"Answer: {results[0].page_content}\n")
else:
    print("No results found.\n")

# ============================================================================
# Retrievers

from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)

results = retriever.batch(
    [
        "What are the relevant laws related to overseas tourists using Paypay but using overseas app to pay in Japan?",
    ],
)

print("Batch retrieval results:")
print(results)


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

results = retriever.batch(
    [
        "What are the relevant laws related to overseas tourists using Paypay but using overseas app to pay in Japan?",
    ],
)

print("Batch retrieval results:")
print(results)

# ============================================================================
# Test direct Pinecone query
test_query = "payment service provider"
query_vector = embeddings.embed_query(test_query)

results = index.query(
    vector=query_vector,
    top_k=3,
    namespace="japanese-legal",
    include_metadata=True
)

print(f"Direct Pinecone query for '{test_query}':")
print(f"Found {len(results['matches'])} results")
for match in results['matches']:
    print(f"Score: {match['score']}")
    print(f"Metadata: {match.get('metadata', {})}\n")

# RAG CHAIN

from langchain.agents import create_agent

from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message


from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

agent = create_agent(model=llm, tools=[], middleware=[prompt_with_context])

query = "What are the relevant laws for overseas payment services?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# Debug: Check what Pinecone actually returns
debug_query = "overseas payment services"
query_vector = embeddings.embed_query(debug_query)
results = index.query(
    vector=query_vector,
    top_k=3,
    namespace="japanese-legal",
    include_metadata=True
)

print(f"\n=== DEBUG: Direct Pinecone results for '{debug_query}' ===")
for match in results['matches']:
    print(f"Score: {match['score']}")
    print(f"Text: {match['metadata'].get('text', '')[:200]}...\n")


# document load