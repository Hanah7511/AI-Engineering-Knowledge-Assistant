import os
from dotenv import load_dotenv
load_dotenv()
print("API KEY LOADED:", os.getenv("OPENAI_API_KEY")[:10] if os.getenv("OPENAI_API_KEY") else "NOT FOUND")
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # or your preferred embeddings
from langchain_community.vectorstores import Chroma  # or Pinecone, FAISS, etc.
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, CSVLoader, JSONLoader

import hashlib
import json
import nbformat
from datetime import datetime



# Configuration
SUPPORTED_EXTENSIONS = [".py", ".md", ".txt", ".json", ".csv", ".yaml", ".yml", ".ipynb"]
PERSIST_DIRECTORY = "data/vector_store"
METADATA_FILE = "data/document_metadata.json"

class MLOpsKnowledgeBase:
    def __init__(self, persist_directory: str = PERSIST_DIRECTORY):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )  # Replace with your preferred embedding model
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""],
            length_function=len,
        )
        
    def load_documents(self, base_path: str = "../../data/documents") -> List[Document]:
        """Load all supported documents from directory"""
        documents = []
        file_metadata = {}
        
        for root, _, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip unwanted directories
                if any(skip in file_path for skip in ["__pycache__", ".git", "node_modules"]):
                    continue
                
                ext = os.path.splitext(file)[1].lower()
                
                if ext not in SUPPORTED_EXTENSIONS:
                    continue
                    
                try:
                    # Create a unique ID for the file
                    file_hash = self._get_file_hash(file_path)
                    file_stat = os.stat(file_path)
                    
                    # Store file metadata
                    file_metadata[file_path] = {
                        "hash": file_hash,
                        "modified": file_stat.st_mtime,
                        "size": file_stat.st_size
                    }
                    
                    # Load based on file type
                    loaded_docs = self._load_single_document(file_path, ext)
                    
                    # Add metadata to each chunk
                    for doc in loaded_docs:
                        doc.metadata.update({
                            "source": file_path,
                            "file_type": ext,
                            "file_hash": file_hash,
                            "loaded_at": datetime.now().isoformat()
                        })
                    
                    documents.extend(loaded_docs)
                    print(f"✓ Loaded: {file}")
                    
                except Exception as e:
                    print(f"✗ Skipped {file}: {str(e)}")
        
        # Save metadata for future incremental updates
        self._save_metadata(file_metadata)
        return documents
    
    def _load_single_document(self, file_path: str, ext: str) -> List[Document]:
        """Load a single document based on its extension"""
        if ext in [".py", ".md", ".txt", ".yaml", ".yml"]:
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()
        
        elif ext == ".csv":
            loader = CSVLoader(file_path)
            return loader.load()
        
        elif ext == ".json":
            try:
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema=".",
                    text_content=False
                )
                return loader.load()
            except:
                print(f"Skipping JSON(jq issue):{file_path}")
                return[]
        
        elif ext == ".ipynb":
            # Handle Jupyter notebooks
            return self._load_notebook(file_path)
        
        return []
    
    def _load_notebook(self, file_path: str) -> List[Document]:
        """Extract code and markdown from Jupyter notebooks"""
        import nbformat
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            
            for i, cell in enumerate(nb.cells):
                if cell.cell_type in ['code', 'markdown']:
                    content = f"Cell {i} ({cell.cell_type}):\n{cell.source}"
                    doc = Document(
                        page_content=content,
                        metadata={
                            "cell_type": cell.cell_type,
                            "cell_index": i,
                            "source": file_path
                        }
                    )
                    documents.append(doc)
        
        return documents
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file content to detect changes"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    
    def _save_metadata(self, metadata: Dict):
        """Save file metadata for incremental updates"""
        os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self) -> Dict:
        """Load previous metadata"""
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def create_vector_store(self, documents: List[Document]):
        """Create and persist vector store from documents"""
        print(f"\n Creating embeddings for {len(documents)} chunks...")
        
        # Split documents into chunks
        splits = self.text_splitter.split_documents(documents)
        print(f" Created {len(splits)} chunks after splitting")
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Persist to disk
        self.vector_store.persist()
        print(f" Vector store saved to {self.persist_directory}")
        
        return self.vector_store
    
    def incremental_update(self, base_path: str = "data/documents"):
        """Only update changed files"""
        old_metadata = self._load_metadata()
        new_documents = []
        
        for root, _, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)
                if "__pycache__" in file_path:
                    continue
                    
                # Check if file has changed
                current_hash = self._get_file_hash(file_path)
                old_hash = old_metadata.get(file_path, {}).get("hash")
                
                if current_hash != old_hash:
                    print(f" Updating: {file}")
                    ext = os.path.splitext(file)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        docs = self._load_single_document(file_path, ext)
                        new_documents.extend(docs)
        
        if new_documents:
            print(f"\n Processing {len(new_documents)} updated documents...")
            # Option 1: Recreate vector store (simpler)
            # self.create_vector_store(new_documents)
            
            # Option 2: Add to existing vector store
            if self.vector_store is None:
                self.load_vector_store()
            
            splits = self.text_splitter.split_documents(new_documents)
            self.vector_store.add_documents(splits)
            self.vector_store.persist()
            print(f" Added {len(splits)} new chunks to vector store")
        else:
            print(" All files are up to date")
    
    def load_vector_store(self):
        """Load existing vector store"""
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f" Loaded vector store with {self.vector_store._collection.count()} chunks")
            return True
        return False
    
    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if not self.vector_store:
            self.load_vector_store()
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
    
    def get_retriever(self, k: int = 5):
        """Get a retriever for use in chains"""
        if not self.vector_store:
            self.load_vector_store()
        
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )


def main():
    """Main execution function"""
    print(" Initializing MLOps Knowledge Base...")
    
    # Initialize knowledge base
    kb = MLOpsKnowledgeBase()
    
    # Check if vector store exists
    if os.path.exists(PERSIST_DIRECTORY):
        print(" Existing vector store found. Loading...")
        kb.load_vector_store()
        
        # Optionally check for updates
        print("\n Checking for file updates...")
        kb.incremental_update()
    else:
        print(" No existing vector store. Creating new one...")
        # Load all documents
        documents = kb.load_documents()
        print(f" Loaded {len(documents)} documents")
        
        # Create vector store
        kb.create_vector_store(documents)
    
    print("\n Knowledge base ready!")
    
    # Example search
    query = "How do we handle model versioning in our projects?"
    print(f"\n Searching for: '{query}'")
    results = kb.search_similar(query, k=3)
    
    for i, (doc, score) in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {score:.3f}) ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content preview: {doc.page_content[:200]}...")


if __name__ == "__main__":
    # Install required packages first:
    # pip install langchain langchain-community chromadb nbformat openai tiktoken
    
    main()