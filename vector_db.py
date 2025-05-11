from typing import List, Dict, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from knowledge_base import get_knowledge_base, save_knowledge_base_to_json

class VectorDBManager:
    def __init__(self, index_name: str = "wellbeing_index"):
        self.index_name = index_name
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the vector store with the knowledge base content."""
        try:
            # Try to load existing index
            if os.path.exists(self.index_name):
                self.vector_store = FAISS.load_local(
                    self.index_name,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                # Create new index from knowledge base
                self._create_vector_store()
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            self._create_vector_store()
    
    def _create_vector_store(self):
        """Create a new vector store from the knowledge base."""
        try:
            # Get knowledge base content
            knowledge_base = get_knowledge_base()
            
            # Prepare documents for vector store
            documents = []
            for entry in knowledge_base:
                # Create a formatted text for each entry
                text = f"Category: {entry['category']}\nTitle: {entry['title']}\n\n{entry['content']}"
                documents.append(text)
            
            # Create vector store
            self.vector_store = FAISS.from_texts(documents, self.embeddings)
            
            # Save the vector store
            self.vector_store.save_local(self.index_name)
            
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise
    
    def semantic_search(self, query: str, top_k: int = 2):
        """Perform semantic search on the vector store and return Document objects."""
        if not self.vector_store:
            self._initialize_vector_store()
        
        try:
            # Search for similar documents
            docs = self.vector_store.similarity_search(query, k=top_k)
            return docs  # Return Document objects directly
        
        except Exception as e:
            print(f"Error performing semantic search: {str(e)}")
            return []
    
    def add_documents(self, documents: List[Dict]):
        """Add new documents to the vector store."""
        try:
            # Format new documents
            texts = []
            for doc in documents:
                text = f"Category: {doc['category']}\nTitle: {doc['title']}\n\n{doc['content']}"
                texts.append(text)
            
            # Add to vector store
            self.vector_store.add_texts(texts)
            
            # Save updated vector store
            self.vector_store.save_local(self.index_name)
            
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            raise

def initialize_vector_db():
    """Initialize the vector database with the knowledge base content."""
    try:
        # Save knowledge base to JSON
        save_knowledge_base_to_json()
        
        # Create vector database manager
        db_manager = VectorDBManager()
        
        return db_manager
    
    except Exception as e:
        print(f"Error initializing vector database: {str(e)}")
        raise

if __name__ == "__main__":
    # Initialize the vector database
    db_manager = initialize_vector_db() 