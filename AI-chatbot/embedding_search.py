import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pickle
import os

class EmbeddingSearchSystem:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding search system
        Uses SentenceTransformers for creating embeddings and FAISS for similarity search
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None
        
    def create_embeddings(self, chunks: List[Dict[str, any]]) -> np.ndarray:
        """
        Create embeddings for all text chunks
        """
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings using SentenceTransformers
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        return embeddings
    
    def build_index(self, chunks: List[Dict[str, any]]) -> None:
        """
        Build FAISS index from document chunks
        """
        self.chunks = chunks
        
        # Create embeddings
        print("Creating embeddings...")
        self.embeddings = self.create_embeddings(chunks)
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Add embeddings to index
        self.index.add(self.embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for most relevant chunks given a query
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Create query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(score)
                chunk['rank'] = i + 1
                results.append(chunk)
        
        return results
    
    def get_context_for_query(self, query: str, max_chunks: int = 3, min_score: float = 0.3) -> Tuple[str, List[Dict[str, any]]]:
        """
        Get relevant context for a query with source references
        """
        search_results = self.search(query, top_k=max_chunks * 2)  # Get more results to filter
        
        # Filter by minimum similarity score
        filtered_results = [
            result for result in search_results 
            if result['similarity_score'] >= min_score
        ][:max_chunks]
        
        if not filtered_results:
            return "No relevant context found.", []
        
        # Combine context from top chunks
        context_parts = []
        source_refs = []
        
        for result in filtered_results:
            text = result['text']
            filename = result.get('filename', 'Unknown')
            page_num = result.get('page_num', 'Unknown')
            score = result['similarity_score']
            
            context_parts.append(f"[Source: {filename}, Page {page_num}]\n{text}")
            source_refs.append({
                'filename': filename,
                'page_num': page_num,
                'similarity_score': score,
                'text_preview': text[:200] + "..." if len(text) > 200 else text
            })
        
        combined_context = "\n\n".join(context_parts)
        
        return combined_context, source_refs
    
    def save_index(self, filepath: str) -> None:
        """
        Save the FAISS index and associated data
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save chunks and embeddings
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'model_name': self.model_name
            }, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """
        Load a previously saved FAISS index
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load chunks and embeddings
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
            saved_model_name = data['model_name']
        
        # Verify model compatibility
        if saved_model_name != self.model_name:
            print(f"Warning: Loaded index was created with {saved_model_name}, "
                  f"but current model is {self.model_name}")
        
        print(f"Index loaded with {self.index.ntotal} vectors")
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the search system
        """
        if not self.chunks:
            return {"status": "No data loaded"}
        
        total_chunks = len(self.chunks)
        total_files = len(set(chunk.get('filename', '') for chunk in self.chunks))
        total_pages = len(set((chunk.get('filename', ''), chunk.get('page_num', 0)) for chunk in self.chunks))
        
        avg_chunk_length = np.mean([len(chunk['text']) for chunk in self.chunks])
        
        return {
            "total_chunks": total_chunks,
            "total_files": total_files,
            "total_pages": total_pages,
            "average_chunk_length": round(avg_chunk_length, 2),
            "model_name": self.model_name,
            "index_size": self.index.ntotal if self.index else 0
        }
