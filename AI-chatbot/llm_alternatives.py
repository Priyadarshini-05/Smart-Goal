import requests
import json
from typing import Dict, List, Optional
import os
import time

class HuggingFaceLLM:
    """
    Free Hugging Face Inference API integration
    No API key required for basic usage
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {}
        
        # Optional: Add HF token for better rate limits
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            self.headers["Authorization"] = f"Bearer {hf_token}"
    
    def generate_answer(self, question: str, context: str, source_refs: List[Dict] = None) -> Dict[str, any]:
        """Generate answer using Hugging Face free API"""
        
        if not context.strip():
            return {
                "answer": "I couldn't find relevant information in your uploaded documents to answer this question.",
                "sources": [],
                "success": True,
                "context_used": False
            }
        
        # Create a simple prompt
        prompt = f"Context: {context[:1000]}\n\nQuestion: {question}\n\nAnswer:"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '').strip()
                    
                    return {
                        "answer": generated_text or "I found relevant information but couldn't generate a complete answer. Please try rephrasing your question.",
                        "sources": source_refs or [],
                        "success": True,
                        "context_used": True,
                        "model": "huggingface-free"
                    }
            
            # Fallback to context-based answer
            return self._fallback_answer(question, context, source_refs)
            
        except Exception as e:
            return self._fallback_answer(question, context, source_refs)
    
    def _fallback_answer(self, question: str, context: str, source_refs: List[Dict] = None) -> Dict[str, any]:
        """Fallback to simple context-based answer"""
        answer = f"Based on your documents:\n\n{context[:600]}..."
        
        if source_refs:
            answer += "\n\nSources:\n"
            for ref in source_refs[:3]:
                answer += f"• {ref['filename']}, Page {ref['page_num']}\n"
        
        return {
            "answer": answer,
            "sources": source_refs or [],
            "success": True,
            "context_used": True,
            "model": "context-based"
        }
    
    def generate_followup_questions(self, question: str, answer: str, context: str) -> List[str]:
        """Generate simple follow-up questions"""
        return [
            "Can you provide more details about this topic?",
            "What are the key points I should remember?",
            "How does this relate to other concepts?"
        ]


class OllamaLLM:
    """
    Local Ollama integration for completely offline usage
    Requires Ollama to be installed locally
    """
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
    
    def generate_answer(self, question: str, context: str, source_refs: List[Dict] = None) -> Dict[str, any]:
        """Generate answer using local Ollama"""
        
        if not context.strip():
            return {
                "answer": "I couldn't find relevant information in your uploaded documents to answer this question.",
                "sources": [],
                "success": True,
                "context_used": False
            }
        
        prompt = f"""You are StudyMate, an AI academic assistant. Answer the question based on the provided context from PDF documents.

Context: {context[:1500]}

Question: {question}

Answer:"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 300
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                return {
                    "answer": generated_text,
                    "sources": source_refs or [],
                    "success": True,
                    "context_used": True,
                    "model": f"ollama-{self.model_name}"
                }
            else:
                return self._fallback_answer(question, context, source_refs)
                
        except requests.exceptions.ConnectionError:
            return {
                "answer": "Ollama is not running. Please start Ollama or use a different LLM option.",
                "sources": [],
                "success": False,
                "error": "Ollama connection failed"
            }
        except Exception as e:
            return self._fallback_answer(question, context, source_refs)
    
    def _fallback_answer(self, question: str, context: str, source_refs: List[Dict] = None) -> Dict[str, any]:
        """Fallback answer when Ollama fails"""
        answer = f"Based on your documents:\n\n{context[:600]}..."
        
        if source_refs:
            answer += "\n\nSources:\n"
            for ref in source_refs[:3]:
                answer += f"• {ref['filename']}, Page {ref['page_num']}\n"
        
        return {
            "answer": answer,
            "sources": source_refs or [],
            "success": True,
            "context_used": True,
            "model": "context-based"
        }
    
    def generate_followup_questions(self, question: str, answer: str, context: str) -> List[str]:
        """Generate follow-up questions"""
        return [
            "Can you explain this concept further?",
            "What are some practical applications?",
            "How does this connect to other topics?"
        ]


class OpenAILLM:
    """
    OpenAI API integration (requires API key but widely available)
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.base_url = "https://api.openai.com/v1"
    
    def generate_answer(self, question: str, context: str, source_refs: List[Dict] = None) -> Dict[str, any]:
        """Generate answer using OpenAI API"""
        
        if not self.api_key:
            return {
                "answer": "OpenAI API key not provided. Please set OPENAI_API_KEY environment variable.",
                "sources": [],
                "success": False,
                "error": "Missing API key"
            }
        
        if not context.strip():
            return {
                "answer": "I couldn't find relevant information in your uploaded documents to answer this question.",
                "sources": [],
                "success": True,
                "context_used": False
            }
        
        messages = [
            {
                "role": "system",
                "content": "You are StudyMate, an AI academic assistant that helps students understand their study materials. Answer questions based only on the provided context from PDF documents."
            },
            {
                "role": "user",
                "content": f"Context from uploaded documents:\n{context}\n\nQuestion: {question}\n\nPlease provide a clear, comprehensive answer based on the context."
            }
        ]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['message']['content'].strip()
                
                return {
                    "answer": generated_text,
                    "sources": source_refs or [],
                    "success": True,
                    "context_used": True,
                    "model": self.model
                }
            else:
                return {
                    "answer": f"OpenAI API error: {response.text}",
                    "sources": [],
                    "success": False,
                    "error": response.text
                }
                
        except Exception as e:
            return {
                "answer": f"Error connecting to OpenAI: {str(e)}",
                "sources": [],
                "success": False,
                "error": str(e)
            }
    
    def generate_followup_questions(self, question: str, answer: str, context: str) -> List[str]:
        """Generate follow-up questions using OpenAI"""
        if not self.api_key:
            return ["Can you explain this in more detail?", "What are the key takeaways?", "How does this apply in practice?"]
        
        messages = [
            {
                "role": "user",
                "content": f"Based on this Q&A about academic content, suggest 3 relevant follow-up questions:\n\nQ: {question}\nA: {answer}\n\nProvide 3 questions, one per line:"
            }
        ]
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": "gpt-3.5-turbo", "messages": messages, "max_tokens": 150, "temperature": 0.3},
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['message']['content'].strip()
                questions = [q.strip() for q in generated_text.split('\n') if q.strip() and '?' in q]
                return questions[:3]
            
        except Exception:
            pass
        
        return ["Can you explain this in more detail?", "What are the key takeaways?", "How does this apply in practice?"]
