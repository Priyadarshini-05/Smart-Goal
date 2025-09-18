import requests
import json
from typing import Dict, List, Optional
import os
from datetime import datetime

from llm_alternatives import HuggingFaceLLM, OllamaLLM, OpenAILLM

class WatsonxLLMIntegration:
    def __init__(self, api_key: str = None, project_id: str = None, url: str = None):
        """
        Initialize Watsonx LLM integration with IBM Granite models
        """
        self.api_key = api_key or os.getenv('WATSONX_API_KEY')
        self.project_id = project_id or os.getenv('WATSONX_PROJECT_ID')
        self.url = url or os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
        
        self.granite_models = {
            "granite-13b-chat": "ibm/granite-13b-chat-v2",
            "granite-13b-instruct": "ibm/granite-13b-instruct-v2", 
            "granite-20b-multilingual": "ibm/granite-20b-multilingual",
            "granite-3b-code-instruct": "ibm/granite-3b-code-instruct",
            "granite-8b-code-instruct": "ibm/granite-8b-code-instruct",
            "granite-20b-code-instruct": "ibm/granite-20b-code-instruct",
            "granite-34b-code-instruct": "ibm/granite-34b-code-instruct"
        }
        
        # Default to the best general-purpose Granite model
        self.default_model = self.granite_models["granite-13b-chat"]
        
        if not all([self.api_key, self.project_id]):
            raise ValueError("API key and project ID are required for Watsonx integration")
        
        self.access_token = None
        self.token_expires_at = None
        
    def _get_access_token(self) -> str:
        """
        Get access token for Watsonx API
        """
        # Check if token is still valid
        if self.access_token and self.token_expires_at:
            if datetime.now().timestamp() < self.token_expires_at - 300:  # 5 min buffer
                return self.access_token
        
        # Get new token
        token_url = "https://iam.cloud.ibm.com/identity/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:apikey",
            "apikey": self.api_key
        }
        
        response = requests.post(token_url, headers=headers, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
            self.token_expires_at = datetime.now().timestamp() + token_data['expires_in']
            return self.access_token
        else:
            raise Exception(f"Failed to get access token: {response.text}")
    
    def generate_answer(self, question: str, context: str, source_refs: List[Dict] = None) -> Dict[str, any]:
        """
        Generate answer using IBM Granite models on Watsonx
        """
        # Create prompt with context
        prompt = self._create_granite_prompt(question, context, source_refs)
        
        # Prepare API request
        url = f"{self.url}/ml/v1/text/generation?version=2023-05-29"
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._get_access_token()}"
        }
        
        body = {
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 600,
                "temperature": 0.2,
                "repetition_penalty": 1.05,
                "stop_sequences": ["Human:", "Question:", "\n\n---", "User:"]
            },
            "model_id": self.default_model,  # Using IBM Granite model
            "project_id": self.project_id
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(body))
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['results'][0]['generated_text'].strip()
                
                return {
                    "answer": generated_text,
                    "sources": source_refs or [],
                    "model": "IBM Granite-13B-Chat",
                    "success": True,
                    "context_used": len(context) > 0
                }
            else:
                return {
                    "answer": f"Error generating answer: {response.text}",
                    "sources": [],
                    "success": False,
                    "error": response.text
                }
                
        except Exception as e:
            return {
                "answer": f"Error connecting to Watsonx: {str(e)}",
                "sources": [],
                "success": False,
                "error": str(e)
            }
    
    def _create_granite_prompt(self, question: str, context: str, source_refs: List[Dict] = None) -> str:
        """
        Create a well-structured prompt optimized for IBM Granite models
        """
        prompt = f"""<|system|>
You are StudyMate, an intelligent academic assistant powered by IBM Granite. You help students understand their study materials by providing clear, accurate answers based on uploaded PDF documents.

Guidelines:
- Answer questions using ONLY the provided context from the documents
- Be precise, educational, and well-structured
- Include page references when available
- If information is insufficient, state this clearly
- Use academic language appropriate for students

<|user|>
Context from documents:
{context}

Question: {question}

<|assistant|>
"""
        return prompt
    
    def generate_code_answer(self, question: str, context: str) -> Dict[str, any]:
        """
        Generate code-related answers using IBM Granite Code models
        """
        prompt = f"""<|system|>
You are a coding assistant powered by IBM Granite Code models. Help with programming questions based on the provided context.

<|user|>
Context: {context}
Question: {question}

<|assistant|>
"""
        
        url = f"{self.url}/ml/v1/text/generation?version=2023-05-29"
        
        headers = {
            "Accept": "application/json", 
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._get_access_token()}"
        }
        
        body = {
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 800,
                "temperature": 0.1,
                "repetition_penalty": 1.02
            },
            "model_id": self.granite_models["granite-20b-code-instruct"],
            "project_id": self.project_id
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(body))
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['results'][0]['generated_text'].strip()
                
                return {
                    "answer": generated_text,
                    "model": "IBM Granite-20B-Code-Instruct",
                    "success": True
                }
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_followup_questions(self, question: str, answer: str, context: str) -> List[str]:
        """
        Generate relevant follow-up questions using IBM Granite
        """
        prompt = f"""<|system|>
Generate 3 relevant follow-up questions that would help a student deepen their understanding of this topic.

<|user|>
Original Question: {question}
Answer: {answer}
Context: {context[:500]}...

Generate exactly 3 follow-up questions:

<|assistant|>
"""

        url = f"{self.url}/ml/v1/text/generation?version=2023-05-29"
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {self._get_access_token()}"
        }
        
        body = {
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 200,
                "temperature": 0.4,
                "stop_sequences": ["\n\n", "<|user|>"]
            },
            "model_id": self.default_model,
            "project_id": self.project_id
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(body))
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['results'][0]['generated_text'].strip()
                
                # Parse follow-up questions
                questions = [q.strip() for q in generated_text.split('\n') if q.strip() and '?' in q]
                return questions[:3]  # Return max 3 questions
            else:
                return []
                
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            return []

    def get_available_models(self) -> Dict[str, str]:
        """
        Get list of available IBM Granite models
        """
        return {
            "Chat Models": {
                "granite-13b-chat": "IBM Granite 13B Chat - Best for conversations",
                "granite-13b-instruct": "IBM Granite 13B Instruct - Best for instructions"
            },
            "Code Models": {
                "granite-3b-code-instruct": "IBM Granite 3B Code - Fast coding assistance", 
                "granite-8b-code-instruct": "IBM Granite 8B Code - Balanced coding",
                "granite-20b-code-instruct": "IBM Granite 20B Code - Advanced coding",
                "granite-34b-code-instruct": "IBM Granite 34B Code - Expert coding"
            },
            "Multilingual": {
                "granite-20b-multilingual": "IBM Granite 20B Multilingual - Multiple languages"
            }
        }

class FallbackLLM:
    """
    Fallback LLM for when Watsonx is not available
    Uses simple rule-based responses
    """
    
    def generate_answer(self, question: str, context: str, source_refs: List[Dict] = None) -> Dict[str, any]:
        """
        Generate a simple answer based on context matching
        """
        if not context.strip():
            return {
                "answer": "I couldn't find relevant information in your uploaded documents to answer this question. Please try rephrasing your question or upload more relevant materials.",
                "sources": [],
                "success": True,
                "context_used": False
            }
        
        # Simple keyword-based answer generation
        answer = f"Based on your uploaded documents, here's what I found:\n\n{context[:800]}..."
        
        if source_refs:
            answer += "\n\nSources:\n"
            for ref in source_refs[:3]:
                answer += f"- {ref['filename']}, Page {ref['page_num']}\n"
        
        return {
            "answer": answer,
            "sources": source_refs or [],
            "success": True,
            "context_used": True,
            "model": "fallback"
        }
    
    def generate_followup_questions(self, question: str, answer: str, context: str) -> List[str]:
        """
        Generate simple follow-up questions
        """
        return [
            "Can you explain this concept in more detail?",
            "What are some examples related to this topic?",
            "How does this relate to other concepts in the document?"
        ]

class LLMManager:
    """
    Manager class that handles multiple LLM options
    Falls back gracefully when services are unavailable
    """
    
    def __init__(self):
        self.llm = None
        self.llm_type = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the best available LLM option"""
        
        if os.getenv('OPENAI_API_KEY'):
            try:
                self.llm = OpenAILLM()
                self.llm_type = "openai"
                return
            except Exception:
                pass
        
        # Try Ollama second (completely free and local)
        try:
            ollama_llm = OllamaLLM()
            # Test if Ollama is running
            test_response = ollama_llm.generate_answer("test", "test context")
            if test_response.get("success") and "connection failed" not in test_response.get("error", ""):
                self.llm = ollama_llm
                self.llm_type = "ollama"
                return
        except Exception:
            pass
        
        # Try Watsonx if credentials are available
        if os.getenv('WATSONX_API_KEY') and os.getenv('WATSONX_PROJECT_ID'):
            try:
                self.llm = WatsonxLLMIntegration()
                self.llm_type = "watsonx"
                return
            except Exception:
                pass
        
        # Fall back to Hugging Face (free, no API key needed)
        try:
            self.llm = HuggingFaceLLM()
            self.llm_type = "huggingface"
            return
        except Exception:
            pass
        
        # Final fallback
        self.llm = FallbackLLM()
        self.llm_type = "fallback"
    
    def generate_answer(self, question: str, context: str, source_refs: List[Dict] = None) -> Dict[str, any]:
        """Generate answer using the available LLM"""
        result = self.llm.generate_answer(question, context, source_refs)
        result["llm_type"] = self.llm_type
        return result
    
    def generate_followup_questions(self, question: str, answer: str, context: str) -> List[str]:
        """Generate follow-up questions"""
        return self.llm.generate_followup_questions(question, answer, context)
    
    def get_llm_info(self) -> Dict[str, str]:
        """Get information about the current LLM"""
        info = {
            "openai": "OpenAI ChatGPT (Recommended - High Quality)",
            "ollama": "Local Ollama (Free, Private, Offline)",
            "watsonx": "IBM Watsonx (Paid, Enterprise)",
            "huggingface": "Hugging Face (Free, Online)",
            "fallback": "Simple Context-Based (Free, Basic)"
        }
        
        return {
            "type": self.llm_type,
            "description": info.get(self.llm_type, "Unknown"),
            "requires_api_key": self.llm_type in ["openai", "watsonx"]
        }
