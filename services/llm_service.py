"""
LLM service for text generation using HuggingFace transformers.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict, Any, Optional
import logging
import warnings

# Suppress warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for text generation using HuggingFace transformers models.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", use_gpu: bool = False):
        """
        Initialize the LLM service.
        
        Args:
            model_name: Name of the HuggingFace model to use
            use_gpu: Whether to use GPU if available
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the language model and tokenizer."""
        try:
            logger.info(f"Loading LLM model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map="auto" if self.use_gpu else None,
                low_cpu_mem_usage=True
            )
            
            if not self.use_gpu:
                self.model = self.model.to(self.device)
            
            # Create pipeline for easier text generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if not self.use_gpu else None,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32
            )
            
            logger.info("LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {str(e)}")
            raise
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: The input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated text response
        """
        try:
            if not self.pipeline:
                raise RuntimeError("LLM model not loaded")
            
            logger.debug(f"Generating response for prompt length: {len(prompt)}")
            
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_length=min(max_length, 1024),  # Limit to prevent memory issues
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
                num_return_sequences=1
            )
            
            if outputs and len(outputs) > 0:
                response = outputs[0]["generated_text"].strip()
                logger.debug(f"Generated response length: {len(response)}")
                return response
            else:
                logger.warning("No response generated")
                return "I apologize, but I couldn't generate a response."
                
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {"model_name": self.model_name, "loaded": False}
        
        return {
            "model_name": self.model_name,
            "loaded": True,
            "device": self.device,
            "use_gpu": self.use_gpu,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 'Unknown')
        }


class SimpleQALLM:
    """
    Simple Question-Answering LLM that works without heavy models.
    Uses a lightweight approach for demonstration purposes.
    """
    
    def __init__(self):
        """Initialize the simple QA system."""
        self.model_name = "Simple Rule-based QA"
        logger.info("Initialized Simple QA LLM")
    
    def generate_response(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        """
        Generate a simple response based on rules and context.
        
        Args:
            prompt: The input prompt containing context and question
            max_length: Maximum length (ignored in this simple version)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Generated response
        """
        try:
            # Extract context and question from prompt
            lines = prompt.split('\n')
            context_lines = []
            question = ""
            
            in_context = False
            for line in lines:
                line = line.strip()
                if line.startswith("Context:"):
                    in_context = True
                    continue
                elif line.startswith("Question:"):
                    in_context = False
                    question = line.replace("Question:", "").strip()
                elif line.startswith("Answer:"):
                    break
                elif in_context and line:
                    context_lines.append(line)
            
            context = " ".join(context_lines)
            
            # Simple keyword-based response generation
            if not context.strip():
                return "The answer is not available on the provided website."
            
            # Convert question and context to lowercase for matching
            question_lower = question.lower()
            context_lower = context.lower()
            
            # Extract key terms from the question
            question_words = set(question_lower.split())
            
            # Remove common words
            stop_words = {'what', 'is', 'are', 'how', 'why', 'when', 'where', 'who', 
                         'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                         'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those'}
            
            key_words = question_words - stop_words
            
            # Find relevant sentences in context
            sentences = [s.strip() for s in context.split('.') if s.strip()]
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                # Check if sentence contains key words from question
                if any(word in sentence_lower for word in key_words):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                # Return the most relevant sentences (up to 3)
                response = '. '.join(relevant_sentences[:3])
                if not response.endswith('.'):
                    response += '.'
                return response
            else:
                # Fallback: return first few sentences of context
                if len(sentences) > 0:
                    response = '. '.join(sentences[:2])
                    if not response.endswith('.'):
                        response += '.'
                    return response
                else:
                    return "The answer is not available on the provided website."
                    
        except Exception as e:
            logger.error(f"Error in simple QA generation: {str(e)}")
            return "The answer is not available on the provided website."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "loaded": True,
            "type": "rule-based",
            "description": "Simple rule-based QA system for demonstration"
        }
