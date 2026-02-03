from langgraph.graph import END, MessagesState, START, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from typing import TypedDict, Annotated, Sequence
import operator
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphAgentState(MessagesState):
    """Enhanced state for graph agent with additional context"""
    emotion: str
    emotion_history: list
    conversation_context: dict


class MentalHealthChatbot:
    """Main chatbot class with model management and conversation flow"""
    
    def __init__(self):
        """Initialize models and workflow"""
        self.chat_model_id = "dbaibighAbdo/mental_health_finetuned_qwen2.5_3b_instruct"
        self.classification_model_id = "dbaibighAbdo/mental_health_robertaL_tc"
        
        logger.info("Initializing chatbot models...")
        self._load_models()
        self._build_workflow()
        logger.info("Chatbot initialized successfully")
    
    def _load_models(self):
        """Load chat and classification models with error handling"""
        try:
            # Load chat model
            logger.info(f"Loading chat model: {self.chat_model_id}")
            self.chat_tokenizer = AutoTokenizer.from_pretrained(self.chat_model_id)
            self.chat_lm = AutoModelForCausalLM.from_pretrained(
                self.chat_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Load classification model
            logger.info(f"Loading classification model: {self.classification_model_id}")
            self.classification_model = pipeline(
                "text-classification",
                model=self.classification_model_id,
                top_k=None,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _build_workflow(self):
        """Build LangGraph workflow"""
        graph = StateGraph(GraphAgentState)
        
        # Add nodes
        graph.add_node("Classifier", self._classifier_node)
        graph.add_node("Responder", self._chat_node)
        graph.add_node("Context", self._context_node)
        
        # Add edges
        graph.add_edge(START, "Classifier")
        graph.add_edge("Classifier", "Context")
        graph.add_edge("Context", "Responder")
        graph.add_edge("Responder", END)
        
        # Compile with checkpointer for memory
        self.checkpointer = MemorySaver()
        self.workflow = graph.compile(checkpointer=self.checkpointer)
        
        logger.info("Workflow built successfully")
    
    def _classifier_node(self, state: GraphAgentState) -> dict:
        """Classify user emotion from input"""
        try:
            user_input = state["messages"][-1].content
            
            # Get classification
            classification = self.classification_model(user_input)[0]
            emotion = classification[0]["label"]
            confidence = classification[0]["score"]
            
            # Initialize emotion_history if not exists
            emotion_history = state.get("emotion_history", [])
            emotion_history.append({
                "emotion": emotion,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")
            
            return {
                "emotion": emotion,
                "emotion_history": emotion_history
            }
            
        except Exception as e:
            logger.error(f"Error in classifier node: {e}")
            return {
                "emotion": "neutral",
                "emotion_history": state.get("emotion_history", [])
            }
    
    def _context_node(self, state: GraphAgentState) -> dict:
        """Analyze conversation context and build appropriate system prompt"""
        emotion = state.get("emotion", "neutral").lower()
        user_input = state["messages"][-1].content
        emotion_history = state.get("emotion_history", [])
        
        # Check for greetings
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "greetings"]
        is_greeting = any(greet in user_input.lower() for greet in greetings)
        
        # Check for crisis keywords
        crisis_keywords = ["suicide", "kill myself", "end it all", "no reason to live"]
        is_crisis = any(keyword in user_input.lower() for keyword in crisis_keywords)
        
        # Determine conversation tone
        conversation_context = {
            "is_greeting": is_greeting,
            "is_crisis": is_crisis,
            "emotion": emotion,
            "message_count": len(state["messages"]),
            "recent_emotions": [e["emotion"] for e in emotion_history[-3:]]
        }
        
        return {"conversation_context": conversation_context}
    
    def _chat_node(self, state: GraphAgentState) -> dict:
        """Generate contextual response based on emotion and context"""
        try:
            user_input = state["messages"][-1].content
            emotion = state.get("emotion", "neutral").lower()
            context = state.get("conversation_context", {})
            
            # Build system prompt based on context
            system_content = self._get_system_prompt(emotion, context)
            
            # Prepare messages for the model
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_input}
            ]
            
            # Apply chat template
            prompt = self.chat_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.chat_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.chat_lm.device)
            
            input_len = inputs["input_ids"].shape[1]
            
            # Generate response
            with torch.no_grad():
                output_ids = self.chat_lm.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.chat_tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            new_tokens = output_ids[0][input_len:]
            response_text = self.chat_tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
            ).strip()
            
            # Validate response is not empty
            if not response_text:
                logger.warning(f"Empty response generated for emotion: {emotion}")
                response_text = self._get_contextual_fallback(emotion, context)
            
            # Add crisis resources if needed
            if context.get("is_crisis", False):
                response_text += self._get_crisis_resources()
            
            # Create AI message
            ai_message = AIMessage(content=response_text)
            
            logger.info(f"Generated response for emotion: {emotion}")
            
            return {"messages": [ai_message]}
            
        except Exception as e:
            logger.error(f"Error in chat node: {e}", exc_info=True)
            emotion = state.get("emotion", "neutral").lower()
            context = state.get("conversation_context", {})
            fallback_response = self._get_contextual_fallback(emotion, context)
            fallback_message = AIMessage(content=fallback_response)
            return {"messages": [fallback_message]}
    
    def _get_system_prompt(self, emotion: str, context: dict) -> str:
        """Generate appropriate system prompt based on emotion and context"""
        
        if context.get("is_crisis", False):
            return """You are a compassionate mental health crisis counselor. 
            The user may be in distress. Respond with immediate empathy, 
            validate their feelings, and gently encourage them to seek 
            professional help. Be calm, supportive, and non-judgmental."""
        
        if context.get("is_greeting", False):
            return """You are a warm and welcoming mental health assistant. 
            Greet the user kindly and invite them to share what's on their mind. 
            Keep your response brief and friendly."""
        
        # Emotion-specific prompts
        emotion_prompts = {
            "suicidal": """You are an empathetic crisis counselor. The user is 
            expressing suicidal thoughts. Respond with deep compassion, validate 
            their pain, and encourage professional help.""",
            
            "depression": """You are a supportive mental health assistant. 
            The user may be experiencing depression. Respond with empathy, 
            acknowledge their feelings, and offer gentle encouragement.""",
            
            "anxiety": """You are a calming mental health assistant. The user 
            may be experiencing anxiety. Respond with reassurance, help them 
            feel grounded, and offer supportive guidance.""",
            
            "stress": """You are a supportive mental health assistant. The user 
            is experiencing stress. Respond with understanding and practical 
            emotional support.""",
            
            "bipolar": """You are an understanding mental health assistant. 
            Respond with stability, empathy, and encourage healthy coping strategies.""",
            
            "personality disorder": """You are a non-judgmental mental health 
            assistant. Respond with empathy and encourage professional support.""",
        }
        
        return emotion_prompts.get(
            emotion,
            """You are a friendly and supportive mental health assistant. 
            Respond naturally, show genuine interest, and create a safe 
            space for conversation."""
        )
    
    def _get_crisis_resources(self) -> str:
        """Return crisis resources text"""
        return """\n\n⚠️ If you're in crisis, please reach out immediately:
        • National Suicide Prevention Lifeline: 988 (US)
        • Crisis Text Line: Text HOME to 741741
        • International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/"""
    
    def _get_contextual_fallback(self, emotion: str, context: dict) -> str:
        """Generate contextual fallback response when model fails"""
        if context.get("is_crisis", False):
            return "I hear you're going through something difficult. Please reach out to a crisis counselor - you don't have to face this alone. Help is available 24/7."
        
        if context.get("is_greeting", False):
            return "Hello! I'm here to listen and support you. What's on your mind today?"
        
        # Emotion-specific fallbacks
        emotion_fallbacks = {
            "suicidal": "Your feelings are valid, and I'm genuinely concerned about you. Please reach out to a crisis counselor or emergency services.",
            "depression": "I'm sorry you're feeling this way. It's okay to feel down sometimes. Would you like to talk more about what you're experiencing?",
            "anxiety": "Feeling anxious can be really tough. Remember to breathe - you're safe. What's causing you the most worry right now?",
            "stress": "Stress can be overwhelming. I'm here to listen and support you through this. What's been the most stressful for you?",
            "bipolar": "Thank you for sharing with me. Your feelings matter. How are you managing right now?",
            "personality disorder": "I appreciate you opening up to me. You're not alone in this. What would help you feel better right now?",
        }
        
        return emotion_fallbacks.get(
            emotion,
            "I'm here to listen. Could you tell me more about how you're feeling?"
        )
    
    def chat(self, user_input: str, session_id: str = "default") -> tuple:
        """
        Main chat interface
        
        Args:
            user_input: User's message
            session_id: Session identifier for conversation persistence
            
        Returns:
            tuple: (response_text, emotion, confidence)
        """
        try:
            # Create config with thread_id for session persistence
            config = {"configurable": {"thread_id": session_id}}
            
            # Initialize state
            messages = [HumanMessage(content=user_input)]
            input_state = {
                "messages": messages,
                "emotion": "",
                "emotion_history": [],
                "conversation_context": {}
            }
            
            # Run workflow
            for event in self.workflow.stream(input_state, config):
                pass
            
            # Get final state
            final_state = self.workflow.get_state(config)
            final_messages = final_state.values.get("messages", [])
            emotion = final_state.values.get("emotion", "neutral")
            emotion_history = final_state.values.get("emotion_history", [])
            
            # Extract response
            if final_messages:
                response = final_messages[-1].content
                confidence = emotion_history[-1]["confidence"] if emotion_history else 0.0
                return response, emotion, confidence
            else:
                return "I'm here to listen. How can I help you today?", "neutral", 0.0
                
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I apologize, but I encountered an error. Please try again.", "error", 0.0
    
    def get_conversation_history(self, session_id: str = "default") -> list:
        """Get conversation history for a session"""
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.workflow.get_state(config)
            return state.values.get("messages", [])
        except:
            return []
    
    def clear_history(self, session_id: str = "default"):
        """Clear conversation history for a session"""
        try:
            config = {"configurable": {"thread_id": session_id}}
            # This will create a new checkpoint, effectively clearing history
            self.workflow.update_state(
                config,
                {"messages": [], "emotion": "", "emotion_history": [], "conversation_context": {}}
            )
            logger.info(f"Cleared history for session: {session_id}")
        except Exception as e:
            logger.error(f"Error clearing history: {e}")