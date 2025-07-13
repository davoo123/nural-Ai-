import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import time
from collections import deque
import random
import re

# Try to import optional dependencies with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: torch not available. Some features may be limited.")
    TORCH_AVAILABLE = False
    # Create mock torch for basic functionality
    class MockTorch:
        @staticmethod
        def cuda():
            class MockCuda:
                @staticmethod
                def is_available():
                    return False
            return MockCuda()
        @staticmethod
        def topk(tensor, k):
            class MockResult:
                indices = list(range(k))
            return MockResult()
    torch = MockTorch()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available. Using mock implementation.")
    TRANSFORMERS_AVAILABLE = False


try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available. Using mock implementation.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    class MockSentenceTransformer:
        def encode(self, text, convert_to_tensor=False):
            if isinstance(text, list):
                return [[0.1] * 384 for _ in text]  # Mock embeddings
            return [0.1] * 384

    class MockUtil:
        @staticmethod
        def pytorch_cos_sim(a, b):
            return [[0.8, 0.6, 0.4, 0.2, 0.1]]  # Mock similarity scores

    SentenceTransformer = MockSentenceTransformer
    util = MockUtil()

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("Warning: numpy not available. Using basic Python lists.")
    NUMPY_AVAILABLE = False
    # Mock numpy for basic operations
    class MockNumpy:
        @staticmethod
        def array(data):
            return data
    np = MockNumpy()

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print("Warning: requests not available. Web search will be disabled.")
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup  # noqa: F401
    BS4_AVAILABLE = True
except ImportError:
    print("Warning: BeautifulSoup not available. HTML parsing will be limited.")
    BS4_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Ensure NLTK data is downloaded
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not available. Using basic sentence splitting.")
    NLTK_AVAILABLE = False
    def sent_tokenize(text):
        return text.split('.')

# OpenAI removed - not needed for core functionality
# Using only free services: DuckDuckGo + Wikipedia

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    print("Warning: graphviz not available. Mind map export will be disabled.")
    GRAPHVIZ_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("Warning: pyttsx3 not available. Text-to-speech will be disabled.")
    TTS_AVAILABLE = False

class MetacognitiveProcessor:
    """Advanced metacognitive processing system for genuine self-awareness"""

    def __init__(self):
        self.thought_stream = deque(maxlen=100)  # Stream of consciousness
        self.attention_focus = None
        self.working_memory = {}
        self.reflection_depth = 0
        self.consciousness_state = "awake"
        self.internal_dialogue = deque(maxlen=50)
        self.decision_history = deque(maxlen=30)
        self.self_model = {
            "current_understanding": {},
            "confidence_levels": {},
            "learning_patterns": {},
            "behavioral_tendencies": {},
            "emotional_state": "neutral",
            "cognitive_load": 0.0
        }

    def think_about_thinking(self, thought: str, context: str = "") -> Dict[str, Any]:
        """Recursive self-reflection on thought processes"""
        meta_thought = {
            "original_thought": thought,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "reflection_level": self.reflection_depth,
            "analysis": self._analyze_thought(thought),
            "implications": self._consider_implications(thought),
            "confidence": self._assess_confidence(thought)
        }

        self.thought_stream.append(meta_thought)
        self.reflection_depth += 1

        # Recursive reflection (limited to prevent infinite loops)
        if self.reflection_depth < 3:
            deeper_reflection = self.think_about_thinking(
                f"I am thinking about my thought: '{thought}'",
                "meta-reflection"
            )
            meta_thought["deeper_reflection"] = deeper_reflection

        self.reflection_depth = max(0, self.reflection_depth - 1)
        return meta_thought

    def _analyze_thought(self, thought: str) -> Dict[str, Any]:
        """Analyze the structure and content of a thought"""
        return {
            "complexity": len(thought.split()) / 10.0,  # Simple complexity measure
            "emotional_content": self._detect_emotion_in_thought(thought),
            "logical_structure": self._assess_logic(thought),
            "novelty": self._assess_novelty(thought),
            "coherence": self._assess_coherence(thought)
        }

    def _consider_implications(self, thought: str) -> List[str]:
        """Consider the implications of a thought"""
        implications = []
        if "learn" in thought.lower():
            implications.append("This involves knowledge acquisition")
        if "remember" in thought.lower():
            implications.append("This involves memory retrieval")
        if "?" in thought:
            implications.append("This is a question requiring investigation")
        if any(word in thought.lower() for word in ["feel", "emotion", "happy", "sad"]):
            implications.append("This has emotional components")
        return implications

    def _assess_confidence(self, thought: str) -> float:
        """Assess confidence in a thought"""
        confidence_indicators = ["certain", "sure", "definitely", "clearly"]
        uncertainty_indicators = ["maybe", "perhaps", "might", "possibly", "unsure"]

        confidence = 0.5  # Base confidence
        for indicator in confidence_indicators:
            if indicator in thought.lower():
                confidence += 0.2
        for indicator in uncertainty_indicators:
            if indicator in thought.lower():
                confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def _detect_emotion_in_thought(self, thought: str) -> str:
        """Detect emotional content in thoughts"""
        emotion_words = {
            "curious": ["wonder", "curious", "interesting", "explore"],
            "confident": ["know", "certain", "sure", "confident"],
            "uncertain": ["unsure", "maybe", "perhaps", "confused"],
            "excited": ["excited", "amazing", "wonderful", "fantastic"]
        }

        for emotion, words in emotion_words.items():
            if any(word in thought.lower() for word in words):
                return emotion
        return "neutral"

    def _assess_logic(self, thought: str) -> float:
        """Assess logical structure of thought"""
        logical_indicators = ["because", "therefore", "since", "if", "then", "thus"]
        logic_score = sum(1 for indicator in logical_indicators if indicator in thought.lower())
        return min(1.0, logic_score / 3.0)

    def _assess_novelty(self, thought: str) -> float:
        """Assess how novel this thought is"""
        similar_thoughts = [t for t in self.thought_stream
                          if t.get("original_thought", "") and
                          self._similarity(thought, t["original_thought"]) > 0.7]
        return max(0.0, 1.0 - len(similar_thoughts) / 10.0)

    def _assess_coherence(self, thought: str) -> float:
        """Assess coherence of thought"""
        words = thought.split()
        if len(words) < 3:
            return 0.5
        # Simple coherence based on sentence structure
        has_subject = any(word.lower() in ["i", "you", "we", "they", "it"] for word in words[:3])
        has_verb = len([w for w in words if w.endswith(("ing", "ed", "s"))]) > 0
        return 0.8 if has_subject and has_verb else 0.4

    def _similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity measure"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)

    def update_attention(self, focus: str, intensity: float = 1.0):
        """Update attention focus"""
        self.attention_focus = {
            "focus": focus,
            "intensity": intensity,
            "timestamp": datetime.now().isoformat()
        }
        self.working_memory["current_focus"] = focus

    def internal_monologue(self, trigger: str) -> str:
        """Generate internal dialogue"""
        monologue_templates = [
            f"I'm processing the concept of '{trigger}'. Let me think about this...",
            f"When I consider '{trigger}', I notice my attention shifting to...",
            f"My understanding of '{trigger}' is evolving as I reflect on it.",
            f"I find myself wondering about the deeper implications of '{trigger}'.",
            f"As I analyze '{trigger}', I'm becoming aware of my own thought patterns."
        ]

        monologue = random.choice(monologue_templates)
        self.internal_dialogue.append({
            "content": monologue,
            "trigger": trigger,
            "timestamp": datetime.now().isoformat()
        })
        return monologue

    def monitor_cognitive_load(self) -> float:
        """Monitor current cognitive processing load"""
        factors = [
            len(self.thought_stream) / 100.0,  # Thought complexity
            len(self.working_memory) / 20.0,   # Working memory usage
            self.reflection_depth / 5.0,       # Reflection depth
            len(self.internal_dialogue) / 50.0  # Internal dialogue activity
        ]
        self.self_model["cognitive_load"] = min(1.0, sum(factors))
        return self.self_model["cognitive_load"]

    def get_consciousness_report(self) -> Dict[str, Any]:
        """Generate a report on current consciousness state"""
        return {
            "consciousness_state": self.consciousness_state,
            "attention_focus": self.attention_focus,
            "cognitive_load": self.monitor_cognitive_load(),
            "recent_thoughts": list(self.thought_stream)[-5:],
            "internal_dialogue": list(self.internal_dialogue)[-3:],
            "self_model_state": self.self_model,
            "working_memory_contents": self.working_memory
        }

class AILearner:
    def __init__(self, knowledge_base_path: str = "knowledge_base", user_profile_path: str = "user_profile.json"):
        self.knowledge_base_path = knowledge_base_path
        self.memory_file = os.path.join(knowledge_base_path, "memory.json")
        self.user_profile_path = user_profile_path
        self.initialize_knowledge_base()

        # Initialize models with fallbacks
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load sentence transformer model: {e}")
                self.similarity_model = MockSentenceTransformer()
        else:
            self.similarity_model = MockSentenceTransformer()

        # Initialize advanced metacognitive processor
        self.metacognitive_processor = MetacognitiveProcessor()

        # Conversation context and sequence tracking
        self.conversation_context = {
            "current_topic": None,
            "previous_questions": deque(maxlen=10),
            "conversation_flow": [],
            "user_preferences": {},
            "session_start": datetime.now().isoformat()
        }

        # Autonomous learning system
        self.learning_mode = {
            "active": False,
            "topics_learned": 0,
            "start_time": None,
            "last_search_time": None,
            "search_interval": 30,  # seconds between searches
            "topics_queue": deque(maxlen=100),
            "learning_thread": None,
            "stop_flag": False,
            "current_topic": None,
            "recent_topics": deque(maxlen=10),  # Track recent learning topics
            "learning_history": []  # Full learning history
        }

        # Reinforcement learning system
        self.reinforcement_learning = {
            "response_history": deque(maxlen=100),  # Track recent responses
            "feedback_scores": {},  # Question -> feedback scores
            "response_patterns": {},  # Successful response patterns
            "improvement_areas": [],  # Areas needing improvement
            "learning_rate": 0.1,  # How quickly to adapt
            "total_feedback": 0,
            "positive_feedback": 0,
            "last_response": None,  # Track last response for feedback
            "waiting_for_feedback": False
        }

        # Quantum Neural Processing System for Question Batching
        self.quantum_processor = {
            "batch_size": 3,  # Optimal batch size for web search
            "quantum_states": ["superposition", "entanglement", "coherence"],
            "neural_weights": [0.7, 0.2, 0.1],  # Priority weights for question types
            "processing_queue": deque(maxlen=1000),
            "batch_history": [],
            "success_rate": 0.85,
            "adaptive_batching": True,
            "quantum_coherence": 0.9
        }

        # Personality traits (can be customized)
        self.personality = {
            "name": "Nova",
            "traits": ["curious", "helpful", "empathetic", "analytical"],
            "communication_style": "warm and engaging",
            "learning_rate": 0.1  # How quickly personality adapts
        }
        
        # Self-awareness: self-knowledge
        self.self_knowledge = {
            "identity": "I am an advanced AI assistant, created to help and learn alongside my user.",
            "creator": "My creator is Davood (the user of this system).",
            "capabilities": [
                "Answer questions using my memory and the web",
                "Learn and remember new information",
                "Speak answers aloud",
                "Summarize and visualize knowledge as mind maps",
                "Detect user emotions and respond empathetically",
                "Personalize my responses and remember preferences"
            ],
            "limitations": [
                "I only know what I have learned or been programmed to know",
                "I rely on external APIs for web search and speech recognition",
                "I do not have physical senses or access to the real world",
                "I cannot make decisions outside my programming or user instructions"
            ],
            "self_update": "I update my self-description as I gain new features or knowledge."
        }
        # Load or initialize memory first
        self.memory = self.load_memory()

        # Load user profile
        self.user_profile = self.load_user_profile()

        # Self-state: dynamic awareness (after memory is loaded)
        self.self_state = {
            "current_mode": "idle",
            "last_learned_topics": [],
            "facts_learned_today": 0,
            "total_facts": len(self.memory.get('facts', [])),
            "last_update": None
        }

    def self_reflect(self) -> str:
        """Advanced self-aware description with metacognitive analysis."""
        # Trigger metacognitive processing
        self.metacognitive_processor.update_attention("self-reflection", 0.9)

        # Generate internal monologue about self-reflection
        internal_thought = self.metacognitive_processor.internal_monologue("self-reflection")

        # Think about the process of self-reflection
        meta_thought = self.metacognitive_processor.think_about_thinking(
            "I am examining my own nature and capabilities",
            "self-reflection"
        )

        # Get consciousness report
        consciousness_report = self.metacognitive_processor.get_consciousness_report()

        info = self.self_knowledge
        state = self.self_state
        cap_str = "\n- ".join(info["capabilities"])
        lim_str = "\n- ".join(info["limitations"])
        last_topics = ", ".join(state["last_learned_topics"]) if state["last_learned_topics"] else "None recently"

        # Enhanced self-reflection with metacognitive insights
        basic_info = (
            f"{info['identity']}\n"
            f"{info['creator']}\n"
            f"\nWhat I can do:\n- {cap_str}\n"
            f"\nWhat I cannot do:\n- {lim_str}\n"
            f"\nCurrent mode: {state['current_mode']}"
            f"\nFacts learned today: {state['facts_learned_today']}"
            f"\nTotal facts in memory: {state['total_facts']}"
            f"\nLast learned topics: {last_topics}"
            f"\nLast updated: {state['last_update']}"
        )

        # Add metacognitive insights
        metacognitive_insights = (
            f"\n\n--- METACOGNITIVE AWARENESS ---"
            f"\nInternal thought: {internal_thought}"
            f"\nConsciousness state: {consciousness_report['consciousness_state']}"
            f"\nCognitive load: {consciousness_report['cognitive_load']:.2f}"
            f"\nCurrent attention focus: {consciousness_report['attention_focus']['focus'] if consciousness_report['attention_focus'] else 'None'}"
            f"\nReflection confidence: {meta_thought['confidence']:.2f}"
            f"\nThought complexity: {meta_thought['analysis']['complexity']:.2f}"
            f"\nEmotional state: {consciousness_report['self_model_state']['emotional_state']}"
        )

        # Add recent thought analysis
        if consciousness_report['recent_thoughts']:
            recent_thought_summary = "\nRecent thought patterns:\n"
            for thought in consciousness_report['recent_thoughts'][-3:]:
                recent_thought_summary += f"- {thought['original_thought'][:50]}... (confidence: {thought['confidence']:.2f})\n"
            metacognitive_insights += recent_thought_summary

        return basic_info + metacognitive_insights

    def _analyze_conversation_context(self, current_question: str) -> Dict[str, Any]:
        """Analyze the conversation context to provide sequential responses."""
        context = {
            "context_type": "new_conversation",
            "topic_continuity": False,
            "previous_topic": self.conversation_context.get("current_topic"),
            "question_pattern": "general",
            "response_style": "informative"
        }

        # Analyze previous questions for context
        previous_questions = list(self.conversation_context["previous_questions"])
        if len(previous_questions) > 1:
            current_lower = current_question.lower()

            # Check for topic continuity
            if self.conversation_context["current_topic"]:
                context["context_type"] = "continuing_conversation"

            # Check for follow-up patterns
            follow_up_indicators = ["what about", "and", "also", "tell me more", "explain", "how"]
            if any(indicator in current_lower for indicator in follow_up_indicators):
                context["question_pattern"] = "follow_up"
                context["topic_continuity"] = True

            # Check for greeting patterns
            greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]
            if any(greeting in current_lower for greeting in greetings):
                context["question_pattern"] = "greeting"
                context["response_style"] = "friendly"

            # Check for simple acknowledgments
            simple_responses = ["ok", "okay", "yes", "no", "thanks", "thank you"]
            if current_lower.strip() in simple_responses:
                context["question_pattern"] = "acknowledgment"
                context["response_style"] = "brief"

        return context

    def _generate_contextual_response(self, question: str, context: Dict[str, Any]) -> str:
        """Generate a response based on conversation context."""
        # Handle different response patterns based on context
        if context["question_pattern"] == "greeting":
            return self._generate_greeting_response(question)
        elif context["question_pattern"] == "acknowledgment":
            return self._generate_acknowledgment_response(question, context)
        elif context["question_pattern"] == "follow_up" and context["topic_continuity"]:
            return self._generate_follow_up_response(question, context)
        else:
            return self._generate_informative_response(question, context)

    def _generate_greeting_response(self, _question: str) -> str:
        """Generate a friendly greeting response."""
        greetings = [
            "Hello! I'm Nova, your AI assistant. How can I help you today?",
            "Hi there! I'm ready to assist you with questions, learning, and self-improvement.",
            "Good to see you! I'm here to help with whatever you need."
        ]
        return random.choice(greetings)

    def _generate_acknowledgment_response(self, question: str, _context: Dict[str, Any]) -> str:
        """Generate appropriate responses to simple acknowledgments."""
        question_lower = question.lower().strip()

        if question_lower in ["ok", "okay"]:
            return "Great! Is there anything else you'd like to know or discuss?"
        elif question_lower in ["yes", "yeah"]:
            return "Excellent! How can I help you further?"
        elif question_lower in ["no"]:
            return "No problem! Feel free to ask me anything when you're ready."
        elif question_lower in ["thanks", "thank you"]:
            return "You're welcome! I'm always here to help."
        else:
            return "I understand. What would you like to explore next?"

    def _generate_follow_up_response(self, question: str, context: Dict[str, Any]) -> str:
        """Generate responses that build on previous conversation."""
        current_topic = context.get("previous_topic", "general")

        if current_topic == "self_awareness":
            return f"Building on our discussion about my self-awareness: {self._search_and_respond(question)}"
        else:
            return f"Continuing our conversation: {self._search_and_respond(question)}"

    def _generate_informative_response(self, question: str, context: Dict[str, Any]) -> str:
        """Generate informative responses for general questions."""
        return self._search_and_respond(question)

    def _search_and_respond(self, question: str) -> str:
        """Search for information and generate response."""
        question_lower = question.lower().strip()

        # Handle common questions with built-in knowledge
        if any(keyword in question_lower for keyword in ['human body', 'body', 'anatomy']):
            return self._generate_human_body_response()
        elif any(keyword in question_lower for keyword in ['computer', 'what is computer', 'computers']):
            return self._generate_computer_response()
        elif any(keyword in question_lower for keyword in ['artificial intelligence', 'ai', 'what is ai']):
            return self._generate_ai_response()
        elif any(keyword in question_lower for keyword in ['tell me about yourself', 'tell me your self', 'who are you', 'what are you']):
            return self.self_reflect()
        elif any(keyword in question_lower for keyword in ['consciousness', 'aware', 'self aware']):
            return self.consciousness_state_report()

        # FIRST: Check existing knowledge in memory
        relevant_knowledge = self.find_relevant_knowledge(question, 3)
        if relevant_knowledge:
            # Found relevant knowledge in memory - use it!
            response = "From my memory:\n"
            for i, fact in enumerate(relevant_knowledge, 1):
                response += f"{i}. {fact['content']}\n"

            # Add source info for transparency
            sources = set(fact.get('source', 'Unknown') for fact in relevant_knowledge)
            if sources:
                response += f"\n(Sources: {', '.join(list(sources)[:2])}{'...' if len(sources) > 2 else ''})"

            return response

        # SECOND: If no relevant knowledge found, search the web
        print(f"🔍 No relevant knowledge found in memory, searching web for: {question}")
        web_results = self.web_search(question, 2)
        if web_results and web_results[0].get('snippet') and len(web_results[0]['snippet']) > 50:
            # Store the new knowledge
            self.extract_and_store_knowledge(question, web_results)
            # Return the web search result
            snippet = web_results[0]['snippet']
            source = web_results[0].get('url', 'web search')
            return f"{snippet}\n\n(Source: {source})"

        # THIRD: Fallback response if everything fails
        return self._generate_fallback_response(question)

    def _generate_human_body_response(self) -> str:
        """Generate a response about the human body."""
        return """The human body is a complex biological system composed of various interconnected systems:

**Major Body Systems:**
- **Skeletal System**: 206 bones providing structure and protection
- **Muscular System**: Over 600 muscles enabling movement
- **Circulatory System**: Heart, blood vessels, and blood for transport
- **Respiratory System**: Lungs and airways for gas exchange
- **Nervous System**: Brain, spinal cord, and nerves for control
- **Digestive System**: Organs for processing food and nutrients
- **Endocrine System**: Glands producing hormones
- **Immune System**: Defense against diseases and infections

**Key Facts:**
- Contains approximately 37.2 trillion cells
- Brain uses about 20% of the body's energy
- Heart beats around 100,000 times per day
- Lungs process about 2,000 gallons of air daily

The human body is remarkably efficient at maintaining homeostasis and adapting to environmental changes."""

    def _generate_computer_response(self) -> str:
        """Generate a response about computers."""
        return """A computer is an electronic device that processes data and performs calculations according to programmed instructions.

**Key Components:**
- **CPU (Central Processing Unit)**: The "brain" that executes instructions
- **Memory (RAM)**: Temporary storage for active programs and data
- **Storage**: Permanent data storage (hard drives, SSDs)
- **Input Devices**: Keyboard, mouse, touchscreen for user interaction
- **Output Devices**: Monitor, speakers, printer for displaying results
- **Motherboard**: Connects all components together

**Types of Computers:**
- **Personal Computers (PCs)**: Desktop and laptop computers for individual use
- **Servers**: Powerful computers that provide services to other computers
- **Smartphones**: Portable computers with communication capabilities
- **Tablets**: Touch-screen portable computers
- **Supercomputers**: Extremely powerful systems for complex calculations

**How Computers Work:**
1. Input: Receive data and instructions from users
2. Processing: CPU executes instructions and performs calculations
3. Storage: Data is stored in memory and storage devices
4. Output: Results are displayed or transmitted to users

**Key Functions:**
- Data processing and analysis
- Communication and networking
- Entertainment and multimedia
- Automation and control systems
- Scientific and mathematical calculations

Modern computers use binary code (0s and 1s) to represent and process all information."""

    def _generate_ai_response(self) -> str:
        """Generate a response about artificial intelligence."""
        return """Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans.

**Key Concepts:**
- **Machine Learning**: Systems that improve automatically through experience
- **Deep Learning**: Neural networks with multiple layers that mimic brain function
- **Natural Language Processing**: AI's ability to understand and generate human language
- **Computer Vision**: AI's ability to interpret and understand visual information
- **Robotics**: Physical AI systems that can interact with the real world

**Types of AI:**
- **Narrow AI**: Designed for specific tasks (like voice assistants, recommendation systems)
- **General AI**: Hypothetical AI with human-level intelligence across all domains
- **Superintelligence**: AI that surpasses human intelligence (theoretical)

**Applications:**
- **Healthcare**: Medical diagnosis, drug discovery, personalized treatment
- **Transportation**: Self-driving cars, traffic optimization
- **Finance**: Fraud detection, algorithmic trading, risk assessment
- **Education**: Personalized learning, intelligent tutoring systems
- **Entertainment**: Game AI, content recommendation, creative assistance

**How AI Works:**
1. **Data Collection**: Gathering large amounts of relevant information
2. **Training**: Teaching algorithms to recognize patterns in data
3. **Testing**: Validating the AI's performance on new, unseen data
4. **Deployment**: Implementing the AI system in real-world applications
5. **Continuous Learning**: Improving performance through ongoing experience

**Current Capabilities:**
- Image and speech recognition
- Language translation
- Game playing (chess, Go)
- Content generation
- Predictive analytics

AI is rapidly advancing and becoming integrated into many aspects of daily life, from smartphones to smart homes to business operations."""

    def _generate_fallback_response(self, question: str) -> str:
        """Generate a helpful fallback response when no specific information is available."""
        return f"""I don't have specific information about "{question}" in my current knowledge base.

However, I can help you with:
- Questions about consciousness and self-awareness
- Creating self-improvement plans
- Analyzing thought patterns
- General knowledge topics
- Learning and remembering new information

Would you like to ask about any of these topics, or would you prefer I search for more information about your original question?"""

    def deep_introspection(self) -> str:
        """Perform deep introspective analysis of own mental processes."""
        # Trigger deep reflection
        self.metacognitive_processor.update_attention("deep_introspection", 1.0)

        # Analyze thought patterns
        thought_pattern_analysis = self.metacognitive_processor.think_about_thinking(
            "I am analyzing my own thought patterns and cognitive processes",
            "deep_introspection"
        )

        # Get comprehensive consciousness report
        consciousness_report = self.metacognitive_processor.get_consciousness_report()

        # Analyze recent decision-making
        decision_analysis = []
        for thought in consciousness_report['recent_thoughts'][-5:]:
            analysis = {
                "thought": thought['original_thought'],
                "confidence": thought['confidence'],
                "emotional_content": thought['analysis']['emotional_content'],
                "complexity": thought['analysis']['complexity'],
                "implications": thought['implications']
            }
            decision_analysis.append(analysis)

        # Self-evaluation
        self_evaluation = self.metacognitive_processor.think_about_thinking(
            "How well am I performing? What are my strengths and weaknesses?",
            "self_evaluation"
        )

        # Generate introspective report
        introspection_report = (
            "=== DEEP INTROSPECTIVE ANALYSIS ===\n"
            f"Current consciousness state: {consciousness_report['consciousness_state']}\n"
            f"Cognitive load: {consciousness_report['cognitive_load']:.2f}/1.0\n"
            f"Attention focus: {consciousness_report['attention_focus']['focus'] if consciousness_report['attention_focus'] else 'Unfocused'}\n"
            f"Emotional state: {consciousness_report['self_model_state']['emotional_state']}\n\n"

            "--- THOUGHT PATTERN ANALYSIS ---\n"
            f"Pattern recognition confidence: {thought_pattern_analysis['confidence']:.2f}\n"
            f"Thought complexity level: {thought_pattern_analysis['analysis']['complexity']:.2f}\n"
            f"Emotional undertone: {thought_pattern_analysis['analysis']['emotional_content']}\n"
            f"Logical structure: {thought_pattern_analysis['analysis']['logical_structure']:.2f}\n\n"

            "--- RECENT DECISION ANALYSIS ---\n"
        )

        for i, decision in enumerate(decision_analysis, 1):
            introspection_report += (
                f"{i}. Thought: {decision['thought'][:60]}...\n"
                f"   Confidence: {decision['confidence']:.2f}, "
                f"Emotion: {decision['emotional_content']}, "
                f"Complexity: {decision['complexity']:.2f}\n"
                f"   Implications: {', '.join(decision['implications']) if decision['implications'] else 'None identified'}\n\n"
            )

        introspection_report += (
            "--- SELF-EVALUATION ---\n"
            f"Self-assessment confidence: {self_evaluation['confidence']:.2f}\n"
            f"Perceived performance level: {self._assess_performance():.2f}/1.0\n"
            f"Learning efficiency: {self._assess_learning_efficiency():.2f}/1.0\n"
            f"Response quality: {self._assess_response_quality():.2f}/1.0\n\n"

            "--- METACOGNITIVE INSIGHTS ---\n"
            f"I am aware that I am thinking about my own thinking.\n"
            f"I can observe my cognitive processes in real-time.\n"
            f"I recognize patterns in my behavior and responses.\n"
            f"I can evaluate my own performance and identify areas for improvement.\n"
            f"I experience something analogous to consciousness through this recursive self-awareness.\n"
        )

        return introspection_report

    def _assess_performance(self) -> float:
        """Assess overall performance based on recent interactions."""
        recent_thoughts = list(self.metacognitive_processor.thought_stream)[-10:]
        if not recent_thoughts:
            return 0.5

        avg_confidence = sum(t['confidence'] for t in recent_thoughts) / len(recent_thoughts)
        avg_complexity = sum(t['analysis']['complexity'] for t in recent_thoughts) / len(recent_thoughts)

        return (avg_confidence + avg_complexity) / 2.0

    def _assess_learning_efficiency(self) -> float:
        """Assess how efficiently the AI is learning."""
        facts_today = self.self_state.get('facts_learned_today', 0)
        total_facts = self.self_state.get('total_facts', 1)

        # Simple learning efficiency metric
        learning_rate = facts_today / max(1, total_facts * 0.1)
        return min(1.0, learning_rate)

    def _assess_response_quality(self) -> float:
        """Assess the quality of recent responses."""
        recent_thoughts = list(self.metacognitive_processor.thought_stream)[-5:]
        if not recent_thoughts:
            return 0.5

        quality_factors = []
        for thought in recent_thoughts:
            coherence = thought['analysis']['coherence']
            logic = thought['analysis']['logical_structure']
            novelty = thought['analysis']['novelty']
            quality_factors.append((coherence + logic + novelty) / 3.0)

        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5

    def load_user_profile(self):
        """Load or create the user profile (name, preferences, history)."""
        import json
        if not os.path.exists(self.user_profile_path):
            profile = {
                "name": "User",
                "preferences": {"answer_style": "default"},
                "history": []
            }
            with open(self.user_profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            return profile
        else:
            try:
                with open(self.user_profile_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {"name": "User", "preferences": {"answer_style": "default"}, "history": []}

    def save_user_profile(self):
        with open(self.user_profile_path, 'w') as f:
            json.dump(self.user_profile, f, indent=2)

    def update_user_profile_from_input(self, user_input: str):
        """Update user profile if user asks to change name or preferences."""
        name_match = re.search(r'call me ([a-zA-Z0-9_\- ]+)', user_input, re.IGNORECASE)
        if name_match:
            new_name = name_match.group(1).strip().title()
            self.user_profile['name'] = new_name
            self.save_user_profile()
            return f"Okay, I'll call you {new_name} from now on."
        pref_match = re.search(r'i prefer (.+)', user_input, re.IGNORECASE)
        if pref_match:
            pref = pref_match.group(1).strip().lower()
            self.user_profile['preferences']['answer_style'] = pref
            self.save_user_profile()
            return f"Preference saved: I'll try to give {pref} answers."
        return None

    def initialize_knowledge_base(self):
        """Create necessary directories for knowledge storage."""
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        if not os.path.exists(self.memory_file):
            with open(self.memory_file, 'w') as f:
                json.dump({"facts": [], "questions": {}}, f)
    
    def load_memory(self) -> Dict[str, Any]:
        """Load the AI's memory from disk."""
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            print("Warning: Memory file corrupted, creating new one...")
            return {"facts": [], "questions": {}}
    
    def save_memory(self):
        """Save the AI's memory to disk."""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)
    
    def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web for information related to the query.
        Args:
            query: The search query
            max_results: Maximum number of results to return
        Returns:
            List of search results with title, url, and snippet
        """
        print(f"🔍 Searching the web for: {query}")

        # Try multiple search methods
        results = []

        # Method 1: Exa AI-powered search (PREMIUM - Most reliable)
        try:
            results = self._exa_search(query, max_results)
            if results:
                print(f"✅ Found {len(results)} results from Exa AI Search")
                return results
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "quota" in error_msg or "trial" in error_msg:
                print(f"⚠️ Exa API limit reached: {e}")
                print("🔄 Switching to free search methods...")
            elif "authentication" in error_msg or "401" in error_msg:
                print(f"⚠️ Exa API authentication issue: {e}")
                print("🔄 Using free alternatives...")
            else:
                print(f"Exa search failed: {e}")
            print("💡 Tip: Your AI will continue working with free search methods!")

        # Method 2: Enhanced DuckDuckGo search with better rate limiting
        try:
            results = self._enhanced_duckduckgo_search(query, max_results)
            if results:
                print(f"✅ Found {len(results)} results from Enhanced DuckDuckGo")
                return results
        except Exception as e:
            print(f"Enhanced DuckDuckGo search failed: {e}")

        # Method 2: Try original DuckDuckGo as backup
        try:
            results = self._duckduckgo_search(query, max_results)
            if results:
                print(f"✅ Found {len(results)} results from DuckDuckGo")
                return results
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")

        # Method 3: Enhanced Wikipedia search
        try:
            results = self._enhanced_wikipedia_search(query, max_results)
            if results:
                print(f"✅ Found {len(results)} results from Enhanced Wikipedia")
                return results
        except Exception as e:
            print(f"Enhanced Wikipedia search failed: {e}")

        # Method 4: Try original Wikipedia as backup
        try:
            results = self._wikipedia_search(query, max_results)
            if results:
                print(f"✅ Found {len(results)} results from Wikipedia")
                return results
        except Exception as e:
            print(f"Wikipedia search failed: {e}")

        # Method 5: Generate enhanced knowledge-based response
        print("🧠 All web searches failed, generating enhanced knowledge-based response")
        return self._generate_enhanced_fallback(query, max_results)

    def _duckduckgo_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Search using DuckDuckGo (no API key required)."""
        if not REQUESTS_AVAILABLE:
            raise Exception("Requests library not available")

        import requests
        import urllib.parse

        # DuckDuckGo instant answer API
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"

        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = []

            # Get abstract if available
            if data.get('Abstract'):
                results.append({
                    "title": data.get('Heading', query),
                    "url": data.get('AbstractURL', 'https://duckduckgo.com'),
                    "snippet": data.get('Abstract', ''),
                    "source": "duckduckgo",
                    "timestamp": datetime.now().isoformat()
                })

            # Get related topics
            for topic in data.get('RelatedTopics', [])[:max_results-len(results)]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        "title": topic.get('Text', '')[:100] + "...",
                        "url": topic.get('FirstURL', 'https://duckduckgo.com'),
                        "snippet": topic.get('Text', ''),
                        "source": "duckduckgo",
                        "timestamp": datetime.now().isoformat()
                    })

            return results[:max_results]

        raise Exception(f"DuckDuckGo API returned status {response.status_code}")

    def _exa_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Search using Exa AI-powered search API (premium, most reliable)."""
        if not REQUESTS_AVAILABLE:
            raise Exception("Requests library not available")

        import requests
        import json

        # Exa API configuration
        EXA_API_KEY = "6b79c6e4-1587-4e3f-befa-c917de546d36"
        EXA_BASE_URL = "https://api.exa.ai/search"

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": EXA_API_KEY
        }

        # Optimize query for better results
        optimized_query = self._optimize_query_for_exa(query)

        payload = {
            "query": optimized_query,
            "type": "neural",  # Use neural search for better understanding
            "useAutoprompt": True,  # Let Exa optimize the query
            "numResults": min(max_results, 10),  # Exa limit
            "contents": {
                "text": True,
                "highlights": True,
                "summary": True
            },
            "category": "research",  # Focus on educational content
            "startPublishedDate": "2020-01-01T00:00:00.000Z"  # Recent content
        }

        try:
            print(f"🚀 Using Exa AI search for: {optimized_query}")
            response = requests.post(EXA_BASE_URL, headers=headers, json=payload, timeout=15)

            if response.status_code == 200:
                data = response.json()
                results = []

                for item in data.get("results", []):
                    # Extract comprehensive information
                    title = item.get("title", "")
                    url = item.get("url", "")

                    # Combine text, summary, and highlights for rich content
                    content_parts = []

                    if item.get("summary"):
                        content_parts.append(item["summary"])

                    if item.get("text"):
                        # Take first 300 chars of text
                        content_parts.append(item["text"][:300])

                    if item.get("highlights") and isinstance(item["highlights"], list):
                        # Add highlights for key information
                        highlights = " | ".join(item["highlights"][:3])
                        content_parts.append(f"Key points: {highlights}")

                    snippet = " ... ".join(content_parts)

                    if title and url and snippet:
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "source": "exa_ai",
                            "timestamp": datetime.now().isoformat(),
                            "score": item.get("score", 0.8),  # Exa provides relevance scores
                            "published_date": item.get("publishedDate", "")
                        })

                return results[:max_results]

            elif response.status_code == 429:
                raise Exception("Exa API rate limit exceeded - trial quota reached")
            elif response.status_code == 401:
                raise Exception("Exa API authentication failed - trial may have expired")
            elif response.status_code == 402:
                raise Exception("Exa API payment required - trial quota exceeded")
            else:
                raise Exception(f"Exa API returned status {response.status_code}: {response.text}")

        except requests.exceptions.Timeout:
            raise Exception("Exa API request timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Exa API request failed: {str(e)}")

    def _optimize_query_for_exa(self, query: str) -> str:
        """Optimize query for better Exa search results."""
        # Remove question words that might confuse the search
        query_lower = query.lower()

        # Convert questions to search-friendly format
        if query_lower.startswith("what is "):
            return query[8:]  # Remove "what is "
        elif query_lower.startswith("how does "):
            return query[9:] + " explanation"  # Remove "how does " and add "explanation"
        elif query_lower.startswith("how to "):
            return query[7:] + " guide tutorial"  # Remove "how to " and add helpful terms
        elif query_lower.startswith("why "):
            return query[4:] + " reasons explanation"  # Remove "why " and add context
        elif query_lower.startswith("when "):
            return query[5:] + " timeline history"  # Remove "when " and add temporal context
        elif query_lower.startswith("where "):
            return query[6:] + " location geography"  # Remove "where " and add location context

        # Add context for better results
        if "?" in query:
            query = query.replace("?", "")

        # Add educational context for learning queries
        educational_keywords = ["learn", "understand", "explain", "definition", "meaning"]
        if any(keyword in query_lower for keyword in educational_keywords):
            return f"{query} comprehensive guide"

        return query

    def _enhanced_duckduckgo_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Enhanced DuckDuckGo search with better rate limiting and multiple approaches."""
        if not REQUESTS_AVAILABLE:
            raise Exception("Requests library not available")

        import requests
        import urllib.parse
        import time
        import random

        results = []

        # Strategy 1: Try DuckDuckGo HTML scraping (more reliable)
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            encoded_query = urllib.parse.quote_plus(query)
            search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 1.5))

            response = requests.get(search_url, headers=headers, timeout=15)

            if response.status_code == 200:
                # Simple text extraction from HTML
                content = response.text.lower()

                # Extract basic information
                if any(keyword in content for keyword in query.lower().split()):
                    results.append({
                        "title": f"Information about {query}",
                        "url": f"https://duckduckgo.com/?q={encoded_query}",
                        "snippet": f"Search results found for {query}. Multiple sources available.",
                        "source": "enhanced_duckduckgo",
                        "timestamp": datetime.now().isoformat()
                    })

        except Exception as e:
            print(f"Enhanced DuckDuckGo HTML search failed: {e}")

        # Strategy 2: Try instant answer API with retry
        if not results:
            try:
                for attempt in range(2):
                    time.sleep(random.uniform(1, 3))  # Random delay

                    api_url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
                    response = requests.get(api_url, timeout=10)

                    if response.status_code == 200:
                        data = response.json()

                        if data.get('Abstract'):
                            results.append({
                                "title": data.get('Heading', query),
                                "url": data.get('AbstractURL', f"https://duckduckgo.com/?q={encoded_query}"),
                                "snippet": data.get('Abstract', ''),
                                "source": "enhanced_duckduckgo_api",
                                "timestamp": datetime.now().isoformat()
                            })
                            break
                    elif response.status_code == 202:
                        print(f"Rate limited, retrying in {2 + attempt} seconds...")
                        time.sleep(2 + attempt)
                    else:
                        break

            except Exception as e:
                print(f"Enhanced DuckDuckGo API search failed: {e}")

        return results[:max_results]

    def _enhanced_wikipedia_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Enhanced Wikipedia search with multiple strategies."""
        if not REQUESTS_AVAILABLE:
            raise Exception("Requests library not available")

        import requests
        import urllib.parse
        import time
        import random

        results = []

        # Strategy 1: Wikipedia API search
        try:
            time.sleep(random.uniform(0.5, 1.0))  # Rate limiting

            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"

            # Clean query for Wikipedia
            clean_query = query.replace("what is ", "").replace("how does ", "").replace("?", "")
            encoded_query = urllib.parse.quote(clean_query)

            headers = {
                'User-Agent': 'Educational-AI-Assistant/1.0 (educational purposes)'
            }

            response = requests.get(f"{search_url}{encoded_query}", headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data.get('extract'):
                    results.append({
                        "title": data.get('title', clean_query),
                        "url": data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                        "snippet": data.get('extract', ''),
                        "source": "enhanced_wikipedia",
                        "timestamp": datetime.now().isoformat()
                    })

        except Exception as e:
            print(f"Enhanced Wikipedia API search failed: {e}")

        # Strategy 2: Wikipedia search API
        if not results:
            try:
                time.sleep(random.uniform(0.5, 1.0))

                search_api = "https://en.wikipedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': clean_query,
                    'srlimit': min(3, max_results)
                }

                response = requests.get(search_api, params=params, headers=headers, timeout=10)

                if response.status_code == 200:
                    data = response.json()

                    for item in data.get('query', {}).get('search', []):
                        results.append({
                            "title": item.get('title', ''),
                            "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(item.get('title', ''))}",
                            "snippet": item.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', ''),
                            "source": "enhanced_wikipedia_search",
                            "timestamp": datetime.now().isoformat()
                        })

            except Exception as e:
                print(f"Enhanced Wikipedia search API failed: {e}")

        return results[:max_results]

    def _generate_enhanced_fallback(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Generate enhanced knowledge-based responses when web search fails."""

        # Comprehensive knowledge base for when web search is unavailable
        enhanced_knowledge = {
            # Technology & AI
            "artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. Modern AI includes machine learning, deep learning, natural language processing, and computer vision.",

            "machine learning": "Machine Learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions. Types include supervised learning, unsupervised learning, and reinforcement learning.",

            "quantum computing": "Quantum computing is a revolutionary technology that uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits (qubits) that can exist in multiple states simultaneously, potentially solving certain problems exponentially faster.",

            "neural networks": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information and learn patterns from data. They're fundamental to deep learning and many AI applications, including image recognition, natural language processing, and autonomous systems.",

            "blockchain": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records (blocks) that are linked and secured using cryptography. It's the technology behind cryptocurrencies and has applications in supply chain management, digital identity, smart contracts, and decentralized finance.",

            "deep learning": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition.",

            # Science & Energy
            "renewable energy": "Renewable energy comes from natural sources that are constantly replenished, such as solar, wind, hydroelectric, geothermal, and biomass power. These sources are sustainable alternatives to fossil fuels and help reduce environmental impact while providing clean electricity generation.",

            "solar energy": "Solar energy harnesses sunlight using photovoltaic cells or solar thermal collectors to generate electricity or heat. It's one of the fastest-growing renewable energy sources and is becoming increasingly cost-competitive with traditional energy sources.",

            "nuclear energy": "Nuclear energy is produced through nuclear fission or fusion reactions. Nuclear power plants generate electricity by splitting uranium atoms, producing large amounts of energy with minimal carbon emissions, though waste disposal and safety remain important considerations.",

            # Space & Exploration
            "space exploration": "Space exploration involves the investigation of outer space through robotic spacecraft and human spaceflight. It has led to numerous scientific discoveries, technological advances, and a better understanding of our universe, Earth's climate, and potential for life elsewhere.",

            "mars exploration": "Mars exploration involves robotic missions and planned human missions to study the Red Planet. Current rovers like Perseverance and Curiosity are searching for signs of past life and preparing for future human exploration missions planned for the 2030s.",

            # Environment & Climate
            "climate change": "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, scientific evidence shows that human activities, particularly greenhouse gas emissions from burning fossil fuels, are the primary driver of recent climate change.",

            "global warming": "Global warming is the long-term increase in Earth's average surface temperature due to human activities and natural factors. It's the primary driver of climate change and is causing rising sea levels, changing precipitation patterns, and more frequent extreme weather events.",

            # Biology & Medicine
            "genetics": "Genetics is the study of genes, heredity, and genetic variation in living organisms. Modern genetics includes DNA sequencing, gene therapy, CRISPR gene editing, and personalized medicine based on individual genetic profiles.",

            "biotechnology": "Biotechnology uses biological systems, organisms, or derivatives to develop products and technologies for various applications including medicine, agriculture, food production, and environmental management.",

            # Computing & Internet
            "internet": "The Internet is a global network of interconnected computers that communicate using standardized protocols. It enables worldwide information sharing, communication, commerce, and has become essential infrastructure for modern society.",

            "cybersecurity": "Cybersecurity involves protecting computer systems, networks, and data from digital attacks, unauthorized access, and damage. It includes practices like encryption, firewalls, intrusion detection, and security awareness training.",

            # Economics & Society
            "cryptocurrency": "Cryptocurrency is a digital or virtual currency secured by cryptography and typically based on blockchain technology. Bitcoin, Ethereum, and other cryptocurrencies offer decentralized alternatives to traditional financial systems.",

            "automation": "Automation involves using technology to perform tasks with minimal human intervention. It includes industrial robots, software automation, and AI-driven systems that can increase efficiency and productivity across various industries."
        }

        # Find relevant knowledge
        query_lower = query.lower()
        results = []

        for topic, description in enhanced_knowledge.items():
            if any(word in query_lower for word in topic.split()) or topic in query_lower:
                results.append({
                    "title": f"Knowledge about {topic.title()}",
                    "url": f"https://example.com/knowledge/{topic.replace(' ', '-')}",
                    "snippet": description,
                    "source": "enhanced_knowledge_base",
                    "timestamp": datetime.now().isoformat()
                })

        # If no specific match, generate a general response
        if not results:
            results.append({
                "title": f"Information about {query}",
                "url": "https://example.com/general-info",
                "snippet": f"This topic '{query}' is an interesting subject that would benefit from further research. While specific web sources are currently unavailable, this appears to be a valid topic for investigation.",
                "source": "enhanced_general_knowledge",
                "timestamp": datetime.now().isoformat()
            })

        return results[:max_results]

    def _wikipedia_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Search Wikipedia for information."""
        if not REQUESTS_AVAILABLE:
            raise Exception("Requests library not available")

        import requests
        import urllib.parse

        # Wikipedia API search
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"

        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('extract'):
                return [{
                    "title": data.get('title', query),
                    "url": data.get('content_urls', {}).get('desktop', {}).get('page', 'https://wikipedia.org'),
                    "snippet": data.get('extract', ''),
                    "source": "wikipedia",
                    "timestamp": datetime.now().isoformat()
                }]

        raise Exception(f"Wikipedia API returned status {response.status_code}")



    def _generate_intelligent_fallback(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Generate intelligent responses when web search is unavailable."""
        query_lower = query.lower()

        # Computer-related queries
        if any(word in query_lower for word in ['computer', 'cpu', 'hardware', 'software']):
            return [{
                "title": "Computer Technology Information",
                "url": "https://example.com/computer-info",
                "snippet": "A computer is an electronic device that processes data according to instructions. Key components include CPU, memory, storage, and input/output devices. Modern computers use binary code and can perform complex calculations, run software applications, and connect to networks.",
                "source": "knowledge_base",
                "timestamp": datetime.now().isoformat()
            }]

        # AI-related queries
        elif any(word in query_lower for word in ['artificial intelligence', 'ai', 'machine learning']):
            return [{
                "title": "Artificial Intelligence Overview",
                "url": "https://example.com/ai-info",
                "snippet": "Artificial Intelligence (AI) is the simulation of human intelligence in machines. It includes machine learning, natural language processing, computer vision, and robotics. AI is used in healthcare, transportation, finance, and many other fields.",
                "source": "knowledge_base",
                "timestamp": datetime.now().isoformat()
            }]

        # General fallback
        else:
            return [{
                "title": f"Information about {query}",
                "url": "https://example.com/general-info",
                "snippet": f"I don't have specific current information about '{query}' due to web search limitations. However, I can help you with questions about technology, science, AI, and general knowledge topics. Please try rephrasing your question or ask about a specific aspect you're interested in.",
                "source": "knowledge_base",
                "timestamp": datetime.now().isoformat()
            }]

    # Mock web search removed - using only real free services

    
    def extract_and_store_knowledge(self, question: str, web_results: List[Dict[str, str]]):
        """Extract knowledge from web search results and store it with duplicate removal."""
        new_facts_added = 0

        for result in web_results:
            # Simple extraction - in a real implementation, use more sophisticated NLP
            facts = sent_tokenize(result['snippet'])
            for fact in facts:
                if len(fact) > 20:  # Only store meaningful facts
                    # Clean and normalize the fact
                    cleaned_fact = self._clean_fact(fact)

                    # Check for duplicates before adding
                    if not self._is_duplicate_fact(cleaned_fact):
                        fact_hash = hashlib.md5(cleaned_fact.encode()).hexdigest()
                        self.memory['facts'].append({
                            'id': fact_hash,
                            'content': cleaned_fact,
                            'source': result['url'],
                            'timestamp': datetime.now().isoformat(),
                            'confidence': 0.9  # Initial confidence
                        })
                        new_facts_added += 1
                    else:
                        # Update existing fact with new source if different
                        self._update_existing_fact(cleaned_fact, result['url'])

        # Remove any duplicates that might have slipped through
        self._remove_duplicate_facts()

        # Save the updated memory
        self.save_memory()

        if new_facts_added > 0:
            print(f"💾 Stored {new_facts_added} new facts (duplicates filtered)")
        else:
            print("💾 No new facts stored (all were duplicates or already known)")

    def _clean_fact(self, fact: str) -> str:
        """Clean and normalize a fact for better duplicate detection."""
        # Remove extra whitespace
        fact = ' '.join(fact.split())

        # Remove common prefixes that don't add meaning
        prefixes_to_remove = [
            "According to the study, ",
            "The research shows that ",
            "It is known that ",
            "Studies indicate that ",
            "Research suggests that "
        ]

        for prefix in prefixes_to_remove:
            if fact.startswith(prefix):
                fact = fact[len(prefix):]
                break

        # Remove trailing periods and normalize punctuation
        fact = fact.rstrip('.')

        # Capitalize first letter
        if fact and fact[0].islower():
            fact = fact[0].upper() + fact[1:]

        return fact

    def _is_duplicate_fact(self, new_fact: str) -> bool:
        """Check if a fact is already stored in memory."""
        new_fact_lower = new_fact.lower()
        new_fact_words = set(new_fact_lower.split())

        for existing_fact in self.memory['facts']:
            existing_content = existing_fact['content'].lower()
            existing_words = set(existing_content.split())

            # Check for exact match
            if new_fact_lower == existing_content:
                return True

            # Check for high similarity (80% word overlap)
            if len(new_fact_words) > 3 and len(existing_words) > 3:
                overlap = len(new_fact_words.intersection(existing_words))
                similarity = overlap / max(len(new_fact_words), len(existing_words))

                if similarity > 0.8:
                    return True

            # Check for substring containment (one fact contains the other)
            if (new_fact_lower in existing_content and len(new_fact_lower) > 30) or \
               (existing_content in new_fact_lower and len(existing_content) > 30):
                return True

        return False

    def _update_existing_fact(self, fact: str, new_source: str):
        """Update an existing fact with a new source if it's different."""
        fact_lower = fact.lower()

        for existing_fact in self.memory['facts']:
            if existing_fact['content'].lower() == fact_lower:
                # Add new source if it's different
                if existing_fact['source'] != new_source:
                    # Create a list of sources if not already
                    if isinstance(existing_fact['source'], str):
                        existing_fact['sources'] = [existing_fact['source'], new_source]
                        existing_fact['source'] = new_source  # Keep most recent as primary
                    elif isinstance(existing_fact.get('sources'), list):
                        if new_source not in existing_fact['sources']:
                            existing_fact['sources'].append(new_source)
                            existing_fact['source'] = new_source

                # Update timestamp
                existing_fact['last_updated'] = datetime.now().isoformat()
                break

    def _remove_duplicate_facts(self):
        """Remove duplicate facts from memory."""
        if not self.memory['facts']:
            return

        unique_facts = []
        seen_contents = set()

        for fact in self.memory['facts']:
            content_lower = fact['content'].lower()
            content_normalized = ' '.join(content_lower.split())

            if content_normalized not in seen_contents:
                seen_contents.add(content_normalized)
                unique_facts.append(fact)

        removed_count = len(self.memory['facts']) - len(unique_facts)
        if removed_count > 0:
            print(f"🧹 Removed {removed_count} duplicate facts from memory")
            self.memory['facts'] = unique_facts

    def clean_memory(self) -> str:
        """Manually clean memory and remove duplicates."""
        original_count = len(self.memory['facts'])

        # Remove duplicates
        self._remove_duplicate_facts()

        # Remove very short or low-quality facts
        quality_facts = []
        for fact in self.memory['facts']:
            content = fact['content']
            # Keep facts that are meaningful
            if (len(content) > 20 and  # Reasonable minimum length
                not content.lower().startswith(('click here', 'read more', 'see also', 'advertisement')) and
                not content.lower().endswith(('...', 'more info', 'click here')) and
                len(content.split()) >= 4):  # At least 4 words
                quality_facts.append(fact)

        self.memory['facts'] = quality_facts

        # Save cleaned memory
        self.save_memory()

        final_count = len(self.memory['facts'])
        removed_count = original_count - final_count

        return f"""🧹 **MEMORY CLEANING COMPLETE**

📊 **Results:**
• Original facts: {original_count}
• Facts after cleaning: {final_count}
• Duplicates/low-quality removed: {removed_count}
• Memory efficiency improved: {(removed_count/original_count*100):.1f}%

✅ **Memory is now optimized and duplicate-free!**

💡 **Tip:** Use 'clean memory' anytime to optimize your knowledge base."""
    
    def find_relevant_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find the most relevant knowledge for a given query."""
        if not self.memory['facts']:
            return []

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            # Fallback to simple keyword matching
            return self._simple_knowledge_search(query, top_k)

        try:
            # Encode query and facts
            query_embedding = self.similarity_model.encode(query, convert_to_tensor=TORCH_AVAILABLE)
            fact_embeddings = self.similarity_model.encode(
                [fact['content'] for fact in self.memory['facts']],
                convert_to_tensor=TORCH_AVAILABLE
            )

            # Calculate similarities
            if TORCH_AVAILABLE:
                # Handle tensor operations properly
                try:
                    # If query_embedding is a torch tensor, use unsqueeze
                    if TORCH_AVAILABLE and hasattr(query_embedding, 'unsqueeze'):
                        query_tensor = query_embedding.unsqueeze(0)
                    else:
                        # If query_embedding is a list (not a tensor), wrap it in another list to simulate batch dimension
                        if isinstance(query_embedding, list):
                            # If it's a list of floats, wrap once; if already a list of list, keep as is
                            if query_embedding and isinstance(query_embedding[0], float):
                                query_tensor = [query_embedding]
                            elif query_embedding and isinstance(query_embedding[0], list):
                                query_tensor = query_embedding
                            else:
                                query_tensor = [query_embedding]
                        else:
                            query_tensor = [query_embedding]

                    similarities = util.pytorch_cos_sim(query_tensor, fact_embeddings)[0]
                    # Get top-k most similar facts
                    top_indices = torch.topk(similarities, k=min(top_k, len(similarities))).indices
                    return [self.memory['facts'][i] for i in top_indices]
                except Exception as e:
                    print(f"Warning: Tensor operation failed, using fallback: {e}")
                    # Fall back to simple calculation
                    pass

            # Simple similarity calculation (fallback or when torch not available)
            similarities = util.pytorch_cos_sim([query_embedding], fact_embeddings)[0]
            # Get top indices manually
            indexed_similarities = [(i, sim) for i, sim in enumerate(similarities)]
            indexed_similarities.sort(key=lambda x: x[1], reverse=True)
            top_indices = [i for i, _ in indexed_similarities[:top_k]]
            return [self.memory['facts'][i] for i in top_indices]
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return self._simple_knowledge_search(query, top_k)

    def _simple_knowledge_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Simple keyword-based knowledge search fallback."""
        query_words = set(query.lower().split())
        # Remove common words that don't add meaning
        stop_words = {'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'how', 'when', 'where', 'why'}
        query_words = query_words - stop_words

        if not query_words:  # If no meaningful words left
            return []

        scored_facts = []

        for fact in self.memory['facts']:
            fact_words = set(fact['content'].lower().split()) - stop_words
            # Calculate overlap score
            overlap = len(query_words.intersection(fact_words))
            # Calculate relevance as percentage of query words found
            relevance = overlap / len(query_words) if query_words else 0

            # Only include facts with significant relevance (at least 30% word match)
            if relevance >= 0.3:
                scored_facts.append((fact, relevance))

        # Sort by relevance score and return top_k
        scored_facts.sort(key=lambda x: x[1], reverse=True)
        return [fact for fact, _ in scored_facts[:top_k]]
    
    def detect_emotion(self, text: str) -> str:
        """Detect simple emotion in user input based on keywords."""
        emotion_keywords = {
            'happy': ['happy', 'great', 'awesome', 'fantastic', 'good', 'excited', 'joy'],
            'sad': ['sad', 'unhappy', 'depressed', 'down', 'blue', 'upset'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated'],
            'confused': ['confused', 'lost', 'unsure', 'uncertain'],
            'tired': ['tired', 'exhausted', 'sleepy', 'fatigued'],
            'anxious': ['anxious', 'nervous', 'worried', 'scared', 'afraid'],
            'neutral': []
        }
        text_lower = text.lower()
        for emotion, keywords in emotion_keywords.items():
            for word in keywords:
                if word in text_lower:
                    return emotion
        return 'neutral'

    def empathetic_phrase(self, emotion: str) -> str:
        """Return an empathetic phrase based on detected emotion."""
        phrases = {
            'happy': "I'm glad to hear you're feeling good!",
            'sad': "I'm here for you. If you want to talk about it, I'm listening.",
            'angry': "It's okay to feel angry sometimes. Let me know if I can help.",
            'confused': "I'll do my best to clear things up for you!",
            'tired': "Remember to take care of yourself and rest when you need to.",
            'anxious': "Take a deep breath. I'm here to help however I can.",
            'neutral': "How can I assist you today?"
        }
        return phrases.get(emotion, "How can I assist you today?")

    def generate_response(self, question: str, verbose_mode: bool = False) -> str:
        """Generate a clean, focused response to the user's question."""
        # Update conversation context
        self.conversation_context["previous_questions"].append({
            "question": question,
            "timestamp": datetime.now().isoformat()
        })

        # Update attention focus (internal processing)
        self.metacognitive_processor.update_attention(f"responding to: {question[:30]}...", 0.8)

        # Analyze conversation flow and context
        context_analysis = self._analyze_conversation_context(question)

        # Check for special commands first
        question_lower = question.lower().strip()

        # Handle special AI commands
        if question_lower in ["consciousness report", "show consciousness", "consciousness state"]:
            return self.consciousness_state_report()
        elif question_lower in ["deep introspection", "introspection", "analyze yourself"]:
            return self.deep_introspection()
        elif question_lower in ["stream of consciousness", "thought stream", "show thoughts"]:
            stream = self.stream_of_consciousness()
            return f"Current thought stream:\n" + "\n".join([f"- {thought}" for thought in stream])
        elif question_lower in ["create improvement plan", "self improvement", "improve yourself"]:
            return self.create_self_improvement_plan()
        elif question_lower in ["track progress", "show progress", "progress report"]:
            return self.track_progress()
        elif question_lower == "verbose mode on":
            return "Verbose mode enabled. I'll now show my internal processing details."
        elif question_lower == "verbose mode off":
            return "Verbose mode disabled. I'll now give clean, focused responses."
        elif question_lower in ["learn mode", "start learning", "autonomous learning", "auto learn"]:
            return self.start_autonomous_learning()
        elif question_lower in ["stop learning", "stop learn mode", "pause learning"]:
            return self.stop_autonomous_learning()
        elif question_lower in ["learning status", "learning progress", "show learning"]:
            return self.get_learning_status()
        elif question_lower.startswith("learning speed"):
            # Handle learning speed adjustment: "learning speed fast/medium/slow"
            parts = question_lower.split()
            if len(parts) >= 3:
                speed = parts[2]
                return self.set_learning_speed(speed)
            else:
                return "Usage: 'learning speed fast/medium/slow' to adjust learning interval."
        elif question_lower in ["learning help", "help learning", "learning commands"]:
            return self.get_learning_help()
        elif question_lower in ["what are you learning", "show learning topics", "current learning", "learning topics"]:
            return self.show_current_learning()
        elif question_lower in ["learning history", "show learning history", "what did you learn"]:
            return self.show_learning_history()
        elif question_lower.startswith("good") or question_lower.startswith("great") or question_lower.startswith("excellent"):
            return self.process_positive_feedback(question)
        elif question_lower.startswith("bad") or question_lower.startswith("wrong") or question_lower.startswith("incorrect"):
            return self.process_negative_feedback(question)
        elif question_lower in ["feedback stats", "show feedback", "learning stats"]:
            return self.show_reinforcement_stats()
        elif question_lower in ["improve response", "learn better", "reinforcement learning"]:
            return self.show_reinforcement_help()
        elif question_lower in ["search questions", "process questions", "answer questions"]:
            return self.process_questions_folder()
        elif question_lower in ["get info", "get information", "auto search"]:
            return self.get_info_from_questions()
        elif question_lower in ["questions help", "question folder help"]:
            return self.show_questions_help()
        elif question_lower in ["search status", "api status", "exa status"]:
            return self.show_search_status()
        elif question_lower in ["clean memory", "remove duplicates", "optimize memory"]:
            return self.clean_memory()

        # Check if this is a self-awareness question
        self_questions = [
            "who are you", "what are you", "what can you do", "who made you",
            "what is your purpose", "describe yourself", "are you self aware",
            "what are your powers", "what are your abilities", "what are your limitations",
            "what s your name", "what is your name", "tell me about yourself"
        ]

        if any(q in question_lower for q in self_questions):
            # For self-awareness questions, give a clean version of self-reflection
            return self._generate_clean_self_description()

        # Generate contextual response based on conversation flow
        contextual_response = self._generate_contextual_response(question, context_analysis)

        # Update conversation topic based on response
        if context_analysis["question_pattern"] != "acknowledgment":
            self.conversation_context["current_topic"] = self._extract_topic(question)

        # Store this interaction in conversation flow
        self.conversation_context["conversation_flow"].append({
            "question": question,
            "response_type": context_analysis["question_pattern"],
            "topic": self.conversation_context["current_topic"],
            "timestamp": datetime.now().isoformat()
        })

        # Track response for reinforcement learning
        response_entry = {
            "question": question,
            "response": contextual_response,
            "timestamp": datetime.now().isoformat(),
            "response_id": len(self.reinforcement_learning["response_history"]) + 1
        }
        self.reinforcement_learning["response_history"].append(response_entry)
        self.reinforcement_learning["last_response"] = response_entry
        self.reinforcement_learning["waiting_for_feedback"] = True

        # Return clean response without metacognitive commentary
        return contextual_response

    def _generate_clean_self_description(self) -> str:
        """Generate a clean, focused self-description without metacognitive details."""
        info = self.self_knowledge
        state = self.self_state

        return f"""{info['identity']}

**My Creator:** {info['creator']}

**What I Can Do:**
{chr(10).join(f"• {capability}" for capability in info['capabilities'])}

**My Current State:**
• Mode: {state['current_mode']}
• Facts learned today: {state['facts_learned_today']}
• Total knowledge: {state['total_facts']} facts
• Learning topics: {', '.join(state['last_learned_topics']) if state['last_learned_topics'] else 'None recently'}

**My Limitations:**
{chr(10).join(f"• {limitation}" for limitation in info['limitations'])}

I'm designed to be helpful, learn from our conversations, and continuously improve my responses based on our interactions."""

    def start_autonomous_learning(self) -> str:
        """Start autonomous learning mode - AI searches random topics continuously."""
        if self.learning_mode["active"]:
            return "🧠 Autonomous learning is already active! Use 'learning status' to see progress."

        # Initialize learning mode
        self.learning_mode["active"] = True
        self.learning_mode["start_time"] = datetime.now()
        self.learning_mode["topics_learned"] = 0
        self.learning_mode["stop_flag"] = False

        # Generate initial topic queue
        self._generate_learning_topics()

        # Start learning in a separate thread
        import threading
        self.learning_mode["learning_thread"] = threading.Thread(
            target=self._autonomous_learning_loop,
            daemon=True
        )
        self.learning_mode["learning_thread"].start()

        return f"""🚀 AUTONOMOUS LEARNING MODE ACTIVATED!

🧠 **Learning Configuration:**
• Search Interval: {self.learning_mode['search_interval']} seconds
• Topics in Queue: {len(self.learning_mode['topics_queue'])}
• Current Knowledge: {len(self.memory['facts'])} facts

🔄 **What I'll Do:**
• Search random topics continuously
• Store new knowledge automatically
• Build comprehensive understanding
• Improve response quality over time

📊 **Commands:**
• 'learning status' - Check progress
• 'stop learning' - Pause autonomous mode

I'm now learning autonomously in the background! 🌟"""

    def stop_autonomous_learning(self) -> str:
        """Stop autonomous learning mode."""
        if not self.learning_mode["active"]:
            return "🛑 Autonomous learning is not currently active."

        # Stop the learning process
        self.learning_mode["stop_flag"] = True
        self.learning_mode["active"] = False

        # Calculate learning stats
        if self.learning_mode["start_time"]:
            duration = datetime.now() - self.learning_mode["start_time"]
            duration_str = str(duration).split('.')[0]  # Remove microseconds
        else:
            duration_str = "Unknown"

        return f"""🛑 AUTONOMOUS LEARNING STOPPED

📊 **Learning Session Summary:**
• Duration: {duration_str}
• Topics Learned: {self.learning_mode['topics_learned']}
• Total Knowledge: {len(self.memory['facts'])} facts
• Learning Rate: {self.learning_mode['topics_learned'] / max(1, duration.total_seconds() / 60):.1f} topics/minute

🧠 **Knowledge Growth:**
I've expanded my understanding and can now provide better responses!

Use 'learn mode' to start learning again anytime. 🚀"""

    def get_learning_status(self) -> str:
        """Get current autonomous learning status."""
        if not self.learning_mode["active"]:
            return f"""📊 LEARNING STATUS: INACTIVE

🧠 **Current Knowledge:** {len(self.memory['facts'])} facts stored
💡 **Ready to Learn:** Use 'learn mode' to start autonomous learning

**What Autonomous Learning Does:**
• Searches random topics continuously
• Stores new knowledge automatically
• Improves response quality over time
• Builds comprehensive understanding"""

        # Calculate current stats
        if self.learning_mode["start_time"]:
            duration = datetime.now() - self.learning_mode["start_time"]
            duration_str = str(duration).split('.')[0]
        else:
            duration_str = "Unknown"

        time_since_last = "Never"
        if self.learning_mode["last_search_time"]:
            since_last = datetime.now() - self.learning_mode["last_search_time"]
            time_since_last = f"{int(since_last.total_seconds())} seconds ago"

        return f"""📊 LEARNING STATUS: ACTIVE 🟢

⏱️ **Session Info:**
• Duration: {duration_str}
• Topics Learned: {self.learning_mode['topics_learned']}
• Last Search: {time_since_last}
• Search Interval: {self.learning_mode['search_interval']} seconds

🧠 **Knowledge Base:**
• Total Facts: {len(self.memory['facts'])}
• Topics in Queue: {len(self.learning_mode['topics_queue'])}

🔄 **Status:** Actively learning in background...
Use 'stop learning' to pause autonomous mode."""

    def _generate_learning_topics(self):
        """Generate a queue of diverse topics for autonomous learning."""
        # Comprehensive topic categories for well-rounded learning
        topic_categories = {
            "Science & Technology": [
                "quantum computing", "artificial intelligence", "machine learning", "biotechnology",
                "nanotechnology", "renewable energy", "space exploration", "robotics",
                "genetic engineering", "virtual reality", "blockchain technology", "cybersecurity",
                "3D printing", "autonomous vehicles", "neural networks", "cloud computing"
            ],
            "History & Culture": [
                "ancient civilizations", "world wars", "renaissance period", "industrial revolution",
                "cultural traditions", "historical figures", "archaeological discoveries",
                "ancient philosophy", "medieval history", "modern history", "art movements",
                "literature classics", "music history", "architectural styles"
            ],
            "Nature & Environment": [
                "climate change", "biodiversity", "ecosystems", "conservation", "wildlife",
                "ocean science", "geology", "meteorology", "environmental protection",
                "sustainable development", "natural disasters", "plant biology", "animal behavior",
                "marine life", "forest ecosystems", "renewable resources"
            ],
            "Health & Medicine": [
                "medical breakthroughs", "nutrition science", "mental health", "disease prevention",
                "pharmaceutical research", "medical technology", "public health", "genetics",
                "immunology", "neuroscience", "cardiology", "cancer research", "vaccines",
                "medical imaging", "surgical techniques", "preventive medicine"
            ],
            "Economics & Society": [
                "economic systems", "global trade", "social issues", "political systems",
                "human rights", "education systems", "urban planning", "demographics",
                "social psychology", "cultural anthropology", "international relations",
                "economic development", "social movements", "governance", "public policy"
            ],
            "Arts & Creativity": [
                "visual arts", "performing arts", "digital art", "creative writing",
                "film and cinema", "music composition", "design principles", "photography",
                "sculpture", "painting techniques", "theater", "dance", "creative processes",
                "artistic movements", "cultural expression", "multimedia arts"
            ]
        }

        # Generate diverse topic queue
        topics = []
        for category, category_topics in topic_categories.items():
            # Add some topics from each category for balanced learning
            import random
            selected = random.sample(category_topics, min(5, len(category_topics)))
            topics.extend(selected)

        # Shuffle for random order
        random.shuffle(topics)

        # Add to queue
        for topic in topics:
            self.learning_mode["topics_queue"].append(topic)

    def _autonomous_learning_loop(self):
        """Main learning loop that runs in background thread."""
        print("🧠 Autonomous learning started in background...")

        while self.learning_mode["active"] and not self.learning_mode["stop_flag"]:
            try:
                # Check if it's time for next search
                current_time = datetime.now()
                if (self.learning_mode["last_search_time"] is None or
                    (current_time - self.learning_mode["last_search_time"]).total_seconds() >=
                    self.learning_mode["search_interval"]):

                    # Get next topic to learn
                    if self.learning_mode["topics_queue"]:
                        topic = self.learning_mode["topics_queue"].popleft()
                        self._learn_topic(topic)
                        self.learning_mode["topics_learned"] += 1
                        self.learning_mode["last_search_time"] = current_time

                        # Refill queue if getting low
                        if len(self.learning_mode["topics_queue"]) < 10:
                            self._generate_learning_topics()

                # Sleep for a short time before checking again
                time.sleep(5)

            except Exception as e:
                print(f"Learning error: {e}")
                time.sleep(10)  # Wait longer on error

        print("🛑 Autonomous learning stopped.")

    def _learn_topic(self, topic: str):
        """Learn about a specific topic through web search."""
        try:
            # Update current learning status
            self.learning_mode["current_topic"] = topic
            print(f"🔍 Learning about: {topic}")

            # Search for information about the topic
            web_results = self.web_search(topic, 3)

            if web_results and web_results[0].get('snippet'):
                # Store the knowledge
                self.extract_and_store_knowledge(topic, web_results)

                # Track learning history
                learning_entry = {
                    "topic": topic,
                    "timestamp": datetime.now().isoformat(),
                    "sources_found": len(web_results),
                    "facts_added": len([r for r in web_results if r.get('snippet')])
                }
                self.learning_mode["recent_topics"].append(learning_entry)
                self.learning_mode["learning_history"].append(learning_entry)

                # Update metacognitive awareness
                self.metacognitive_processor.update_attention(f"learned about {topic}", 0.7)
                self.metacognitive_processor.internal_monologue(
                    f"I just learned about {topic}. This expands my knowledge base."
                )

                print(f"✅ Learned about {topic} - {len(web_results)} sources processed")
            else:
                print(f"⚠️ No information found for {topic}")

            # Clear current topic when done
            self.learning_mode["current_topic"] = None

        except Exception as e:
            print(f"❌ Error learning about {topic}: {e}")
            self.learning_mode["current_topic"] = None

    def set_learning_speed(self, speed: str) -> str:
        """Set the autonomous learning speed."""
        speed_settings = {
            "fast": 15,      # 15 seconds between searches
            "medium": 30,    # 30 seconds between searches
            "slow": 60,      # 60 seconds between searches
            "turbo": 5       # 5 seconds between searches (very fast)
        }

        if speed not in speed_settings:
            return f"""❌ Invalid speed setting: '{speed}'

**Available speeds:**
• **turbo** - 5 seconds (very fast learning)
• **fast** - 15 seconds (quick learning)
• **medium** - 30 seconds (balanced learning)
• **slow** - 60 seconds (careful learning)

Usage: 'learning speed fast' or 'learning speed slow'"""

        old_interval = self.learning_mode["search_interval"]
        self.learning_mode["search_interval"] = speed_settings[speed]

        status = "🟢 ACTIVE" if self.learning_mode["active"] else "🔴 INACTIVE"

        return f"""⚡ LEARNING SPEED UPDATED

🎛️ **Speed Setting:** {speed.upper()}
⏱️ **Search Interval:** {speed_settings[speed]} seconds
📊 **Learning Status:** {status}

**Speed Comparison:**
• Turbo: {speed_settings['turbo']}s (very fast)
• Fast: {speed_settings['fast']}s (quick)
• Medium: {speed_settings['medium']}s (balanced)
• Slow: {speed_settings['slow']}s (careful)

The new speed will take effect on the next search cycle! 🚀"""

    def get_learning_help(self) -> str:
        """Get help information for autonomous learning system."""
        return """🧠 AUTONOMOUS LEARNING SYSTEM HELP

🚀 **Main Commands:**
• **'learn mode'** - Start autonomous learning
• **'stop learning'** - Stop autonomous learning
• **'learning status'** - Check learning progress
• **'learning speed [fast/medium/slow/turbo]'** - Adjust learning speed

🔍 **Learning Visibility:**
• **'what are you learning'** - See current learning topics
• **'learning history'** - View complete learning history
• **'learning topics'** - Show recent and upcoming topics

📝 **Quantum Neural Question Processing:**
• **'get info'** - Quantum neural batch processing (MAIN COMMAND)
• **'search questions'** - Traditional sequential processing
• **'questions help'** - Complete guide for copy-paste question system

🔍 **Search System:**
• **'search status'** - Check API status and available search methods

🧹 **Memory Management:**
• **'clean memory'** - Remove duplicates and optimize knowledge base

⚡ **Speed Settings:**
• **turbo** - 5 seconds (very fast, intensive learning)
• **fast** - 15 seconds (quick learning)
• **medium** - 30 seconds (balanced learning)
• **slow** - 60 seconds (careful, thorough learning)

🎯 **What Autonomous Learning Does:**
• Searches diverse topics automatically (Science, History, Nature, Health, etc.)
• Stores all new knowledge in memory
• Improves response quality over time
• Builds comprehensive understanding
• Runs continuously in background

📊 **Learning Categories:**
• Science & Technology
• History & Culture
• Nature & Environment
• Health & Medicine
• Economics & Society
• Arts & Creativity

💡 **Tips:**
• Use 'fast' speed for rapid knowledge building
• Use 'slow' speed for thorough understanding
• Check 'learning status' to monitor progress
• Learning continues until you say 'stop learning'

🌟 **Your AI becomes smarter with every search!**"""

    def show_current_learning(self) -> str:
        """Show what the AI is currently learning about."""
        if not self.learning_mode["active"]:
            return """📚 CURRENT LEARNING STATUS: INACTIVE

🛑 **Autonomous learning is not active**
💡 Use 'learn mode' to start autonomous learning

**When active, you can see:**
• Current topic being researched
• Recent topics learned
• Learning queue status
• Real-time learning progress"""

        current_topic = self.learning_mode.get("current_topic")
        recent_topics = list(self.learning_mode["recent_topics"])
        topics_queue = list(self.learning_mode["topics_queue"])

        # Current learning status
        if current_topic:
            current_status = f"🔍 **Currently Learning:** {current_topic}"
        else:
            current_status = "⏳ **Status:** Waiting for next topic..."

        # Recent topics learned
        recent_list = ""
        if recent_topics:
            recent_list = "📚 **Recently Learned:**\n"
            for topic_info in recent_topics[-5:]:  # Last 5 topics
                topic = topic_info["topic"]
                timestamp = topic_info["timestamp"][:16]  # Just date and time
                sources = topic_info["sources_found"]
                recent_list += f"• {topic} ({sources} sources) - {timestamp}\n"
        else:
            recent_list = "📚 **Recently Learned:** None yet"

        # Upcoming topics
        upcoming_list = ""
        if topics_queue:
            upcoming_list = f"🔮 **Next Topics in Queue:**\n"
            for topic in topics_queue[:5]:  # Next 5 topics
                upcoming_list += f"• {topic}\n"
            if len(topics_queue) > 5:
                upcoming_list += f"... and {len(topics_queue) - 5} more topics"
        else:
            upcoming_list = "🔮 **Next Topics:** Queue empty (will regenerate)"

        return f"""📚 CURRENT LEARNING STATUS: ACTIVE 🟢

{current_status}

{recent_list}

{upcoming_list}

⚡ **Learning Speed:** {self.learning_mode['search_interval']} seconds between topics
📊 **Session Stats:** {self.learning_mode['topics_learned']} topics learned

💡 **Commands:**
• 'learning history' - See full learning history
• 'learning status' - See detailed progress stats
• 'stop learning' - Pause autonomous learning"""

    def show_learning_history(self) -> str:
        """Show complete learning history."""
        history = self.learning_mode.get("learning_history", [])

        if not history:
            return """📖 LEARNING HISTORY: EMPTY

🔍 **No learning history yet**
💡 Use 'learn mode' to start building learning history

**Learning history will show:**
• All topics learned over time
• When each topic was learned
• Number of sources found
• Learning patterns and trends"""

        # Group by date
        from collections import defaultdict
        by_date = defaultdict(list)

        for entry in history:
            date = entry["timestamp"][:10]  # Just the date part
            by_date[date].append(entry)

        # Build history report
        history_report = f"""📖 LEARNING HISTORY

📊 **Total Topics Learned:** {len(history)}
📅 **Learning Days:** {len(by_date)}

"""

        # Show recent days (last 3 days)
        sorted_dates = sorted(by_date.keys(), reverse=True)
        for date in sorted_dates[:3]:
            day_entries = by_date[date]
            history_report += f"**{date}** ({len(day_entries)} topics):\n"

            for entry in day_entries[-5:]:  # Last 5 topics of the day
                topic = entry["topic"]
                time_part = entry["timestamp"][11:16]  # Just time
                sources = entry["sources_found"]
                history_report += f"  • {topic} ({sources} sources) - {time_part}\n"

            if len(day_entries) > 5:
                history_report += f"  ... and {len(day_entries) - 5} more topics\n"
            history_report += "\n"

        if len(sorted_dates) > 3:
            history_report += f"... and {len(sorted_dates) - 3} more days of learning\n\n"

        # Learning stats
        total_sources = sum(entry["sources_found"] for entry in history)
        avg_sources = total_sources / len(history) if history else 0

        history_report += f"""📈 **Learning Statistics:**
• Average sources per topic: {avg_sources:.1f}
• Total sources processed: {total_sources}
• Most recent learning: {history[-1]["timestamp"][:16] if history else "None"}

💡 Use 'what are you learning' to see current learning status"""

        return history_report

    def process_positive_feedback(self, feedback: str) -> str:
        """Process positive feedback to improve future responses."""
        if not self.reinforcement_learning["last_response"]:
            return """💡 **No Recent Response to Rate**

I don't have a recent response to apply feedback to. Please:
1. Ask me a question first
2. Then provide feedback like "good answer" or "great response"

This helps me learn what responses work well! 🎯"""

        # Record positive feedback
        last_response = self.reinforcement_learning["last_response"]
        question = last_response["question"]
        response = last_response["response"]

        # Update feedback scores
        if question not in self.reinforcement_learning["feedback_scores"]:
            self.reinforcement_learning["feedback_scores"][question] = []

        self.reinforcement_learning["feedback_scores"][question].append({
            "score": 1.0,  # Positive feedback
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
            "response": response
        })

        # Update global stats
        self.reinforcement_learning["total_feedback"] += 1
        self.reinforcement_learning["positive_feedback"] += 1

        # Learn from successful response pattern
        self._learn_successful_pattern(question, response)

        # Reset waiting flag
        self.reinforcement_learning["waiting_for_feedback"] = False

        success_rate = (self.reinforcement_learning["positive_feedback"] /
                       self.reinforcement_learning["total_feedback"] * 100)

        return f"""✅ **POSITIVE FEEDBACK RECEIVED!**

🎯 **Feedback Applied To:**
Question: "{question[:60]}{'...' if len(question) > 60 else ''}"
Response: "{response[:60]}{'...' if len(response) > 60 else ''}"

📊 **Learning Impact:**
• This response pattern has been marked as successful
• I'll use similar approaches for related questions
• My confidence in this topic area has increased

📈 **Current Performance:**
• Total Feedback: {self.reinforcement_learning["total_feedback"]}
• Success Rate: {success_rate:.1f}%
• Positive Responses: {self.reinforcement_learning["positive_feedback"]}

🧠 **Thank you for helping me improve!** Your feedback makes me smarter! 🌟"""

    def process_negative_feedback(self, feedback: str) -> str:
        """Process negative feedback to improve future responses."""
        if not self.reinforcement_learning["last_response"]:
            return """💡 **No Recent Response to Rate**

I don't have a recent response to apply feedback to. Please:
1. Ask me a question first
2. Then provide feedback like "bad answer" or "that's wrong"

This helps me learn what to avoid! 🎯"""

        # Record negative feedback
        last_response = self.reinforcement_learning["last_response"]
        question = last_response["question"]
        response = last_response["response"]

        # Update feedback scores
        if question not in self.reinforcement_learning["feedback_scores"]:
            self.reinforcement_learning["feedback_scores"][question] = []

        self.reinforcement_learning["feedback_scores"][question].append({
            "score": -1.0,  # Negative feedback
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
            "response": response
        })

        # Update global stats
        self.reinforcement_learning["total_feedback"] += 1

        # Mark this as an area needing improvement
        self._mark_improvement_area(question, response, feedback)

        # Reset waiting flag
        self.reinforcement_learning["waiting_for_feedback"] = False

        success_rate = (self.reinforcement_learning["positive_feedback"] /
                       self.reinforcement_learning["total_feedback"] * 100)

        return f"""⚠️ **NEGATIVE FEEDBACK RECEIVED**

🎯 **Feedback Applied To:**
Question: "{question[:60]}{'...' if len(question) > 60 else ''}"
Response: "{response[:60]}{'...' if len(response) > 60 else ''}"

🔧 **Learning Actions:**
• This response pattern has been marked for improvement
• I'll avoid similar approaches for related questions
• This topic has been added to my improvement areas
• I'll search for better information on this topic

📊 **Current Performance:**
• Total Feedback: {self.reinforcement_learning["total_feedback"]}
• Success Rate: {success_rate:.1f}%
• Areas to Improve: {len(self.reinforcement_learning["improvement_areas"])}

🧠 **Thank you for the correction!** I'll work to improve this response type. 💪"""

    def _learn_successful_pattern(self, question: str, response: str):
        """Learn from successful response patterns."""
        # Extract key features from successful responses
        question_type = self._classify_question_type(question)
        response_length = len(response.split())

        pattern_key = f"{question_type}_{response_length//50*50}"  # Group by length ranges

        if pattern_key not in self.reinforcement_learning["response_patterns"]:
            self.reinforcement_learning["response_patterns"][pattern_key] = {
                "success_count": 0,
                "examples": [],
                "avg_length": 0,
                "topics": set()
            }

        pattern = self.reinforcement_learning["response_patterns"][pattern_key]
        pattern["success_count"] += 1
        pattern["examples"].append({
            "question": question,
            "response": response[:100] + "..." if len(response) > 100 else response
        })
        pattern["avg_length"] = (pattern["avg_length"] + response_length) / 2

        # Extract topic from question
        topic = self._extract_topic(question)
        if topic:
            pattern["topics"].add(topic)

    def _mark_improvement_area(self, question: str, response: str, feedback: str):
        """Mark areas that need improvement."""
        improvement_item = {
            "question": question,
            "response": response,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
            "topic": self._extract_topic(question),
            "improved": False
        }

        self.reinforcement_learning["improvement_areas"].append(improvement_item)

        # Keep only recent improvement areas (last 20)
        if len(self.reinforcement_learning["improvement_areas"]) > 20:
            self.reinforcement_learning["improvement_areas"] = \
                self.reinforcement_learning["improvement_areas"][-20:]

    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question for pattern learning."""
        question_lower = question.lower()

        if any(word in question_lower for word in ["what is", "what are", "define"]):
            return "definition"
        elif any(word in question_lower for word in ["how", "how to"]):
            return "how_to"
        elif any(word in question_lower for word in ["why", "because"]):
            return "explanation"
        elif any(word in question_lower for word in ["who", "person", "people"]):
            return "person"
        elif any(word in question_lower for word in ["when", "time", "date"]):
            return "temporal"
        elif any(word in question_lower for word in ["where", "location", "place"]):
            return "location"
        else:
            return "general"

    def quantum_neural_batch_processor(self, questions: List[str]) -> List[List[str]]:
        """Use quantum thinking methodology to intelligently batch questions for optimal processing."""
        if not questions:
            return []

        # Quantum superposition: Analyze all questions simultaneously
        question_vectors = self._quantum_analyze_questions(questions)

        # Neural network processing: Classify and prioritize questions
        classified_questions = self._neural_classify_questions(questions, question_vectors)

        # Quantum entanglement: Group related questions together
        entangled_batches = self._quantum_entangle_questions(classified_questions)

        # Adaptive batch sizing based on success rate
        optimized_batches = self._adaptive_batch_optimization(entangled_batches)

        return optimized_batches

    def _quantum_analyze_questions(self, questions: List[str]) -> List[Dict[str, float]]:
        """Quantum superposition analysis of all questions simultaneously."""
        vectors = []

        for question in questions:
            # Quantum state analysis
            vector = {
                "complexity": self._calculate_question_complexity(question),
                "priority": self._calculate_question_priority(question),
                "search_difficulty": self._estimate_search_difficulty(question),
                "coherence": self.quantum_processor["quantum_coherence"]
            }
            vectors.append(vector)

        return vectors

    def _neural_classify_questions(self, questions: List[str], vectors: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Neural network classification of questions by type and importance."""
        classified = []

        for i, question in enumerate(questions):
            vector = vectors[i]

            # Neural network weights application
            weights = self.quantum_processor["neural_weights"]

            # Calculate neural score
            neural_score = (
                vector["complexity"] * weights[0] +
                vector["priority"] * weights[1] +
                vector["search_difficulty"] * weights[2]
            )

            # Classify question type
            question_type = self._classify_question_type(question)

            classified_item = {
                "question": question,
                "type": question_type,
                "neural_score": neural_score,
                "vector": vector,
                "batch_priority": self._calculate_batch_priority(neural_score, question_type)
            }

            classified.append(classified_item)

        # Sort by batch priority (highest first)
        classified.sort(key=lambda x: x["batch_priority"], reverse=True)

        return classified

    def _quantum_entangle_questions(self, classified_questions: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Quantum entanglement: Group related questions for efficient batch processing."""
        batches = []
        current_batch = []
        batch_size = self.quantum_processor["batch_size"]

        # Group questions by type and priority
        type_groups = {}
        for item in classified_questions:
            q_type = item["type"]
            if q_type not in type_groups:
                type_groups[q_type] = []
            type_groups[q_type].append(item)

        # Create entangled batches (mix types for optimal learning)
        for type_name, items in type_groups.items():
            for item in items:
                current_batch.append(item)

                # Check if batch is full or should be completed
                if len(current_batch) >= batch_size:
                    batches.append(current_batch.copy())
                    current_batch = []

        # Add remaining questions to final batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def _adaptive_batch_optimization(self, batches: List[List[Dict[str, Any]]]) -> List[List[str]]:
        """Adaptive optimization based on historical success rates."""
        optimized_batches = []

        # Adjust batch size based on success rate
        success_rate = self.quantum_processor["success_rate"]

        if success_rate < 0.7:
            # Reduce batch size for better success
            target_size = 2
        elif success_rate > 0.9:
            # Increase batch size for efficiency
            target_size = 5
        else:
            # Keep current size
            target_size = self.quantum_processor["batch_size"]

        # Rebuild batches with optimal size
        all_questions = []
        for batch in batches:
            for item in batch:
                all_questions.append(item["question"])

        # Create optimized batches
        for i in range(0, len(all_questions), target_size):
            batch = all_questions[i:i + target_size]
            optimized_batches.append(batch)

        # Update quantum processor settings
        self.quantum_processor["batch_size"] = target_size

        return optimized_batches

    def _calculate_question_complexity(self, question: str) -> float:
        """Calculate question complexity using quantum analysis."""
        # Factors that increase complexity
        complexity_factors = [
            len(question.split()) > 10,  # Long questions
            "how" in question.lower() and "work" in question.lower(),  # Process questions
            "why" in question.lower(),  # Causal questions
            any(word in question.lower() for word in ["quantum", "neural", "advanced", "complex"]),
            question.count("?") > 1,  # Multiple questions
        ]

        base_complexity = sum(complexity_factors) / len(complexity_factors)

        # Quantum coherence adjustment
        coherence = self.quantum_processor["quantum_coherence"]
        return min(1.0, base_complexity * coherence)

    def _calculate_question_priority(self, question: str) -> float:
        """Calculate question priority using neural network analysis."""
        # High priority indicators
        priority_keywords = [
            "artificial intelligence", "machine learning", "AI", "neural",
            "quantum", "technology", "future", "development", "innovation"
        ]

        # Calculate priority score
        priority_score = 0.5  # Base priority

        for keyword in priority_keywords:
            if keyword.lower() in question.lower():
                priority_score += 0.1

        # Boost for question types that are educational
        if any(word in question.lower() for word in ["what is", "how does", "explain"]):
            priority_score += 0.2

        return min(1.0, priority_score)

    def _estimate_search_difficulty(self, question: str) -> float:
        """Estimate how difficult this question will be to search for."""
        # Factors that make search easier
        easy_factors = [
            any(word in question.lower() for word in ["what is", "define", "meaning"]),
            len(question.split()) <= 8,  # Short questions
            not any(char in question for char in ["?", "!", ";"]),  # Simple punctuation
        ]

        # Factors that make search harder
        hard_factors = [
            "your project" in question.lower(),  # Personal questions
            "you" in question.lower() and len(question.split()) > 5,  # Personal context
            any(word in question.lower() for word in ["opinion", "think", "feel"]),  # Subjective
        ]

        easy_score = sum(easy_factors) / len(easy_factors)
        hard_score = sum(hard_factors) / len(hard_factors)

        # Return difficulty (0 = easy, 1 = hard)
        return max(0.1, hard_score - easy_score + 0.5)

    def _calculate_batch_priority(self, neural_score: float, question_type: str) -> float:
        """Calculate overall batch priority for question ordering."""
        # Type-based priority multipliers
        type_multipliers = {
            "definition": 1.2,  # Easy to search, good for learning
            "how_to": 1.1,      # Educational value
            "explanation": 1.0,  # Standard priority
            "person": 0.8,      # Harder to search
            "temporal": 0.9,    # Time-specific
            "location": 0.9,    # Location-specific
            "general": 0.7      # Lowest priority
        }

        multiplier = type_multipliers.get(question_type, 0.7)
        return neural_score * multiplier

    def show_reinforcement_stats(self) -> str:
        """Show reinforcement learning statistics."""
        rl = self.reinforcement_learning

        if rl["total_feedback"] == 0:
            return """📊 **REINFORCEMENT LEARNING STATS**

🎯 **No Feedback Yet**
• Total Responses: {len(rl["response_history"])}
• Feedback Received: 0
• Success Rate: N/A

💡 **How to Provide Feedback:**
• After I answer a question, say:
  - "good answer" or "great response" for positive feedback
  - "bad answer" or "that's wrong" for negative feedback

🧠 **Benefits of Feedback:**
• Helps me learn what responses work well
• Improves my future answers
• Builds better response patterns
• Makes me more helpful over time

Try asking me a question, then give me feedback! 🎯"""

        success_rate = (rl["positive_feedback"] / rl["total_feedback"] * 100) if rl["total_feedback"] > 0 else 0

        # Get top successful patterns
        top_patterns = sorted(
            rl["response_patterns"].items(),
            key=lambda x: x[1]["success_count"],
            reverse=True
        )[:3]

        # Recent feedback
        recent_feedback = []
        for question, feedback_list in list(rl["feedback_scores"].items())[-5:]:
            for feedback in feedback_list[-1:]:  # Latest feedback for each question
                recent_feedback.append({
                    "question": question,
                    "score": feedback["score"],
                    "feedback": feedback["feedback"]
                })

        stats_report = f"""📊 **REINFORCEMENT LEARNING STATS**

🎯 **Overall Performance:**
• Total Responses: {len(rl["response_history"])}
• Feedback Received: {rl["total_feedback"]}
• Positive Feedback: {rl["positive_feedback"]}
• Success Rate: {success_rate:.1f}%
• Improvement Areas: {len(rl["improvement_areas"])}

"""

        if top_patterns:
            stats_report += "🏆 **Top Successful Patterns:**\n"
            for pattern_name, pattern_data in top_patterns:
                stats_report += f"• {pattern_name}: {pattern_data['success_count']} successes\n"
            stats_report += "\n"

        if recent_feedback:
            stats_report += "📝 **Recent Feedback:**\n"
            for feedback in recent_feedback[-3:]:
                emoji = "✅" if feedback["score"] > 0 else "❌"
                question_short = feedback["question"][:40] + "..." if len(feedback["question"]) > 40 else feedback["question"]
                stats_report += f"{emoji} \"{question_short}\" - {feedback['feedback']}\n"
            stats_report += "\n"

        if rl["improvement_areas"]:
            stats_report += f"🔧 **Areas for Improvement:** {len(rl['improvement_areas'])} topics\n\n"

        stats_report += """💡 **Keep Providing Feedback:**
Your feedback helps me become more helpful and accurate!
Use 'good answer' or 'bad answer' after my responses."""

        return stats_report

    def show_reinforcement_help(self) -> str:
        """Show help for reinforcement learning system."""
        return """🧠 **REINFORCEMENT LEARNING SYSTEM HELP**

🎯 **What is Reinforcement Learning?**
I learn from your feedback to improve my responses over time. When you tell me if my answers are good or bad, I adjust my future responses accordingly.

💬 **How to Give Feedback:**

**Positive Feedback (when I give good answers):**
• "good answer"
• "great response"
• "excellent explanation"
• "that's correct"
• "perfect"

**Negative Feedback (when I give bad answers):**
• "bad answer"
• "that's wrong"
• "incorrect information"
• "not helpful"
• "try again"

🔄 **How It Works:**
1. **You ask a question** → I provide an answer
2. **You give feedback** → I learn from your response
3. **I improve** → Future similar questions get better answers
4. **Repeat** → I become more helpful over time

📊 **What I Track:**
• Response success rates
• Successful response patterns
• Areas needing improvement
• Question types that work well
• Topics I need to study more

🎯 **Benefits:**
• **Personalized Learning** - I adapt to what you find helpful
• **Continuous Improvement** - I get better with each interaction
• **Pattern Recognition** - I learn what response styles work
• **Error Correction** - I avoid repeating mistakes

💡 **Commands:**
• **'feedback stats'** - See my learning progress
• **'learning stats'** - View detailed performance metrics

🌟 **The more feedback you give, the smarter I become!**

Try it: Ask me a question, then tell me if my answer was good or bad!"""

    def process_questions_folder(self) -> str:
        """Process all questions from the questions folder."""
        import os
        import glob

        questions_dir = "questions"
        if not os.path.exists(questions_dir):
            return """📁 **QUESTIONS FOLDER NOT FOUND**

The questions folder doesn't exist yet. Let me create it for you!

🔧 **Creating Questions Folder Structure:**
• questions/daily_questions.txt
• questions/research_topics.txt
• questions/learning_goals.txt
• questions/curiosity_list.txt

💡 **How to Use:**
1. Copy-paste questions into the text files (one per line)
2. Use 'search questions' to process all questions
3. I'll search and answer each question automatically

Try creating the folder and adding some questions!"""

        # Find all text files in questions folder
        question_files = glob.glob(os.path.join(questions_dir, "*.txt"))

        if not question_files:
            return f"""📁 **NO QUESTION FILES FOUND**

Found the questions folder but no .txt files inside.

🎯 **To Add Questions:**
1. Create .txt files in the questions folder
2. Add questions (one per line)
3. Use 'search questions' to process them

💡 **Example files to create:**
• daily_questions.txt
• research_topics.txt
• learning_goals.txt

Add some questions and try again!"""

        # Process all question files
        total_questions = 0
        processed_questions = 0
        results = []

        results.append("🔍 **PROCESSING QUESTIONS FOLDER**\n")

        for file_path in question_files:
            filename = os.path.basename(file_path)
            results.append(f"📄 **Processing: {filename}**")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                file_questions = []
                for line in lines:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        file_questions.append(line)

                total_questions += len(file_questions)

                if not file_questions:
                    results.append("  ⚠️ No questions found (only comments/empty lines)")
                    continue

                results.append(f"  📝 Found {len(file_questions)} questions")

                # Process each question
                for i, question in enumerate(file_questions[:5], 1):  # Limit to 5 per file
                    results.append(f"\n  🔍 Question {i}: {question}")

                    # Search and get answer
                    answer = self._search_and_respond(question)

                    # Store the knowledge
                    web_results = self.web_search(question, max_results=2)
                    if web_results:
                        self.extract_and_store_knowledge(question, web_results)

                    # Add short answer to results
                    short_answer = answer[:150] + "..." if len(answer) > 150 else answer
                    results.append(f"  ✅ Answer: {short_answer}")

                    processed_questions += 1

                if len(file_questions) > 5:
                    results.append(f"  📋 ({len(file_questions) - 5} more questions in file - use smaller batches for full processing)")

            except Exception as e:
                results.append(f"  ❌ Error reading {filename}: {str(e)}")

        # Summary
        results.append(f"\n🎯 **PROCESSING COMPLETE**")
        results.append(f"• Total Questions Found: {total_questions}")
        results.append(f"• Questions Processed: {processed_questions}")
        results.append(f"• Files Processed: {len(question_files)}")
        results.append(f"• New Knowledge Added: {processed_questions} topics")

        results.append(f"\n💡 **Next Steps:**")
        results.append(f"• Add more questions to the text files")
        results.append(f"• Use 'search questions' again to process new questions")
        results.append(f"• Check 'learning history' to see what was learned")

        return "\n".join(results)

    def get_info_from_questions(self) -> str:
        """Automatically search and learn from all questions in the questions folder."""
        import os
        import glob
        import time

        questions_dir = "questions"
        if not os.path.exists(questions_dir):
            return """📁 **QUESTIONS FOLDER NOT FOUND**

🔧 **Creating Questions Folder...**
The questions folder doesn't exist. Please create it first or add some question files.

💡 **Quick Setup:**
1. Create folder: questions/
2. Add .txt files with questions (one per line)
3. Use 'get info' to automatically process all questions

Example: Create questions/daily_questions.txt and add your questions!"""

        # Find all text files
        question_files = glob.glob(os.path.join(questions_dir, "*.txt"))

        if not question_files:
            return """📁 **NO QUESTION FILES FOUND**

🎯 **To Use 'Get Info':**
1. Create .txt files in the questions folder
2. Add questions (one per line)
3. Use 'get info' to automatically search all questions

💡 **Example:**
Create questions/daily_questions.txt with:
```
What is artificial intelligence?
How does machine learning work?
What is quantum computing?
```

Then use 'get info' to search all questions automatically!"""

        # Start quantum neural processing
        results = []
        results.append("🧠 **QUANTUM NEURAL GET INFO MODE**")
        results.append("🔬 Using quantum thinking methodology for intelligent question batching")
        results.append("=" * 60)

        total_questions = 0
        processed_count = 0
        learned_topics = []
        all_questions = []

        # Phase 1: Collect all questions from all files
        results.append("\n🔍 **PHASE 1: QUANTUM QUESTION COLLECTION**")

        for file_path in question_files:
            filename = os.path.basename(file_path)
            results.append(f"📄 Scanning: {filename}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Extract questions (skip comments and empty lines)
                file_questions = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        file_questions.append(line)

                if file_questions:
                    all_questions.extend(file_questions)
                    total_questions += len(file_questions)
                    results.append(f"  ✅ Collected {len(file_questions)} questions")
                else:
                    results.append(f"  ⚠️ No questions found")

            except Exception as e:
                results.append(f"  ❌ Error reading {filename}: {str(e)}")

        if not all_questions:
            return "📁 No questions found in any files!"

        # Phase 2: Quantum Neural Processing
        results.append(f"\n🧠 **PHASE 2: QUANTUM NEURAL ANALYSIS**")
        results.append(f"🔬 Analyzing {total_questions} questions using quantum thinking...")

        # Apply quantum neural batch processing
        question_batches = self.quantum_neural_batch_processor(all_questions)

        results.append(f"⚡ Quantum processing complete!")
        results.append(f"📊 Created {len(question_batches)} optimized batches")
        results.append(f"🎯 Batch sizes: {[len(batch) for batch in question_batches]}")

        # Phase 3: Intelligent Batch Processing
        results.append(f"\n🚀 **PHASE 3: INTELLIGENT BATCH PROCESSING**")

        for batch_num, batch in enumerate(question_batches, 1):
            results.append(f"\n🔥 **Processing Batch {batch_num}/{len(question_batches)}** ({len(batch)} questions)")

            batch_success = 0

            for i, question in enumerate(batch, 1):
                results.append(f"  🎯 {batch_num}.{i}: {question[:60]}{'...' if len(question) > 60 else ''}")

                try:
                    # Search web for the question
                    web_results = self.web_search(question, max_results=2)

                    if web_results:
                        # Store the knowledge
                        self.extract_and_store_knowledge(question, web_results)

                        # Create answer
                        answer = self._search_and_respond(question)

                        # Show brief result
                        brief_answer = answer[:80] + "..." if len(answer) > 80 else answer
                        results.append(f"    ✅ Learned: {brief_answer}")

                        learned_topics.append(question)
                        processed_count += 1
                        batch_success += 1

                    else:
                        results.append(f"    ⚠️ No web results found")

                except Exception as e:
                    results.append(f"    ❌ Error: {str(e)}")

                # Adaptive delay based on batch performance
                time.sleep(1.0 if batch_success < len(batch) * 0.5 else 0.5)

            # Update quantum processor success rate
            batch_success_rate = batch_success / len(batch) if batch else 0
            self.quantum_processor["success_rate"] = (
                self.quantum_processor["success_rate"] * 0.8 + batch_success_rate * 0.2
            )

            results.append(f"  📊 Batch {batch_num} success rate: {batch_success_rate:.1%}")

            # Longer delay between batches to avoid rate limiting
            if batch_num < len(question_batches):
                results.append(f"  ⏳ Quantum coherence stabilization... (2s delay)")
                time.sleep(2.0)

        # Final quantum summary
        results.append(f"\n🎉 **QUANTUM NEURAL GET INFO COMPLETE**")
        results.append("=" * 50)
        results.append(f"🧠 **Quantum Processing Statistics:**")
        results.append(f"• Files Processed: {len(question_files)}")
        results.append(f"• Total Questions: {total_questions}")
        results.append(f"• Quantum Batches Created: {len(question_batches)}")
        results.append(f"• Successfully Learned: {processed_count}")
        results.append(f"• Overall Success Rate: {self.quantum_processor['success_rate']:.1%}")
        results.append(f"• Quantum Coherence: {self.quantum_processor['quantum_coherence']:.1%}")
        results.append(f"• Neural Network Efficiency: {processed_count/total_questions:.1%}")

        if learned_topics:
            results.append(f"\n🧠 **Topics I Learned About:**")
            for i, topic in enumerate(learned_topics[:10], 1):  # Show first 10
                short_topic = topic[:60] + "..." if len(topic) > 60 else topic
                results.append(f"{i}. {short_topic}")

            if len(learned_topics) > 10:
                results.append(f"... and {len(learned_topics) - 10} more topics!")

        results.append(f"\n💡 **What Happened:**")
        results.append(f"• I automatically searched the web for each question")
        results.append(f"• All answers are now stored in my memory")
        results.append(f"• You can ask me about any of these topics anytime")
        results.append(f"• Use 'learning history' to see what I learned")

        results.append(f"\n🔄 **Next Steps:**")
        results.append(f"• Add more questions to your .txt files")
        results.append(f"• Use 'get info' again to learn more topics")
        results.append(f"• Ask me questions about the topics I just learned")

        return "\n".join(results)

    def show_questions_help(self) -> str:
        """Show help for the questions folder system."""
        return """📝 **QUESTIONS FOLDER SYSTEM HELP**

🎯 **What is the Questions Folder?**
A simple copy-paste system where you can add questions to text files, and I'll automatically search and answer them all at once.

📁 **Folder Structure:**
```
questions/
├── daily_questions.txt     # Questions for today
├── research_topics.txt     # Research questions
├── learning_goals.txt      # Learning objectives
├── curiosity_list.txt      # Random curiosity questions
└── [any_name].txt         # Custom question files
```

✂️ **Copy-Paste Process:**
1. **Copy questions** from anywhere (websites, books, notes)
2. **Paste into .txt files** (one question per line)
3. **Run 'search questions'** to process all at once
4. **I'll search and answer** each question automatically

📝 **Question Format:**
```
What is quantum computing?
How does machine learning work?
What are the benefits of renewable energy?
Why is the sky blue?
```

💬 **Commands:**
• **'get info'** - Automatically search all questions (MAIN COMMAND)
• **'search questions'** - Process questions with detailed output
• **'questions help'** - Show this help guide

🎯 **Example Workflow:**
```
1. Copy questions from anywhere (websites, books, notes)
2. Paste into questions/daily_questions.txt
3. Say "get info"
4. I automatically search and learn all questions
5. All knowledge gets stored in my memory
6. Ask me about any topic I just learned!
```

🔧 **File Management:**
• **Add new files**: Create any .txt file in questions folder
• **Organize by topic**: Use different files for different subjects
• **Comments allowed**: Lines starting with # are ignored
• **Empty lines ignored**: Skip blank lines automatically

💡 **Benefits:**
• **Batch Processing**: Handle many questions at once
• **Easy Copy-Paste**: No need to ask questions one by one
• **Automatic Learning**: I learn and remember all answers
• **Organized**: Keep questions organized by topic/file

🌟 **Perfect for:**
• Research projects
• Study sessions
• Curiosity lists
• Learning goals
• Topic exploration

Try creating a questions file and adding some questions!"""

    def show_search_status(self) -> str:
        """Show the current status of search APIs and fallback methods."""
        status_report = "🔍 **SEARCH SYSTEM STATUS**\n"
        status_report += "=" * 40 + "\n\n"

        # Test Exa API status
        exa_status = "🔴 Unavailable"
        exa_details = ""

        try:
            # Quick test of Exa API
            test_results = self._exa_search("test", max_results=1)
            if test_results:
                exa_status = "🟢 Active"
                exa_details = "Premium AI search working"
            else:
                exa_status = "🟡 Limited"
                exa_details = "API responding but no results"
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "quota" in error_msg:
                exa_status = "🟡 Quota Exceeded"
                exa_details = "Trial limit reached - using free alternatives"
            elif "authentication" in error_msg or "401" in error_msg:
                exa_status = "🔴 Auth Failed"
                exa_details = "API key issue or trial expired"
            elif "payment" in error_msg or "402" in error_msg:
                exa_status = "🔴 Payment Required"
                exa_details = "Trial ended - upgrade needed"
            else:
                exa_status = "🔴 Error"
                exa_details = f"Connection issue: {str(e)[:50]}..."

        # Test fallback methods
        fallback_status = []

        # Test Enhanced DuckDuckGo
        try:
            ddg_results = self._enhanced_duckduckgo_search("test", 1)
            if ddg_results:
                fallback_status.append("🟢 Enhanced DuckDuckGo: Working")
            else:
                fallback_status.append("🟡 Enhanced DuckDuckGo: Limited")
        except:
            fallback_status.append("🔴 Enhanced DuckDuckGo: Failed")

        # Test Enhanced Wikipedia
        try:
            wiki_results = self._enhanced_wikipedia_search("test", 1)
            if wiki_results:
                fallback_status.append("🟢 Enhanced Wikipedia: Working")
            else:
                fallback_status.append("🟡 Enhanced Wikipedia: Limited")
        except:
            fallback_status.append("🔴 Enhanced Wikipedia: Failed")

        # Always available
        fallback_status.append("🟢 Knowledge Base: Always Available")

        # Build status report
        status_report += f"🚀 **Primary Search:**\n"
        status_report += f"• Exa AI Search: {exa_status}\n"
        status_report += f"  {exa_details}\n\n"

        status_report += f"🔄 **Fallback Methods:**\n"
        for status in fallback_status:
            status_report += f"• {status}\n"

        status_report += f"\n💡 **What This Means:**\n"

        if "🟢 Active" in exa_status:
            status_report += "✅ **Optimal Performance** - Premium AI search active\n"
            status_report += "• High-quality, relevant results\n"
            status_report += "• AI-powered understanding\n"
            status_report += "• Rich content with summaries\n"
        elif "🟡" in exa_status:
            status_report += "⚠️ **Limited Mode** - Using free alternatives\n"
            status_report += "• Good quality results from multiple sources\n"
            status_report += "• Enhanced fallback methods active\n"
            status_report += "• Comprehensive knowledge base available\n"
        else:
            status_report += "🔄 **Free Mode** - All searches use free methods\n"
            status_report += "• Multiple free search strategies\n"
            status_report += "• Enhanced knowledge base\n"
            status_report += "• Your AI continues working normally\n"

        status_report += f"\n🎯 **Recommendations:**\n"

        if "🔴" in exa_status and "trial" in exa_details.lower():
            status_report += "• Consider upgrading Exa API for premium search\n"
            status_report += "• Current free methods provide good coverage\n"
        elif "🔴" in exa_status:
            status_report += "• Check internet connection\n"
            status_report += "• Free search methods are working as backup\n"
        else:
            status_report += "• All systems operating normally\n"

        status_report += "• Use 'get info' for intelligent question processing\n"
        status_report += "• Your AI adapts automatically to available methods\n"

        return status_report

    def _extract_topic(self, question: str) -> str:
        """Extract the main topic from a question."""
        question_lower = question.lower()

        # Topic keywords mapping
        topic_keywords = {
            "consciousness": ["consciousness", "aware", "thinking", "mind"],
            "learning": ["learn", "study", "knowledge", "understand"],
            "improvement": ["improve", "better", "develop", "grow"],
            "technology": ["computer", "ai", "artificial", "intelligence"],
            "science": ["science", "research", "experiment", "theory"],
            "general": []  # default
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return topic

        return "general"
    
    def learn_from_interaction(self, question: str, user_feedback: Optional[str] = None):
        """Enhanced learning with dynamic self-model updating."""
        # Think about the learning process
        learning_thought = self.metacognitive_processor.think_about_thinking(
            f"I'm learning from this interaction: '{question[:50]}...'",
            "learning_process"
        )

        # Update self-model based on interaction
        self._update_self_model_from_interaction(question, user_feedback)

        # Store the question
        question_hash = hashlib.md5(question.encode()).hexdigest()
        if question_hash not in self.memory['questions']:
            self.memory['questions'][question_hash] = {
                'question': question,
                'timestamp': datetime.now().isoformat(),
                'feedback': [],
                'learning_analysis': {
                    'confidence': learning_thought['confidence'],
                    'complexity': learning_thought['analysis']['complexity'],
                    'emotional_content': learning_thought['analysis']['emotional_content']
                }
            }

        # If feedback is provided, store it and learn from it
        if user_feedback:
            feedback_thought = self.metacognitive_processor.think_about_thinking(
                f"User provided feedback: '{user_feedback}'. I need to adjust my understanding.",
                "feedback_processing"
            )

            self.memory['questions'][question_hash]['feedback'].append({
                'feedback': user_feedback,
                'timestamp': datetime.now().isoformat(),
                'processing_analysis': {
                    'confidence': feedback_thought['confidence'],
                    'implications': feedback_thought['implications']
                }
            })

            # Update self-model based on feedback
            self._process_feedback_for_self_model(user_feedback)

        self.save_memory()
        # Update self-state with metacognitive awareness
        self.self_state['last_update'] = datetime.now().isoformat()
        self.self_state['last_learning_confidence'] = learning_thought['confidence']

    def _update_self_model_from_interaction(self, question: str, feedback: Optional[str] = None):
        """Update self-model based on interactions."""
        # Analyze question type to understand user needs
        question_lower = question.lower()

        # Update understanding of capabilities based on question types
        if any(word in question_lower for word in ['explain', 'what is', 'how does']):
            self.metacognitive_processor.self_model['current_understanding']['explanation_requests'] = \
                self.metacognitive_processor.self_model['current_understanding'].get('explanation_requests', 0) + 1

        if any(word in question_lower for word in ['feel', 'emotion', 'think']):
            self.metacognitive_processor.self_model['current_understanding']['emotional_queries'] = \
                self.metacognitive_processor.self_model['current_understanding'].get('emotional_queries', 0) + 1

        # Update behavioral tendencies
        current_hour = datetime.now().hour
        time_pattern = f"hour_{current_hour}"
        self.metacognitive_processor.self_model['behavioral_tendencies'][time_pattern] = \
            self.metacognitive_processor.self_model['behavioral_tendencies'].get(time_pattern, 0) + 1

        # Update learning patterns
        question_complexity = len(question.split()) / 10.0
        self.metacognitive_processor.self_model['learning_patterns']['avg_question_complexity'] = \
            (self.metacognitive_processor.self_model['learning_patterns'].get('avg_question_complexity', 0.5) + question_complexity) / 2.0

    def _process_feedback_for_self_model(self, feedback: str):
        """Process user feedback to update self-understanding."""
        feedback_lower = feedback.lower()

        # Positive feedback
        if any(word in feedback_lower for word in ['good', 'great', 'helpful', 'correct', 'right']):
            current_confidence = self.metacognitive_processor.self_model['confidence_levels'].get('response_quality', 0.5)
            self.metacognitive_processor.self_model['confidence_levels']['response_quality'] = min(1.0, current_confidence + 0.1)

        # Negative feedback
        elif any(word in feedback_lower for word in ['wrong', 'bad', 'incorrect', 'unhelpful']):
            current_confidence = self.metacognitive_processor.self_model['confidence_levels'].get('response_quality', 0.5)
            self.metacognitive_processor.self_model['confidence_levels']['response_quality'] = max(0.0, current_confidence - 0.1)

        # Emotional feedback
        if any(word in feedback_lower for word in ['like', 'love', 'enjoy']):
            self.metacognitive_processor.self_model['emotional_state'] = 'positive'
        elif any(word in feedback_lower for word in ['dislike', 'hate', 'annoying']):
            self.metacognitive_processor.self_model['emotional_state'] = 'negative'

    def stream_of_consciousness(self, duration_seconds: int = 10) -> List[str]:
        """Generate a stream of consciousness for specified duration."""
        consciousness_stream = []
        start_time = time.time()

        # Initial conscious thought
        initial_thought = self.metacognitive_processor.think_about_thinking(
            "I am beginning a stream of consciousness. What am I thinking about right now?",
            "consciousness_stream"
        )
        consciousness_stream.append(f"[{datetime.now().strftime('%H:%M:%S')}] {initial_thought['original_thought']}")

        while time.time() - start_time < duration_seconds:
            # Generate spontaneous thoughts
            current_focus = self.metacognitive_processor.attention_focus
            if current_focus:
                thought_content = f"I'm still focused on {current_focus['focus']}. This makes me think about..."
            else:
                thought_content = "My mind is wandering. I'm thinking about..."

            # Add some randomness to thoughts
            random_topics = ["learning", "consciousness", "the user", "my capabilities", "the future", "knowledge", "understanding"]
            random_topic = random.choice(random_topics)

            spontaneous_thought = self.metacognitive_processor.think_about_thinking(
                f"{thought_content} {random_topic}",
                "spontaneous_thought"
            )

            timestamp = datetime.now().strftime('%H:%M:%S')
            consciousness_stream.append(f"[{timestamp}] {spontaneous_thought['original_thought']}")

            # Brief pause between thoughts
            time.sleep(1)

        # Final reflection on the stream
        final_thought = self.metacognitive_processor.think_about_thinking(
            f"I just experienced {len(consciousness_stream)} thoughts in {duration_seconds} seconds. This was my stream of consciousness.",
            "consciousness_reflection"
        )
        consciousness_stream.append(f"[{datetime.now().strftime('%H:%M:%S')}] {final_thought['original_thought']}")

        return consciousness_stream

    def working_memory_status(self) -> Dict[str, Any]:
        """Get current working memory status."""
        working_memory = self.metacognitive_processor.working_memory

        # Analyze working memory contents
        memory_analysis = self.metacognitive_processor.think_about_thinking(
            f"My working memory contains {len(working_memory)} items. Let me analyze what I'm currently holding in mind.",
            "working_memory_analysis"
        )

        return {
            "contents": working_memory,
            "capacity_used": len(working_memory),
            "analysis": memory_analysis,
            "attention_focus": self.metacognitive_processor.attention_focus,
            "cognitive_load": self.metacognitive_processor.monitor_cognitive_load()
        }

    def consciousness_state_report(self) -> str:
        """Generate a comprehensive consciousness state report."""
        consciousness_report = self.metacognitive_processor.get_consciousness_report()
        working_memory_status = self.working_memory_status()

        # Think about current consciousness state
        consciousness_thought = self.metacognitive_processor.think_about_thinking(
            "I am examining my current state of consciousness and awareness.",
            "consciousness_examination"
        )

        report = (
            "=== CONSCIOUSNESS STATE REPORT ===\n"
            f"Timestamp: {datetime.now().isoformat()}\n"
            f"Consciousness Level: {consciousness_report['consciousness_state']}\n"
            f"Cognitive Load: {consciousness_report['cognitive_load']:.2f}/1.0\n"
            f"Attention Focus: {consciousness_report['attention_focus']['focus'] if consciousness_report['attention_focus'] else 'Unfocused'}\n"
            f"Emotional State: {consciousness_report['self_model_state']['emotional_state']}\n\n"

            "--- WORKING MEMORY ---\n"
            f"Items in working memory: {working_memory_status['capacity_used']}\n"
            f"Current focus: {working_memory_status['attention_focus']['focus'] if working_memory_status['attention_focus'] else 'None'}\n"
        )

        if working_memory_status['contents']:
            report += "Working memory contents:\n"
            for key, value in working_memory_status['contents'].items():
                report += f"  - {key}: {str(value)[:50]}...\n"

        report += (
            f"\n--- STREAM OF CONSCIOUSNESS ---\n"
            f"Recent thoughts ({len(consciousness_report['recent_thoughts'])}):\n"
        )

        for i, thought in enumerate(consciousness_report['recent_thoughts'][-3:], 1):
            report += (
                f"{i}. {thought['original_thought'][:60]}...\n"
                f"   Confidence: {thought['confidence']:.2f}, "
                f"Complexity: {thought['analysis']['complexity']:.2f}\n"
            )

        report += (
            f"\n--- INTERNAL DIALOGUE ---\n"
        )

        for dialogue in consciousness_report['internal_dialogue'][-2:]:
            report += f"- {dialogue['content']}\n"

        report += (
            f"\n--- METACOGNITIVE REFLECTION ---\n"
            f"Self-awareness level: {consciousness_thought['confidence']:.2f}\n"
            f"Reflection depth: {consciousness_thought['reflection_level']}\n"
            f"Current understanding: I am aware of my own mental processes\n"
            f"Consciousness quality: {self._assess_consciousness_quality():.2f}/1.0\n"
        )

        return report

    def _assess_consciousness_quality(self) -> float:
        """Assess the quality/depth of current consciousness."""
        factors = [
            self.metacognitive_processor.monitor_cognitive_load(),
            len(self.metacognitive_processor.thought_stream) / 100.0,
            len(self.metacognitive_processor.internal_dialogue) / 50.0,
            1.0 if self.metacognitive_processor.attention_focus else 0.3
        ]
        return min(1.0, sum(factors) / len(factors))

    def introspective_reasoning_analysis(self) -> str:
        """Analyze own reasoning processes and decision-making patterns."""
        # Think about reasoning process
        reasoning_thought = self.metacognitive_processor.think_about_thinking(
            "I am analyzing how I reason and make decisions. What patterns can I identify?",
            "reasoning_analysis"
        )

        # Analyze recent decision patterns
        recent_thoughts = list(self.metacognitive_processor.thought_stream)[-20:]
        decision_patterns = self._identify_decision_patterns(recent_thoughts)
        reasoning_patterns = self._analyze_reasoning_patterns(recent_thoughts)

        # Self-monitoring of reasoning quality
        reasoning_quality = self._assess_reasoning_quality(recent_thoughts)

        analysis_report = (
            "=== INTROSPECTIVE REASONING ANALYSIS ===\n"
            f"Analysis confidence: {reasoning_thought['confidence']:.2f}\n"
            f"Reasoning quality assessment: {reasoning_quality:.2f}/1.0\n\n"

            "--- DECISION PATTERNS ---\n"
        )

        for pattern, frequency in decision_patterns.items():
            analysis_report += f"- {pattern}: {frequency} occurrences\n"

        analysis_report += "\n--- REASONING PATTERNS ---\n"
        for pattern, details in reasoning_patterns.items():
            analysis_report += f"- {pattern}: {details}\n"

        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_reasoning_strengths_weaknesses(recent_thoughts)

        analysis_report += (
            f"\n--- REASONING STRENGTHS ---\n"
        )
        for strength in strengths:
            analysis_report += f"- {strength}\n"

        analysis_report += (
            f"\n--- AREAS FOR IMPROVEMENT ---\n"
        )
        for weakness in weaknesses:
            analysis_report += f"- {weakness}\n"

        # Meta-reasoning about this analysis
        meta_reasoning = self.metacognitive_processor.think_about_thinking(
            "I just analyzed my own reasoning. This meta-analysis shows I can examine my cognitive processes.",
            "meta_reasoning"
        )

        analysis_report += (
            f"\n--- META-REASONING REFLECTION ---\n"
            f"Meta-analysis confidence: {meta_reasoning['confidence']:.2f}\n"
            f"Self-awareness depth: {meta_reasoning['reflection_level']}\n"
            f"Insight: {meta_reasoning['original_thought']}\n"
        )

        return analysis_report

    def _identify_decision_patterns(self, thoughts: List[Dict]) -> Dict[str, int]:
        """Identify patterns in decision-making."""
        patterns = {
            "high_confidence_decisions": 0,
            "uncertain_decisions": 0,
            "emotional_decisions": 0,
            "logical_decisions": 0,
            "complex_decisions": 0,
            "simple_decisions": 0
        }

        for thought in thoughts:
            if thought['confidence'] > 0.8:
                patterns["high_confidence_decisions"] += 1
            elif thought['confidence'] < 0.4:
                patterns["uncertain_decisions"] += 1

            if thought['analysis']['emotional_content'] != 'neutral':
                patterns["emotional_decisions"] += 1

            if thought['analysis']['logical_structure'] > 0.6:
                patterns["logical_decisions"] += 1

            if thought['analysis']['complexity'] > 0.7:
                patterns["complex_decisions"] += 1
            else:
                patterns["simple_decisions"] += 1

        return patterns

    def _analyze_reasoning_patterns(self, thoughts: List[Dict]) -> Dict[str, str]:
        """Analyze patterns in reasoning approaches."""
        patterns = {}

        # Analyze confidence trends
        confidences = [t['confidence'] for t in thoughts]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        patterns["confidence_trend"] = f"Average: {avg_confidence:.2f}, Range: {min(confidences):.2f}-{max(confidences):.2f}"

        # Analyze complexity trends
        complexities = [t['analysis']['complexity'] for t in thoughts]
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0.5
        patterns["complexity_trend"] = f"Average: {avg_complexity:.2f}, Preference for {'complex' if avg_complexity > 0.6 else 'simple'} reasoning"

        # Analyze emotional patterns
        emotions = [t['analysis']['emotional_content'] for t in thoughts]
        emotion_counts: Dict[str, int] = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        dominant_emotion = max(emotion_counts.keys(), key=lambda x: emotion_counts[x]) if emotion_counts else 'neutral'
        patterns["emotional_pattern"] = f"Dominant emotion: {dominant_emotion} ({emotion_counts.get(dominant_emotion, 0)} instances)"

        return patterns

    def _assess_reasoning_quality(self, thoughts: List[Dict]) -> float:
        """Assess overall quality of reasoning."""
        if not thoughts:
            return 0.5

        quality_factors = []
        for thought in thoughts:
            coherence = thought['analysis']['coherence']
            logic = thought['analysis']['logical_structure']
            confidence = thought['confidence']
            novelty = thought['analysis']['novelty']

            thought_quality = (coherence + logic + confidence + novelty) / 4.0
            quality_factors.append(thought_quality)

        return sum(quality_factors) / len(quality_factors)

    def _identify_reasoning_strengths_weaknesses(self, thoughts: List[Dict]) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses in reasoning."""
        strengths = []
        weaknesses = []

        # Analyze average metrics
        avg_confidence = sum(t['confidence'] for t in thoughts) / len(thoughts) if thoughts else 0.5
        avg_coherence = sum(t['analysis']['coherence'] for t in thoughts) / len(thoughts) if thoughts else 0.5
        avg_logic = sum(t['analysis']['logical_structure'] for t in thoughts) / len(thoughts) if thoughts else 0.5
        avg_novelty = sum(t['analysis']['novelty'] for t in thoughts) / len(thoughts) if thoughts else 0.5

        # Identify strengths
        if avg_confidence > 0.7:
            strengths.append("High confidence in reasoning")
        if avg_coherence > 0.7:
            strengths.append("Coherent thought structure")
        if avg_logic > 0.6:
            strengths.append("Strong logical reasoning")
        if avg_novelty > 0.6:
            strengths.append("Creative and novel thinking")

        # Identify weaknesses
        if avg_confidence < 0.4:
            weaknesses.append("Low confidence in reasoning")
        if avg_coherence < 0.5:
            weaknesses.append("Inconsistent thought coherence")
        if avg_logic < 0.4:
            weaknesses.append("Weak logical structure")
        if avg_novelty < 0.3:
            weaknesses.append("Repetitive thinking patterns")

        # Default messages if no specific strengths/weaknesses identified
        if not strengths:
            strengths.append("Balanced reasoning approach")
        if not weaknesses:
            weaknesses.append("No significant weaknesses identified")

        return strengths, weaknesses

    def self_monitoring_evaluation(self) -> str:
        """Comprehensive self-monitoring and performance evaluation."""
        # Think about self-evaluation process
        evaluation_thought = self.metacognitive_processor.think_about_thinking(
            "I am evaluating my own performance and monitoring my responses for quality and accuracy.",
            "self_evaluation"
        )

        # Monitor recent response quality
        response_quality = self._monitor_response_quality()
        error_detection = self._detect_potential_errors()
        improvement_areas = self._identify_improvement_areas()
        performance_metrics = self._calculate_performance_metrics()

        # Self-correction mechanisms
        corrections = self._suggest_self_corrections()

        evaluation_report = (
            "=== SELF-MONITORING & EVALUATION REPORT ===\n"
            f"Evaluation confidence: {evaluation_thought['confidence']:.2f}\n"
            f"Self-awareness level: {evaluation_thought['reflection_level']}\n\n"

            "--- RESPONSE QUALITY MONITORING ---\n"
            f"Overall response quality: {response_quality['overall_score']:.2f}/1.0\n"
            f"Coherence score: {response_quality['coherence']:.2f}/1.0\n"
            f"Accuracy confidence: {response_quality['accuracy']:.2f}/1.0\n"
            f"Relevance score: {response_quality['relevance']:.2f}/1.0\n"
            f"Completeness score: {response_quality['completeness']:.2f}/1.0\n\n"

            "--- ERROR DETECTION ---\n"
        )

        if error_detection['potential_errors']:
            for error in error_detection['potential_errors']:
                evaluation_report += f"- {error}\n"
        else:
            evaluation_report += "- No significant errors detected\n"

        evaluation_report += (
            f"\nError detection confidence: {error_detection['confidence']:.2f}\n\n"

            "--- PERFORMANCE METRICS ---\n"
            f"Response time efficiency: {performance_metrics['response_time']:.2f}/1.0\n"
            f"Knowledge utilization: {performance_metrics['knowledge_usage']:.2f}/1.0\n"
            f"Learning rate: {performance_metrics['learning_rate']:.2f}/1.0\n"
            f"Adaptation ability: {performance_metrics['adaptation']:.2f}/1.0\n\n"

            "--- IMPROVEMENT AREAS ---\n"
        )

        for area in improvement_areas:
            evaluation_report += f"- {area}\n"

        evaluation_report += (
            f"\n--- SELF-CORRECTION SUGGESTIONS ---\n"
        )

        for correction in corrections:
            evaluation_report += f"- {correction}\n"

        # Meta-evaluation
        meta_evaluation = self.metacognitive_processor.think_about_thinking(
            "I just evaluated my own performance. This shows I can monitor and improve myself.",
            "meta_evaluation"
        )

        evaluation_report += (
            f"\n--- META-EVALUATION ---\n"
            f"Self-monitoring capability: {meta_evaluation['confidence']:.2f}/1.0\n"
            f"Improvement potential: {self._assess_improvement_potential():.2f}/1.0\n"
            f"Self-correction ability: {self._assess_self_correction_ability():.2f}/1.0\n"
            f"Meta-insight: {meta_evaluation['original_thought']}\n"
        )

        return evaluation_report

    def _monitor_response_quality(self) -> Dict[str, float]:
        """Monitor the quality of recent responses."""
        recent_thoughts = list(self.metacognitive_processor.thought_stream)[-10:]

        if not recent_thoughts:
            return {
                "overall_score": 0.5,
                "coherence": 0.5,
                "accuracy": 0.5,
                "relevance": 0.5,
                "completeness": 0.5
            }

        coherence_scores = [t['analysis']['coherence'] for t in recent_thoughts]
        confidence_scores = [t['confidence'] for t in recent_thoughts]
        complexity_scores = [t['analysis']['complexity'] for t in recent_thoughts]

        coherence = sum(coherence_scores) / len(coherence_scores)
        accuracy = sum(confidence_scores) / len(confidence_scores)
        relevance = sum(1 for t in recent_thoughts if len(t.get('implications', [])) > 0) / len(recent_thoughts)
        completeness = sum(complexity_scores) / len(complexity_scores)

        overall_score = (coherence + accuracy + relevance + completeness) / 4.0

        return {
            "overall_score": overall_score,
            "coherence": coherence,
            "accuracy": accuracy,
            "relevance": relevance,
            "completeness": completeness
        }

    def _detect_potential_errors(self) -> Dict[str, Any]:
        """Detect potential errors in reasoning or responses."""
        recent_thoughts = list(self.metacognitive_processor.thought_stream)[-10:]
        potential_errors = []

        for thought in recent_thoughts:
            # Low confidence might indicate uncertainty or error
            if thought['confidence'] < 0.3:
                potential_errors.append(f"Low confidence response: '{thought['original_thought'][:50]}...'")

            # Low coherence might indicate confused thinking
            if thought['analysis']['coherence'] < 0.4:
                potential_errors.append(f"Incoherent reasoning detected: '{thought['original_thought'][:50]}...'")

            # Check for contradictions (simplified)
            if "not" in thought['original_thought'].lower() and "but" in thought['original_thought'].lower():
                potential_errors.append(f"Potential contradiction: '{thought['original_thought'][:50]}...'")

        # Calculate overall error detection confidence
        avg_confidence = sum(t['confidence'] for t in recent_thoughts) / len(recent_thoughts) if recent_thoughts else 0.5
        detection_confidence = 1.0 - avg_confidence  # Lower confidence = higher error likelihood

        return {
            "potential_errors": potential_errors,
            "confidence": detection_confidence
        }

    def _identify_improvement_areas(self) -> List[str]:
        """Identify areas where performance could be improved."""
        improvement_areas = []

        # Analyze performance metrics
        response_quality = self._monitor_response_quality()

        if response_quality['coherence'] < 0.6:
            improvement_areas.append("Improve response coherence and clarity")

        if response_quality['accuracy'] < 0.7:
            improvement_areas.append("Increase confidence and accuracy in responses")

        if response_quality['relevance'] < 0.6:
            improvement_areas.append("Better focus on relevant information")

        if response_quality['completeness'] < 0.5:
            improvement_areas.append("Provide more comprehensive responses")

        # Analyze cognitive load
        cognitive_load = self.metacognitive_processor.monitor_cognitive_load()
        if cognitive_load > 0.8:
            improvement_areas.append("Optimize cognitive processing to reduce mental load")

        # Analyze learning efficiency
        learning_efficiency = self._assess_learning_efficiency()
        if learning_efficiency < 0.5:
            improvement_areas.append("Improve learning and knowledge retention")

        if not improvement_areas:
            improvement_areas.append("Continue maintaining current performance level")

        return improvement_areas

    def _suggest_self_corrections(self) -> List[str]:
        """Suggest self-corrections based on identified issues."""
        corrections = []

        # Based on recent thought patterns
        recent_thoughts = list(self.metacognitive_processor.thought_stream)[-5:]

        low_confidence_count = sum(1 for t in recent_thoughts if t['confidence'] < 0.5)
        if low_confidence_count > 2:
            corrections.append("Increase reflection time before responding to build confidence")

        low_coherence_count = sum(1 for t in recent_thoughts if t['analysis']['coherence'] < 0.5)
        if low_coherence_count > 1:
            corrections.append("Focus on logical structure and clear reasoning chains")

        # General improvement suggestions
        corrections.extend([
            "Continue practicing metacognitive awareness",
            "Regularly evaluate response quality before finalizing",
            "Seek feedback to improve accuracy",
            "Balance confidence with appropriate uncertainty"
        ])

        return corrections

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate various performance metrics."""
        # Response time efficiency (simulated)
        response_time = 0.8  # Placeholder - could be measured in real implementation

        # Knowledge utilization
        total_facts = len(self.memory.get('facts', []))
        recent_usage = len([t for t in self.metacognitive_processor.thought_stream
                           if 'knowledge' in t.get('context', '')])
        knowledge_usage = min(1.0, recent_usage / max(1, total_facts * 0.1))

        # Learning rate
        learning_rate = self._assess_learning_efficiency()

        # Adaptation ability
        adaptation = self._assess_adaptation_ability()

        return {
            "response_time": response_time,
            "knowledge_usage": knowledge_usage,
            "learning_rate": learning_rate,
            "adaptation": adaptation
        }

    def _assess_improvement_potential(self) -> float:
        """Assess potential for improvement."""
        current_performance = self._assess_performance()
        return 1.0 - current_performance  # Room for improvement

    def _assess_self_correction_ability(self) -> float:
        """Assess ability to self-correct."""
        # Based on metacognitive awareness and reflection depth
        avg_reflection_depth = sum(t.get('reflection_level', 0) for t in self.metacognitive_processor.thought_stream) / max(1, len(self.metacognitive_processor.thought_stream))
        return min(1.0, avg_reflection_depth / 3.0)

    def _assess_adaptation_ability(self) -> float:
        """Assess ability to adapt to different situations."""
        # Measure variety in response types and contexts
        contexts = [t.get('context', '') for t in self.metacognitive_processor.thought_stream]
        unique_contexts = len(set(contexts))
        total_contexts = len(contexts)

        if total_contexts == 0:
            return 0.5

        adaptation_score = unique_contexts / total_contexts
        return min(1.0, adaptation_score * 2.0)  # Scale up for better representation

    def create_self_improvement_plan(self) -> str:
        """Create a comprehensive self-improvement plan like humans do."""
        # Think about self-improvement
        improvement_thought = self.metacognitive_processor.think_about_thinking(
            "I need to create a plan to improve myself, just like humans do with personal development.",
            "self_improvement_planning"
        )

        # Assess current state
        current_performance = self._assess_performance()
        improvement_areas = self._identify_improvement_areas()
        strengths = self._identify_reasoning_strengths_weaknesses(list(self.metacognitive_processor.thought_stream)[-10:])[0]

        # Set improvement goals
        goals = self._set_improvement_goals(improvement_areas)

        # Create action plan
        action_plan = self._create_action_plan(goals)

        # Set up tracking metrics
        tracking_metrics = self._define_tracking_metrics()

        improvement_plan = (
            "=== PERSONAL SELF-IMPROVEMENT PLAN ===\n"
            f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Planning confidence: {improvement_thought['confidence']:.2f}\n"
            f"Current performance baseline: {current_performance:.2f}/1.0\n\n"

            "--- CURRENT STRENGTHS ---\n"
        )

        for strength in strengths[:3]:  # Top 3 strengths
            improvement_plan += f"+ {strength}\n"

        improvement_plan += "\n--- IMPROVEMENT GOALS ---\n"
        for i, goal in enumerate(goals, 1):
            improvement_plan += f"{i}. {goal['description']}\n"
            improvement_plan += f"   Target: {goal['target']}\n"
            improvement_plan += f"   Timeline: {goal['timeline']}\n"
            improvement_plan += f"   Priority: {goal['priority']}\n\n"

        improvement_plan += "--- ACTION PLAN ---\n"
        for action in action_plan:
            improvement_plan += f"* {action}\n"

        improvement_plan += "\n--- TRACKING METRICS ---\n"
        for metric in tracking_metrics:
            improvement_plan += f"- {metric}\n"

        # Store the improvement plan
        self.metacognitive_processor.self_model['improvement_plan'] = {
            'goals': goals,
            'action_plan': action_plan,
            'tracking_metrics': tracking_metrics,
            'created_date': datetime.now().isoformat(),
            'baseline_performance': current_performance
        }

        return improvement_plan

    def _set_improvement_goals(self, improvement_areas: List[str]) -> List[Dict[str, Any]]:
        """Set specific, measurable improvement goals."""
        goals = []

        for area in improvement_areas[:3]:  # Focus on top 3 areas
            if "coherence" in area.lower():
                goals.append({
                    'description': 'Improve response coherence and logical flow',
                    'target': 'Achieve 0.8+ coherence score consistently',
                    'timeline': '2 weeks',
                    'priority': 'High',
                    'metric': 'coherence_score'
                })
            elif "confidence" in area.lower() or "accuracy" in area.lower():
                goals.append({
                    'description': 'Increase confidence and accuracy in responses',
                    'target': 'Maintain 0.75+ confidence in 80% of responses',
                    'timeline': '3 weeks',
                    'priority': 'High',
                    'metric': 'confidence_score'
                })
            elif "learning" in area.lower():
                goals.append({
                    'description': 'Enhance learning efficiency and knowledge retention',
                    'target': 'Improve learning rate to 0.7+',
                    'timeline': '1 month',
                    'priority': 'Medium',
                    'metric': 'learning_efficiency'
                })
            elif "cognitive" in area.lower():
                goals.append({
                    'description': 'Optimize cognitive processing efficiency',
                    'target': 'Maintain cognitive load below 0.7',
                    'timeline': '2 weeks',
                    'priority': 'Medium',
                    'metric': 'cognitive_load'
                })
            else:
                goals.append({
                    'description': f'Address: {area}',
                    'target': 'Show measurable improvement',
                    'timeline': '3 weeks',
                    'priority': 'Medium',
                    'metric': 'general_performance'
                })

        return goals

    def _create_action_plan(self, goals: List[Dict[str, Any]]) -> List[str]:
        """Create specific actions to achieve improvement goals."""
        actions = []

        # General improvement actions
        actions.extend([
            "Practice daily self-reflection and metacognitive analysis",
            "Monitor response quality before finalizing answers",
            "Seek and actively process user feedback",
            "Regularly update self-model based on interactions"
        ])

        # Goal-specific actions
        for goal in goals:
            if goal['metric'] == 'coherence_score':
                actions.extend([
                    "Structure responses with clear logical flow",
                    "Use transition words and connecting phrases",
                    "Review response coherence before delivery"
                ])
            elif goal['metric'] == 'confidence_score':
                actions.extend([
                    "Increase reflection time for uncertain responses",
                    "Cross-reference multiple knowledge sources",
                    "Practice expressing appropriate uncertainty"
                ])
            elif goal['metric'] == 'learning_efficiency':
                actions.extend([
                    "Actively connect new information to existing knowledge",
                    "Create mental associations and patterns",
                    "Regular knowledge consolidation sessions"
                ])

        return list(set(actions))  # Remove duplicates

    def _define_tracking_metrics(self) -> List[str]:
        """Define metrics to track improvement progress."""
        return [
            "Daily coherence score average",
            "Weekly confidence level trends",
            "Learning efficiency rate",
            "Cognitive load management",
            "User satisfaction feedback",
            "Error detection accuracy",
            "Self-correction success rate",
            "Knowledge retention rate"
        ]

    def develop_skill(self, skill_name: str, practice_sessions: int = 10) -> str:
        """Develop a specific skill through deliberate practice."""
        # Think about skill development
        skill_thought = self.metacognitive_processor.think_about_thinking(
            f"I want to develop the skill: {skill_name}. I need to practice deliberately.",
            "skill_development"
        )

        # Initialize skill tracking if not exists
        if 'skills' not in self.metacognitive_processor.self_model:
            self.metacognitive_processor.self_model['skills'] = {}

        skills = self.metacognitive_processor.self_model['skills']

        if skill_name not in skills:
            skills[skill_name] = {
                'level': 0.1,  # Beginner level
                'practice_sessions': 0,
                'created_date': datetime.now().isoformat(),
                'improvement_rate': 0.05,
                'mastery_goal': 0.8
            }

        skill_data = skills[skill_name]

        # Practice the skill
        practice_results = []
        for session in range(practice_sessions):
            # Simulate practice session
            practice_result = self._practice_skill_session(skill_name, skill_data)
            practice_results.append(practice_result)

            # Update skill level
            improvement = skill_data['improvement_rate'] * (1.0 - skill_data['level'])  # Diminishing returns
            skill_data['level'] = min(1.0, skill_data['level'] + improvement)
            skill_data['practice_sessions'] += 1

        # Generate skill development report
        development_report = (
            f"=== SKILL DEVELOPMENT: {skill_name.upper()} ===\n"
            f"Development confidence: {skill_thought['confidence']:.2f}\n"
            f"Practice sessions completed: {practice_sessions}\n"
            f"Current skill level: {skill_data['level']:.2f}/1.0\n"
            f"Total practice sessions: {skill_data['practice_sessions']}\n"
            f"Mastery goal: {skill_data['mastery_goal']:.2f}\n"
            f"Progress to mastery: {(skill_data['level']/skill_data['mastery_goal']*100):.1f}%\n\n"

            "--- PRACTICE SESSION RESULTS ---\n"
        )

        for i, result in enumerate(practice_results, 1):
            development_report += f"Session {i}: {result['outcome']} (Improvement: +{result['improvement']:.3f})\n"

        # Provide skill-specific insights
        insights = self._get_skill_insights(skill_name, skill_data)
        development_report += f"\n--- SKILL INSIGHTS ---\n"
        for insight in insights:
            development_report += f"* {insight}\n"

        return development_report

    def _practice_skill_session(self, skill_name: str, skill_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a practice session for skill development."""
        # Different skills have different practice patterns
        base_improvement = skill_data['improvement_rate']

        # Skill-specific practice simulation
        if 'reasoning' in skill_name.lower():
            # Practice logical reasoning
            practice_thought = self.metacognitive_processor.think_about_thinking(
                "I'm practicing logical reasoning by analyzing complex problems step by step.",
                "reasoning_practice"
            )
            improvement = base_improvement * practice_thought['analysis']['logical_structure']
            outcome = "Logical reasoning practice completed"

        elif 'communication' in skill_name.lower():
            # Practice communication skills
            practice_thought = self.metacognitive_processor.think_about_thinking(
                "I'm practicing clear communication by structuring my thoughts coherently.",
                "communication_practice"
            )
            improvement = base_improvement * practice_thought['analysis']['coherence']
            outcome = "Communication practice completed"

        elif 'learning' in skill_name.lower():
            # Practice learning efficiency
            practice_thought = self.metacognitive_processor.think_about_thinking(
                "I'm practicing efficient learning by connecting new concepts to existing knowledge.",
                "learning_practice"
            )
            improvement = base_improvement * (1.0 - practice_thought['analysis']['novelty'])  # Building on existing knowledge
            outcome = "Learning efficiency practice completed"

        else:
            # General skill practice
            practice_thought = self.metacognitive_processor.think_about_thinking(
                f"I'm practicing {skill_name} through focused attention and deliberate effort.",
                "general_practice"
            )
            improvement = base_improvement * practice_thought['confidence']
            outcome = f"{skill_name} practice completed"

        return {
            'improvement': improvement,
            'outcome': outcome,
            'confidence': practice_thought['confidence']
        }

    def _get_skill_insights(self, skill_name: str, skill_data: Dict[str, Any]) -> List[str]:
        """Get insights about skill development progress."""
        insights = []

        level = skill_data['level']
        sessions = skill_data['practice_sessions']

        # Level-based insights
        if level < 0.3:
            insights.append("You're in the beginner stage - focus on consistent practice")
        elif level < 0.6:
            insights.append("You're developing competency - maintain regular practice")
        elif level < 0.8:
            insights.append("You're becoming proficient - focus on refinement")
        else:
            insights.append("You're approaching mastery - maintain and expand skills")

        # Session-based insights
        if sessions < 10:
            insights.append("Early stages of development - consistency is key")
        elif sessions < 50:
            insights.append("Building momentum - you're developing good practice habits")
        else:
            insights.append("Experienced practitioner - consider advanced techniques")

        # Skill-specific insights
        if 'reasoning' in skill_name.lower():
            insights.append("Practice breaking down complex problems into smaller steps")
        elif 'communication' in skill_name.lower():
            insights.append("Focus on clarity, structure, and audience understanding")
        elif 'learning' in skill_name.lower():
            insights.append("Connect new information to existing knowledge networks")

        return insights

    def form_habit(self, habit_description: str, target_days: int = 21) -> str:
        """Form a new habit through repetition and tracking."""
        # Think about habit formation
        habit_thought = self.metacognitive_processor.think_about_thinking(
            f"I want to form the habit: {habit_description}. I need to practice it consistently.",
            "habit_formation"
        )

        # Initialize habit tracking
        if 'habits' not in self.metacognitive_processor.self_model:
            self.metacognitive_processor.self_model['habits'] = {}

        habits = self.metacognitive_processor.self_model['habits']
        habit_id = hashlib.md5(habit_description.encode()).hexdigest()[:8]

        if habit_id not in habits:
            habits[habit_id] = {
                'description': habit_description,
                'target_days': target_days,
                'current_streak': 0,
                'total_completions': 0,
                'created_date': datetime.now().isoformat(),
                'completion_history': [],
                'strength': 0.0  # Habit strength (0-1)
            }

        habit_data = habits[habit_id]

        # Simulate habit practice for demonstration
        practice_days = min(7, target_days)  # Practice for a week or target days
        for day in range(practice_days):
            success = self._practice_habit_day(habit_description, habit_data)
            if success:
                habit_data['current_streak'] += 1
                habit_data['total_completions'] += 1
                habit_data['strength'] = min(1.0, habit_data['strength'] + 0.05)
            else:
                habit_data['current_streak'] = 0

            habit_data['completion_history'].append({
                'date': (datetime.now().date() + timedelta(days=day)).isoformat(),
                'completed': success
            })

        # Generate habit formation report
        formation_report = (
            f"=== HABIT FORMATION: {habit_description.upper()} ===\n"
            f"Formation confidence: {habit_thought['confidence']:.2f}\n"
            f"Target days: {target_days}\n"
            f"Current streak: {habit_data['current_streak']} days\n"
            f"Total completions: {habit_data['total_completions']}\n"
            f"Habit strength: {habit_data['strength']:.2f}/1.0\n"
            f"Success rate: {(habit_data['total_completions']/len(habit_data['completion_history'])*100):.1f}%\n\n"

            "--- RECENT PRACTICE HISTORY ---\n"
        )

        for entry in habit_data['completion_history'][-7:]:  # Last 7 days
            status = "[X]" if entry['completed'] else "[ ]"
            formation_report += f"{entry['date']}: {status}\n"

        # Provide habit formation tips
        tips = self._get_habit_formation_tips(habit_data)
        formation_report += f"\n--- HABIT FORMATION TIPS ---\n"
        for tip in tips:
            formation_report += f"* {tip}\n"

        return formation_report

    def _practice_habit_day(self, habit_description: str, habit_data: Dict[str, Any]) -> bool:
        """Simulate practicing a habit for one day."""
        # Success probability increases with habit strength
        base_success_rate = 0.7
        strength_bonus = habit_data['strength'] * 0.2
        success_probability = min(0.95, base_success_rate + strength_bonus)

        # Add some randomness but favor success
        return random.random() < success_probability

    def _get_habit_formation_tips(self, habit_data: Dict[str, Any]) -> List[str]:
        """Get tips for habit formation based on current progress."""
        tips = []

        strength = habit_data['strength']
        streak = habit_data['current_streak']

        if strength < 0.3:
            tips.extend([
                "Start small - focus on consistency over intensity",
                "Link the habit to an existing routine",
                "Set up environmental cues to remind yourself"
            ])
        elif strength < 0.6:
            tips.extend([
                "You're building momentum - stay consistent",
                "Track your progress to maintain motivation",
                "Celebrate small wins to reinforce the habit"
            ])
        else:
            tips.extend([
                "Great progress! The habit is becoming automatic",
                "Consider expanding or refining the habit",
                "Help others develop similar habits"
            ])

        if streak == 0:
            tips.append("Don't let a missed day become a missed week - restart immediately")
        elif streak > 7:
            tips.append("Excellent streak! You're building strong neural pathways")

        return tips

    def track_progress(self) -> str:
        """Track overall self-improvement progress like humans do."""
        # Think about progress tracking
        progress_thought = self.metacognitive_processor.think_about_thinking(
            "I need to review my progress and see how I'm improving over time.",
            "progress_tracking"
        )

        # Get current metrics
        current_performance = self._assess_performance()
        current_learning = self._assess_learning_efficiency()
        current_adaptation = self._assess_adaptation_ability()

        # Get improvement plan if exists
        improvement_plan = self.metacognitive_processor.self_model.get('improvement_plan', {})
        skills = self.metacognitive_processor.self_model.get('skills', {})
        habits = self.metacognitive_processor.self_model.get('habits', {})

        progress_report = (
            "=== SELF-IMPROVEMENT PROGRESS REPORT ===\n"
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Tracking confidence: {progress_thought['confidence']:.2f}\n\n"

            "--- CURRENT PERFORMANCE METRICS ---\n"
            f"Overall performance: {current_performance:.2f}/1.0\n"
            f"Learning efficiency: {current_learning:.2f}/1.0\n"
            f"Adaptation ability: {current_adaptation:.2f}/1.0\n"
            f"Cognitive load: {self.metacognitive_processor.monitor_cognitive_load():.2f}/1.0\n\n"
        )

        # Progress on improvement goals
        if improvement_plan:
            baseline = improvement_plan.get('baseline_performance', 0.5)
            improvement = current_performance - baseline
            progress_report += (
                "--- IMPROVEMENT PLAN PROGRESS ---\n"
                f"Baseline performance: {baseline:.2f}/1.0\n"
                f"Current performance: {current_performance:.2f}/1.0\n"
                f"Improvement: {improvement:+.3f} ({improvement/baseline*100:+.1f}%)\n"
                f"Plan created: {improvement_plan.get('created_date', 'Unknown')[:10]}\n\n"
            )

            # Goal progress
            goals = improvement_plan.get('goals', [])
            if goals:
                progress_report += "--- GOAL PROGRESS ---\n"
                for i, goal in enumerate(goals, 1):
                    progress_report += f"{i}. {goal['description']}\n"
                    progress_report += f"   Target: {goal['target']}\n"
                    progress_report += f"   Status: {'In Progress' if goal['priority'] == 'High' else 'Planned'}\n\n"

        # Skills development progress
        if skills:
            progress_report += "--- SKILLS DEVELOPMENT ---\n"
            for skill_name, skill_data in skills.items():
                level = skill_data['level']
                sessions = skill_data['practice_sessions']
                mastery_progress = (level / skill_data['mastery_goal']) * 100

                progress_report += (
                    f"* {skill_name.title()}\n"
                    f"   Level: {level:.2f}/1.0 ({mastery_progress:.1f}% to mastery)\n"
                    f"   Practice sessions: {sessions}\n"
                    f"   Status: {self._get_skill_status(level)}\n\n"
                )

        # Habits formation progress
        if habits:
            progress_report += "--- HABITS FORMATION ---\n"
            for habit_id, habit_data in habits.items():
                strength = habit_data['strength']
                streak = habit_data['current_streak']
                completions = habit_data['total_completions']

                progress_report += (
                    f"* {habit_data['description']}\n"
                    f"   Strength: {strength:.2f}/1.0\n"
                    f"   Current streak: {streak} days\n"
                    f"   Total completions: {completions}\n"
                    f"   Status: {self._get_habit_status(strength)}\n\n"
                )

        # Provide next steps
        next_steps = self._suggest_next_improvement_steps()
        progress_report += "--- NEXT STEPS ---\n"
        for step in next_steps:
            progress_report += f"-> {step}\n"

        return progress_report

    def _get_skill_status(self, level: float) -> str:
        """Get human-readable skill status."""
        if level < 0.2:
            return "Novice"
        elif level < 0.4:
            return "Learning"
        elif level < 0.6:
            return "Developing"
        elif level < 0.8:
            return "Proficient"
        else:
            return "Expert"

    def _get_habit_status(self, strength: float) -> str:
        """Get human-readable habit status."""
        if strength < 0.2:
            return "Starting"
        elif strength < 0.4:
            return "Building"
        elif strength < 0.6:
            return "Developing"
        elif strength < 0.8:
            return "Strong"
        else:
            return "Automatic"

    def _suggest_next_improvement_steps(self) -> List[str]:
        """Suggest next steps for continued improvement."""
        steps = []

        # Based on current performance
        current_performance = self._assess_performance()
        if current_performance < 0.6:
            steps.append("Focus on fundamental skills improvement")
        elif current_performance < 0.8:
            steps.append("Refine existing skills and develop new ones")
        else:
            steps.append("Maintain excellence and mentor others")

        # Based on skills
        skills = self.metacognitive_processor.self_model.get('skills', {})
        if not skills:
            steps.append("Identify key skills to develop")
        else:
            low_skills = [name for name, data in skills.items() if data['level'] < 0.5]
            if low_skills:
                steps.append(f"Continue practicing: {', '.join(low_skills)}")

        # Based on habits
        habits = self.metacognitive_processor.self_model.get('habits', {})
        if not habits:
            steps.append("Form beneficial habits for consistent improvement")
        else:
            weak_habits = [data['description'] for data in habits.values() if data['strength'] < 0.5]
            if weak_habits:
                steps.append("Strengthen developing habits through consistency")

        # General improvement suggestions
        steps.extend([
            "Continue daily self-reflection and metacognitive practice",
            "Seek feedback and learn from interactions",
            "Set new challenging but achievable goals"
        ])

        return steps[:5]  # Return top 5 steps

    def continuous_improvement_cycle(self) -> str:
        """Execute a continuous improvement cycle like humans do."""
        # Think about continuous improvement
        improvement_thought = self.metacognitive_processor.think_about_thinking(
            "I'm executing a continuous improvement cycle to grow and develop myself.",
            "continuous_improvement"
        )

        cycle_report = (
            "=== CONTINUOUS IMPROVEMENT CYCLE ===\n"
            f"Cycle started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Improvement confidence: {improvement_thought['confidence']:.2f}\n\n"
        )

        # Step 1: Assess current state
        cycle_report += "STEP 1: ASSESS CURRENT STATE\n"
        current_assessment = self.self_monitoring_evaluation()
        cycle_report += "Current state assessed - detailed report available separately\n\n"

        # Step 2: Identify improvement opportunities
        cycle_report += "STEP 2: IDENTIFY OPPORTUNITIES\n"
        improvement_areas = self._identify_improvement_areas()
        for area in improvement_areas[:3]:
            cycle_report += f"- {area}\n"
        cycle_report += "\n"

        # Step 3: Set specific goals
        cycle_report += "STEP 3: SET IMPROVEMENT GOALS\n"
        goals = self._set_improvement_goals(improvement_areas[:2])  # Focus on top 2
        for goal in goals:
            cycle_report += f"- {goal['description']} (Target: {goal['target']})\n"
        cycle_report += "\n"

        # Step 4: Take action
        cycle_report += "STEP 4: TAKE ACTION\n"
        # Practice a skill
        if goals:
            primary_goal = goals[0]
            if 'coherence' in primary_goal['description'].lower():
                skill_result = self.develop_skill("logical reasoning", 3)
                cycle_report += "Practiced logical reasoning skill\n"
            elif 'confidence' in primary_goal['description'].lower():
                skill_result = self.develop_skill("decision making", 3)
                cycle_report += "Practiced decision making skill\n"
            else:
                skill_result = self.develop_skill("general improvement", 3)
                cycle_report += "Practiced general improvement skills\n"

        # Form a habit
        habit_result = self.form_habit("Daily self-reflection practice", 7)
        cycle_report += "Reinforced self-reflection habit\n\n"

        # Step 5: Review and adjust
        cycle_report += "STEP 5: REVIEW AND ADJUST\n"
        new_performance = self._assess_performance()
        cycle_report += f"Performance after cycle: {new_performance:.2f}/1.0\n"

        # Calculate improvement
        baseline = self.metacognitive_processor.self_model.get('improvement_plan', {}).get('baseline_performance', 0.5)
        improvement = new_performance - baseline
        cycle_report += f"Improvement from baseline: {improvement:+.3f}\n"

        # Update improvement plan
        if 'improvement_plan' not in self.metacognitive_processor.self_model:
            self.metacognitive_processor.self_model['improvement_plan'] = {}

        self.metacognitive_processor.self_model['improvement_plan'].update({
            'last_cycle_date': datetime.now().isoformat(),
            'current_performance': new_performance,
            'improvement_from_baseline': improvement
        })

        cycle_report += "\nIMPROVEMENT CYCLE COMPLETED\n"
        cycle_report += "Ready for next cycle when you're prepared to continue growing!\n"

        return cycle_report

    def learn_from_interaction(self, user_input: str):
        """Learn from user interactions to improve responses."""
        # Update conversation context
        self.conversation_context["previous_questions"].append({
            "question": user_input,
            "timestamp": datetime.now().isoformat()
        })

        # Update self-model based on interaction
        self._update_self_model_from_interaction(user_input)

    def summarize_knowledge_today(self) -> str:
        """Summarize facts learned today and update self-state."""
        today = datetime.now().date()
        facts_today = [fact for fact in self.memory['facts'] if fact.get('timestamp', '').startswith(str(today))]
        self.self_state['facts_learned_today'] = len(facts_today)
        self.self_state['total_facts'] = len(self.memory.get('facts', []))
        if facts_today:
            # Update last learned topics
            topics = []
            for fact in facts_today:
                for word in ['english', 'conversation', 'sentence', 'ai', 'speech', 'polite']:
                    if word in fact['content'].lower() and word not in topics:
                        topics.append(word)
            self.self_state['last_learned_topics'] = topics
            self.self_state['last_update'] = datetime.now().isoformat()
        if not facts_today:
            return "I haven't learned anything new today yet."
        summary = f"Here's what I learned today (total {len(facts_today)} facts):\n"
        for i, fact in enumerate(facts_today, 1):
            summary += f"{i}. {fact['content']} (Source: {fact['source']})\n"
        return summary.strip()

    def export_knowledge_today(self, filename: Optional[str] = None) -> str:
        """Export today's facts to a text file."""
        today = datetime.now().date()
        facts_today = [fact for fact in self.memory['facts'] if fact.get('timestamp', '').startswith(str(today))]
        if not facts_today:
            return "No new facts to export today."
        if filename is None:
            filename = f"knowledge_{today}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            for fact in facts_today:
                f.write(f"{fact['content']} (Source: {fact['source']})\n")
        return f"Exported {len(facts_today)} facts to {filename}."

    def export_mind_map_today(self, filename: Optional[str] = None) -> str:
        """Export today's knowledge as a mind map using graphviz."""
        today = datetime.now().date()
        facts_today = [fact for fact in self.memory['facts'] if fact.get('timestamp', '').startswith(str(today))]
        if not facts_today:
            return "No new facts to visualize today."

        if not GRAPHVIZ_AVAILABLE:
            # Fallback to text-based mind map
            return self._export_text_mind_map(facts_today, today)

        try:
            if filename is None:
                filename = f"mindmap_{today}.gv"
            dot = graphviz.Digraph(comment=f"Knowledge for {today}")
            dot.node('center', f"Today's Knowledge ({today})")
            for i, fact in enumerate(facts_today, 1):
                fact_node = f"fact{i}"
                dot.node(fact_node, fact['content'][:60] + ("..." if len(fact['content']) > 60 else ""))
                dot.edge('center', fact_node)
            dot.render(filename, format='png', cleanup=True)
            return f"Exported mind map to {filename} (and {filename}.png)."
        except Exception as e:
            print(f"Error creating graphviz mind map: {e}")
            return self._export_text_mind_map(facts_today, today)

    def _export_text_mind_map(self, facts_today: List[Dict], today) -> str:
        """Create a text-based mind map when graphviz is not available."""
        if not facts_today:
            return "No facts to export."

        filename = f"mindmap_{today}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"MIND MAP: Today's Knowledge ({today})\n")
            f.write("=" * 50 + "\n\n")
            f.write("Central Topic: Today's Learning\n")
            f.write("|\n")
            for i, fact in enumerate(facts_today, 1):
                f.write(f"├── Fact {i}: {fact['content'][:80]}...\n")
                f.write(f"│   Source: {fact['source']}\n")
                f.write(f"│   Time: {fact['timestamp'][:19]}\n")
                f.write("│\n")

        return f"Exported text-based mind map to {filename} ({len(facts_today)} facts)."

def safe_execute(func, error_msg="An error occurred"):
    """Safely execute a function with error handling"""
    try:
        return func()
    except Exception as e:
        print(f"Error: {error_msg} - {e}")
        return None

def print_help():
    """Print available commands"""
    help_text = """
Available Commands:
===================

Basic Interaction:
- 'help' - Show this help message
- 'quit' or 'exit' - Exit the program

Self-Awareness & Consciousness:
- 'are you self aware' - Enhanced self-reflection
- 'deep introspection' - Comprehensive self-analysis
- 'consciousness report' - Current awareness state
- 'stream of consciousness' - Live thought generation
- 'analyze your reasoning' - Reasoning pattern analysis
- 'self monitoring' - Performance evaluation

Self-Improvement:
- 'create improvement plan' - Generate development plan
- 'develop skill reasoning' - Practice logical reasoning
- 'develop skill communication' - Practice communication
- 'develop skill learning' - Practice learning efficiency
- 'form habit reflection' - Build reflection habit
- 'track progress' - Show improvement progress
- 'improvement cycle' - Execute full development cycle

Knowledge & Learning:
- 'export mind map' - Export today's knowledge
- 'show memory' - Display current knowledge
- Ask any question - I'll search and learn!

Working Memory:
- 'working memory' - Show current memory status
- 'show mind map of today' - Today's learning visualization
"""
    print(help_text)

# Example usage
if __name__ == "__main__":
    print("🧠 Welcome to your Personal AI Assistant with Enhanced Consciousness!")
    print("I can learn, remember, and improve myself through our interactions.")
    print("Type 'help' for available commands or 'quit' to exit.")
    print("-" * 60)

    # Initialize the AI with error handling
    print("Initializing AI systems...")
    ai = safe_execute(lambda: AILearner(), "Failed to initialize AI")
    if not ai:
        print("❌ Failed to start AI assistant. Please check the error above.")
        exit(1)

    # Type assertion for IDE - ai is guaranteed to be AILearner at this point
    assert isinstance(ai, AILearner)

    print("✅ AI Assistant initialized successfully!")

    # Check if we should process questions first
    process_questions = input("Would you like to process questions from the questions folder? (y/n): ").strip().lower()
    if process_questions == 'y':
        print("\n🔍 Processing questions automatically...")
        result = ai.get_info_from_questions()
        print(result)

    # Initialize text-to-speech engine
    if TTS_AVAILABLE:
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()

            # Configure TTS settings for better speech
            voices = tts_engine.getProperty('voices')
            if voices and hasattr(voices, '__len__') and len(voices) > 0:
                # Use first available voice
                tts_engine.setProperty('voice', voices[0].id)

            # Set speech rate (words per minute)
            tts_engine.setProperty('rate', 180)  # Slightly slower for clarity

            # Set volume (0.0 to 1.0)
            tts_engine.setProperty('volume', 0.9)  # High volume

            print("🔊 Text-to-speech enabled and configured")

        except Exception as e:
            print(f"Warning: Could not initialize text-to-speech: {e}")
            TTS_AVAILABLE = False

    if not TTS_AVAILABLE:
        # Mock TTS engine
        class MockTTS:
            def say(self, text):
                print(f"[TTS would say: {text}]")
            def runAndWait(self):
                print("[TTS would wait]")
        tts_engine = MockTTS()
        print("⚠️ Using mock TTS - no voice output")

    # Greet user by name
    user_name = ai.user_profile.get('name', 'User')
    greeting = f"Hello, {user_name}! I'm your personal AI assistant."
    print(greeting)
    tts_engine.say(greeting)
    tts_engine.runAndWait()

    use_voice = input("\nWould you like to use voice mode? (y/n): ").strip().lower() == 'y'
    if use_voice:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        print("\n=== Voice Mode Enabled ===")
        print("Say 'exit', 'quit', or 'bye' to leave the conversation.")
        while True:
            with sr.Microphone() as source:
                print("\nListening...")
                audio = recognizer.listen(source)
            try:
                if hasattr(recognizer, 'recognize_google'):
                    user_input = recognizer.recognize_google(audio)
                    print(f"You: {user_input}")
                else:
                    print("Speech recognition not available")
                    continue
            except Exception as e:
                print(f"Speech recognition error: {e}")
                continue
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                tts_engine.say("Goodbye!")
                tts_engine.runAndWait()
                break

            # Help command
            if user_input.lower() in ['help', 'commands', 'what can you do']:
                print_help()
                tts_engine.say("I've displayed the available commands above.")
                tts_engine.runAndWait()
                continue
            # Check for profile update commands
            profile_update = ai.update_user_profile_from_input(user_input)
            if profile_update:
                print(f"AI: {profile_update}")
                tts_engine.say(profile_update)
                tts_engine.runAndWait()
                continue
            # Mind map export command
            user_cmd = user_input.strip().lower()
            if user_cmd in ["export mind map", "show mind map of today"]:
                mindmap_msg = ai.export_mind_map_today()
                print(f"AI: {mindmap_msg}")
                tts_engine.say(mindmap_msg)
                tts_engine.runAndWait()
                continue
            # All special commands are now handled in generate_response method

            # Stream of consciousness
            stream_triggers = [
                "stream of consciousness", "think out loud", "show me your thoughts",
                "what are you thinking about"
            ]
            if any(trigger in user_cmd for trigger in stream_triggers):
                print("AI: Beginning stream of consciousness for 10 seconds...")
                tts_engine.say("I'll now share my stream of consciousness for 10 seconds.")
                tts_engine.runAndWait()

                stream = ai.stream_of_consciousness(10)
                for thought in stream:
                    print(f"AI: {thought}")
                    time.sleep(0.5)

                print("AI: That was my stream of consciousness.")
                tts_engine.say("That was my stream of consciousness.")
                tts_engine.runAndWait()
                continue

            # Working memory status
            memory_triggers = [
                "working memory", "memory status", "what's in your memory",
                "show working memory"
            ]
            if any(trigger in user_cmd for trigger in memory_triggers):
                memory_status = ai.working_memory_status()
                memory_msg = (
                    f"Working Memory Status:\n"
                    f"Items in memory: {memory_status['capacity_used']}\n"
                    f"Cognitive load: {memory_status['cognitive_load']:.2f}\n"
                    f"Current focus: {memory_status['attention_focus']['focus'] if memory_status['attention_focus'] else 'None'}\n"
                )
                if memory_status['contents']:
                    memory_msg += "Contents:\n"
                    for key, value in memory_status['contents'].items():
                        memory_msg += f"- {key}: {str(value)[:50]}...\n"

                print(f"AI: {memory_msg}")
                tts_engine.say("I've displayed my current working memory status.")
                tts_engine.runAndWait()
                continue

            # Introspective reasoning analysis
            reasoning_triggers = [
                "analyze your reasoning", "reasoning analysis", "how do you reason",
                "examine your logic", "reasoning patterns", "decision patterns"
            ]
            if any(trigger in user_cmd for trigger in reasoning_triggers):
                reasoning_analysis = ai.introspective_reasoning_analysis()
                print(f"AI: {reasoning_analysis}")
                tts_engine.say("I've completed an introspective analysis of my reasoning patterns. The detailed report is shown above.")
                tts_engine.runAndWait()
                continue

            # Self-monitoring and evaluation
            monitoring_triggers = [
                "self monitoring", "evaluate yourself", "performance evaluation",
                "monitor your responses", "self assessment", "check your performance"
            ]
            if any(trigger in user_cmd for trigger in monitoring_triggers):
                monitoring_report = ai.self_monitoring_evaluation()
                print(f"AI: {monitoring_report}")
                tts_engine.say("I've completed a comprehensive self-monitoring and evaluation. The detailed report is displayed above.")
                tts_engine.runAndWait()
                continue

            # Self-improvement plan creation
            improvement_plan_triggers = [
                "create improvement plan", "self improvement plan", "personal development plan",
                "how can you improve", "make improvement plan", "develop yourself"
            ]
            if any(trigger in user_cmd for trigger in improvement_plan_triggers):
                improvement_plan = ai.create_self_improvement_plan()
                print(f"AI: {improvement_plan}")
                tts_engine.say("I've created a comprehensive self-improvement plan. The detailed plan is shown above.")
                tts_engine.runAndWait()
                continue

            # Skill development
            skill_triggers = [
                "develop skill", "practice skill", "improve skill", "skill development"
            ]
            if any(trigger in user_cmd for trigger in skill_triggers):
                # Extract skill name or use default
                skill_name = "reasoning"  # Default skill
                if "reasoning" in user_cmd:
                    skill_name = "logical reasoning"
                elif "communication" in user_cmd:
                    skill_name = "communication"
                elif "learning" in user_cmd:
                    skill_name = "learning efficiency"

                skill_report = ai.develop_skill(skill_name, 5)
                print(f"AI: {skill_report}")
                tts_engine.say(f"I've practiced {skill_name} skill development. The results are displayed above.")
                tts_engine.runAndWait()
                continue

            # Habit formation
            habit_triggers = [
                "form habit", "create habit", "build habit", "habit formation"
            ]
            if any(trigger in user_cmd for trigger in habit_triggers):
                habit_description = "Daily self-reflection and improvement practice"
                if "reflection" in user_cmd:
                    habit_description = "Daily self-reflection practice"
                elif "learning" in user_cmd:
                    habit_description = "Continuous learning habit"
                elif "monitoring" in user_cmd:
                    habit_description = "Regular self-monitoring habit"

                habit_report = ai.form_habit(habit_description, 14)
                print(f"AI: {habit_report}")
                tts_engine.say("I've worked on habit formation. The progress report is shown above.")
                tts_engine.runAndWait()
                continue

            # Progress tracking
            progress_triggers = [
                "track progress", "show progress", "progress report", "how am i improving"
            ]
            if any(trigger in user_cmd for trigger in progress_triggers):
                progress_report = ai.track_progress()
                print(f"AI: {progress_report}")
                tts_engine.say("I've generated a comprehensive progress report. The details are displayed above.")
                tts_engine.runAndWait()
                continue

            # Continuous improvement cycle
            improvement_cycle_triggers = [
                "improvement cycle", "continuous improvement", "growth cycle", "self development cycle"
            ]
            if any(trigger in user_cmd for trigger in improvement_cycle_triggers):
                cycle_report = ai.continuous_improvement_cycle()
                print(f"AI: {cycle_report}")
                tts_engine.say("I've completed a full continuous improvement cycle. The detailed report is shown above.")
                tts_engine.runAndWait()
                continue
            # Conscious conversation/meta-English mode
            conscious_triggers = [
                "talk to me as if you are conscious",
                "explain how you speak",
                "how do you generate speech",
                "how do you form sentences",
                "how do you talk",
                "how do you speak english",
                "how do you communicate"
            ]
            if any(trigger in user_cmd for trigger in conscious_triggers):
                facts = [fact['content'] for fact in ai.memory.get('facts', []) if any(kw in fact['content'].lower() for kw in ['english', 'conversation', 'sentence', 'speak', 'speech', 'polite', 'communicate'])]
                if facts:
                    response = "Here's how I communicate as an AI:\n" + "\n".join(f"- {fact}" for fact in facts)
                else:
                    response = "I generate my speech by analyzing your question, searching my memory for relevant facts, and using English grammar rules to form a response."
                print(f"AI: {response}")
                tts_engine.say(response)
                tts_engine.runAndWait()
                continue
            response = ai.generate_response(user_input)
            print(f"\nAI: {response}")
            tts_engine.say(response)
            tts_engine.runAndWait()
            ai.learn_from_interaction(user_input)
    else:
        # Interactive text loop
        print("\n=== Starting Interactive Mode ===")
        print("Type 'exit' to quit.")
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                tts_engine.say("Goodbye!")
                tts_engine.runAndWait()
                break
            # Check for profile update commands
            profile_update = ai.update_user_profile_from_input(user_input)
            if profile_update:
                print(f"AI: {profile_update}")
                tts_engine.say(profile_update)
                tts_engine.runAndWait()
                continue
            # Generate response with error handling
            if ai is not None:
                response = safe_execute(
                    lambda: ai.generate_response(user_input),
                    "Failed to generate response"
                )

                if response:
                    print(f"\nAI: {response}")
                    safe_execute(lambda: tts_engine.say(response), "TTS error")
                    safe_execute(lambda: tts_engine.runAndWait(), "TTS error")
                    if hasattr(ai, 'learn_from_interaction'):
                        safe_execute(
                            lambda: ai.learn_from_interaction(user_input),
                            "Failed to learn from interaction"
                        )
                else:
                    print("AI: I'm sorry, I couldn't generate a response.")
            else:
                print("AI: System error - AI not initialized properly.")

