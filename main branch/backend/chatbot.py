# backend/chatbot.py - Fixed Implementation with Proper Session Management

import random
import re
import json
from typing import Tuple, List, Dict, Optional
import datetime

class MultilingualChatbot:
    def __init__(self):
        self.conversation_history = []
        self.user_name = None
        self.preferred_languages = []
        self.context = {}
        self.session_id = None
        
        # Project-focused responses for multilingual assistant
        self.responses = {
            'greeting': [
                "Hello! I'm your multilingual AI assistant created by Vedang. How can I help you with language tasks today?",
                "Hi there! Ready to help you translate, analyze text, or process speech. What language challenge can I solve for you?",
                "Greetings! I'm here to break down language barriers with translation and text analysis. What would you like to work on?",
                "Hey! I'm your language companion, built by Vedang to make multilingual communication seamless. How can I assist you?"
            ],
            'capabilities': [
                "I specialize in: 🌐 Real-time translation between 50+ languages, 🎤 Speech-to-text conversion, 🔊 Text-to-speech synthesis, 😊 Sentiment analysis, and 📝 Text processing!",
                "My core features include multilingual translation, voice recognition, audio synthesis, emotion detection in text, and language analysis tools!",
                "I'm equipped with translation engines, speech processing, sentiment analysis, and text-to-speech capabilities across multiple languages!",
                "I can translate text instantly, convert speech to text, synthesize natural-sounding speech, analyze emotions in text, and process multiple languages simultaneously!"
            ],
            'translation': [
                "I can translate between over 50 languages instantly! Just tell me the source and target language, and provide the text you'd like translated.",
                "Translation is my specialty! Which languages would you like to translate between? I support major world languages plus many regional ones.",
                "Ready to translate! Simply specify: 'Translate [text] from [source language] to [target language]' and I'll handle it immediately.",
                "I offer accurate translations with context awareness. What language pair are you working with today?"
            ],
            'speech_processing': [
                "I can convert speech to text and text to speech in multiple languages! Upload an audio file or provide text for synthesis.",
                "My speech processing handles voice recognition and audio generation. What would you like to do - transcribe audio or create speech?",
                "Speech-to-text and text-to-speech are both available! I can work with various audio formats and languages for both directions.",
                "Voice processing is ready! I can transcribe your recordings or convert your text into natural-sounding speech."
            ],
            'sentiment_analysis': [
                "I can analyze the emotional tone of text in multiple languages! Share the text and I'll detect sentiment, emotions, and mood.",
                "Sentiment analysis helps understand the emotional context of text. Provide any text and I'll analyze its emotional tone and intensity.",
                "I detect emotions like happiness, sadness, anger, fear, and neutrality in text across different languages. What text should I analyze?",
                "Text emotion analysis is available! I can identify sentiment polarity, emotional intensity, and contextual mood in your content."
            ],
            'languages': [
                "I support 50+ languages including English, Spanish, French, German, Chinese, Japanese, Arabic, Hindi, Portuguese, Russian, and many more!",
                "My language coverage includes European, Asian, African, and American languages. Which specific languages are you interested in?",
                "I work with major world languages plus regional dialects. From Romance to Germanic, Sino-Tibetan to Afroasiatic language families!",
                "Languages I handle: European (English, Spanish, French, German, Italian), Asian (Chinese, Japanese, Korean, Hindi), Arabic, and 40+ others!"
            ],
            'help': [
                "I'm here to help! You can ask me to translate text, analyze sentiment, convert speech to text, or synthesize speech. Created by Vedang to make your multilingual experience awesome!",
                "Sure! I can assist with: 🌐 Translations, 🎤 Speech processing, 😊 Sentiment analysis, 🔊 Audio synthesis! Developed by Vedang with love for language and technology!",
                "Happy to help! Try asking me to translate something, analyze the mood of a text, or convert between speech and text! Built by Vedang to connect you with the world!",
                "Of course! I excel at language translation, emotion analysis, voice processing, and multilingual communication. Designed by Vedang to break down language barriers!"
            ],
            'creator': [
                "I'm your multilingual assistant, created by Vedang who loves combining language and technology to make communication seamless and fun!",
                "This multilingual assistant is proudly designed and developed by Vedang to help you connect with the world in any language!",
                "Built by Vedang with passion for breaking language barriers through AI and making global communication accessible to everyone!",
                "Vedang created me to solve real-world language challenges - from translation to sentiment analysis to speech processing!"
            ],
            'examples': [
                "Try: 'Translate Hello how are you from English to Spanish' or 'Analyze sentiment: I love this beautiful day!' or 'Convert this text to speech'",
                "Examples: 'What languages do you support?', 'Translate bonjour to English', 'Analyze the emotion in this text: [your text]'",
                "Sample requests: 'Speech to text processing', 'Translate from French to German', 'Sentiment analysis of customer feedback'",
                "You can ask: 'Support Hindi translation?', 'Analyze mood: [text]', 'Text-to-speech in Spanish', 'Voice recognition features'"
            ],
            'goodbye': [
                "Goodbye! Thanks for using the multilingual assistant. Come back anytime for translation, analysis, or speech processing!",
                "Farewell! Hope I helped with your language tasks today. I'm always here for multilingual support!",
                "See you later! It was great helping with your language needs. Return anytime for more linguistic assistance!",
                "Take care! Thanks for trying out the multilingual features. I'm ready whenever you need language help again!"
            ]
        }
        
        # Project-focused keywords for intent recognition
        self.intent_keywords = {
            'greeting': ['hello', 'hi', 'hey', 'greetings', 'start'],
            'capabilities': ['what can you do', 'features', 'abilities', 'functions', 'what do you offer'],
            'translation': ['translate', 'translation', 'convert language', 'language conversion', 'from english to', 'to spanish'],
            'speech_processing': ['speech to text', 'text to speech', 'voice', 'audio', 'speech recognition', 'voice synthesis'],
            'sentiment_analysis': ['sentiment', 'emotion', 'mood', 'feeling', 'analyze text', 'emotional tone'],
            'languages': ['what languages', 'which languages', 'language support', 'supported languages'],
            'help': ['help', 'how to use', 'guide', 'instructions', 'examples'],
            'creator': ['who made you', 'who created', 'developer', 'vedang', 'creator'],
            'examples': ['example', 'sample', 'demo', 'show me how', 'how to'],
            'goodbye': ['goodbye', 'bye', 'exit', 'quit', 'end']
        }

        # Supported languages list
        self.supported_languages = [
            'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Chinese', 'Japanese', 
            'Korean', 'Arabic', 'Hindi', 'Russian', 'Dutch', 'Swedish', 'Norwegian', 'Danish',
            'Finnish', 'Polish', 'Czech', 'Hungarian', 'Romanian', 'Bulgarian', 'Croatian',
            'Greek', 'Turkish', 'Hebrew', 'Thai', 'Vietnamese', 'Indonesian', 'Malay'
        ]

    def detect_intent(self, message: str) -> str:
        """Detect user intent focused on project features"""
        message_lower = message.lower()
        
        # Check for translation patterns
        if re.search(r'translate.*from.*to|from.*to.*translate', message_lower):
            return 'translation'
        
        # Check for speech patterns
        if any(term in message_lower for term in ['speech', 'voice', 'audio', 'sound']):
            return 'speech_processing'
        
        # Check for sentiment patterns
        if any(term in message_lower for term in ['sentiment', 'emotion', 'mood', 'feel']):
            return 'sentiment_analysis'
        
        # Standard keyword matching
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return intent
        
        return 'general'

    def extract_language_info(self, message: str) -> Dict:
        """Extract language information from user message"""
        message_lower = message.lower()
        info = {'source_lang': None, 'target_lang': None, 'text_to_process': None}
        
        # Pattern matching for translation requests
        patterns = [
            r'translate\s+"([^"]+)"\s+from\s+(\w+)\s+to\s+(\w+)',
            r'translate\s+([^f]+)\s+from\s+(\w+)\s+to\s+(\w+)',
            r'from\s+(\w+)\s+to\s+(\w+):\s*(.+)',
            r'(\w+)\s+to\s+(\w+):\s*(.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:
                    if 'translate' in pattern:
                        info['text_to_process'] = match.group(1).strip()
                        info['source_lang'] = match.group(2).capitalize()
                        info['target_lang'] = match.group(3).capitalize()
                    else:
                        info['source_lang'] = match.group(1).capitalize()
                        info['target_lang'] = match.group(2).capitalize()
                        info['text_to_process'] = match.group(3).strip()
                break
        
        return info

    def get_contextual_response(self, message: str, intent: str) -> str:
        """Generate contextual response based on project features"""
        
        # Handle specific translation requests
        if intent == 'translation':
            lang_info = self.extract_language_info(message)
            if lang_info['text_to_process'] and lang_info['source_lang'] and lang_info['target_lang']:
                return f"I can help translate '{lang_info['text_to_process']}' from {lang_info['source_lang']} to {lang_info['target_lang']}. In a full implementation, this would connect to the translation API. What other translations do you need?"
            else:
                return random.choice(self.responses['translation'])
        
        # Handle sentiment analysis requests
        if intent == 'sentiment_analysis':
            # Look for text to analyze
            analyze_patterns = [
                r'analyze[:\s]+"([^"]+)"',
                r'analyze[:\s]+(.+)',
                r'sentiment[:\s]+"([^"]+)"',
                r'sentiment[:\s]+(.+)'
            ]
            
            for pattern in analyze_patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    text_to_analyze = match.group(1).strip()
                    return f"Analyzing sentiment for: '{text_to_analyze}'. In a full implementation, this would return detailed emotional analysis including positivity, negativity, and specific emotions detected. What other text would you like me to analyze?"
            
            return random.choice(self.responses['sentiment_analysis'])
        
        # Handle language support queries
        if intent == 'languages':
            sample_languages = random.sample(self.supported_languages, 8)
            return f"I support {len(self.supported_languages)}+ languages including: {', '.join(sample_languages[:6])}, and many more! Which specific language are you interested in working with?"
        
        # Use predefined responses for other intents
        if intent in self.responses:
            return random.choice(self.responses[intent])
        
        # For general queries, keep focused on project scope
        return self.generate_focused_response(message)

    def generate_focused_response(self, message: str) -> str:
        """Generate responses focused on multilingual assistant features"""
        
        # Check if user is asking about specific features
        if any(word in message.lower() for word in ['api', 'integration', 'code', 'development']):
            return "I'm designed for end-users to interact with language features! For API integration and development details, you'd want to check the technical documentation. How can I help you with translation, sentiment analysis, or speech processing right now?"
        
        # Default responses stay within project scope
        focused_responses = [
            "I'm specialized in language processing tasks! Try asking me to translate text, analyze sentiment, or explain my speech processing capabilities.",
            "As a multilingual assistant, I'm here to help with translation, emotion analysis, and speech conversion. What language task can I help you with?",
            "My expertise is in breaking down language barriers! Ask me about translation between languages, sentiment analysis, or voice processing features.",
            "I focus on multilingual communication tools. Would you like to try translation, text sentiment analysis, or learn about speech processing?",
            "Let me help you with language-related tasks! I can translate, analyze emotions in text, or process speech. What interests you most?"
        ]
        
        return random.choice(focused_responses)

    def add_to_history(self, user_input: str, bot_response: str, intent: str):
        """Add conversation to history"""
        timestamp = datetime.datetime.now()
        self.conversation_history.extend([
            {'user': user_input, 'timestamp': timestamp},
            {'bot': bot_response, 'timestamp': timestamp, 'intent': intent}
        ])

# Global chatbot instance (better approach)
_global_chatbot = None

def get_global_chatbot() -> MultilingualChatbot:
    """Get or create global chatbot instance"""
    global _global_chatbot
    if _global_chatbot is None:
        _global_chatbot = MultilingualChatbot()
    return _global_chatbot

def reset_chatbot():
    """Reset the global chatbot instance"""
    global _global_chatbot
    _global_chatbot = None

# SOLUTION 1: Using Global Instance (Recommended for simple cases)
def get_chatbot_response_v1(user_input: str, history: List[Dict] = None) -> Tuple[str, Dict]:
    """
    Get chatbot response using global instance approach
    """
    chatbot = get_global_chatbot()
    
    # Detect intent
    intent = chatbot.detect_intent(user_input)
    
    # Generate response
    response = chatbot.get_contextual_response(user_input, intent)
    
    # Add to conversation history
    chatbot.add_to_history(user_input, response, intent)
    
    # Prepare metadata
    metadata = {
        'intent': intent,
        'supported_features': ['translation', 'sentiment_analysis', 'speech_processing'],
        'conversation_length': len(chatbot.conversation_history),
        'timestamp': datetime.datetime.now().isoformat(),
        'project_focus': 'multilingual_assistant'
    }
    
    return response, metadata

# SOLUTION 2: Using Session-based approach (Recommended for Streamlit)
def get_chatbot_response_v2(user_input: str, session_state=None) -> Tuple[str, Dict]:
    """
    Get chatbot response using session state (for Streamlit)
    Usage in Streamlit: response, metadata = get_chatbot_response_v2(user_input, st.session_state)
    """
    
    # Initialize chatbot in session state if not exists
    if session_state is not None:
        if 'chatbot' not in session_state:
            session_state.chatbot = MultilingualChatbot()
        chatbot = session_state.chatbot
    else:
        # Fallback to global instance
        chatbot = get_global_chatbot()
    
    # Detect intent
    intent = chatbot.detect_intent(user_input)
    
    # Generate response
    response = chatbot.get_contextual_response(user_input, intent)
    
    # Add to conversation history
    chatbot.add_to_history(user_input, response, intent)
    
    # Prepare metadata
    metadata = {
        'intent': intent,
        'supported_features': ['translation', 'sentiment_analysis', 'speech_processing'],
        'conversation_length': len(chatbot.conversation_history),
        'timestamp': datetime.datetime.now().isoformat(),
        'project_focus': 'multilingual_assistant'
    }
    
    return response, metadata

# SOLUTION 3: Factory pattern with session management
class ChatbotManager:
    """Manages chatbot instances with proper session handling"""
    
    def __init__(self):
        self.sessions = {}
    
    def get_chatbot(self, session_id: str = "default") -> MultilingualChatbot:
        """Get or create chatbot for session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = MultilingualChatbot()
            self.sessions[session_id].session_id = session_id
        return self.sessions[session_id]
    
    def reset_session(self, session_id: str = "default"):
        """Reset specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def clear_all_sessions(self):
        """Clear all sessions"""
        self.sessions.clear()

# Global manager instance
_chatbot_manager = ChatbotManager()

def get_chatbot_response_v3(user_input: str, session_id: str = "default") -> Tuple[str, Dict]:
    """
    Get chatbot response using manager pattern
    """
    chatbot = _chatbot_manager.get_chatbot(session_id)
    
    # Detect intent
    intent = chatbot.detect_intent(user_input)
    
    # Generate response
    response = chatbot.get_contextual_response(user_input, intent)
    
    # Add to conversation history
    chatbot.add_to_history(user_input, response, intent)
    
    # Prepare metadata
    metadata = {
        'intent': intent,
        'supported_features': ['translation', 'sentiment_analysis', 'speech_processing'],
        'conversation_length': len(chatbot.conversation_history),
        'timestamp': datetime.datetime.now().isoformat(),
        'project_focus': 'multilingual_assistant',
        'session_id': session_id
    }
    
    return response, metadata

# Main function (choose your preferred approach)
def get_chatbot_response(user_input: str, session_state=None, session_id: str = "default") -> Tuple[str, Dict]:
    """
    Main chatbot response function - uses session state if available, otherwise falls back to session manager
    """
    if session_state is not None:
        return get_chatbot_response_v2(user_input, session_state)
    else:
        return get_chatbot_response_v3(user_input, session_id)

# Utility functions
def clear_conversation_history(session_state=None, session_id: str = "default"):
    """Clear conversation history"""
    if session_state is not None and hasattr(session_state, 'chatbot'):
        session_state.chatbot.conversation_history = []
    else:
        _chatbot_manager.reset_session(session_id)

def get_conversation_history(session_state=None, session_id: str = "default") -> List[Dict]:
    """Get conversation history"""
    if session_state is not None and hasattr(session_state, 'chatbot'):
        return session_state.chatbot.conversation_history
    else:
        chatbot = _chatbot_manager.get_chatbot(session_id)
        return chatbot.conversation_history

# Simplified function for quick implementation (unchanged)
def quick_multilingual_response(user_input: str) -> Tuple[str, Dict]:
    """Quick focused responses for multilingual assistant"""
    
    focused_responses = {
        'hello': "Hello! I'm your multilingual assistant by Vedang. Ready to help with translation, sentiment analysis, or speech processing!",
        'translate': "I can translate between 50+ languages! Specify: 'Translate [text] from [language] to [language]' and I'll help you.",
        'sentiment': "I analyze emotions in text across multiple languages! Share your text and I'll detect the sentiment and mood.",
        'speech': "I handle speech-to-text and text-to-speech in multiple languages! What audio processing do you need?",
        'languages': "I support major world languages: English, Spanish, French, German, Chinese, Japanese, Arabic, Hindi, and 40+ more!",
        'help': "I specialize in: 🌐 Translation, 😊 Sentiment Analysis, 🎤 Speech Processing. Created by Vedang to break language barriers!",
        'capabilities': "My core features: Real-time translation, emotion analysis, voice recognition, and speech synthesis across 50+ languages!",
        'bye': "Goodbye! Thanks for using the multilingual assistant. Come back for more translation and language processing!"
    }
    
    user_lower = user_input.lower()
    
    for key, response in focused_responses.items():
        if key in user_lower:
            return response, {'intent': key, 'project_scope': 'multilingual_features'}
    
    # Default focused response
    return "I'm your multilingual assistant! I can help with translation, sentiment analysis, or speech processing. What language task would you like to try?", {'intent': 'general', 'project_scope': 'multilingual_features'}

# Usage examples for different approaches:
if __name__ == "__main__":
    print("=== Testing Different Chatbot Approaches ===\n")
    
    test_messages = [
        "Hello!",
        "What can you do?",
        "Translate hello from English to Spanish",
        "Analyze sentiment: I love this product!",
        "What languages do you support?",
        "Goodbye!"
    ]
    
    print("--- Approach 1: Global Instance ---")
    for message in test_messages:
        response, metadata = get_chatbot_response_v1(message)
        print(f"User: {message}")
        print(f"Bot: {response}")
        print(f"Intent: {metadata['intent']}")
        print("-" * 40)
    
    print("\n--- Approach 3: Session Manager ---")
    for message in test_messages:
        response, metadata = get_chatbot_response_v3(message, "test_session")
        print(f"User: {message}")
        print(f"Bot: {response}")
        print(f"Intent: {metadata['intent']}")
        print("-" * 40)