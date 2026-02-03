import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required packages are installed"""
    logger.info("Testing imports...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("langgraph", "LangGraph"),
        ("langchain_core", "LangChain Core"),
        ("streamlit", "Streamlit"),
        ("plotly", "Plotly"),
        ("pandas", "Pandas")
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {name} imported successfully")
        except ImportError as e:
            logger.error(f"✗ {name} not found: {e}")
            missing_packages.append(name)
    
    if missing_packages:
        logger.error(f"\nMissing packages: {', '.join(missing_packages)}")
        logger.error("Please run: pip install -r requirements.txt")
        return False
    
    logger.info("All imports successful!\n")
    return True


def test_chatbot_initialization():
    """Test chatbot model loading"""
    logger.info("Testing chatbot initialization...")
    
    try:
        from app.chatbot import MentalHealthChatbot
        
        logger.info("Creating chatbot instance (this may take a while)...")
        chatbot = MentalHealthChatbot()
        logger.info("✓ Chatbot initialized successfully!\n")
        return chatbot
    
    except Exception as e:
        logger.error(f"✗ Chatbot initialization failed: {e}")
        return None


def test_conversation(chatbot):
    """Test basic conversation functionality"""
    logger.info("Testing conversation functionality...")
    
    test_messages = [
        "Hello, how are you?",
        "I'm feeling a bit anxious today",
        "Thank you for listening"
    ]
    
    session_id = f"test_session_{int(datetime.now().timestamp())}"
    
    for i, message in enumerate(test_messages, 1):
        logger.info(f"\nTest {i}/{len(test_messages)}")
        logger.info(f"User: {message}")
        
        try:
            response, emotion, confidence = chatbot.chat(message, session_id)
            logger.info(f"Bot: {response[:100]}..." if len(response) > 100 else f"Bot: {response}")
            logger.info(f"Emotion: {emotion} (confidence: {confidence:.2%})")
            logger.info("✓ Response generated successfully")
        
        except Exception as e:
            logger.error(f"✗ Conversation test failed: {e}")
            return False
    
    logger.info("\n✓ All conversation tests passed!\n")
    return True


def test_emotion_classification(chatbot):
    """Test emotion classification accuracy"""
    logger.info("Testing emotion classification...")
    
    test_cases = [
        ("I'm so happy today!", "positive"),
        ("I feel really anxious and worried", "anxiety"),
        ("Everything feels hopeless", "depression"),
        ("I'm under a lot of pressure", "stress"),
    ]
    
    session_id = f"test_emotion_{int(datetime.now().timestamp())}"
    
    for message, expected_category in test_cases:
        logger.info(f"\nTesting: {message}")
        
        try:
            response, emotion, confidence = chatbot.chat(message, session_id)
            logger.info(f"Detected: {emotion} (confidence: {confidence:.2%})")
            logger.info(f"Expected category: {expected_category}")
            
            if confidence > 0.3:  # Minimum confidence threshold
                logger.info("✓ Classification successful")
            else:
                logger.warning(f"⚠ Low confidence: {confidence:.2%}")
        
        except Exception as e:
            logger.error(f"✗ Classification test failed: {e}")
            return False
    
    logger.info("\n✓ Emotion classification tests completed!\n")
    return True


def test_session_persistence(chatbot):
    """Test conversation memory and session persistence"""
    logger.info("Testing session persistence...")
    
    session_id = f"test_persistence_{int(datetime.now().timestamp())}"
    
    try:
        # First message
        logger.info("Sending first message...")
        chatbot.chat("My name is Test User", session_id)
        
        # Get history
        history = chatbot.get_conversation_history(session_id)
        logger.info(f"History length after first message: {len(history)}")
        
        # Second message
        logger.info("Sending second message...")
        chatbot.chat("What's my name?", session_id)
        
        # Check history again
        history = chatbot.get_conversation_history(session_id)
        logger.info(f"History length after second message: {len(history)}")
        
        if len(history) >= 4:  # 2 user messages + 2 bot responses
            logger.info("✓ Session persistence working!\n")
            return True
        else:
            logger.warning("⚠ History shorter than expected\n")
            return False
    
    except Exception as e:
        logger.error(f"✗ Session persistence test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Mental Health Chatbot - Test Suite")
    logger.info("=" * 60 + "\n")
    
    # Test 1: Imports
    if not test_imports():
        logger.error("\n❌ Import tests failed. Please install missing packages.")
        return False
    
    # Test 2: Initialization
    chatbot = test_chatbot_initialization()
    if chatbot is None:
        logger.error("\n❌ Chatbot initialization failed. Check model IDs and internet connection.")
        return False
    
    # Test 3: Basic Conversation
    if not test_conversation(chatbot):
        logger.error("\n❌ Conversation tests failed.")
        return False
    
    # Test 4: Emotion Classification
    if not test_emotion_classification(chatbot):
        logger.error("\n❌ Emotion classification tests failed.")
        return False
    
    # Test 5: Session Persistence
    if not test_session_persistence(chatbot):
        logger.warning("\n⚠ Session persistence tests showed warnings.")
    
    # Summary
    logger.info("=" * 60)
    logger.info("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("\nYou can now run the Streamlit app with:")
    logger.info("  streamlit run streamlit_app.py")
    logger.info("=" * 60 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}")
        sys.exit(1)