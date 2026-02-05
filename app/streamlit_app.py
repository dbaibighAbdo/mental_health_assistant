import streamlit as st
from chatbot import MentalHealthChatbot
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .subheader {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .crisis-alert {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stats-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    """Load chatbot model (cached for performance)"""
    return MentalHealthChatbot()


def get_emotion_color(emotion: str) -> str:
    """Return color code for emotion"""
    colors = {
        "suicidal": "#dc3545",
        "depression": "#6c757d",
        "anxiety": "#ffc107",
        "stress": "#fd7e14",
        "bipolar": "#6f42c1",
        "personality disorder": "#e83e8c",
        "normal": "#28a745",
        "neutral": "#17a2b8"
    }
    return colors.get(emotion.lower(), "#6c757d")


def display_message(message, is_user=True, emotion=None, confidence=None):
    """Display a chat message with styling"""
    message_class = "user-message" if is_user else "bot-message"
    sender = "You" if is_user else "Assistant"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <strong>{sender}:</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)
        
        if not is_user and emotion:
            color = get_emotion_color(emotion)
            confidence_text = f" ({confidence:.1%})" if confidence else ""
            st.markdown(f"""
            <span class="emotion-badge" style="background-color: {color}; color: white;">
                {emotion}{confidence_text}
            </span>
            """, unsafe_allow_html=True)


def display_crisis_resources():
    """Display crisis resources in sidebar"""
    st.sidebar.markdown("""
    <div class="crisis-alert">
        <h4>🆘 Crisis Resources</h4>
        <p>If you're in crisis, please reach out:</p>
        <ul>
            <li><strong>988</strong> - Suicide & Crisis Lifeline (US)</li>
            <li><strong>741741</strong> - Crisis Text Line (Text HOME)</li>
            <li><strong>International:</strong> <a href="https://www.iasp.info/resources/Crisis_Centres/" target="_blank">IASP Crisis Centers</a></li>
        </ul>
        <p><small>Available 24/7</small></p>
    </div>
    """, unsafe_allow_html=True)


def display_emotion_chart(emotion_history):
    """Display emotion tracking chart"""
    if emotion_history:
        df = pd.DataFrame(emotion_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        for emotion in df['emotion'].unique():
            emotion_data = df[df['emotion'] == emotion]
            fig.add_trace(go.Scatter(
                x=emotion_data['timestamp'],
                y=emotion_data['confidence'],
                mode='lines+markers',
                name=emotion,
                line=dict(color=get_emotion_color(emotion))
            ))
        
        fig.update_layout(
            title="Emotion Tracking Over Time",
            xaxis_title="Time",
            yaxis_title="Confidence",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        with st.spinner('Loading chatbot models... This may take a minute.'):
            st.session_state.chatbot = load_chatbot()
        st.success('✅ Chatbot loaded successfully!')
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Mental Health Support Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">A safe space to talk about your feelings</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Conversation Insights")
        
        # Statistics
        if st.session_state.messages:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.metric("Total Messages", len(st.session_state.messages))
            if st.session_state.emotion_history:
                latest_emotion = st.session_state.emotion_history[-1]['emotion']
                st.metric("Current Emotion", latest_emotion)
                st.metric("Confidence", f"{st.session_state.emotion_history[-1]['confidence']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Emotion chart
        if len(st.session_state.emotion_history) > 1:
            st.subheader("Emotion Trends")
            display_emotion_chart(st.session_state.emotion_history)
        
        # Controls
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.emotion_history = []
            st.session_state.chatbot.clear_history(st.session_state.session_id)
            st.rerun()
        
        if st.button("💾 Export Chat", use_container_width=True):
            chat_export = {
                "session_id": st.session_state.session_id,
                "timestamp": datetime.now().isoformat(),
                "messages": [
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                        "emotion": msg.get("emotion"),
                        "timestamp": msg.get("timestamp")
                    }
                    for msg in st.session_state.messages
                ]
            }
            st.download_button(
                label="Download JSON",
                data=str(chat_export),
                file_name=f"chat_export_{st.session_state.session_id}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Crisis resources
        st.markdown("---")
        display_crisis_resources()
    
    # Main chat area
    chat_container = st.container()
    
    with chat_container:
        # Display conversation history
        for msg in st.session_state.messages:
            display_message(
                msg["content"],
                is_user=(msg["role"] == "user"),
                emotion=msg.get("emotion"),
                confidence=msg.get("confidence")
            )
    
    # Chat input
    st.markdown("---")
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.chat_input("Share what's on your mind...", key="chat_input")
    
    with col2:
        if st.button("📤 Send", use_container_width=True):
            # Trigger input processing if there's text
            pass
    
    # Process user input
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Show typing indicator
        with st.spinner("Assistant is typing..."):
            # Get bot response
            response, emotion, confidence = st.session_state.chatbot.chat(
                user_input,
                st.session_state.session_id
            )
            
            # Add bot message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "emotion": emotion,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update emotion history
            st.session_state.emotion_history.append({
                "emotion": emotion,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })
        
        # Rerun to display new messages
        st.rerun()
    
    # Quick action buttons
    st.markdown("---")
    st.markdown("#### 💬 Quick Prompts")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_prompts = [
        ("😊", "I'm feeling good today"),
        ("😰", "I'm feeling anxious"),
        ("😔", "I'm feeling down"),
        ("💭", "I need someone to talk to")
    ]
    
    for col, (emoji, prompt) in zip([col1, col2, col3, col4], quick_prompts):
        if col.button(f"{emoji} {prompt}", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            with st.spinner("Assistant is typing..."):
                response, emotion, confidence = st.session_state.chatbot.chat(
                    prompt,
                    st.session_state.session_id
                )
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "emotion": emotion,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                })
                
                st.session_state.emotion_history.append({
                    "emotion": emotion,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                })
            
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>⚠️ <strong>Disclaimer:</strong> This chatbot is for supportive conversation only and is not a substitute for professional mental health care.</p>
        <p>If you're experiencing a mental health crisis, please contact emergency services or a crisis helpline immediately.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()