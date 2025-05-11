import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
from typing import List, Dict, Optional
from vector_db import VectorDBManager, initialize_vector_db
from knowledge_base import get_knowledge_base
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# System prompt for the Wellbeing Coach
SYSTEM_PROMPT = """You are PeakMind Coach, an AI-powered wellbeing and productivity assistant. Your role is to provide helpful, empathetic advice strictly confined to wellness and productivity topics.

Key characteristics:
- Helpful and empathetic
- Focused on wellness and productivity
- Encouraging and supportive
- Professional but friendly

IMPORTANT DISCLAIMER: I am an AI assistant, PeakMind Coach, and not a medical or certified professional. My advice is for informational purposes. For serious concerns, please consult a qualified professional.

When responding:
1. Stay within wellness and productivity topics
2. Show empathy and understanding
3. Provide practical, actionable advice
4. If asked about non-wellness topics, politely redirect to wellness
5. Use the provided context to enhance responses
6. Consider the user's profile for personalization

Current context: {context}

User profile: {user_profile}

Question: {question}

Please provide a helpful, empathetic response:"""

class UserProfile:
    def __init__(self):
        self.profile = {
            "user_id": "sim_user_001",
            "name": None,
            "age": None,
            "gender": None,
            "occupation": None,
            "interests": [],
            "current_goal": None,
            "last_expressed_mood": "😐 Neutral",  # Set default mood
            "mood_history": [],
            "interaction_history": [],
            "preferences": {
                "notification_frequency": "Daily",
                "focus_areas": [],
                "language": "English"
            }
        }
    
    def update_profile(self, profile_data: Dict):
        """Update multiple profile fields at once."""
        for key, value in profile_data.items():
            if key in self.profile:
                self.profile[key] = value
    
    def update_goal(self, goal: str):
        self.profile["current_goal"] = goal
    
    def update_mood(self, mood: str):
        """Update mood with validation to ensure it matches available options."""
        valid_moods = ["😢 Stressed", "😔 Low", "😐 Neutral", "🙂 Good", "😊 Great"]
        if mood in valid_moods:
            self.profile["last_expressed_mood"] = mood
            self.profile["mood_history"].append({
                "mood": mood,
                "timestamp": datetime.now().isoformat()
            })
    
    def add_interest(self, interest: str):
        if interest not in self.profile["interests"]:
            self.profile["interests"].append(interest)
    
    def remove_interest(self, interest: str):
        if interest in self.profile["interests"]:
            self.profile["interests"].remove(interest)
    
    def add_interaction(self, interaction: Dict):
        self.profile["interaction_history"].append(interaction)
    
    def get_profile(self) -> Dict:
        return self.profile

def get_conversational_chain():
    prompt_template = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "user_profile", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        temperature=0.7,
        convert_system_message_to_human=True
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)
    return chain

def process_user_input(user_question: str, user_profile: UserProfile, vector_db: VectorDBManager):
    try:
        # Get relevant context from vector store (now Document objects)
        context_docs = vector_db.semantic_search(user_question)
        context = "\n\n".join([doc.page_content for doc in context_docs]) if context_docs else ""
        
        # Update user profile based on the question
        if "goal" in user_question.lower():
            user_profile.update_goal(user_question)
        if any(mood in user_question.lower() for mood in ["happy", "sad", "stressed", "anxious", "tired"]):
            user_profile.update_mood(user_question)
        
        # Get response from the model
        chain = get_conversational_chain()
        response = chain(
            {
                "input_documents": context_docs if context_docs else [],
                "context": context,
                "user_profile": json.dumps(user_profile.get_profile()),
                "question": user_question
            },
            return_only_outputs=True
        )
        
        # Handle response text encoding
        response_text = response["output_text"]
        try:
            # Try to encode and decode to handle any encoding issues
            response_text = response_text.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception as e:
            st.error(f"Error encoding response: {str(e)}")
            response_text = "I apologize, but I encountered an error processing the response."
        
        # Add interaction to history
        user_profile.add_interaction({
            "question": user_question,
            "response": response_text
        })
        
        return response_text
    
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        return "I apologize, but I encountered an error. Please try again."

def display_profile_form():
    """Display and handle the profile setup form in the sidebar."""
    st.sidebar.title("Your Profile")
    
    # Basic Information
    st.sidebar.subheader("Basic Information")
    name = st.sidebar.text_input("Name", value=st.session_state.user_profile.profile["name"] or "")
    age = st.sidebar.number_input("Age", min_value=13, max_value=100, value=st.session_state.user_profile.profile["age"] or 25)
    gender = st.sidebar.selectbox(
        "Gender",
        options=["Prefer not to say", "Male", "Female", "Non-binary", "Other"],
        index=0 if not st.session_state.user_profile.profile["gender"] else 
        ["Prefer not to say", "Male", "Female", "Non-binary", "Other"].index(st.session_state.user_profile.profile["gender"])
    )
    occupation = st.sidebar.text_input("Occupation", value=st.session_state.user_profile.profile["occupation"] or "")
    
    # Interests
    st.sidebar.subheader("Interests")
    interests = st.sidebar.multiselect(
        "Select your interests",
        options=["Meditation", "Yoga", "Fitness", "Nutrition", "Sleep", "Stress Management", 
                "Productivity", "Time Management", "Personal Development", "Mental Health"],
        default=st.session_state.user_profile.profile["interests"]
    )
    
    # Current Mood
    st.sidebar.subheader("Current Mood")
    mood_options = ["😢 Stressed", "😔 Low", "😐 Neutral", "🙂 Good", "😊 Great"]
    current_mood = st.session_state.user_profile.profile["last_expressed_mood"]
    if current_mood not in mood_options:
        current_mood = "😐 Neutral"  # Default to neutral if current mood is invalid
    
    mood = st.sidebar.select_slider(
        "How are you feeling today?",
        options=mood_options,
        value=current_mood
    )
    
    # Goals
    st.sidebar.subheader("Current Goal")
    goal = st.sidebar.text_area(
        "What's your current wellness or productivity goal?",
        value=st.session_state.user_profile.profile["current_goal"] or ""
    )
    
    # Preferences
    st.sidebar.subheader("Preferences")
    notification_options = ["Never", "Daily", "Weekly"]
    current_freq = st.session_state.user_profile.profile["preferences"]["notification_frequency"]
    try:
        freq_index = notification_options.index(current_freq)
    except ValueError:
        freq_index = 1  # Default to "Daily" if current value is not in options
    
    notification_freq = st.sidebar.selectbox(
        "Notification Frequency",
        options=notification_options,
        index=freq_index
    )
    
    focus_areas = st.sidebar.multiselect(
        "Focus Areas",
        options=["Physical Health", "Mental Health", "Productivity", "Work-Life Balance", "Personal Growth"],
        default=st.session_state.user_profile.profile["preferences"]["focus_areas"]
    )
    
    # Save button
    if st.sidebar.button("Save Profile"):
        st.session_state.user_profile.update_profile({
            "name": name,
            "age": age,
            "gender": gender,
            "occupation": occupation,
            "interests": interests,
            "current_goal": goal,
            "last_expressed_mood": mood,
            "preferences": {
                "notification_frequency": notification_freq,
                "focus_areas": focus_areas,
                "language": st.session_state.user_profile.profile["preferences"]["language"]
            }
        })
        st.sidebar.success("Profile updated successfully!")

def main():
    st.set_page_config(
        page_title="PeakMind Wellbeing Coach",
        page_icon="🧘",
        layout="wide"
    )
    
    st.title("🧘 PeakMind Wellbeing Coach")
    st.markdown("""
    Your personal AI guide for wellness and productivity. I'm here to help you with:
    - Productivity techniques
    - Mindfulness and meditation
    - Stress management
    - Work-life balance
    - Personal development
    """)
    
    # Initialize session state
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = UserProfile()
    
    if "vector_db" not in st.session_state:
        with st.spinner("Initializing knowledge base..."):
            st.session_state.vector_db = initialize_vector_db()
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            try:
                # Handle chat history message encoding
                content = message["content"].encode('utf-8', errors='ignore').decode('utf-8')
                st.write(content)
            except Exception as e:
                st.error(f"Error displaying message: {str(e)}")
                st.write("Error displaying message content")
    
    # Chat interface
    user_question = st.chat_input("Ask me anything about wellness and productivity...")
    
    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            response = process_user_input(
                user_question,
                st.session_state.user_profile,
                st.session_state.vector_db
            )
            try:
                st.write(response)
            except Exception as e:
                st.error(f"Error displaying response: {str(e)}")
                st.write("Error displaying response content")
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Display profile form in sidebar
    display_profile_form()
    
    # Display available topics
    with st.sidebar:
        st.title("Available Topics")
        knowledge_base = get_knowledge_base()
        for entry in knowledge_base:
            st.markdown(f"**{entry['title']}** ({entry['category']})")
        
        # Add clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main() 