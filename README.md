# PeakMind Wellbeing Coach

An AI-powered chatbot serving as a personal guide for wellness and productivity. This project showcases practical application of the Gemini API, effective prompt engineering, a local vector database for knowledge retrieval, and a well-structured Retrieval-Augmented Generation (RAG) pipeline.

## Features

- 🤖 AI-powered wellbeing and productivity coaching
- 📚 Knowledge base with wellness and productivity content
- 🔍 Semantic search using FAISS vector database
- 👤 User profile and personalization
- 💬 Natural conversation interface
- 🎯 Domain-specific responses

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd PeakMindWellbeingCoach
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run WellbeingCoachApp.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Project Structure

- `WellbeingCoachApp.py`: Main application file
- `knowledge_base.py`: Knowledge base content and management
- `vector_db.py`: Vector database management
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (create this file)

## RAG Pipeline

The application implements a Retrieval-Augmented Generation (RAG) pipeline:

1. User Query Input
   - User submits a question through the chat interface

2. Query Preprocessing
   - Question is analyzed for user profile updates
   - Semantic search query is prepared

3. Information Retrieval
   - Vector database is searched for relevant content
   - Top-k most similar documents are retrieved

4. Augmented Prompt Construction
   - Retrieved context is combined with system prompt
   - User profile information is included
   - Question is formatted for the model

5. LLM Call
   - Gemini Pro model generates response
   - Response is contextualized with retrieved information

6. Response Post-processing
   - Response is formatted for display
   - User profile is updated
   - Interaction is logged

## Knowledge Base

The knowledge base includes information about:
- Productivity techniques (Pomodoro, time blocking)
- Mindfulness and meditation
- Stress management
- Work-life balance
- Personal development

## Ethical Considerations

- The chatbot is designed to provide general wellness and productivity advice only
- It is not a replacement for professional medical or mental health services
- Users are advised to consult qualified professionals for serious concerns
- The system includes appropriate disclaimers and redirections

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 