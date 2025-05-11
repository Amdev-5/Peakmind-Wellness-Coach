from typing import List, Dict
import json

# Knowledge base entries for wellness and productivity
KNOWLEDGE_BASE = [
    {
        "category": "Productivity Techniques",
        "title": "Pomodoro Technique",
        "content": """The Pomodoro Technique is a time management method developed by Francesco Cirillo in the late 1980s. It uses a timer to break work into intervals, traditionally 25 minutes in length, separated by short breaks. Each interval is known as a "pomodoro", the Italian word for tomato, after the tomato-shaped kitchen timer Cirillo used as a university student.

Key steps:
1. Choose a task to work on
2. Set the timer for 25 minutes
3. Work on the task until the timer rings
4. Take a short 5-minute break
5. After four pomodoros, take a longer break (15-30 minutes)

Benefits:
- Improved focus and concentration
- Reduced mental fatigue
- Better time management
- Increased productivity
- Clear work/break boundaries"""
    },
    {
        "category": "Productivity Techniques",
        "title": "Time Blocking",
        "content": """Time blocking is a productivity method where you divide your day into blocks of time. Each block is dedicated to accomplishing a specific task or group of tasks.

How to implement:
1. Review your tasks and priorities
2. Estimate time needed for each task
3. Create blocks in your calendar
4. Include buffer time between blocks
5. Stick to the schedule

Benefits:
- Better focus on important tasks
- Reduced context switching
- Clear daily structure
- Improved work-life balance
- More realistic time estimates"""
    },
    {
        "category": "Mindfulness",
        "title": "Basic Meditation",
        "content": """Meditation is a practice where an individual uses a technique to train attention and awareness, achieving a mentally clear and emotionally calm state.

Simple meditation steps:
1. Find a quiet, comfortable space
2. Sit in a comfortable position
3. Close your eyes
4. Focus on your breath
5. When your mind wanders, gently bring it back to your breath
6. Start with 5 minutes and gradually increase

Benefits:
- Reduced stress and anxiety
- Improved focus and concentration
- Better emotional regulation
- Increased self-awareness
- Better sleep quality"""
    },
    {
        "category": "Stress Management",
        "title": "Deep Breathing Exercise",
        "content": """Deep breathing is a simple but effective way to reduce stress and anxiety. It helps activate the body's relaxation response.

4-7-8 Breathing Technique:
1. Inhale quietly through your nose for 4 seconds
2. Hold your breath for 7 seconds
3. Exhale completely through your mouth for 8 seconds
4. Repeat 4 times

Benefits:
- Immediate stress relief
- Lower blood pressure
- Reduced anxiety
- Better oxygen flow
- Improved focus"""
    },
    {
        "category": "Work-Life Balance",
        "title": "Setting Boundaries",
        "content": """Setting healthy boundaries is crucial for maintaining work-life balance and preventing burnout.

Key strategies:
1. Define clear work hours
2. Learn to say no
3. Take regular breaks
4. Separate work and personal spaces
5. Set communication expectations
6. Prioritize self-care

Benefits:
- Reduced stress
- Better relationships
- Increased productivity
- Improved mental health
- More time for personal life"""
    }
]

def get_knowledge_base() -> List[Dict]:
    """Return the knowledge base entries."""
    return KNOWLEDGE_BASE

def save_knowledge_base_to_json(filename: str = "knowledge_base.json"):
    """Save the knowledge base to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(KNOWLEDGE_BASE, f, indent=2)

def load_knowledge_base_from_json(filename: str = "knowledge_base.json") -> List[Dict]:
    """Load the knowledge base from a JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return KNOWLEDGE_BASE

if __name__ == "__main__":
    # Save the knowledge base to a JSON file
    save_knowledge_base_to_json() 