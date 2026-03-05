

## 📋 Installation

### Prerequisites
- Python 3.13.3
- pip (Python package installer)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- **FastAPI** - Web framework for building APIs
- **LangChain** - AI framework for language model interactions
- **LangGraph** - Graph-based workflow orchestration
- **Google Generative AI** - Access to Gemini language models
- **Redis** - Data persistence and memory management
- And other supporting libraries

### Step 2: Set Up Environment Variables

Create a `.env` file in the project root with your API keys:

```
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 🚀 Running the Application

### Start the FastAPI Server

```bash
fastapi dev app.py
```

The server will start on `http://localhost:8000`

**What this does:**
- Initializes connection to Google's Gemini AI models
- Connects to Redis database for memory storage
- Sets up all API endpoints and listening ports
- Enables hot reload for development (auto-restarts on code changes)

---

##  API Endpoints Documentation

### 1. **Home / User Welcome** (still working on it I need somehelp from u in this endpoint it shouldnt take taht much )
**Endpoint:** `GET /`

**What it does:**
- the system  prompt for every user that contain his perferences name and everything 

**Input:**
```
user_id: string (your unique user ID)
```

**Output:**
```json
a string
```

**Long-term Memory Access:**
- Retrieves saved user profile and preferences
- Personalizes greeting based on past interactions
- Maintains learning history and goals

---

### 2. **Quiz Generation**
**Endpoint:** `POST /quiz/`
**where to access it from**
from either the chatbot directly and the chat bot willl fill the iuntput or accing it via a form or something like that 
**What it does:**
- Generates 30 multiple-choice quiz questions
- Customized by subject and topic
- AI-designed to accurately assess understanding
- Engaging and educational questions

**Input:**
```
subject: string (e.g., "Mathematics", "Science", "History")
topic: string (e.g., "Photosynthesis", "Algebra", "World War II")
```

**Output:**
```json
{
  "quiz": [
    {
      "question": "What is photosynthesis?",
      "options": [
        "Process of converting light to energy",
        "Breaking down food for energy",
        "Moving water through plants",
        "Creating new plant cells"
      ]
    },
    ...
  ]
}
```

---

### 3. **Learning Document Generation**
**Endpoint:** `POST /document/
**notes** need to ask issam but I probably believe that he wants summary instrad of a document
**What it does:**
- Creates comprehensive study materials on any topic
- Structures content into clear modules
- Includes engaging introductions and examples
- Provides motivational conclusions
- Designed for different age levels

**Input:**
```
topic: string (e.g., "The Solar System", "Photosynthesis")
```

**Output:**
```json
{
  "title": "Understanding Photosynthesis",
  "topic": "Photosynthesis",
  "introduction": "Learn how plants make their own food...",
  "modules": [
    {
      "module_name": "Introduction to Photosynthesis",
      "introduction": "...",
      "content": "...",
      "sub_modules": [
        {
          "sub_module_title": "Light-Dependent Reactions",
          "content": "...",
          "conclusion": "..."
        }
      ],
      "conclusion": "..."
    }
  ],
  "conclusion": "You now understand photosynthesis!",
  "motivation": "Keep exploring the amazing world of plants!"
}
```

**Features:**
- **Adaptive Learning:** Adjusts content complexity based on topic
- **Structured Learning:** Breaks complex topics into digestible modules
- **Sub-modules:** For complex topics requiring deeper exploration
- **Engagement Focus:** Uses real-world examples and analogies

---

### 4. **Interactive Chatbot**
**Endpoint:** `POST /chatbot/`

**What it does:**
- Maintains multi-turn conversations with students
- Accesses long-term memory for personalized responses
- if u upload the pdf file it wil transfer it into a script than a tts model(I did figure right now that it probably need to be separate)
- Can process PDF uploads for document-based learning
- Remembers conversation context within a thread
- Routes users to appropriate tools/pages as needed

**Input:**
```
user_input: string (your message to the chatbot)
thr_id: string (conversation thread ID - maintains chat history)
usr_id: string (your user ID for long term memory )
file: UploadFile (optional - PDF document to analyze) for now to turn  a pdf into a script
```

**Output:**
```json
{
  "aimessage": {
    "messages": [
      {
        "type": "HumanMessage",
        "content": "Can you explain photosynthesis?"
      },
      {
        "type": "AIMessage",
        "content": "Of course! Photosynthesis is how plants..."
      }
    ]
  }
}
it can also transfer u to the quiz or to the document endpoint
```

**Memory Capabilities:**
- **Long-term Memory:** Remembers user preferences, interests, learning style
- **Short-term Memory:** Tracks current conversation context within a thread
- **PDF Processing:** Can extract and explain content from uploaded documents
- **Personalization:** Adapts tone and complexity to each student

---

### 5. **Student Evaluation & Feedback**
**Endpoint:** `POST /evaluate/`

**What it does:**
- Evaluates quiz answers and provides detailed feedback
- Generates personalized study recommendations
- Creates learning documents based on weak areas
- Produces avatar script for voice feedback (if needed)
- Assesses student level and suggests difficulty for next quiz

**Input:**
```json
{
  "question_1": "A",
  "question_2": "B",
  "question_3": "C",
  ...
}
```

**Output:**
```json
{
  "feedback": {
    "correct_answers": 24,
    "wrong_answers": 6,
    "eval": [
      {
        "wrong": "C",
        "corrections": ["The correct answer is A", "This is because..."],
        "explanation": "Imagine it like a game where...",
        "advice": "Remember to focus on..."
      }
    ],
    "what_should_i_do_to_improve": "Practice more questions about...",
    "level": "Advanced",
    "difficulty_recommendation": "Hard",
    "document": [
      {
        "title": "Mastering Topic X",
        ...
      }
    ]
  },
  "script": "Here's a voice script explaining your results..."
}
```

**Features:**
- **Detailed Feedback:** Fun, game-like explanations for wrong answers
- **Performance Analysis:** Shows accuracy, areas of strength and weakness
- **Personalized Documents:** Generates study materials for weak areas
- **Next Steps:** Recommends difficulty level and improvement strategies
- **Encouragement:** Positive reinforcement and growth mindset messaging

---

### 6. **Code Correction**
**Endpoint:** `POST /code/correct/`

**What it does:**
- Analyzes code for syntax and logical errors
- Provides corrected version with explanation
- Identifies inefficiencies and improvements
- Explains why changes were made

**Input:**
```
code: string (your code snippet)
role: string (e.g., "Python Developer", "JavaScript Expert", "Data Scientist")
```

**Output:**
```json
{
  "correction": "Here's the corrected code:\n\n[corrected code]\n\nChanges made:\n- Fixed syntax error on line 5\n- Improved efficiency by using list comprehension"
}
```

---

### 7. **Code Review**
**Endpoint:** `POST /code/review/`

**What it does:**
- Reviews code for best practices
- Assesses readability and maintainability
- Provides suggestions for improvement
- Returns refactored version with explanations

**Input:**
```
code: string (your code snippet)
role: string (e.g., "Senior Python Developer", "Full-stack Engineer")
```

**Output:**
```json
{
  "review": "Here's my code review:\n\n[feedback]\n\nImproved version:\n[refactored code]\n\nKey improvements:\n- Better naming conventions\n- Improved code structure\n- Enhanced readability"
}
```

---

## 🧠 System Features

### AI Models Used
- **Gemini 2.5 Flash** - Main conversational model
- **Gemini 2.5 Flash TTS** - Text-to-speech for audio responses
- **Gemini Embedding** - For semantic understanding

### Database & Memory
- **Redis:** Stores user information, conversation history, and long-term preferences
- **Short-term Memory:** Maintains context within individual conversation threads
- **Long-term Memory:** Remembers user preferences across all sessions
  - User name and age
  - Learning interests and favorite topics
  - Communication tone preferences
  - Learning style (visual, step-by-step, interactive, etc.)
  - Educational goals

### Safety Features
- **Content Moderation:** Automatic safety checks for child-appropriate content
- **Religion Policy:** Educational discussion of major religions in respectful context
- **Violence Policy:** Historical context allowed, but no graphic descriptions
- **Grooming Detection:** Prevents inappropriate interactions with minors

---

## 📝 Example Workflow

1. **User Logs In** → GET `/` retrieves their profile and preferences
2. **User Chooses Topic** → POST `/quiz/` generates 30 practice questions
3. **User Takes Quiz** → Answers saved in thread session
4. **User Gets Evaluated** → POST `/evaluate/` provides feedback and study materials
5. **User Needs Help** → POST `/chatbot/` engages with AI learning companion
6. **User Reviews Code** → POST `/code/review/` gets expert feedback

---

## 🔧 Environment Setup

Make sure you have:
- ✅ Python 3.8+
- ✅ Google API Key (for Gemini models)
- ✅ Redis connection string (for memory storage)
- ✅ requirements.txt installed

## 📚 Support

For issues or questions about specific endpoints, check the endpoint documentation above for:
- What input format is expected
- What output you'll receive
- Which memory features are being used
- Special behaviors or edge cases
