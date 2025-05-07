import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq, GroqError
from uuid import uuid4, UUID
from typing import List, Dict
from datetime import datetime
from difflib import SequenceMatcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Car Mechanical Chatbot",
    description="""
    
    مرحباً👋
    أنا مستعد لإستقبال اي سؤال حول سياراتك 😊
    """,
    version="1.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Chat sessions and question history
chat_sessions: Dict[str, List[Dict]] = {}
question_history: Dict[str, List[Dict]] = {}
MAX_MESSAGES = 100
MAX_HISTORY = 1000

# Request model
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

# Validate UUID
def is_valid_uuid(session_id: str) -> bool:
    try:
        UUID(session_id)
        return True
    except ValueError:
        return False

# Check if questions are related
def are_questions_related(current_message: str, previous_message: str) -> bool:
    if not previous_message:
        return False
    similarity = SequenceMatcher(None, current_message, previous_message).ratio()
    return similarity > 0.5

# Endpoint to start a new session
@app.get("/start_session")
async def start_session():
    session_id = str(uuid4())
    chat_sessions[session_id] = [
                {
                    "role": "system",
                    "content": " أنت مساعد ميكانيكي ذكي وخبير في أنظمة السيارات. أجب باللغة العربية بأسلوب وجمل قصيرة وواضحة ومنظمة على شكل نقاط عند الحاجة."
                }
    ]
    question_history[session_id] = []
    logger.info(f"New session started: {session_id}")
    return {"session_id": session_id}

# Chat endpoint
@app.post("/chat")
async def chat(chat_request: ChatRequest):
    logger.info("Received chat request")
    try:
        # Generate new session_id if not provided
        session_id = chat_request.session_id if chat_request.session_id else str(uuid4())
        
        # Validate session_id if provided
        if chat_request.session_id and not is_valid_uuid(chat_request.session_id):
            raise HTTPException(status_code=400, detail="Invalid session_id")

        current_message = chat_request.message.strip()

        # Initialize session if it doesn't exist
        if session_id not in chat_sessions:
            chat_sessions[session_id] = [
                {
                    "role": "system",
                    "content": " أنت مساعد ميكانيكي ذكي وخبير في أنظمة السيارات. أجب باللغة العربية بأسلوب وجمل قصيرة وواضحة ومنظمة على شكل نقاط عند الحاجة."
                }
            ]
            question_history[session_id] = []

        # Add question to history
        question_history[session_id].append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": current_message
        })
        question_history[session_id] = question_history[session_id][-MAX_HISTORY:]

        # Get last user message from session
        last_user_message = ""
        for msg in reversed(chat_sessions[session_id]):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break

        # Check if the question is related
        if not are_questions_related(current_message, last_user_message):
            chat_sessions[session_id] = [
                {
                    "role": "system",
                    "content": " أنت مساعد ميكانيكي ذكي وخبير في أنظمة السيارات. أجب باللغة العربية بأسلوب وجمل قصيرة وواضحة ومنظمة على شكل نقاط عند الحاجة."
                }
            ]

        # Add new question to session
        chat_sessions[session_id].append({
            "role": "user",
            "content": current_message
        })

        # Call Groq to get response
        chat_completion = client.chat.completions.create(
            messages=chat_sessions[session_id],
            model="gemma2-9b-it",
            max_tokens=300,
            temperature=0.5,
        )

        response_content = chat_completion.choices[0].message.content
        chat_sessions[session_id].append({
            "role": "assistant",
            "content": response_content
        })

        # Trim session
        chat_sessions[session_id] = chat_sessions[session_id][-MAX_MESSAGES:]

        logger.info(f"Request processed successfully for session {session_id}")
        return {"session_id": session_id, "response": response_content}

    except GroqError as e:
        logger.error(f"Groq API error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")