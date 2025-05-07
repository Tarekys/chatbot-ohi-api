import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq, GroqError
from uuid import UUID

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تحميل المتغيرات البيئية
load_dotenv()

# إنشاء تطبيق FastAPI
app = FastAPI(
    title="Car Mechanic Chatbot API",
    description="API لدردشة باللغة العربية مع مساعد ميكانيكي ذكي",
    version="1.0.0"
)

# إعداد CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إنشاء عميل Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# جلسات المحادثة
chat_sessions = {}
MAX_MESSAGES = 100

# نموذج الطلب
class ChatRequest(BaseModel):
    message: str
    session_id: str

# التحقق من UUID
def is_valid_uuid(session_id: str) -> bool:
    try:
        UUID(session_id)
        return True
    except ValueError:
        return False

# تنظيف النص
def clean_response(text: str) -> str:
    import re
    text = re.sub(r"[*•\-]+", "", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()

# نقطة نهاية المحادثة
@app.post("/chat")
async def chat(chat_request: ChatRequest):
    logger.info(f"Received request for session {chat_request.session_id}")
    try:
        if not is_valid_uuid(chat_request.session_id):
            raise HTTPException(status_code=400, detail="Invalid session_id")

        session_id = chat_request.session_id
        if session_id not in chat_sessions:
            chat_sessions[session_id] = [
                {
                    "role": "system",
                    "content": "أنت مساعد ميكانيكي ذكي وخبير في أنظمة السيارات. أجب باللغة العربية بأسلوب وجمل قصيرة وواضحة ومنظمة على شكل نقاط عند الحاجة."
                }
            ]

        chat_sessions[session_id].append({
            "role": "user",
            "content": chat_request.message
        })

        chat_completion = client.chat.completions.create(
            messages=chat_sessions[session_id],
            model="gemma-7b-it",
            max_tokens=300,
            temperature=0.7,
        )

        response_content = clean_response(chat_completion.choices[0].message.content)
        chat_sessions[session_id].append({
            "role": "assistant",
            "content": response_content
        })

        # اقتصاص الجلسة
        chat_sessions[session_id] = chat_sessions[session_id][-MAX_MESSAGES:]

        logger.info("Request processed successfully")
        return {"response": response_content}

    except GroqError as e:
        logger.error(f"Groq API error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
