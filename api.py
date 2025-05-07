import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# جلسات المحادثة (ذاكرة مؤقتة)
chat_sessions = {}

# API endpoint
class ChatRequest(BaseModel):
    message: str
    session_id: str

def clean_response(text: str) -> str:
    """تنظيف النص من التنسيقات غير المرغوبة"""
    import re
    text = re.sub(r"[*•\-]+", "", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    text = text.strip()
    return text

# نقطة النهاية للمحادثة
@app.post("/chat")
async def chat(chat_request: ChatRequest):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("key HF not found")

        # إنشاء أو تحديث سجل الجلسة
        session_id = chat_request.session_id
        if session_id not in chat_sessions:
            chat_sessions[session_id] = [
                {
                    "role": "system",
                    "content": "أنت مساعد ميكانيكي ذكي وخبير في أنظمة السيارات. أجب باللغة العربية بأسلوب و جمل قصيرة وواضحة ومنظمة على شكل نقاط عند الحاجة."
                }
            ]

        # إضافة رسالة المستخدم
        chat_sessions[session_id].append({
            "role": "user",
            "content": chat_request.message
        })

        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=chat_sessions[session_id],
            model="gemma-7b-it",
            max_tokens=200,
            temperature=0.7,
        )

        # حفظ رد المساعد في السجل
        response_content = chat_completion.choices[0].message.content
        chat_sessions[session_id].append({
            "role": "assistant",
            "content": response_content
        })

        return {"response": response_content}

    except Exception as e:
        return {"error": str(e)}
