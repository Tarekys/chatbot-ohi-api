from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import os
from dotenv import load_dotenv

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

class ChatRequest(BaseModel):
    message: str

def clean_response(text: str) -> str:
    """تنظيف النص من التنسيقات غير المرغوبة"""
    import re
    text = re.sub(r"[*•\-]+", "", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    text = text.strip()
    return text

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    try:
        api_key = os.getenv("Key") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("key Hugging face not found")
        
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "أنت مساعد ميكانيكي ذكي وخبير في أنظمة السيارات. أجب باللغة العربية بأسلوب طبيعي وواضح، باختصار و بجمل ذات فائدة."
                },
                {
                    "role": "user",
                    "content": chat_request.message
                }
            ],
            model="gemma2-9b-it",
            max_tokens=300,
            temperature=0.7,
        )

        raw_response = chat_completion.choices[0].message.content
        cleaned_response = clean_response(raw_response)

        return {"response": cleaned_response}
    
    except Exception as e:
        return {"error": str(e)}
