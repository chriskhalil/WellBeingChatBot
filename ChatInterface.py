from fastapi import FastAPI, HTTPException, Depends, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import re
import secrets
from collections import defaultdict
import uvicorn
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('./data.env')

# Custom imports
from Chatbot import WellbeingChatbot, GroqAdapter
from Chatbot import load_prompt_from_file, count_tokens

# Initialize FastAPI app
app = FastAPI(
    title="SYD Wellbeing Chatbot API",
    description="Wellbeing 24/7",
    version="0.1A"
)

security = HTTPBearer(auto_error=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

RATE_LIMIT_MESSAGES = 20
RATE_LIMIT_WINDOW = 3600


# USE A REAL DATABSE IN PRODUCTIN FOR BOTH RATE LIMITING AND USERS.
# User database     
USERS_DB = {
    "user123": {
        "api_key": "syd_000001",
        "name": "Aya",
        "active": True
    }
}

# Rate limiting storage
rate_limit_store: Dict[str, List[datetime]] = defaultdict(list)

# Chatbot instances per user
chatbot_instances: Dict[str, WellbeingChatbot] = {}

# ============================================================================
# MODELS
# ============================================================================

class Message(BaseModel):
    role: str
    content: str
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        if v not in ['user', 'assistant']:
            raise ValueError("Role must be 'user' or 'assistant'")
        return v

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = None
    history: Optional[List[Message]] = []
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()

class ChatResponse(BaseModel):
    success: bool
    message: str
    response: str
    conversation_id: str
    timestamp: datetime
    tokens_used: Optional[int] = None
    wellbeing_mode_activated: bool = False

class RateLimitInfo(BaseModel):
    messages_used: int
    messages_remaining: int
    limit: int
    window_seconds: int

# ============================================================================
# Trivial Cleaning
# ============================================================================

def sanitize_message(message: str) -> Dict[str, any]:
    """Sanitize and validate incoming message."""
    clean_msg = message.strip()
    
    if len(clean_msg) < 1:
        return {"valid": False, "error": "Message too short"}
    
    if len(clean_msg) > 4000:
        return {"valid": False, "error": "Message exceeds maximum length"}
    
    malicious_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'eval\s*\(',
        r'\.\./',
        r'union\s+select',
        r'drop\s+table',
        r'exec\s*\(',
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, clean_msg, re.IGNORECASE):
            return {"valid": False, "error": "Message contains potentially malicious content"}
    
    clean_msg = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', clean_msg)
    
    if re.search(r'(.)\1{50,}', clean_msg):
        return {"valid": False, "error": "Message contains excessive repetition"}
    
    return {"valid": True, "sanitized": clean_msg}

# ============================================================================
# AUTHENTICATION  - Prevent UnAuthorized Access
# ============================================================================

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict:
    """Verify API key and return user information."""
    api_key = credentials.credentials
    
    for user_id, user_data in USERS_DB.items():
        if user_data["api_key"] == api_key:
            if not user_data["active"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is inactive"
                )
            return {"user_id": user_id, **user_data}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "Bearer"}
    )

# ============================================================================
# RATE LIMITING  - Avoid Abuse
# ============================================================================

def check_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded rate limit."""
    now = datetime.now()
    cutoff_time = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    
    rate_limit_store[user_id] = [
        ts for ts in rate_limit_store[user_id] 
        if ts > cutoff_time
    ]
    
    if len(rate_limit_store[user_id]) >= RATE_LIMIT_MESSAGES:
        return False
    
    rate_limit_store[user_id].append(now)
    return True

def get_rate_limit_info(user_id: str) -> RateLimitInfo:
    """Get current rate limit status for user."""
    now = datetime.now()
    cutoff_time = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    
    rate_limit_store[user_id] = [
        ts for ts in rate_limit_store[user_id] 
        if ts > cutoff_time
    ]
    
    messages_used = len(rate_limit_store[user_id])
    messages_remaining = max(0, RATE_LIMIT_MESSAGES - messages_used)
    
    return RateLimitInfo(
        messages_used=messages_used,
        messages_remaining=messages_remaining,
        limit=RATE_LIMIT_MESSAGES,
        window_seconds=RATE_LIMIT_WINDOW
    )

# ============================================================================
# CHATBOT INITIALIZATION
# ============================================================================

def get_or_create_chatbot(user_id: str) -> WellbeingChatbot:
    """Get existing chatbot instance or create new one for user."""
    if user_id not in chatbot_instances:
        # Load API key from environment
        load_dotenv(dotenv_path='./data.env')
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY not found in data.env file")
        
        adapter = GroqAdapter(
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            api_key=groq_api_key
        )
        
        system_message = load_prompt_from_file('./prompts/sys_prompt.ini')
        
        chatbot_instances[user_id] = WellbeingChatbot(
            adapter=adapter,
            system_message=system_message,
            memory_max_tokens=4000,
            memory_max_messages=8,
            enable_wellbeing_mode=True,
            wellbeing_prompts_dir="./prompts"
        )
        
        logging.info(f"Created new chatbot instance for user: {user_id}")
    
    return chatbot_instances[user_id]

# ============================================================================
# API ENDPOINTS - ALL REQUIRE AUTHENTICATION
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: Dict = Depends(verify_api_key)
):
    """Main chat endpoint - REQUIRES API KEY."""
    user_id = user["user_id"]
    
    # Check rate limit
    if not check_rate_limit(user_id):
        rate_info = get_rate_limit_info(user_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. {rate_info.messages_used}/{rate_info.limit} messages used."
        )
    
    # Sanitize message
    sanitization_result = sanitize_message(request.message)
    if not sanitization_result["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=sanitization_result["error"]
        )
    
    sanitized_message = sanitization_result["sanitized"]
    conversation_id = request.conversation_id or f"{user_id}_{secrets.token_urlsafe(8)}"
    
    try:
        chatbot = get_or_create_chatbot(user_id)
        
        if request.history:
            chatbot.memory.clear()
            for msg in request.history:
                chatbot.memory.add_message(msg.role, msg.content)
        
        wellbeing_activated = False
        if chatbot.wellbeing_engine:
            wellbeing_activated = chatbot.wellbeing_engine.detect_need_for_intervention(
                sanitized_message,
                chatbot.memory.get_messages()
            )
        
        response_text = chatbot.chat(sanitized_message)
        tokens_used = count_tokens(sanitized_message) + count_tokens(response_text)
        
        return ChatResponse(
            success=True,
            message="Response generated successfully",
            response=response_text,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            tokens_used=tokens_used,
            wellbeing_mode_activated=wellbeing_activated
        )
    
    except Exception as e:
        logging.error(f"Error for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )

@app.get("/rate-limit", response_model=RateLimitInfo)
def get_rate_limit_status(user: Dict = Depends(verify_api_key)):
    """Get rate limit status - REQUIRES API KEY."""
    return get_rate_limit_info(user["user_id"])

@app.post("/clear-session")
def clear_session(user: Dict = Depends(verify_api_key)):
    """Clear conversation history - REQUIRES API KEY."""
    user_id = user["user_id"]
    
    if user_id in chatbot_instances:
        chatbot_instances[user_id].memory.clear()
        return {
            "success": True,
            "message": "Session cleared successfully",
            "timestamp": datetime.now()
        }
    
    return {
        "success": True,
        "message": "No active session to clear",
        "timestamp": datetime.now()
    }

@app.delete("/session")
def delete_session(user: Dict = Depends(verify_api_key)):
    """Delete chatbot instance - REQUIRES API KEY."""
    user_id = user["user_id"]
    
    if user_id in chatbot_instances:
        del chatbot_instances[user_id]
        return {
            "success": True,
            "message": "Session deleted successfully",
            "timestamp": datetime.now()
        }
    
    return {
        "success": True,
        "message": "No active session to delete",
        "timestamp": datetime.now()
    }

@app.get("/session-info")
def get_session_info(user: Dict = Depends(verify_api_key)):
    """Get session information - REQUIRES API KEY."""
    user_id = user["user_id"]
    
    if user_id not in chatbot_instances:
        return {
            "active": False,
            "message": "No active session"
        }
    
    chatbot = chatbot_instances[user_id]
    history = chatbot.memory.get_messages()
    
    return {
        "active": True,
        "message_count": len(history),
        "wellbeing_mode_enabled": chatbot.wellbeing_engine is not None,
        "timestamp": datetime.now()
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        "ChatInterface:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )