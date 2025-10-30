## LLM INTERFACE WITH MEMORY
## file written by ck
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import re

# Typing imports
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

# Third-party imports
import openai
import tiktoken
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

################# Document Loading and IO Handling code #############################
from typing import Dict, Pattern

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    #cl100k is for gpt4 and 3.5 totally useless for lamma  just here for some stats.  gpt4o and + uses o200k
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except KeyError:
        print(f"Warning: Model '{model}' not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def load_prompt_from_file(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        
        with open(path, 'r', encoding=encoding) as f:
            prompt_content = f.read()
        
        if not prompt_content.strip():
            logging.warning(f"Warning: Prompt file is empty: {file_path}")
        
        return prompt_content
    
    except UnicodeDecodeError as e:
        raise IOError(f"Error decoding file with {encoding} encoding: {str(e)}")
    except Exception as e:
        raise IOError(f"Error reading prompt file {file_path}: {str(e)}")


############### LLM Adapter Classes and API Access ######################
# Base class used with polymorph for adapter pattern.
# improves modularity and encapsulate provider logic by only providing the needed interface get_llm
class LLMAdapter(ABC):
    def __init__(self, model_name: str, temperature: float, api_key: str):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key

    @abstractmethod
    def get_llm(self) -> Any:
        pass
    
class OpenAIAdapter(LLMAdapter):
    def __init__(
        self, 
        model_name: str, 
        temperature: float, 
        api_key: str,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        streaming: bool = False
    ):
        super().__init__(model_name, temperature, api_key)
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.streaming = streaming

    def get_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            streaming=self.streaming
        )
    
class AnthropicAdapter(LLMAdapter):
    def get_llm(self) -> ChatAnthropic:
        return ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key
        )

class GroqAdapter(LLMAdapter):
    def get_llm(self) -> ChatGroq:
        return ChatGroq(
            model_name=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key
        )

class CohereAdapter(LLMAdapter):
    def get_llm(self) -> ChatCohere:
        return ChatCohere(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key
        )

class RateLimiter:
    def __init__(self, max_calls: int, time_frame: float):
        self.max_calls = max_calls
        self.time_frame = time_frame
        self.calls = []

    def wait(self):
        current_time = time.time()
        self.calls = [call for call in self.calls if current_time - call < self.time_frame]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.calls[0] + self.time_frame - current_time
            time.sleep(max(0, sleep_time))
        self.calls.append(time.time())


############### Memory Management ######################

### manage user conversation history. LLMS calls are amnesic
### handling of token limit and window for short-long chat.
### provide basic CRUD functionality in real production ust be using REDIS or a database SQL/NSQL ?
class ConversationMemory:

    def __init__(self, max_tokens: int = 4000, max_messages: int = 20):
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.message_history: List[Dict[str, str]] = []
        
    def add_message(self, role: str, content: str):
        self.message_history.append({"role": role, "content": content})
        self._trim_history()
    
    def _trim_history(self):
        # First trim by message count
        if len(self.message_history) > self.max_messages:
            self.message_history = self.message_history[-self.max_messages:]
        
        # Then trim by token count
        total_tokens = sum(count_tokens(msg["content"]) for msg in self.message_history)
        while total_tokens > self.max_tokens and len(self.message_history) > 2:
            # Remove oldest messages (but keep at least 2)
            self.message_history.pop(0)
            total_tokens = sum(count_tokens(msg["content"]) for msg in self.message_history)
    
    def get_messages(self) -> List[Dict[str, str]]:
        return self.message_history.copy()
    
    def clear(self):
        self.message_history.clear()
    
    def to_langchain_messages(self) -> List[Union[HumanMessage, AIMessage]]:
        messages = []
        for msg in self.message_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages

## Based on Conversation memory. Handle multiple sessions.Did it before so why remove it. maybe i will use it ?
class SessionMemoryManager:    
    def __init__(self):
        self.sessions: Dict[str, ConversationMemory] = {}
    
    def get_session(self, session_id: str, max_tokens: int = 4000, 
                    max_messages: int = 20) -> ConversationMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory(max_tokens, max_messages)
        return self.sessions[session_id]
    
    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def clear_all(self):
        self.sessions.clear()

