from LlmToolKit import * 

# Wellbeing Assistant with Mood Detection and Plan Generation
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from pathlib import Path
import tiktoken
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Reuse your existing adapters and memory classes
from typing import Any, Union
import time
############### Wellbeing Mode Components ###############

### Decoupled wellbeing mode that can be plugged into any chatbot. Performs mood detection, plan generation, and validation.
class WellbeingModeEngine:
  
    # Intervention detection keywords and patterns SAMPLE LIST.
    #  REAL PRODUCTION SHOULD BE IN A DATABASE AND WITH A SMALLER modell trained for detection maybe BERT ?
    DISTRESS_KEYWORDS = {
        'high': ['suicide', 'kill myself', 'end it all', 'want to die', 'no point living', 
                 'better off dead', 'self-harm', 'hurt myself'],
        'medium': ['depressed', 'anxious', 'panic', 'cant cope', "can't cope", 'overwhelmed', 
                   'hopeless', 'helpless', 'worthless', 'hate myself', 'stressed out', 
                   'breaking down', 'falling apart'],
        'low': ['sad', 'worried', 'stressed', 'tired', 'exhausted', 'struggling', 
                'difficult', 'hard time', 'not okay', 'feeling down']
    }
    
    def __init__(self, llm_adapter, rate_limiter, prompts_dir: Union[str, Path] = "./prompts"):
        self.llm = llm_adapter.get_llm()
        self.rate_limiter = rate_limiter
        self.parser = StrOutputParser()
        self.prompts_dir = Path(prompts_dir)
        
        # Load prompts from disk
        self._load_prompts()
    
    def _load_prompts(self):
        """Load all prompts from .ini files"""
        try:
            mood_prompt_text = load_prompt_from_file(self.prompts_dir / "mood_analysis.ini")
            self.mood_prompt = ChatPromptTemplate.from_messages([
                ("human", mood_prompt_text)
            ])
            
            plan_prompt_text = load_prompt_from_file(self.prompts_dir / "plan_generation.ini")
            self.plan_prompt = ChatPromptTemplate.from_messages([
                ("human", plan_prompt_text)
            ])
            
            validation_prompt_text = load_prompt_from_file(self.prompts_dir / "plan_validation.ini")
            self.validation_prompt = ChatPromptTemplate.from_messages([
                ("human", validation_prompt_text)
            ])
            
            # Compile chains
            self.mood_chain = self.mood_prompt | self.llm | self.parser
            self.plan_chain = self.plan_prompt | self.llm | self.parser
            self.validation_chain = self.validation_prompt | self.llm | self.parser
            
            logging.info("Successfully loaded wellbeing prompts from disk")
            
        except FileNotFoundError as e:
            logging.error(f"Failed to load prompt files: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error loading prompts: {str(e)}")
            raise
    
    def detect_need_for_intervention(self, message: str, history: List[Dict]) -> bool:
        
        #Fast keyword-based detection to determine if wellbeing mode should activate.
        # Returns True if intervention is needed.
        message_lower = message.lower()
        
        # Check high-priority keywords first (most critical)
        for keyword in self.DISTRESS_KEYWORDS['high']:
            if keyword in message_lower:
                logging.info(f"High-priority intervention trigger: '{keyword}'")
                return True
        
        # Check medium-priority keywords
        for keyword in self.DISTRESS_KEYWORDS['medium']:
            if keyword in message_lower:
                logging.info(f"Medium-priority intervention trigger: '{keyword}'")
                return True
        
        # Check low-priority keywords (require at least 2 mentions or pattern)
        low_matches = sum(1 for keyword in self.DISTRESS_KEYWORDS['low'] 
                         if keyword in message_lower)
        if low_matches >= 2:
            logging.info(f"Low-priority intervention trigger: {low_matches} keywords")
            return True
        
        # Check conversation history for patterns (last 3 messages)
        if len(history) >= 2:
            recent_messages = ' '.join([msg.get('content', '') for msg in history[-3:]])
            recent_lower = recent_messages.lower()
            
            # Count total distress keywords in recent context
            total_distress = sum(
                1 for level in self.DISTRESS_KEYWORDS.values() 
                for keyword in level 
                if keyword in recent_lower
            )
            if total_distress >= 3:
                logging.info(f"Pattern-based intervention trigger: {total_distress} keywords in history")
                return True
        
        return False
    
    def mood_analyzer(self, user_message: str, history: List[Dict]) -> Dict[str, Any]:
        
        # Analyzes the user's emotional state using LLM.
       # Returns structured mood analysis as dictionary.
        self.rate_limiter.wait()
        
        # Format history for prompt
        history_text = self._format_history(history, max_messages=5)
        
        response = self.mood_chain.invoke({
            "history": history_text,
            "message": user_message
        })
        
        mood_analysis = self._safe_json_parse(response)
        
        # Validate required fields
        required_fields = ['primary_emotion', 'severity', 'risk_level', 'needs']
        if not all(field in mood_analysis for field in required_fields):
            raise ValueError(f"Mood analysis missing required fields. Got: {list(mood_analysis.keys())}")
        
        logging.info(f"Mood analysis: {mood_analysis.get('primary_emotion')} "
                    f"(severity: {mood_analysis.get('severity')}, "
                    f"risk: {mood_analysis.get('risk_level')})")
        
        return mood_analysis
    
    def plan_generation(self, mood_analysis: Dict[str, Any], history: List[Dict]) -> Dict[str, Any]:

        # Generates a personalized wellbeing plan based on mood analysis.
       # Returns structured plan as dictiona
        self.rate_limiter.wait()
        

        ### IN PRODUCTION USE  RAG/ API FOR WEBMD, MAYOCLINIC/ USE PRIVATE VECTOR DATABASE ...... AND APPEND TO THE CONTEXT
        # Format inputs
        history_text = self._format_history(history, max_messages=5)
        mood_text = json.dumps(mood_analysis, indent=2)
        
        response = self.plan_chain.invoke({
            "mood_analysis": mood_text,
            "history": history_text
        })
        
        plan = self._safe_json_parse(response)
        
        required_fields = ['immediate_actions', 'personalized_message']
        if not all(field in plan for field in required_fields):
            raise ValueError(f"Plan missing required fields. Got: {list(plan.keys())}")
        
        logging.info(f"Generated plan with {len(plan.get('immediate_actions', []))} immediate actions")
        
        return plan
    
    def validation(self, plan: Dict[str, Any], mood_analysis: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        
        # Validates the generated plan for safety and feasibility.
        #Returns (is_valid, validation_result) tuple.
        self.rate_limiter.wait()
        
        # Format inputs
        plan_text = json.dumps(plan, indent=2)
        mood_text = json.dumps(mood_analysis, indent=2)
        
        # Invoke LLM
        response = self.validation_chain.invoke({
            "mood_analysis": mood_text,
            "plan": plan_text
        })
        
        # Parse JSON response
        validation_result = self._safe_json_parse(response)
        
        # Validate response structure
        required_fields = ['is_valid', 'safety_score', 'feasibility_score']
        if not all(field in validation_result for field in required_fields):
            raise ValueError(f"Validation result missing required fields. Got: {list(validation_result.keys())}")
        
        # Determine validity
        is_valid = validation_result.get('is_valid', False)
        safety_score = validation_result.get('safety_score', 0)
        feasibility_score = validation_result.get('feasibility_score', 0)
        
        # Override if scores are too low
        if safety_score < 8 or feasibility_score < 8:
            is_valid = False
        
        logging.info(f"Validation result: valid={is_valid}, "
                    f"safety={safety_score}, feasibility={feasibility_score}")
        
        return is_valid, validation_result
    
    def execute_wellbeing_mode(
        self, 
        user_message: str, 
        history: List[Dict]
    ) -> str:
        #Main entry point: Execute full wellbeing workflow with validation loop.
        #Returns the final empathetic message to show user.
        # Uses a very simple linear self reasoning. This can be extended wih Graphs and MCT
        logging.info("=== Starting Wellbeing Mode ===")
        
        # Step 1: Assess mood
        mood_analysis = self.mood_analyzer(user_message, history)
        
        # Step 2 & 3: Generate and validate plan (with retry loop)
        max_attempts = 3
        plan = None
        validation_result = None
        
        for attempt in range(1, max_attempts + 1):
            logging.info(f"Plan generation attempt {attempt}/{max_attempts}")
            
            # Generate plan
            plan = self.plan_generation(mood_analysis, history)
            
            # Validate plan
            is_valid, validation_result = self.validation(plan, mood_analysis)
            
            if is_valid:
                logging.info("Plan validated successfully")
                break
            else:
                logging.warning(f"Plan validation failed: {validation_result.get('issues', [])}")
                
                # On last attempt, raise exception
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Failed to generate valid plan after {max_attempts} attempts. "
                        f"Last validation issues: {validation_result.get('issues', [])}"
                    )
                
                # Inject feedback into next generation
                mood_analysis['validation_feedback'] = validation_result.get('suggestions', [])
        
        # Step 4: Format final response for user
        final_message = self._format_final_response(plan, mood_analysis)
        
        logging.info("=== Wellbeing Mode Complete ===")
        return final_message
    
    # Helper methods
    
    def _format_history(self, history: List[Dict], max_messages: int = 5) -> str:
        #"Format conversation history for prompts
        if not history:
            return "No previous conversation."
        
        recent = history[-max_messages:]
        formatted = []
        for msg in recent:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted)
    
    #Safely parse JSON from LLM response. Raises exception if parsing fails.
    def _safe_json_parse(self, response: str) -> Dict[str, Any]:
        try:
            # Try direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find JSON object in text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            raise ValueError(f"No valid JSON found in LLM response: {response[:200]}...")
    
    #Format the plan into a user-friendly message
    def _format_final_response(self, plan: Dict[str, Any], mood_analysis: Dict[str, Any]) -> str:
        response_parts = []
        
        # Start with personalized message
        personalized = plan.get('personalized_message', '')
        if personalized:
            response_parts.append(personalized)
            response_parts.append("")  # Blank line
        
        # Immediate actions
        immediate = plan.get('immediate_actions', [])
        if immediate:
            response_parts.append("**Right Now (next 5-10 minutes):**")
            for i, action in enumerate(immediate, 1):
                response_parts.append(f"{i}. {action['action']}")
                response_parts.append(f"   *Why this helps: {action['rationale']}*")
            response_parts.append("")
        
        # Short-term actions
        short_term = plan.get('short_term_actions', [])
        if short_term:
            response_parts.append("**Today:**")
            for i, action in enumerate(short_term, 1):
                response_parts.append(f"{i}. {action['action']}")
            response_parts.append("")
        
        # Ongoing practices
        ongoing = plan.get('ongoing_practices', [])
        if ongoing:
            response_parts.append("**This Week:**")
            for i, practice in enumerate(ongoing, 1):
                response_parts.append(f"{i}. {practice['practice']} ({practice['frequency']})")
            response_parts.append("")
        
        # Emergency resources (only if present and needed)
        emergency = plan.get('emergency_resources', [])
        severity = mood_analysis.get('severity', 0)
        risk_level = mood_analysis.get('risk_level', 'low')
        
        # Only include crisis resources if high severity/risk OR if explicitly included in plan
        # YES THE NUMBER ARE RANDOM EXPERT NEED TO VALIDATE AND GIVE FEEDBACK
        if emergency and (severity >= 8 or risk_level == 'high' or len(emergency) > 0):
            response_parts.append("**🆘 If You Need Immediate Help:**")
            for resource in emergency:
                response_parts.append(f"• **{resource['resource']}**: {resource['contact']}")
                response_parts.append(f"  _{resource['when']}_")
            response_parts.append("")
        
        # Closing
        response_parts.append("---")
        response_parts.append("*You don't have to do everything at once. Even one small step is meaningful. NOTE: THIS IS NOT A MEDICAL ADVICE PLEASE CONSULT A REAL DOCTOR FOR MEDICAL ADVIVCES*")
        
        return "\n".join(response_parts)


############### Enhanced Chatbot with Wellbeing Mode ###############

class WellbeingChatbot:
    
    def __init__(
        self,
        adapter,
        system_message: str,
        memory_max_tokens: int = 4000,
        memory_max_messages: int = 8,
        enable_wellbeing_mode: bool = True,
        wellbeing_prompts_dir: Union[str, Path] = "./prompts"
    ):
        self.llm = adapter.get_llm()
        self.system_message = system_message
        self.rate_limiter = RateLimiter(20, 60)
        
        self.memory = ConversationMemory(memory_max_tokens, memory_max_messages)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        # Wellbeing mode (can be disabled)
        self.wellbeing_engine = None
        if enable_wellbeing_mode:
            self.wellbeing_engine = WellbeingModeEngine(
                adapter, 
                self.rate_limiter,
                prompts_dir=wellbeing_prompts_dir
            )
    
    def chat(self, user_input: str) -> str:

        history = self.memory.get_messages()
        
        # Check if wellbeing mode is needed
        needs_intervention = False
        if self.wellbeing_engine:
            needs_intervention = self.wellbeing_engine.detect_need_for_intervention(
                user_input, 
                history
            )
        
        # Route to appropriate mode AUTOMATIC SWITCH
        if needs_intervention and self.wellbeing_engine:
            try:
                response = self.wellbeing_engine.execute_wellbeing_mode(user_input, history)
            except Exception as e:
                logging.error(f"Wellbeing mode failed: {str(e)}")
                raise RuntimeError(f"Wellbeing service unavailable: {str(e)}")
        else:
            # Normal chat mode
            try:
                self.rate_limiter.wait()
                history_msgs = self.memory.to_langchain_messages()
                response = self.chain.invoke({
                    "history": history_msgs,
                    "input": user_input
                })
            except Exception as e:
                logging.error(f"Chat service failed: {str(e)}")
                raise RuntimeError(f"Chat service unavailable: {str(e)}")
        
        # Save to memory
        self.memory.add_message("user", user_input)
        self.memory.add_message("assistant", response)
        
        return response
    
    def disable_wellbeing_mode(self):
        """Turn off wellbeing mode"""
        self.wellbeing_engine = None
    
    def enable_wellbeing_mode(self, adapter, prompts_dir: Union[str, Path] = "./prompts"):
        """Turn on wellbeing mode"""
        self.wellbeing_engine = WellbeingModeEngine(
            adapter, 
            self.rate_limiter,
            prompts_dir=prompts_dir
        )