import json
import time
import asyncio
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

class Intent(Enum):
    ELIGIBILITY = "eligibility"
    DEALER_SEARCH = "dealer_search"
    NORMAL_QA = "normal_qa"

@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    details: Optional[Dict] = None

class LLMIntentDetector:
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("API_KEY_1")
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        
        try:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                verify=False,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            print("✓ Intent Detection LLM client initialized")
        except Exception as e:
            print(f"✗ Intent Detection LLM client failed: {e}")
            self.client = None
    
    async def detect_intent(self, query: str) -> IntentResult:
        
        if not query or not query.strip():
            return IntentResult(
                intent=Intent.NORMAL_QA,
                confidence=0.5,
                details={"reason": "empty_query"}
            )
        
        if not self.client or not self.api_key:
            
            return IntentResult(
                intent=Intent.NORMAL_QA,
                confidence=0.3,
                details={"reason": "llm_not_available"}
            )
        
        try:
            
            prompt = self._create_intent_prompt(query)
            
            
            response = await self._call_llm(prompt)
            
            
            intent_result = self._parse_llm_response(response, query)
            
            return intent_result
            
        except Exception as e:
            print(f"Intent detection error: {e}")
            
            return IntentResult(
                intent=Intent.NORMAL_QA,
                confidence=0.2,
                details={"reason": f"error: {str(e)[:50]}"}
            )
    
    def _create_intent_prompt(self, query: str) -> str:
        """Create a focused prompt for intent classification"""
        return f"""You are an intent classifier for a Bajaj Finance REMI customer service bot.

Analyze this user query and classify it into exactly ONE of these three categories:

1. **ELIGIBILITY**: Questions about checking eligibility, verification of mobile numbers, REMI qualification status, credit checks, or asking "am I eligible"

2. **DEALER_SEARCH**: Questions about finding stores, dealers, shops, locations, addresses, or searching by pincode/area

3. **NORMAL_QA**: All other questions including product information, general inquiries, greetings, complaints, payment info, documents, how REMI works, etc.

User Query: "{query}"

Respond with ONLY this exact JSON format (no other text):
{{
    "intent": "eligibility" | "dealer_search" | "normal_qa",
    "confidence": 0.95,
    "reasoning": "brief reason for classification"
}}

Think step by step:
1. Does this query ask about checking eligibility or mobile verification? → ELIGIBILITY
2. Does this query ask about finding stores/dealers/locations? → DEALER_SEARCH  
3. Everything else → NORMAL_QA

JSON Response:"""

    async def _call_llm(self, prompt: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-flash-latest:generateContent"
            f"?key={self.api_key}"
        )

        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 80,
                "topP": 0.8,
                "topK": 40,
                # 👇 this is the important part
                "response_mime_type": "application/json"
            }
        }

        response = await self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()

        if "candidates" in data and data["candidates"]:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            print("Raw Intent Response:", repr(content))
            return content.strip()
        else:
            raise Exception("No content in LLM response")
        
    def _parse_llm_response(self, response: str, original_query: str) -> IntentResult:
        """
        Parse LLM JSON response into IntentResult.
        Tries direct JSON first, falls back to extracting JSON substring.
        """
        try:
            # 1. First, try to parse the whole thing directly (JSON mode case)
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # 2. Fallback: try to extract JSON substring
                json_start = response.find('{')
                json_end = response.rfind('}') + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    data = json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found in cleaned response")

            # ---- At this point, `data` is a dict ----
            intent_str = str(data.get("intent", "")).lower().strip()

            if intent_str == "eligibility":
                intent = Intent.ELIGIBILITY
            elif intent_str == "dealer_search":
                intent = Intent.DEALER_SEARCH
            else:
                intent = Intent.NORMAL_QA

            try:
                confidence = float(data.get("confidence", 0.7))
            except (TypeError, ValueError):
                confidence = 0.7

            reasoning = data.get("reasoning", "No reasoning provided")

            return IntentResult(
                intent=intent,
                confidence=confidence,
                details={
                    "reasoning": reasoning,
                    "raw_response": response,
                    "query": original_query,
                },
            )

        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response!r}")

            return IntentResult(
                intent=Intent.NORMAL_QA,
                confidence=0.3,
                details={
                    "reason": f"parse_error: {str(e)}",
                    "raw_response": response,
                },
            )

    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
    
    def is_available(self) -> bool:
        """Check if intent detection is available"""
        return self.client is not None and self.api_key is not None


intent_detector = None

def initialize_intent_detector():
    """Initialize the global intent detector"""
    global intent_detector
    try:
        intent_detector = LLMIntentDetector()
        print("✓ Intent detector initialized")
        return True
    except Exception as e:
        print(f"✗ Intent detector initialization failed: {e}")
        intent_detector = None
        return False


initialize_intent_detector()

async def get_query_intent(query: str) -> IntentResult:
    """
    Main function to get intent of a query
    This is the only function you need to call from outside
    """
    global intent_detector
    
    if not intent_detector:
        
        if not initialize_intent_detector():
            return IntentResult(
                intent=Intent.NORMAL_QA,
                confidence=0.1,
                details={"reason": "detector_not_available"}
            )
    
    return await intent_detector.detect_intent(query)


async def test_intent_detection():
    """Test the intent detection with sample queries"""
    test_queries = [
        "Check my eligibility for REMI",
        "Am I eligible for the loan?", 
        "My mobile number is 9876543210",
        "Find dealers near me",
        "Show me stores in 411001",
        "Dealers in pune",
        "What is REMI?",
        "How does EMI work?",
        "Hello, I need help"
    ]
    
    print("\n" + "="*60)
    print("TESTING INTENT DETECTION")
    print("="*60)
    
    for query in test_queries:
        result = await get_query_intent(query)
        print(f"\nQuery: '{query}'")
        print(f"Intent: {result.intent.value}")
        print(f"Confidence: {result.confidence:.2f}")
        if result.details and 'reasoning' in result.details:
            print(f"Reasoning: {result.details['reasoning']}")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(test_intent_detection())