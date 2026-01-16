from io import BytesIO
import uuid
from typing import Optional,List, Annotated
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict
from llm2 import generate_followup_suggestions, search_exact_pincode, fallback_to_llm, extract_mobile_from_text, format_eligibility_response, parse_query_for_pincode_and_category, map_category_terms, map_category_term, dealers_df, memory,qa_chain, parse_suggestions,get_static_suggestions
from chain import create_rag_chain_csv
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import os
import time
import asyncio
from langchain_groq import ChatGroq
from langgraph.types import interrupt,Command
from langgraph.checkpoint.memory import InMemorySaver
import dotenv
from supabase import Client, create_client
from operator import add
from langchain_nvidia import ChatNVIDIA
from utils import generate_followup_suggestions, check_remi_eligibility

dotenv.load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)
nvapi = os.getenv("NV_API_KEY")

llm_client = ChatGroq(
    model= "openai/gpt-oss-120b",
    api_key= os.getenv("GROQ_API_KEY") 
)

suggestion_client = ChatNVIDIA(
    model="nvidia/nemotron-3-nano-30b-a3b",
    api_key= nvapi
)


class Conv(TypedDict):
    user: str
    assistant: str

class Intent(BaseModel):
    intent:Literal["dealer_query","eligibility","rag_faq"] = Field(..., description="The step to be taken for the user query")

router = llm_client.with_structured_output(Intent)

class State(TypedDict):
    user_query:Optional[str]
    intent:Optional[str]
    output:Optional[str]
    chat_history:Annotated[List[Conv], add]
    suggestion: Optional[List[str]]

class Feedback(BaseModel):
    pincode:Optional[str] = Field(default=None,
        description="6-digit area pincode if present in the query.",
    )
    category:Optional[str] = Field(default=None,
        description="Store category if present in the query. Category includes words like Bike Accessories, Cycle, Eyewear, tyres, Apparels, Furnishing, Gym, Spas, Paints, Hardware, Toys, Vehicle, Watches, Water Purifier, Small Appliances, Footwear etc.",
    )
    feedback: str = Field(
        description="If anything is missing or not available in the query, ask the user to provide it.",
    )

class FeedbackEligibility(BaseModel):
    mob: Optional[str] = Field(
        default=None,
        description="The 10-digit mobile number extracted from the query. Keep it None if not found."
    )
    feedback: str = Field(
        description="A polite, conversational question asking the user for their mobile number if 'mob' is missing. Do NOT repeat the user query."
    )
    
    
def append_turn(
    conversation: list[Conv],
    user: str,
    assistant: str
) -> list[Conv]:
    return conversation + [{"user": user, "assistant": assistant}]


def load_chat_history()->list:
    global memory
    if not memory:
        return []
    try:
        return memory.load_memory_variables({})['chat_history',[]] or []
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []
def save_chat_history(query:str,response:str):
    global memory
    print("saving to memory:", memory)
    if not memory:
        return
    try:
        memory.save_context({"input": query}, {"output": response})
    except Exception as e:
        print(f"Memory save failed: {e}")

async def detect_intent(state: State):
    conversation = state.get("chat_history", [])
    query = state["user_query"]
    print("conversation inside the detect_intent:", conversation)
    print("previous intent", state.get("intent"))
    messages = [
        SystemMessage(
            content=(
                "You are an intent classifier.\n"
                "eligibility: questions about whether user qualifies for remi, whether user can apply for loan.\n"
                "dealer_query: questions about dealers, stores, locations, to buy anything.\n"
                "rag_faq: general FAQs, product info, how-to, policy details, anything related to payments, documents or some criteria, etc\n" \
                "Understand the broader concept of the query asked by the user.\n"
                "Use conversation context if relevant.\n"
                "Classify into: eligibility, dealer_query, rag_faq."
            )
        )
    ]

    for turn in conversation[-6:]:
        messages.append(HumanMessage(content=turn["user"]))
        messages.append(AIMessage(content=turn["assistant"]))

    messages.append(HumanMessage(content=query))

    decision = await router.ainvoke(messages)

    return {"intent": decision.intent,
            "chat_history":conversation}

def route_decision(state:State):
    print("Routing based on intent", state.get("intent"))
    intent = state.get("intent")
    if intent == "eligibility":
        return "eligibility_flow_node"
    elif intent == "dealer_query":
        return "dealer_flow_node"
    elif intent == "rag_faq":
        return "rag_flow_node"
    else:
        print("Unknown intent, defaulting to rag_flow")
        return "rag_flow_node"

async def rag_flow_node(state:State):
    query = state["user_query"]
    return await rag_flow_async(state)

async def dealer_flow_node(state:State):
    query = state["user_query"]
    return await dealer_flow_async(state)

async def eligibility_flow_node(state:State):
    query = state["user_query"]
    return await eligibility_flow_async(state)
    
async def eligibility_flow_async(state: State):
    query = state["user_query"]
    conversation = state.get("chat_history", [])
    
    llm_ele = ChatGroq(model="llama-3.3-70b-versatile")
    evaluator_ele = llm_ele.with_structured_output(FeedbackEligibility)
    
    # We only look at the current query for the mobile number
    evaluation = evaluator_ele.invoke([
        SystemMessage(content="Extract 10-digit mobile. If missing, ask for it."),
        HumanMessage(content=query)
    ])

    if not evaluation.mob:
        # Return the feedback question to the frontend immediately
        resp = evaluation.feedback or "Please provide your 10-digit mobile number."
        return {
            "output": resp,
            "intent": "eligibility",
            "suggestion": get_static_suggestions("eligibility"),
            "chat_history": [{"user": query, "assistant": resp}]
        }

    found, name = check_remi_eligibility(evaluation.mob.strip())
    if found and name:
        response_text = format_eligibility_response(found, name, evaluation.mob.strip())
    else:
        response_text = "I could not find your mobile number in our records. Please check the number."
    
    suggestion = await generate_followup_suggestions(chat_history=conversation, current_query=query, context="eligibility")
        
    return {"output": response_text, "intent": "eligibility","suggestion":suggestion,
            "chat_history": [{"user": query, "assistant": response_text}]
            }

async def dealer_flow_async(state: State):
    query = state["user_query"]
    conversation = state.get("chat_history", [])
    
    llm_client = ChatGroq(model="llama-3.3-70b-versatile")
    evaluator = llm_client.with_structured_output(Feedback)

    message = [
        SystemMessage(content="Extract 6-digit pincode and category.Use the conversation history to fill in the missing details. If missing, ask politely."),
    ]
    for turn in conversation[-6:]:
        message.append(HumanMessage(content=turn["user"]))
        message.append(AIMessage(content=turn["assistant"]))
    message.append(HumanMessage(content=query))

    feedback = evaluator.invoke(message)

    # Check if we are missing either piece of info
    if not feedback.pincode or not feedback.category:
        resp = feedback.feedback or "Please provide the missing information."
        return {
            "output": resp,
            "intent": "dealer_query",
            "suggestion" : get_static_suggestions("dealer_query"),
            "chat_history": [{"user": query, "assistant": resp}]
        }

    # If we have both, proceed with dealer lookup
    mapped = map_category_terms(feedback.category.strip())
    exact = search_exact_pincode(dealers_df, feedback.pincode.strip(), mapped, top_n=5)

    if not exact:
        response = f"Sorry, I couldn't find any stores for {mapped} in {feedback.pincode}."
    else:
        dealers_info = [f"{i+1}. {r.get('Dealer_Name')} – {r.get('Dealer_Address')}" for i, r in enumerate(exact)]
        response = f"I found {len(exact)} {mapped} stores in {feedback.pincode}:\n" + "\n".join(dealers_info)

    suggestion = await generate_followup_suggestions(chat_history=conversation, current_query=query, context="dealer_query")
    return {"output": response, "intent": "dealer_query", "suggestion": suggestion, "chat_history": [{"user": query, "assistant": response}]}

async def rag_flow_async(state:State):
    print("Entering RAG flow")
    query = state["user_query"]
    global qa_chain, memory
    if qa_chain:
        print("searching knowledge base")
        try:
            conversation = state.get("chat_history",[])
            formatted_history = []
            for turn in conversation:
                formatted_history.append(HumanMessage(content=turn["user"]))
                formatted_history.append(AIMessage(content=turn["assistant"]))

            print("chat history:", conversation)
            start_time = time.time()
            res = await qa_chain.ainvoke({"input": query, "chat_history": formatted_history})
            response =  (
            res.get("answer")
            or res.get("result")
            or res.get("output_text")
            or res.get("response")
            or res.get("text")
        )
            if memory:
                memory.save_context({"input": query}, {"output": response})
            end_time = time.time()
            
            not_found_phrases = ["i don't know", "i do not know", "no information", "cannot find", "no data", "not in the context"]
            found_in_rag = not any(phrase in response.lower() for phrase in not_found_phrases)
            
            suggestion = await generate_followup_suggestions(chat_history=conversation, current_query=query, context="rag_faq")
            if found_in_rag:
                
                return {
                    "output": response,
                    "suggestion":suggestion,
                    "chat_history": [{"user": query, "assistant": response}]
                }
            else:
                print("not found in rag, using semantic fallback")
                fallback_response = await fallback_to_llm(query)
                return {
                    "output": fallback_response,
                    "suggestion":suggestion,
                    "chat_history": [{"user": query, "assistant": fallback_response}]
                }
        except Exception as e:
            print(f"RAG error: {e}")
            fallback_response = await fallback_to_llm(query)
            return {
                "output": fallback_response,
                "chat_history": [{"user": query, "assistant": fallback_response}]
            }
    else:
        print("no rag, using llm")
        fallback_response = await fallback_to_llm(query)
        return {
            "output": fallback_response,
        }

router_builder = StateGraph(State)
print("Building router workflow...")

router_builder.add_node("detect_intent", detect_intent)
router_builder.add_node("eligibility_flow_node", eligibility_flow_node)
router_builder.add_node("dealer_flow_node", dealer_flow_node)
router_builder.add_node("rag_flow_node", rag_flow_node)

router_builder.add_edge(START, "detect_intent")

router_builder.add_conditional_edges(
    "detect_intent",
    route_decision,
    {
        "eligibility_flow_node": "eligibility_flow_node",
        "dealer_flow_node": "dealer_flow_node",
        "rag_flow_node": "rag_flow_node",
    },
)

router_builder.add_edge("eligibility_flow_node", END)
router_builder.add_edge("dealer_flow_node", END)
router_builder.add_edge("rag_flow_node", END)

graph = router_builder.compile()


def chat_once(user_query: str, conversation: List[Conv]) -> State:
    state_out = graph.invoke({"user_query": user_query,
                              "chat_history": conversation})
    return {
        "reply": state_out.get("output"),
        "conversation": state_out.get("chat_history"),
        "intent": state_out.get("intent"),
    }

# if __name__ == "__main__":
#     print("Starting conversational router. Type 'exit' or 'quit' to end.\n")
    
#     conversation: List[Conv] = []

#     try:
#         while True:
#             try:
#                 user_text = input("You: ").strip()
#             except EOFError:
#                 print("\nEOF received — exiting.")
#                 break

#             if not user_text:
#                 continue

#             if user_text.lower() in ("exit", "quit", "bye"):
#                 print("Bot: Bye! Ending conversation.")
#                 break


#             try:
#                 state_out = graph.invoke({"user_query":user_text,
#                                           "chat_history":conversation})

#                 conversation = state_out.get("chat_history")


#                 print("State output:\n", state_out.get("output"))

#             except Exception as e:
#                 print(f"Bot: Sorry, an internal error occurred: {e}")
#                 continue

#             intent = state_out.get("intent")
#             output = state_out.get("output")

#             if intent:
#                 print(f"Intent: {intent}")

#             if output:
#                 print(f"Bot: {output}")
#             else:
#                 # print(output)
#                 print("Bot:",output)

#     except KeyboardInterrupt:
#         print("\nKeyboardInterrupt — exiting.")