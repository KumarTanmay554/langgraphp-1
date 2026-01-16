import dotenv
from supabase import Client, create_client
from operator import add
from langchain_nvidia import ChatNVIDIA
import os
from typing import List, Optional
from langchain_groq import ChatGroq
from llm2 import get_static_suggestions, parse_suggestions
import streamlit as st

dotenv.load_dotenv()

url = os.environ.get("SUPABASE_URL") or st.secrets["SUPABASE_URL"]
key = os.environ.get("SUPABASE_KEY") or st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)
nvapi = os.environ.get("NV_API_KEY") or st.secrets["NV_API_KEY"]

llm_client = ChatGroq(
    model= "openai/gpt-oss-120b",
    api_key= os.environ.get("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
)

suggestion_client = ChatNVIDIA(
    model="nvidia/nemotron-3-nano-30b-a3b",
    api_key= nvapi
)

async def generate_followup_suggestions(chat_history, current_query=None, context=None):
    
    if not suggestion_client:
        return get_static_suggestions(context)
    
    try:
        if not chat_history:
            return get_static_suggestions(context)
        
        # Build context
        history_text = ""
        if isinstance(chat_history, list):
            for msg in chat_history[-1:]: 
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    history_text += f"{role}: {content}\n"
                else:
                    role = "assistant" if getattr(msg, 'type', 'user') == "ai" else "user"
                    content = getattr(msg, 'content', str(msg))
                    history_text += f"{role}: {content}\n"
        
        if current_query:
            history_text += f"user: {current_query}\n"
        
        system_prompt = f"""
        Based on the following conversation history, generate 3-4 relevant follow-up questions that a user might ask about Bajaj Finance REMI services, dealers, or products.

        Guidelines:
        - Questions should be natural and contextually relevant
        - Questions should be very short and to the point
        - Focus on Bajaj Finance REMI products and services
        - Include questions about dealer locations, eligibility, products, or services
        - Keep questions concise and clear
        - Only 2-3 questions
        - Don't mention about giving follow up questions explicitly
        - Do NOT write things like: ‘Here are 3 things…’, ‘You could also ask…’, ‘Some follow-up questions are…’. Avoid such patterns completely.
        Conversation History:
        {history_text}"""
        
        suggestions = ""
        for i in suggestion_client.stream([{"role":"user","content":system_prompt}]):
            if i.additional_kwargs and "content" in i.additional_kwargs:
                suggestions += i.additional_kwargs["content"]
            if i.content:
                suggestions += i.content
        return parse_suggestions(suggestions)
        
    except Exception as e:
        print(f"Error generating follow-up suggestions: {e}")
        return get_static_suggestions(context)

def check_remi_eligibility(mobile:str)->tuple[bool,Optional[str]]:
    if not mobile:
        return (False,None)
    try:
        print("mobile check")
        try:
            mobile = int(mobile)
            print("mobile after int:", mobile)
        except ValueError:
            print("mobile not int")
            return (False,None)
        res = (supabase.table("customer_data").select("Name").eq("Mobile_Number", mobile).limit(1).execute())
        print("supabase response:", res)
        data = res.data
        if not data:
            print("no data")
            return False,None
        name = data[0].get("Name")
        return True,name
    except Exception as e:  
        print("Supabase error: ",e)
        return False, None
    
async def fallback_to_llm(query, chat_history):
    if llm_client:
        try:
            
            # response = llm_client.generate_from_messages(messages + [{"role": "user", "content": query}])
            response = llm_client.invoke([])
            # print("using llm fallback response")
            
            # Save to memory
            # if memory:
            #     memory.save_context({"input": query}, {"output": response})
                
            # Speak the response
            if len(response) > 300:
                chunks = [response[i:i+300] for i in range(0, len(response), 300)]
                for chunk in chunks:
                    # await speak(chunk)
                    print(chunk)
                    return response
            else:
                # await speak(response)
                return response
        except Exception as e:
            # await speak(f"Sorry, I couldn't find an answer to that question. {str(e)}")
            print(f"LLM fallback error: {e}")
    else:
        # await speak("LLM fallback not available right now.")
        print("LLM fallback not available right now.")