from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any,Optional
from test import graph, Conv
import uuid
import uvicorn

app = FastAPI(debug=True)

SESSION:Dict[str, list] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_query: str
    # session_id:str|None = None
    chat_history:List[Conv] = []

class ChatResponse(BaseModel):
    reply: str
    intent: str|None = None
    chat_history: List[Conv]
    suggestion: Optional[List[str]]

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # if not req.session_id:
    #     session_id = str(uuid.uuid4())
    #     SESSION[session_id] = []
    # else:
    #     session_id = req.session_id
    #     if session_id not in SESSION:
    #         SESSION[session_id] = []
    # conversation = SESSION[session_id]
    # state_out = chat_once(req.user_query, conversation)
    # SESSION[session_id] = state_out["conversation"]

    # return{
    #     "reply": state_out["reply"],
    #     "intent": state_out["intent"],
    #     "conversation": state_out["conversation"],
    #     "session_id": session_id
    # }
    try:
        print("Received chat request:", req)
        print("OG Chat history:", req.chat_history)
        clean_history = [
            {
                "user": conv["user"],
                "assistant": conv["assistant"]
            } for conv in req.chat_history
        ]
        inputs = {
            "user_query": req.user_query,
            "chat_history": clean_history
        }
        print("Inputs to graph:", inputs)
        result = await graph.ainvoke(inputs)
        print("Graph result:", result)

        # new_history = result.get("chat_history", [])
        # new_history.append({
        #     "user":req.user_query,
        #     "assistant": result.get("output")
        # })
        # print("New chat history:", new_history)
        return{
            "reply": result.get("output"),
            "intent": result.get("intent"),
            "suggestion":result.get("suggestion"),
            "chat_history" : result.get("chat_history")
        }
    except Exception as e:
        print(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)