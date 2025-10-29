from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import build_graph 
import uvicorn
from langchain_core.messages import HumanMessage
import json

app = FastAPI(
    title="College Chatbot Agent",
    description="An agentic RAG chatbot with streaming and conversation history",
    version="1.1.0"
)

agent_app = build_graph()

class QueryRequest(BaseModel):
    query: str
    thread_id: str  

@app.get("/")
def read_root():
    """
    Root endpoint to check if the server is running.
    """
    return {"message": "College Chatbot API is running. POST to /chat"}

@app.post("/chat")
async def chat(request: QueryRequest):
    """
    Main chat endpoint.
    It streams the agent's LLM tokens as they are generated.
    It uses the `thread_id` to maintain conversation history.
    """
    
    inputs = {"messages": [HumanMessage(content=request.query)]}
    
    config = {"configurable": {"thread_id": request.thread_id}}

    async def event_stream():
        """
        The async generator function that streams events.
        """
        try:
            async for event in agent_app.astream_events(inputs, config, version="v1"):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    
                    node_name = event["name"]
                    
                    if node_name in ("generate_answer", "generate_synthesized_answer", "handle_chat"):
                    
                        chunk = event["data"]["chunk"]
                        if chunk.content:
                            yield f"data: {json.dumps(chunk.content)}\n\n"
                            
                # You can uncomment this to see the agent's "thoughts"
                # if kind == "on_chain_end":
                #    if event["name"] in ("rewrite_query", "classify_query"):
                #        yield f"event: log\ndata: {json.dumps(event['data']['output'])}\n\n"


            yield "event: end\ndata: [END]\n\n"
            
        except Exception as e:
            print(f"Error in stream: {e}")
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)