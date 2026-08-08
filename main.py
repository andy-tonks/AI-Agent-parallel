"""
Parallel Agent Execution in LangGraph
Running multiple agents simultaneously
"""

import os
from typing_extensions import TypedDict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/chat")
def chat_page():
    with open("chat.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html, media_type="text/html; charset=utf-8")


# @app.get("/chat")
# def chat_page():
#     with open("chat.html", "r", encoding="utf-8") as f:
#         return HTMLResponse(f.read())


# -----------------------------
# LLM Setup
# -----------------------------
ChatOpenAI.openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)


# -----------------------------
# LangGraph State
# -----------------------------
class ParallelState(TypedDict):
    query: str
    research_result: str
    creative_result: str
    technical_result: str
    final_synthesis: str


# -----------------------------
# Graph Construction
# -----------------------------
def create_parallel_research():
    """Three research agents working in parallel."""

    def research_agent(state: ParallelState) -> dict:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an academic researcher. Provide factual, well-sourced information."
                ),
                HumanMessage(content=f"Research this topic: {state['query']}"),
            ]
        )
        return {"research_result": response.content}

    def creative_agent(state: ParallelState) -> dict:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a creative thinker. Provide novel perspectives and ideas."
                ),
                HumanMessage(content=f"Give creative insights on: {state['query']}"),
            ]
        )
        return {"creative_result": response.content}

    def technical_agent(state: ParallelState) -> dict:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a technical analyst. Provide practical, implementation-focused insights."
                ),
                HumanMessage(content=f"Analyze technically: {state['query']}"),
            ]
        )
        return {"technical_result": response.content}

    def synthesize(state: ParallelState) -> dict:
        synthesis_prompt = f"""Synthesize these three perspectives into a comprehensive response:

        RESEARCH: {state['research_result']}

        CREATIVE: {state['creative_result']}

        TECHNICAL: {state['technical_result']}

        Create a unified, well-structured response."""
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an expert synthesizer. Combine multiple perspectives into coherent insights.  Use new paragraphs to format the response.content and provide a clear conclusion."
                ),
                HumanMessage(content=synthesis_prompt),
            ]
        )
        return {"final_synthesis": response.content}

    graph = StateGraph(ParallelState)

    # Nodes with simple "overwrite" reducers
    graph.add_node("research", research_agent, reducer=lambda old, new: new)
    graph.add_node("creative", creative_agent, reducer=lambda old, new: new)
    graph.add_node("technical", technical_agent, reducer=lambda old, new: new)
    graph.add_node("synthesize", synthesize, reducer=lambda old, new: new)

    # Fan-out from START
    graph.add_edge(START, "research")
    graph.add_edge(START, "creative")
    graph.add_edge(START, "technical")

    # Fan-in to synthesize
    graph.add_edge("research", "synthesize")
    graph.add_edge("creative", "synthesize")
    graph.add_edge("technical", "synthesize")

    graph.add_edge("synthesize", END)

    return graph.compile()


# Create the agent once, for reuse in FastAPI
qa = create_parallel_research()


# -----------------------------
# FastAPI Endpoint
# -----------------------------
@app.post("/ask")
def ask_question(payload: dict):
    # Accept either "query" or "message" from the frontend
    user_input = payload.get("query") or payload.get("message")
    if not user_input:
        return {"error": "Missing 'query' or 'message' in request body."}

    result = qa.invoke(
        {
            "query": user_input,
            "research_result": "",
            "creative_result": "",
            "technical_result": "",
            "final_synthesis": "",
        }
    )

    return {
        "research": str(result.get("research_result", "")),
        "creative": str(result.get("creative_result", "")),
        "technical": str(result.get("technical_result", "")),
        "answer": str(result.get("final_synthesis", "")),
    }


# -----------------------------
# Optional CLI Demo
# -----------------------------
def demo_parallel_execution():

    print("Parallel Agent Execution Demo:\n")

    result = qa.invoke(
        {
            "query": "qual é o futuro da advocacia previdenciária em São José dos Campos?",
            "research_result": "",
            "creative_result": "",
            "technical_result": "",
            "final_synthesis": "",
        }
    )

    print("Individual Perspectives:")
    print(f"\n[Research]\n\n{result['research_result'][:300]}...")
    print(f"\n[Creative]\n\n{result['creative_result'][:300]}...")
    print(f"\n[Technical]\n\n{result['technical_result'][:300]}...")
    print(f"\n{'=' * 50}")
    print(f"[SYNTHESIZED]\n\n{result['final_synthesis']}")



if __name__ == "__main__":
    demo_parallel_execution()
    # Run server:
    # uvicorn main:app --reload
    # Open:
    # http://127.0.0.1:8000/chat
