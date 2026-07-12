"""
Parallel Agent Execution in LangGraph
Running multiple agents simultaneously
"""

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import asyncio
import os

ChatOpenAI.openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


class ParallelState(TypedDict):
    query: str
    research_result: str
    creative_result: str
    technical_result: str
    final_synthesis: str


def create_parallel_research():
    """Three research agents working in parallel."""

    def research_agent(state: ParallelState) -> dict:
        """Academic/factual research."""
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
        """Creative perspectives."""
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
        """Technical analysis."""
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
        """Combine all perspectives."""
        synthesis_prompt = f"""Synthesize these three perspectives into a comprehensive response:

        RESEARCH: {state['research_result']}

        CREATIVE: {state['creative_result']}

        TECHNICAL: {state['technical_result']}

        Create a unified, well-structured response."""

        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an expert synthesizer. Combine multiple perspectives into coherent insights."
                ),
                HumanMessage(content=synthesis_prompt),
            ]
        )
        return {"final_synthesis": response.content}

    graph = StateGraph(ParallelState)

    graph.add_node("research", research_agent)
    graph.add_node("creative", creative_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("synthesize", synthesize)

    # Fan-out: START goes to all three agents
    graph.add_edge(START, "research")
    graph.add_edge(START, "creative")
    graph.add_edge(START, "technical")

    graph.add_edge("research", "synthesize")
    graph.add_edge("creative", "synthesize")
    graph.add_edge("technical", "synthesize")

    graph.add_edge("synthesize", END)

    return graph.compile()


def demo_parallel_execution():
    """Demo parallel agent execution."""

    agent = create_parallel_research()

    print("Parallel Agent Execution Demo:\n")

    result = agent.invoke(
        {
            "query": "qual e o futuro da advogada do previdencia em São jose dos campos",
            "research_result": "",
            "creative_result": "",
            "technical_result": "",
            "final_synthesis": "",
        }
    )

    print("Individual Perspectives:")
    print(f"\n[Research]\n{result['research_result'][:300]}...")
    print(f"\n[Creative]\n{result['creative_result'][:300]}...")
    print(f"\n[Technical]\n{result['technical_result'][:300]}...")

    print(f"\n{'='*50}")
    print(f"[SYNTHESIZED]\n{result['final_synthesis']}")



if __name__ == "__main__":
    demo_parallel_execution()

    #demo_map_reduce()
