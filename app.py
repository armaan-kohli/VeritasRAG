import operator
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# 1. Define the State
class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    faithfulness_score: float
    critique: str
    iterations: int

# 2. Define the "Critic" Output Structure
class QualityReview(BaseModel):
    score: float = Field(description="Score between 0 and 1. 1 is perfectly faithful to source.")
    critique: str = Field(description="Explanation of any hallucinations or missing info.")

# --- NODES ---

def retrieve(state: AgentState):
    print("---RETRIEVING---")
    # Generic placeholder for your Vector DB (Pinecone/Chroma)
    return {"context": ["Snippet 1 from doc...", "Snippet 2 from doc..."]}

def generate_answer(state: AgentState):
    print("---GENERATING---")
    # The LLM draft based on state['context']
    return {"answer": "The contract ends on Jan 1st.", "iterations": state.get("iterations", 0) + 1}

def critique_answer(state: AgentState):
    print("---CRITIQUING---")
    # Second LLM call (The Critic) comparing state['answer'] vs state['context']
    # If the LLM sees the date is wrong, it gives a low score.
    review = QualityReview(score=0.4, critique="Date in snippet is Dec 31, but answer says Jan 1.")
    return {"faithfulness_score": review.score, "critique": review.critique}

# 3. Define the Router (The "Decision Maker")
def decide_to_finish(state: AgentState):
    if state["faithfulness_score"] > 0.8 or state["iterations"] > 3:
        return "end"
    else:
        return "generate" # Loop back to fix it

# 4. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate_answer)
workflow.add_node("critique", critique_answer)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "critique")

workflow.add_conditional_edges(
    "critique",
    decide_to_finish,
    {
        "end": END,
        "generate": "generate"
    }
)

app = workflow.compile()