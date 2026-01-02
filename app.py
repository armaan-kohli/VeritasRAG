import operator
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import os

# New Imports for Retrieval & Generation
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

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
    
    if "GOOGLE_API_KEY" not in os.environ:
        print("Warning: GOOGLE_API_KEY not set. Returning mock context.")
        return {"context": ["Mock Context: Section 4.2 penalty is $500."]}

    try:
        # 1. Connect to the existing DB
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        # 2. Search
        # k=3 means "get top 3 most relevant chunks"
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(state["question"])
        
        # 3. Format for the agent
        context_text = [doc.page_content for doc in docs]
        
        # Fallback if nothing found (prevent empty context errors downstream)
        if not context_text:
             context_text = ["No relevant documents found."]
             
        return {"context": context_text}
        
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return {"context": ["Error retrieving documents."]}

def generate_answer(state: AgentState):
    print("---GENERATING---")
    
    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0)
    
    # Create prompt
    context_str = "\n".join(state["context"])
    prompt = f"""
    Answer the question based ONLY on the context below.
    
    Context:
    {context_str}
    
    Question: {state['question']}
    
    Answer:
    """
    
    response = llm.invoke(prompt)
    return {"answer": response.content, "iterations": state.get("iterations", 0) + 1}

def critique_answer(state: AgentState):
    print("---CRITIQUING---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0)
    structured_llm = llm.with_structured_output(QualityReview)
    
    prompt = f"""
    You are a strict fact-checker. 
    Compare the Answer to the Context.
    
    Context: {state['context']}
    Answer: {state['answer']}
    
    Give a score (0.0 to 1.0) on how faithful the answer is to the context.
    If there is any information in the answer NOT in the context, penalize the score.
    """
    
    review = structured_llm.invoke(prompt)
    return {"faithfulness_score": review.score, "critique": review.critique}

# 3. Define the Router (The "Decision Maker")
def decide_to_finish(state: AgentState):
    # End if score is high OR if we've looped too many times (to prevent infinite loops)
    if state["faithfulness_score"] > 0.8 or state["iterations"] > 2:
        return "end"
    else:
        return "generate" 

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
