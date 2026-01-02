import json
import asyncio
import os
from pydantic import BaseModel, Field
import google.generativeai as genai
from golden_dataset import GOLDEN_SET
from app import app  # Import your actual RAG pipeline

# 1. Define the Evaluation Criteria
class EvalResult(BaseModel):
    faithfulness_score: int = Field(description="Score 1-5: 5 means every claim is backed by context, 1 means hallucination.")
    relevance_score: int = Field(description="Score 1-5: 5 means it answered the question perfectly, 1 means it missed the point.")
    reasoning: str = Field(description="Detailed explanation of why the score was given.")

# Configure Gemini
if "GOOGLE_API_KEY" not in os.environ:
    print("Warning: GOOGLE_API_KEY not set. Gemini calls will fail.")
else:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

from golden_dataset import GOLDEN_SET

# 3. The Judge Logic
async def judge_output(question, ground_truth, ai_answer, context):
    print(f"Judging response for: {question[:50]}...")
    
    prompt = f"""
    You are an expert legal auditor. Compare the AI's Answer against the Ground Truth and the provided Context.
    
    Question: {question}
    Context: {context}
    Ground Truth: {ground_truth}
    AI Answer: {ai_answer}
    
    Evaluate based on:
    1. Faithfulness: Is the AI answer supported ONLY by the context?
    2. Relevance: Did it answer the specific question asked?
    """
    
    model = genai.GenerativeModel(
        model_name="gemini-3-pro-preview", 
        system_instruction="You are a strict grading judge.",
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": EvalResult
        }
    )
    
    response = await model.generate_content_async(prompt)
    return EvalResult.model_validate_json(response.text)

# 4. The Runner
async def run_evaluation():
    results = []
    
    for entry in GOLDEN_SET:
        print(f"\nProcessing Question: {entry['question']}")
        
        # Step A: Run your actual RAG pipeline
        # invoke returns the final state of the graph
        final_state = await app.ainvoke({"question": entry['question']})
        
        ai_answer = final_state['answer']
        # Context is a list of strings, joining them for the judge
        context_used = "\n\n".join(final_state['context'])
        
        print(f"  -> AI Answer: {ai_answer[:100]}...")
        
        # Step B: Judge it
        score = await judge_output(entry['question'], entry['ground_truth'], ai_answer, context_used)
        results.append(score)

    # 5. Summarize
    avg_faith = sum(r.faithfulness_score for r in results) / len(results)
    avg_rel = sum(r.relevance_score for r in results) / len(results)
    
    print("\n--- EVALUATION COMPLETE ---")
    print(f"Average Faithfulness: {avg_faith}/5")
    print(f"Average Relevance: {avg_rel}/5")
    
    for i, res in enumerate(results):
        print(f"\nCase {i+1} Reasoning: {res.reasoning}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
