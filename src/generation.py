from openai import OpenAI
from config import settings

client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def generate_reasoned_answer(query: str, context: list):
    """Generate a reasoned answer based on context using DeepSeek."""
    try:
        if not context:
            return {
                "thought": "No context provided",
                "answer": "Sorry, no relevant documents were found to answer your question. Please upload PDFs first."
            }
        
        formatted_context = "\n\n".join(context)
        
        response = client.chat.completions.create(
            model="deepseek-chat",  # Using deepseek-chat for general availability
            messages=[
                {"role": "system", "content": "You are a professional RAG assistant. Use the provided context to reason and answer."},
                {"role": "user", "content": f"Context: {formatted_context}\n\nQuestion: {query}"}
            ]
        )
        
        # Extract thinking and answer from response
        thinking_content = getattr(response.choices[0].message, 'reasoning_content', None)
        answer_content = response.choices[0].message.content
        
        return {
            "thought": thinking_content or "Reasoning not available",
            "answer": answer_content
        }
    except Exception as e:
        print(f"Error generating answer: {e}")
        return {
            "thought": "An error occurred while processing",
            "answer": f"Sorry, I encountered an error: {str(e)}"
        }