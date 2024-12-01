from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
load_dotenv()


# FastAPI app initialization
app = FastAPI()

# Configure OpenAI/NVIDIA API key
API_KEY = os.getenv("NVIDIA_API_KEY")  # Store your key as an environment variable
openai.api_key = API_KEY

# Chat request and response schema
class ChatRequest(BaseModel):
    user_message: str  # The user's input
    temperature: float = 0.5
    top_p: float = 0.7
    max_tokens: int = 1024

class ChatResponse(BaseModel):
    assistant_message: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    API endpoint for jungle-related chatbot interaction.
    """
    try:
        # System message to guide the chatbot's behavior
        system_message = {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant who only answers questions about jungles, "
                "national parks, wildlife sanctuaries, animals, and plants found in jungles. "
                "If a question is not related to jungles, kindly inform the user politely."
            )
        }
        
        # User message
        user_message = {"role": "user", "content": request.user_message}

        # Call NVIDIA/OpenAI API
        response = openai.ChatCompletion.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",  # You can update this model as required
            messages=[system_message, user_message],
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )

        # Extract the assistant's reply
        assistant_message = response['choices'][0]['message']['content']
        return ChatResponse(assistant_message=assistant_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
