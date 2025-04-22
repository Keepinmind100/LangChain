from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import GEMINI_API_KEY

def get_gemini_model(model_name="gemini-pro", temperature=0.7):
    """
    Initialize and return a Gemini chat model
    """
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=GEMINI_API_KEY,
    ) 