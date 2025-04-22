from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from src.utils.gemini import get_gemini_model

def create_conversation_chain(prompt_template: str, input_variables: list[str]):
    """
    Create a basic conversation chain with Gemini model
    """
    llm = get_gemini_model()
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=input_variables
    )
    return LLMChain(llm=llm, prompt=prompt) 