import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# .env 파일 로드 (프로젝트 루트 디렉토리 기준)
dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

def main():
    # 환경변수에서 API 키 가져오기
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    # Gemini 모델 초기화
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.7,
        google_api_key=api_key
    )

    # 프롬프트 템플릿 설정
    template = """
    다음 질문에 대해 친절하게 답변해주세요:
    
    질문: {question}
    
    답변:"""

    prompt = PromptTemplate(
        input_variables=["question"],
        template=template
    )

    # LangChain 체인 생성
    chain = LLMChain(llm=llm, prompt=prompt)

    # 체인 실행
    result = chain.invoke({"question": "안녕하세요! LangChain과 Gemini에 대해 설명해주세요."})
    print(result["text"])

if __name__ == "__main__":
    main()