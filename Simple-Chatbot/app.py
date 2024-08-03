from langchain_huggingface import HuggingFaceEndpoint 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st 
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
## Langsmith tracking
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question: {question}")
    ]
)

## streamlit framework

st.title('Langchain Learning with HuggingFace')
input_text = st.text_input("Search the topic u want")

## HuggingFace llm
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, temperature= 0.7, model_kwargs={
        "max_length": 128,
        "token": os.getenv("HUGGINGFACE_API_KEY")
    })

output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    try:
        result = chain.invoke({'question': input_text})
        st.write(result)
    except Exception as e:
        st.write(f"Error: {e}")
