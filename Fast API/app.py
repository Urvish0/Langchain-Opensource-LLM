from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceEndpoint 
from langserve import add_routes
import uvicorn
import os 
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACE_API_KEY")
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"An error occurred: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred.", "details": str(exc)}
    )

try:
    hf_endpoint = HuggingFaceEndpoint(
        repo_id=repo_id, 
        temperature=0.7, 
        model_kwargs={
            "max_length": 128,
            "token": os.getenv("HUGGINGFACE_API_KEY")
        }
    )
except Exception as e:
    logger.error(f"Failed to initialize HuggingFaceEndpoint: {str(e)}", exc_info=True)
    raise

try:
    llm = Ollama(model="codegemma:7b")
except Exception as e:
    logger.error(f"Failed to initialize Ollama: {str(e)}", exc_info=True)
    raise

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a python code for {topic}")

add_routes(
    app,
    prompt1 | hf_endpoint,
    path="/essay"
)

add_routes(
    app,
    prompt2 | llm,
    path="/code"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
