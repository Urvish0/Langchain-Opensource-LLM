import requests
import streamlit as st 
import json

def get_response(url, input_text):
    try:
        response = requests.post(url, json={"input": {'topic': input_text}})
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        data = response.json()
        if isinstance(data, dict) and 'output' in data:
            return data['output']
        else:
            return f"Unexpected response structure: {data}"
    except requests.exceptions.RequestException as e:
        if e.response is not None:
            try:
                error_data = e.response.json()
                return f"Server error: {error_data.get('message', str(e))}\nDetails: {error_data.get('details', 'No details provided')}"
            except json.JSONDecodeError:
                return f"Server error: {str(e)}\nResponse: {e.response.text}"
        else:
            return f"Request failed: {str(e)}"
    except json.JSONDecodeError:
        return f"Failed to decode JSON. Response content: {response.text}"

def get_huggingface_response(input_text):
    return get_response("http://localhost:8000/essay/invoke", input_text)

def get_ollama_response(input_text):
    return get_response("http://localhost:8000/code/invoke", input_text)

st.title('Langchain Demo API')
input_text = st.text_input("Write an essay on")
input_text1 = st.text_input("Write a python code for")

if input_text:
    response = get_huggingface_response(input_text)
    st.write(response)
    
if input_text1:
    response = get_ollama_response(input_text1)
    st.write(response)
