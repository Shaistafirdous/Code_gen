import streamlit as st
import openai
from openai import OpenAI

# Replace with your actual OpenAI API key
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

question = st.text_input("Type your question here...")
with st.sidebar:
    st. title("Get your text converted to Code")
if question:
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0613:personal::8p1RYc64",
        messages=[{"role": "user", "content": question}],
    )
    generated_code = response.choices[0].message.content

    # Generate algorithm
    prompt = f""" Your AI assistant's task is to produce a step-by-step algorithm for the code delimited by ```\
    ``` code:{generated_code} ``` """
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )

    algorithm = completion.choices[0].message.content

    # Create two columns
    col1, col2 = st.columns(2)

    # Display algorithm in the first column
    col1.header("Algorithm")
    col1.write(algorithm)

    # Display generated code in the second column
    col2.header("Generated Code")
    col2.code(generated_code, language="python")
