import streamlit as st
import openai
from openai import OpenAI
import black
import transformers
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import autopep8

model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

from transformers import TFAutoModelForSeq2SeqLM

model_path = "Vishwasv007/eng-hin"
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)

client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

question = st.text_input("Type your question here...")
with st.sidebar:
    st. title("Get your text converted to Code")
    
generated_code = ""
algorithm = ""
translated_algorithm = ""

if question:
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0613:personal::8p1RYc64",
        messages=[{"role": "user", "content": question}],
    )
    generated_code = response.choices[0].message.content
    formatted_code = autopep8.fix_code(generated_code)

    formatted_code = autopep8.fix_code(generated_code)

    # Generate algorithm
    prompt = f""" Your AI assistant's task is to produce a explaination for the code delimited by ``` which can further be used for language translation also\
    ``` code:{generated_code} ``` """
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
    )

    algorithm = completion.choices[0].message.content

    # Display code, algorithm, and translation columns
col1, col2, col3 = st.columns([1, 1, 1])

# Display algorithm in the first column
col1.header("Algorithm")
col1.write(algorithm)

# Display generated code in the second column
col2.header("Generated Code")
col2.write(formatted_code, language="python")

translate_button_pressed = st.button("Translate")

# If translate button is pressed and algorithm exists, translate it and show in third column
if translate_button_pressed and algorithm:
    tokenized = tokenizer([algorithm], return_tensors='tf')
    out = model.generate(**tokenized, max_length=218)
    with tokenizer.as_target_tokenizer():
        translated_algorithm = tokenizer.decode(out[0], skip_special_tokens=True)

    # Display translated algorithm in the third column
    col3.header("Translated Algorithm")
    col3.write(translated_algorithm)
