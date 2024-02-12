import streamlit as st
import openai
from openai import OpenAI
import transformers
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import autopep8

model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
OPENAI_API_KEY="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"
client = OpenAI(api_key=OPENAI_API_KEY)


# Generate algorithm
def generate_algorithm(generated_code):
    prompt = f""" Your AI assistant's task is to produce a explaination for the code delimited by ``` which can further be used for language translation also\
    ``` code:{generated_code} ``` """
    completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
        )
    return completion.choices[0].message.content
from transformers import TFAutoModelForSeq2SeqLM

model_path = "Vishwasv007/eng-hin"
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)


translated_algo=""

question = st.text_input("Type your question here...")
response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0613:personal::8p1RYc64",
        messages=[{"role": "user", "content": question}],
)

with st.sidebar:
    st. title("Get your text converted to Code")


if question:
    generated_code = response.choices[0].message.content
    formatted_code = autopep8.fix_code(generated_code)

    algorithm = generate_algorithm(generated_code)

    tokenized = tokenizer([algorithm], return_tensors='np')
    out = model.generate(**tokenized, max_length=218)
    with tokenizer.as_target_tokenizer():
        translated_algorithm = tokenizer.decode(out[0], skip_special_tokens=True)
    translated_algo=translated_algorithm

    # Create two columns
    col1, col2 ,col3= st.columns(3)

    # Display algorithm in the first column
    col1.header("Algorithm")
    col1.write(algorithm)

    # Display generated code in the second column
    col2.header("Generated Code")
    col2.code(formatted_code, language="python")

    col3.header("translated algorithm")
    if col3.button("Translate"):
        # Display translated algorithm in the same column
        col3.write(translated_algo)
