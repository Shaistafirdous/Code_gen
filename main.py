import streamlit as st
import openai
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import autopep8

# Initialize OpenAI client
client = openai.Client(st.secrets.get("OPENAI_API_KEY"))

# Initialize tokenizer and model for translation
model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

translation_model_path = "Vishwasv007/eng-hin"
translation_model = TFAutoModelForSeq2SeqLM.from_pretrained(translation_model_path)

question = st.text_input("Type your question here...")

with st.sidebar:
    st.title("Get your text converted to Code")

if question:
    # Generate code
    response = client.chat_completion.create(
        model="ft:gpt-3.5-turbo-0613:personal::8p1RYc64",
        messages=[{"role": "user", "content": question}],
    )
    generated_code = response.choices[0].message.content
    formatted_code = autopep8.fix_code(generated_code)

    # Generate algorithm
    prompt = f"""Your AI assistant's task is to produce an explanation for the code delimited by ``` which can further be used for language translation also\n``` code:{generated_code} ```"""
    completion = client.chat_completion.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
    )
    algorithm = completion.choices[0].message.content

    # Display code and algorithm
    st.header("Algorithm")
    st.write(algorithm)

    st.header("Generated Code")
    st.code(formatted_code, language="python")

if st.button("Translate"):
    # Translate algorithm
    tokenized = tokenizer([algorithm], return_tensors='tf')
    translated = translation_model.generate(**tokenized, max_length=218)
    with tokenizer.as_target_tokenizer():
        translated_algorithm = tokenizer.decode(
            translated[0], skip_special_tokens=True
        )

    # Display translated algorithm
    st.header("Translated Algorithm")
    st.write(translated_algorithm)

