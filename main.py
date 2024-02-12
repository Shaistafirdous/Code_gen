import streamlit as st
import openai
from openai import OpenAI
import black
import transformers
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import AdamWeightDecay
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

from transformers import TFAutoModelForSeq2SeqLM

model_path = "C:/Users/Vishwas/Desktop/personal/major project/firdous/tf_model/"
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)

OPENAI_API_KEY='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'
# Replace with your actual OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)

question = st.text_input("Type your question here...")
with st.sidebar:
    st. title("Get your text converted to Code")
if question:
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0613:personal::8p1RYc64",
        messages=[{"role": "user", "content": question}],
    )
    generated_code = response.choices[0].message.content

    #formatted_code = black.format_str(generated_code, mode=black.FileMode())

    # Generate algorithm
    prompt = f""" Your AI assistant's task is to produce a explaination for the code delimited by ``` which can further be used for language translation also\
    ``` code:{generated_code} ``` """
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
    )

    algorithm = completion.choices[0].message.content

    if st.button("Translate"):
            tokenized = tokenizer([algorithm], return_tensors='np')
            out = model.generate(**tokenized, max_length=218)
            with tokenizer.as_target_tokenizer():
                translated_algorithm = tokenizer.decode(out[0], skip_special_tokens=True)

            # Display translated algorithm in the same column
            st.header("Translated Algorithm")
            st.write(translated_algorithm)

    # Create two columns
    col1, col2 = st.columns(2)

    # Display algorithm in the first column
    col1.header("Algorithm")
    col1.write(algorithm)

    # Display generated code in the second column
    col2.header("Generated Code")
    col2.code(generated_code, language="python")
