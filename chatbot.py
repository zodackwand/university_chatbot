import streamlit as st
import argparse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
import openai
from dotenv import load_dotenv
import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()
# Установите ваш API ключ OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    st.title('Ask Student Handbook')

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input('Ask your question about university rules and processes:')

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Prepare the DB.
        embedding_function = OpenAIEmbeddings(openai_api_key=openai.api_key)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(prompt, k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            answer = "Unable to find matching results."
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            formatted_prompt = prompt_template.format(context=context_text, question=prompt)

            model = OpenAI(openai_api_key=openai.api_key)
            response_text = model.predict(formatted_prompt)

            sources = [doc.metadata.get("source", None) for doc, _score in results]
            answer = response_text

        st.chat_message('assistant').markdown(answer)
        st.session_state.messages.append({'role': 'assistant', 'content': answer})

if __name__ == "__main__":
    main()