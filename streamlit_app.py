import streamlit as st
from groq import Groq
from langchain.chat_models import ChatGroq
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Page title
st.set_page_config(page_title='🦜🔗 Text Summarization App')
st.title('🦜🔗 Text Summarization App')
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)
def generate_response(txt):
    # Instantiate the LLM model
    #llm = ChatGroq(model_name="llama3-8b-8192", temperature=0, groq_api_key=groq_api_key)

    #llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0, groq_api_key=st.secrets["GROQ_API_KEY"])
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)


# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    #openai_api_key = st.text_input('OpenAI API Key', type = 'password', disabled=not txt_input)
    #submitted = st.form_submit_button('Submit')
    #if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(txt_input)
            result.append(response)
           # del openai_api_key

if len(result):
    st.info(response)
