import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders.csv_loader import CSVLoader
import requests
from PIL import Image
import pydeck as pdk
import os
import json

st.set_page_config(
   page_title="FoodGPT - Nagpur Based Food Recommendation System.",
   page_icon="🍊",
   layout="wide",
   initial_sidebar_state="expanded",
)

from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='data.csv')
documents = loader.load()

text_splitter  = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':"cpu"})

vector_store = FAISS.from_documents(text_chunks,embeddings)

llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",
                    config={'max_new_tokens':128,'temperature':0.01})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                              memory=memory)

# Sidebar for user input
st.sidebar.title("FoodGPT!🍊")
st.sidebar.info("FoodGPT : A Nagpur Based Food Recommendation Chat! Recommends you the best locally recognized brands for your cravings! As this system is backed with LLMA-2 on hand picked data.")
github_link = "[GitHub]()"
st.sidebar.info("To contribute and Sponser - " + github_link)

st.title("FoodGPT: A Nagpur based Food Recommendation Bot! 🍊")

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello!I'm FoodGPT, Ask me anything about Nagpur's Food."]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hello!"]

reply_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Question:", placeholder="Ask anything about Nagpur's Food Joints or cravings", key='input')
        image_upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        submit_button = st.form_submit_button(label='Send')

    try:
        if submit_button and user_input:
            output = chain({"question": user_input, "chat_history": st.session_state['history']})["answer"]
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if st.session_state['generated']:
    with reply_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
            message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


API_URL = "https://api-inference.huggingface.co/models/Prasanna18/indian-food-classification"
HEADERS = {"Authorization": "Bearer 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"}

def query_image_classification(image_bytes):
    try:
        response = requests.post(API_URL, headers=HEADERS, data=image_bytes)
        result = response.json()
        return result
    except Exception as e:
        st.error(f"An error occurred during image classification: {str(e)}")
        return None

if image_upload:
    image_bytes = image_upload.read()
    
    classification_result = query_image_classification(image_bytes)
    
    if classification_result:
        st.image(image_upload, caption="Uploaded Image", use_column_width=True)
        
        if isinstance(classification_result, list) and classification_result:
            # Ensure that classification_result is a list of results and not empty
            best_label = max(classification_result, key=lambda x: x.get('score', 0))
            
            if 'label' in best_label:
                st.header("Image Classification Result:")
                st.write(f"Classified as: {best_label['label']}")
            else:
                st.error("Invalid classification result format. Missing 'label' key.")
        else:
            st.error("Invalid classification result format or empty result list.")
    else:
        st.error("No classification result received.")


import pydeck as pdk

st.title("Nagpur Map")
center = [21.1458, 79.0882]

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": center[0],
            "longitude": center[1],
            "zoom": 13,
            "pitch": 10,
        },
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=[{"position": center, "tooltip": "Nagpur"}],
                get_position="position",
                get_radius=10000,
                get_color=[255, 0, 0],
                pickable=True,
            ),
        ],
    )
)


