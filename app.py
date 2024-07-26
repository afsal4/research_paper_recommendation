# imports
from streamlit_pdf_viewer import pdf_viewer
import streamlit as st
import os
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from streamlit import session_state as ss
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import io
import torch
from langchain_community.vectorstores import FAISS 
from readme_page import create
from prompts import *

dotenv.load_dotenv()

SIMILARITY_THRESHOLD = 0.65
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 40
INDEX_NAME = 'recomentation'
pinecone_api_key = os.getenv('PINECONE_API_KEY')

LINK = 'https://arxiv.org/pdf/'

if 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

if 'PINECONE_API_KEY' not in os.environ:
    os.environ['PINECONE_API_KEY'] = pinecone_api_key


def navigator(page_name):
    st.session_state.page_name = page_name

def get_session(session_id):
    return st.session_state.get(session_id, ChatMessageHistory())

def reset_session():
    st.session_state.message_history = []


def query_page():
    if 'links' in st.session_state:
        for i in range(len(st.session_state['links'])):
            if i in st.session_state and st.session_state[i]:
                link = st.session_state['links'][i]
                st.session_state['curr_link'] = link
                navigator('pdf_qna_page')
                pdf_qna_page(link)
                return 

    if 'message_history' not in st.session_state:
        st.session_state.message_history = []
    model = ChatGoogleGenerativeAI(model='gemini-pro', convert_system_message_to_human=True, stop=['A:'])

    # few shot prompting + chain of thought 
    template = ChatPromptTemplate.from_messages([('system',
        '''
        You are a chat assitant for helping the user to find the reaserch paper 

        Only ask three one question at a time.
        
        After asking three questions (only one at a time), create a query to search in database

        
        Example:
        `````
        Input: Sentiment analysis usig transformers network
        Answer: Is there any unique network you are interested in?
        Input: no i dont care about the transformer architecture as long as it is good for getting accurate res 
        Answer: Is there anything else you want to hadd wich help me find the best paper for you
        Input: No, I dont have any other points
        (transformer should be related to deeplearning, he doesnt care about the architecture so any architecture with high acc for predicting the sentiment of the task)
        Answer: Query: The transformer network in sentiment analysis tasks which comes with good performance
        `````
        Begin! 
        '''), 
        MessagesPlaceholder('messages')
        ])

    parser = StrOutputParser()
    chain = template | model | parser
    with_message_history = RunnableWithMessageHistory(chain, get_session)

    st.title('Research Paper Recommender')
    search_method = st.radio('##### Search methods', ['Context Search', 'Hybrid Search', 'Tf-Idf Search'], index=0)

    human_message = st.chat_input('Enter here:')
    if human_message:

        config = {'configurable': {'session_id': 'sgd'}}

        # can add some part for cleaning text 

        st.session_state.message_history.append(HumanMessage(human_message))
        ai_message = with_message_history.invoke(st.session_state.message_history, config=config)
        st.session_state.message_history.append(AIMessage(ai_message))

        for message in st.session_state['message_history']:

            if type(message) == HumanMessage:
                role = 'user'
            elif 'Query' not in message:
                role = 'ai'

            message = message.content
            with st.chat_message(role):
                st.write(message)

        if 'Query:' in ai_message:
            result_query = ai_message.replace('Query:', '').strip()
            reset_session()
            recommended_links, recommended_titles = get_links(result_query, search_method)
            if not recommended_links:
                st.write('There are no similar files that you are looking for')
                return 
            st.session_state['links'] = recommended_links

            for i, title in enumerate(recommended_titles):
                st.button(title, key=i)



def main():
    # st.write(st.session_state)
    if 'page_name' not in st.session_state:
        st.session_state.page_name = 'query_page'

    # st.write(st.session_state)
    home = st.sidebar.button('Chat')
    arch_button = st.sidebar.button('Research Paper Recommendation System Architecture')
    eval = st.sidebar.button('Evaluation')
    res_button = st.sidebar.button('Observations and Results')

    flg = not (any([home, arch_button, eval, res_button]))
    if home or flg:
        match st.session_state['page_name']:
            case 'query_page':
                query_page()

            case 'pdf_qna_page':
                link = st.session_state['curr_link']
                pdf_qna_page(link)
    elif arch_button:
        create('arch')
    elif eval:
        create('eval')
    elif res_button:
        create('res')


def reset_links():
    for link in st.session_state['links']:
        st.session_state[link] = False


def pdf_qna_page(link):
    'Page for Pdf Qna and summary'
    back_button = st.button('Back_button')
    if back_button == True:
        reset_links()
        navigator('query_page')
        st.rerun()
    
    
    path = create_pdf_viewer(link)
    qna_pdf_rag(path)

    


def load_and_get_file(query):
    "Loads the vector data base and returns the similarity search result"
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    db = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return db.similarity_search_with_score(query)


def get_correct_ids(similar_docs):
    ''''Return dictionary of id and score after correction of id'''
    res_id = [{i[0].metadata['id']: [i[1], i[0].metadata['title']]} for i in similar_docs]
    corrected_ids = {}
    for documents in res_id:
        doc_id = list(documents.keys())[0]
        tmp_id = doc_id
        tmp_id_splits = str(tmp_id).split('.')
        
        while len(tmp_id_splits[0]) < 4:
            tmp_id_splits[0] = '0' + tmp_id_splits[0]

        while len(tmp_id_splits[1]) < 4:
            tmp_id_splits[1] = tmp_id_splits[1] + '0'

        tmp_id = '.'.join(tmp_id_splits)
        corrected_ids[tmp_id] = documents[doc_id]
    return corrected_ids

def create_link(ids, min_thresh):
    '''Returns a list of links in the order of most similar to less similar if it pass threshold'''

    sorted_list = sorted(ids, key=lambda x: ids[x][0], reverse=True)
    pdf_links = []
    titles = []

    for id in sorted_list:
        score, title = ids[id]
        if score > min_thresh:
            pdf_links.append(LINK+str(id))
            titles.append(title)
    return pdf_links, titles
    
def get_links(query, method='Context Search'): 
    'Given a query sentence, search for similar docs and returns the download links of paper'
    match method:
        case 'Context Search':
            similar_docs = load_and_get_file(query)
            corrected_ids = get_correct_ids(similar_docs)
            pdf_ordered_links, ordered_titles = create_link(corrected_ids, SIMILARITY_THRESHOLD)
        case 'Hybrid Search':
            pdf_ordered_links, ordered_titles = get_hybrid_res(query)
        case 'Tf-Idf Search':
            pdf_ordered_links, ordered_titles = get_tf_idf_res(query)

    return pdf_ordered_links, ordered_titles

def get_hybrid_res(query):
    print('hybrid, \n\n\n\n\n')
    index_name = 'recommendation-hybrid-test'
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    bm25_path = 'data/bm25_data.json'

    bm25encoder = BM25Encoder().load(bm25_path)
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    pc_t = PineconeHybridSearchRetriever(index=index, embeddings=embeddings, sparse_encoder=bm25encoder, top_k=4)
    recommendations = pc_t.invoke(query)
    pdf_links = []
    titles = []
    for results in recommendations:
        id = results.metadata['id']
        title = results.metadata['title']
        pdf_links.append(LINK+str(id))
        titles.append(title)
    return pdf_links, titles


def get_tf_idf_res(query):
    print('tfidf, \n\n\n\n\n')

    index_name = 'recommendation-tfidf-bigram'
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    corpus = torch.load('data/preprocessed_documents.pt')
    vectorize = TfidfVectorizer(max_df=0.97, min_df=0.005, ngram_range = (1, 2))
    vectorize.fit(corpus)

    query_vec = vectorize.transform([query]).toarray()
    query_vec = query_vec.squeeze()
    val = [float(i) for i in query_vec]

    recommendations = index.query(vector=val, top_k=4, include_metadata=True)

    pdf_links = []
    titles = []
    for results in recommendations['matches']:
        id = results['metadata']['id']
        title = results['metadata']['title']
        pdf_links.append(LINK+str(id))
        titles.append(title)
    return pdf_links, titles


def get_text_chunks_langchain(text):
    'Convert String to doc loader'
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs

def create_pdf_viewer(link):
    """Download the PDF from the provided link and show it in Streamlit."""
    # Define the path for the PDF file
    path = 'static_files/pdf.pdf'
    # Download the PDF file
    try:
        response = requests.get(link)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.HTTPError as e:
        st.write(f'HTTP Error: {e}')
        return
    except requests.exceptions.RequestException as e:
        st.write(f'Error downloading PDF: {e}')
        return

    # Load the content into a BytesIO object
    pdf_bytes = io.BytesIO(response.content)

    # Write the PDF to a file
    with open(path, 'wb') as file:
        file.write(pdf_bytes.read())

    pdf_viewer(path)
    return path

def get_pdf_doc_as_chunks(path, chunk_size=600, chunk_overlap=40):
    pdf_loader = PyPDFLoader(file_path=path)
    pdf_documents = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(pdf_documents)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    vectorstore.save_local('tmp_pdf_vectorstore')
    return vectorstore


def handle_userinput(user_question, db, llm):
    try:
        ai_message = manual_prompt(user_question, st.session_state.chat_history, db, llm)
        st.session_state.chat_history.append(HumanMessage(user_question))
        st.session_state.chat_history.append(AIMessage(ai_message))
        # st.write(st.session_state)

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                with st.chat_message('user'):
                    st.write(message.content)
            else:
                with st.chat_message('assistant'):
                    st.write(message.content)
    except Exception as e:
        st.error(f"Error processing question: {e}")

def qna_pdf_rag(path):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.chat_input('Enter')
    if user_question:
        if 'prev_link' not in st.session_state or st.session_state['prev_link'] != st.session_state['curr_link']:
            st.session_state['prev_link'] = st.session_state['curr_link']
            st.session_state['chat_history'] = []
            chunks = get_pdf_doc_as_chunks(path)
            db = get_vectorstore(chunks)
        else:
            embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
            db = FAISS.load_local('tmp_pdf_vectorstore', embeddings, allow_dangerous_deserialization=True)

        


        llm = ChatGoogleGenerativeAI(model='gemini-pro', convert_system_message_to_human=True)
        handle_userinput(user_question, db, llm)


if __name__ == '__main__':
    main()


