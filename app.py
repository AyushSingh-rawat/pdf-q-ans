import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate



# creating custom template to guide llm model
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
#    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

   os.environ['HF_TOKEN']=os.getenv("HUGGINGFACEHUB_API_TOKEN")
   embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   vectorstore = FAISS.from_texts(texts = text_chunks, embedding= embeddings)
   return vectorstore

# def get_conversation_chain(vectorstore):
#    os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
#    groq_api_key=os.getenv("GROQ_API_KEY")
#    llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

#    memory = ConversationBufferMemory(
#       memory_key= 'chat_history', return_messages= True)
#    conversation_chain = ConversationalRetrievalChain.from_llm(
#       llm = llm,
#       retriever = vectorstore.as_retriever(),
#       memory = memory
#    )

#    return conversation_chain

# generating conversation chain  
def get_conversationchain(vectorstore):
    os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
    groq_api_key=os.getenv("GROQ_API_KEY")
    llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True,
                                      output_key='answer') # using conversation buffer memory to hold past information
    conversation_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vectorstore.as_retriever(),
                                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                                memory=memory)
    return conversation_chain



def handle_userinput(user_question):
    
    response = st.session_state.conversation({'question': user_question})
    
    # Update the chat history in session state
    st.session_state.chat_history = response['chat_history']

    # Display each message in the chat history
    for i, message in enumerate(st.session_state.chat_history):
        # Clean the response to remove "According to the provided context,"
        cleaned_message = message.content.replace("According to the provided context, ", "")
        
        # Use the cleaned message in the display
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", cleaned_message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", cleaned_message), unsafe_allow_html=True)

def main():
   load_dotenv()
   st.set_page_config(page_title = "Chat with multiple PDFs", page_icon = ":books:")
   st.write(css, unsafe_allow_html= True)

   if "conversation" not in st.session_state:
      st.session_state.conversation = None
   
   if "chat_history" not in st.session_state:
      st.session_state.chat_history = None

   st.header("Chat with multiple PDFs :books:")
   user_question =  st.text_input("Ask a question about your documents:")

   if user_question:
      handle_userinput(user_question)

   with st.sidebar:
      st.subheader("Your Documents")
      pdf_docs =st.file_uploader("upload your Pdfs here and click on 'PROCESS'", accept_multiple_files= True)
      if st.button("PROCESS"):
        with st.spinner("Processing"):
           #get pdf text
           raw_text = get_pdf_text(pdf_docs)


           #get the text chunks
           text_chunks = get_text_chunks(raw_text)

           #create vector store
           vectorstore = get_vectorstore(text_chunks)

           #create conversation chain
           st.session_state.conversation = get_conversationchain(vectorstore)




if __name__=='__main__':
   main()



