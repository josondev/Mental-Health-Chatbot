import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec, PineconeApiException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

# --- 1. Load Environment Variables and API Keys ---
load_dotenv()

# --- 2. Initialize RAG Components (Models, Vector Store, Chains) ---
# This section contains potentially expensive operations.
# For production, consider Streamlit's caching for some of these.

@st.cache_resource # Cache the Pinecone connection and related resources
def get_rag_components():
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    spec = ServerlessSpec(
        cloud=os.getenv("PINECONE_CLOUD"), # Default to 'aws' if not set
        region=os.getenv("PINECONE_REGION") # Default to 'us-east-1' if not set
    )

    # Initialize Embeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Pinecone Index
    index_name = "medical-chatbot"
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        try:
            # Determine embedding dimension correctly
            # For all-MiniLM-L6-v2, the dimension is 384
            dimension = embeddings_model.client.get_sentence_embedding_dimension()
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=spec
            )
            st.success(f"Pinecone index '{index_name}' created successfully.")
        except PineconeApiException as e:
            st.error(f"Failed to create Pinecone index: {e}")
            st.stop() # Stop execution if index creation fails
    
    try:
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings_model
        )
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5} # Fetch top 5 relevant documents
        )
    except Exception as e:
        st.error(f"Failed to connect to Pinecone index '{index_name}': {e}")
        st.stop()

    # Initialize LLM
    # Use a known valid model name, e.g., "gemini-1.5-flash-latest" or "gemini-pro"
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=0.1,
        max_tokens=1024,
        google_api_key=os.getenv("GOOGLE_API_KEY"), 
        streaming=True
    )

    # Prompt for rephrasing query based on chat history
    contextualize_question_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_question_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=base_retriever,
        prompt=contextualize_q_prompt
    )

    # Prompt for answering the question based on retrieved context
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. And DON'T answer questions outside mental health domain. "
        "If you don't know the answer, say that you don't know. "
        "Keep the answer concise."
        "\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Final RAG chain
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=question_answer_chain
    )
    return rag_chain

# --- 3. Streamlit App UI and Interaction Logic ---
st.set_page_config(page_title="Mental Health RAG Chatbot", layout="wide")
st.title("💬 Mental Health RAG Chatbot")
st.caption("Powered by LangChain, Google Gemini, Pinecone, and Streamlit")

# Load RAG components (cached)
try:
    rag_chain = get_rag_components()
except Exception as e:
    st.error(f"Error initializing RAG components: {e}")
    st.stop()


# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [] # This will store HumanMessage and AIMessage objects

# Display existing chat messages from history
for message_obj in st.session_state.messages:
    role = "user" if isinstance(message_obj, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message_obj.content)

# Accept user input
if user_query := st.chat_input("Ask any question related to Mental Health:"):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Prepare chat history for the RAG chain
    # This should be the history *before* the current user_query is added
    history_for_chain = list(st.session_state.messages) # Make a copy to avoid modifying during iteration if issues arise

    # Add current user message to the main message list for display and next turn's history
    st.session_state.messages.append(HumanMessage(content=user_query))

    # Display AI response (streaming)
    with st.chat_message("assistant"):
        response_placeholder = st.empty() # Create a placeholder for the streamed response
        full_ai_response_content = ""
        
        try:
            # Stream the response from the RAG chain
            for chunk in rag_chain.stream({
                "input": user_query,
                "chat_history": history_for_chain 
            }):
                if "answer" in chunk and chunk["answer"] is not None:
                    full_ai_response_content += chunk["answer"]
                    response_placeholder.markdown(full_ai_response_content + "▌") # Display with a blinking cursor
            
            response_placeholder.markdown(full_ai_response_content) # Display final response
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            response_placeholder.error(error_message)
            full_ai_response_content = error_message # Store error as AI response for history

    # Add AI's full response (or error) to session state messages
    st.session_state.messages.append(AIMessage(content=full_ai_response_content))

    # Optional: Force a rerun if needed, though Streamlit usually handles it
    # st.rerun()
