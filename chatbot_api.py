# Mental Health Chatbot - FastAPI Backend

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec, PineconeApiException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Mental Health Chatbot API",
    description="AI-powered mental health support chatbot",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    chat_history: List[ChatMessage]

# Global variables
rag_chain = None

@app.on_event("startup")
async def startup_event():
    global rag_chain

    try:
        # Initialize components (same as your code)
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        spec = ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1")
        )

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        index_name = "medical-chatbot"
        try:
            pc.Index(index_name).describe_index_stats()
        except PineconeApiException:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=spec
            )

        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1,
            max_tokens=1024,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

        contextualize_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualized_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=base_retriever,
            prompt=contextualized_q_prompt,
        )

        system_prompt = (
            "You are a compassionate mental health assistant. "
            "Use the following context to answer the question. "
            "Focus ONLY on mental health topics. "
            "If you don't know, say so. Be empathetic."
            "\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=question_answer_chain,
        )

        print("Chatbot initialized")

    except Exception as e:
        print(f"Error: {e}")
        raise

@app.get("/")
def root():
    return {
        "name": "Mental Health Chatbot API",
        "status": "online"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "chatbot_ready": rag_chain is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not rag_chain:
            raise HTTPException(status_code=500, detail="Chatbot not initialized")

        chat_history = []
        for msg in request.chat_history:
            if msg.role == "user":
                chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                chat_history.append(AIMessage(content=msg.content))

        result = rag_chain.invoke({
            "input": request.message,
            "chat_history": chat_history
        })

        response_text = result.get("answer", "Sorry, I couldn't respond.")

        updated_history = request.chat_history + [
            ChatMessage(role="user", content=request.message),
            ChatMessage(role="assistant", content=response_text)
        ]

        return ChatResponse(
            response=response_text,
            chat_history=updated_history
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
