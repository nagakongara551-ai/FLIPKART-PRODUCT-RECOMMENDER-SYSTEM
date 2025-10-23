from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flipkart.config import Config

class RAGChainBuilder:
    """
    Constructs a complete, conversational Retrieval-Augmented Generation (RAG) pipeline.

    This class sets up a chain that uses a vector store for document retrieval and
    the Groq LLM for answering questions, while maintaining chat history for context-aware
    follow-up questions.

    Attributes:
        vector_store (VectorStore): The document source (e.g., AstraDBVectorStore).
        model (ChatGroq): The LLM used for question rephrasing and answer generation.
        history_store (dict): An in-memory dictionary to store chat histories,
                              keyed by session ID.
    """
    def __init__(self,vector_store):
        """
        Initializes the RAGChainBuilder with a vector store and LLM.

        Args:
            vector_store (VectorStore): The vector store instance containing the documents.
        """
        self.vector_store=vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL , temperature=0.5)
        self.history_store={}

    def _get_history(self,session_id:str) -> BaseChatMessageHistory:
        """
        Retrieves or creates a new chat history object for a given session ID.

        Args:
            session_id (str): A unique identifier for the user's conversation session.

        Returns:
            BaseChatMessageHistory: The chat history object for the session.
        """
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
    
    def build_chain(self):
        """
        Constructs and returns the final conversational RAG chain.

        The chain consists of three main parts:
        1. Contextualization: Rewriting the user's question using history.
        2. Retrieval: Fetching the top 3 relevant documents (k=3).
        3. Generation: Stuffing context and question into a prompt for the LLM to answer.

        Returns:
            RunnableWithMessageHistory: The full, stateful RAG chain, ready for invocation.
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k":3})

        # Prompt for the first stage: Rephrase the question
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        # Prompt for the second stage: Generate the final answer
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're an e-commerce bot answering product-related queries using reviews and titles.
                             Stick to context. Be concise and helpful.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}"""),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        # 1. Chain for making the retriever history-aware
        history_aware_retriever = create_history_aware_retriever(
            self.model , retriever , context_prompt
        )

        # 2. Chain for generating the final answer from documents
        question_answer_chain = create_stuff_documents_chain(
            self.model , qa_prompt
        )

        # 3. Final RAG chain combining retrieval and answering
        rag_chain = create_retrieval_chain(
            history_aware_retriever,question_answer_chain
        )

        # 4. Wrap the RAG chain with session history management
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )