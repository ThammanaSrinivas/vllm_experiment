import logging
import requests
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import VLLMOpenAI
from utils import timeout, get_or_create_embeddings

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, config):
        self.config = config
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        
    def setup(self):
        """Initialize RAG system components."""
        logger.info("Setting up RAG System")
        self._validate_directory()
        self._setup_embeddings()
        self._setup_vectorstore()
        self._setup_llm()

    def _validate_directory(self):
        if not self.config.pdf_dir.exists() or not self.config.pdf_dir.is_dir():
            raise FileNotFoundError(f"Invalid PDF directory: {self.config.pdf_dir}")

    def _setup_embeddings(self):
        self.embeddings = get_or_create_embeddings(
            self.config.persist_directory,
            self.config.embeddings_model_name
        )

    def _setup_vectorstore(self):
        loader = PyPDFDirectoryLoader(str(self.config.pdf_dir))
        
        # More aggressive splitting for better semantic chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            keep_separator=True,
            add_start_index=True
        )
        
        chunks = splitter.split_documents(loader.load())
        
        # Add metadata for better filtering
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'content_preview': chunk.page_content[:100]
            })
        
        self.vectorstore = Chroma.from_documents(
            chunks, 
            self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def _setup_llm(self):
        """Initialize connection to vLLM API server."""
        try:
            self.llm = VLLMOpenAI(
                openai_api_base=self.config.vllm_api_url,
                model_name=self.config.llm_model_path,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            logger.info("vLLM client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM client: {e}")
            raise

    def query(self, query_text: str) -> str:
        """Process a query using the RAG system."""
        with timeout(self.config.response_timeout):
            # Two-stage retrieval
            # 1. Get more candidates first
            initial_docs = self.vectorstore.similarity_search_with_score(
                query_text,
                k=self.config.fetch_k * 2  # Get more initial candidates
            )
            
            # 2. Filter and re-rank
            filtered_docs = []
            query_terms = set(query_text.lower().split())
            
            for doc, score in initial_docs:
                doc_terms = set(doc.page_content.lower().split())
                term_overlap = len(query_terms & doc_terms)
                if term_overlap > 0 or score > 0.8:  # Keep if good term match or high similarity
                    filtered_docs.append((doc, term_overlap, score))
            
            # Rank by combination of term overlap and similarity
            retrieved_docs = [
                doc for doc, _, _ in sorted(
                    filtered_docs,
                    key=lambda x: (x[1] * 2 + x[2]),  # Weight term matches more heavily
                    reverse=True
                )[:self.config.return_k]
            ]

            context = "\n".join(doc.page_content for doc in retrieved_docs)
            if len(context) > self.config.max_input_tokens * 4:
                context = context[:self.config.max_input_tokens * 4] + "..."

            print("="*10 + "debugging" + "="*10)
            print(context)
            print("="*10 + "debugging" + "="*10)

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Be concise and accurate."),
                ("user", "Use this context to answer the question:\n\nContext:\n{context}\n\nQuestion: {query}")
            ])

            return (
                {"context": lambda _: context, "query": RunnablePassthrough()}
                | prompt 
                | self.llm 
                | StrOutputParser()
            ).invoke(query_text)