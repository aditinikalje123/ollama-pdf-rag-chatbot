"""
Secure PDF RAG Assistant

Features:
- Multiple PDF upload
- Normal PDF text extraction using PyMuPDF
- OCR fallback for scanned PDFs
- Local Ollama embeddings when running locally
- HuggingFace embeddings when deployed with Groq
- ChromaDB vector database
- Local Ollama LLM when running locally
- Groq cloud LLM when GROQ_API_KEY is available
- Source filename and page numbers
- PDF viewer
- Render-friendly lazy loading and caching
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import io
import warnings

from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pymupdf

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq


# ============================================================
# SETTINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*torch.classes.*"
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

PERSIST_DIRECTORY = os.path.join("data", "vectors")

GROQ_MODEL = "openai/gpt-oss-20b"

OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"

OLLAMA_DEFAULT_MODEL = "llama3.2:3b"

HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Secure PDF RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ============================================================
# ENVIRONMENT
# ============================================================

def using_groq() -> bool:
    """
    Return True when GROQ_API_KEY is available.
    """

    return bool(os.getenv("GROQ_API_KEY"))


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():
    """
    Create the embedding model only when it is actually needed.

    Cloud / Render:
        HuggingFace embeddings

    Local:
        Ollama embeddings
    """

    if using_groq():

        logger.info(
            "Initializing HuggingFace embeddings for cloud deployment"
        )

        # Lazy import:
        # This prevents the HuggingFace stack from loading
        # just to display the Streamlit page.
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=HF_EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

    logger.info(
        "Initializing local Ollama embeddings"
    )

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL
    )


# ============================================================
# LLM
# ============================================================

@st.cache_resource
def get_llm(selected_model: str):
    """
    Return Groq LLM when GROQ_API_KEY exists.

    Otherwise return local Ollama LLM.
    """

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if groq_api_key:

        logger.info(
            f"Initializing Groq model: {GROQ_MODEL}"
        )

        return ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
            groq_api_key=groq_api_key,
        )

    logger.info(
        f"Initializing local Ollama model: {selected_model}"
    )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=selected_model,
        temperature=0,
    )


# ============================================================
# OLLAMA MODEL NAMES
# ============================================================

def extract_model_names(
    models_info: Any
) -> Tuple[str, ...]:
    """
    Extract available Ollama model names.
    """

    try:

        if hasattr(
            models_info,
            "models"
        ):

            model_names = tuple(
                model.model
                for model in models_info.models
            )

        else:

            model_names = tuple()

        logger.info(
            f"Available Ollama models: {model_names}"
        )

        return model_names

    except Exception as e:

        logger.error(
            f"Error extracting Ollama model names: {e}"
        )

        return tuple()


# ============================================================
# PDF TEXT + OCR EXTRACTION
# ============================================================

def extract_pdf_documents(
    path: str,
    file_name: str
) -> Tuple[List[Document], int]:
    """
    Extract text from PDF.

    First uses PyMuPDF text extraction.

    If a page contains no readable text,
    OCR using Tesseract is attempted.
    """

    logger.info(
        f"Starting PDF extraction: {file_name}"
    )

    pdf = pymupdf.open(path)

    page_count = len(pdf)

    documents = []

    for page_number, page in enumerate(pdf):

        page_text = page.get_text(
            "text"
        ).strip()

        # ----------------------------------------------------
        # NORMAL TEXT PDF
        # ----------------------------------------------------

        if page_text:

            logger.info(
                f"Page {page_number + 1}: normal text detected"
            )

            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "page": page_number + 1,
                        "pdf_name": file_name,
                        "extraction_method": "text",
                    },
                )
            )

            continue

        # ----------------------------------------------------
        # OCR FALLBACK
        # ----------------------------------------------------

        logger.info(
            f"Page {page_number + 1}: "
            f"no text found. Trying OCR..."
        )

        try:

            import pytesseract
            from PIL import Image

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(
                    2,
                    2
                ),
                alpha=False,
            )

            image_bytes = pix.tobytes(
                "png"
            )

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            ocr_text = (
                pytesseract
                .image_to_string(image)
                .strip()
            )

            if ocr_text:

                logger.info(
                    f"OCR succeeded on page "
                    f"{page_number + 1}"
                )

                documents.append(
                    Document(
                        page_content=ocr_text,
                        metadata={
                            "page": page_number + 1,
                            "pdf_name": file_name,
                            "extraction_method": "OCR",
                        },
                    )
                )

            else:

                logger.warning(
                    f"OCR found no text on page "
                    f"{page_number + 1}"
                )

        except ImportError:

            pdf.close()

            raise RuntimeError(
                "pytesseract is not installed.\n\n"
                "Run:\n"
                "pip install pytesseract"
            )

        except Exception as e:

            logger.error(
                f"OCR failed on page "
                f"{page_number + 1}: {e}"
            )

    pdf.close()

    logger.info(
        f"PDF extraction completed. "
        f"Readable pages: {len(documents)}"
    )

    return documents, page_count


# ============================================================
# CHUNK DOCUMENTS
# ============================================================

def split_documents(
    documents: List[Document]
) -> List[Document]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=7500,
        chunk_overlap=100,
    )

    chunks = (
        text_splitter
        .split_documents(documents)
    )

    logger.info(
        f"Created {len(chunks)} chunks"
    )

    return chunks


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vector_db(
    file_upload
) -> Chroma:

    logger.info(
        f"Creating vector database for "
        f"{file_upload.name}"
    )

    temp_dir = tempfile.mkdtemp()

    try:

        path = os.path.join(
            temp_dir,
            file_upload.name,
        )

        with open(
            path,
            "wb"
        ) as f:

            f.write(
                file_upload.getvalue()
            )

        data, _ = extract_pdf_documents(
            path,
            file_upload.name,
        )

        if not data:

            raise ValueError(
                f"No readable text could be extracted "
                f"from {file_upload.name}."
            )

        chunks = split_documents(
            data
        )

        if not chunks:

            raise ValueError(
                f"No text chunks were created "
                f"from {file_upload.name}."
            )

        embeddings = get_embeddings()

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        logger.info(
            "Generating embeddings..."
        )

        embedding_vectors = (
            embeddings
            .embed_documents(texts)
        )

        if (
            not embedding_vectors
            or len(embedding_vectors)
            != len(texts)
        ):

            raise ValueError(
                "Embedding generation failed."
            )

        collection_name = (
            f"pdf_{abs(hash(file_upload.name))}"
        )

        vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )

        vector_db.add_texts(
            texts=texts,
            metadatas=[
                chunk.metadata
                for chunk in chunks
            ],
            embeddings=embedding_vectors,
        )

        logger.info(
            "Vector database created successfully"
        )

        return vector_db

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


# ============================================================
# GENERATE PDF ID
# ============================================================

def generate_pdf_id(
    file_upload
) -> str:

    timestamp = datetime.now().isoformat()

    return (
        f"pdf_"
        f"{abs(hash(file_upload.name + timestamp))}"
    )


# ============================================================
# PROCESS AND STORE PDF
# ============================================================

def process_and_store_pdf(
    file_upload,
    pdf_id: str,
    is_sample: bool = False,
):

    logger.info(
        f"Processing PDF: {file_upload.name}"
    )

    temp_dir = tempfile.mkdtemp()

    try:

        path = os.path.join(
            temp_dir,
            file_upload.name,
        )

        file_bytes = (
            file_upload.getvalue()
        )

        with open(
            path,
            "wb"
        ) as f:

            f.write(file_bytes)

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        data, page_count = (
            extract_pdf_documents(
                path,
                file_upload.name,
            )
        )

        if not data:

            raise ValueError(
                f"No readable text could be extracted "
                f"from {file_upload.name}."
            )

        # ----------------------------------------------------
        # Split into chunks
        # ----------------------------------------------------

        chunks = split_documents(
            data
        )

        if not chunks:

            raise ValueError(
                "No text chunks were created."
            )

        # ----------------------------------------------------
        # Add metadata
        # ----------------------------------------------------

        for i, chunk in enumerate(chunks):

            chunk.metadata.update(
                {
                    "pdf_id": pdf_id,
                    "pdf_name": file_upload.name,
                    "chunk_index": i,
                    "source_file": file_upload.name,
                }
            )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        embeddings = get_embeddings()

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        logger.info(
            "Generating embeddings..."
        )

        embedding_vectors = (
            embeddings
            .embed_documents(texts)
        )

        if (
            not embedding_vectors
            or len(embedding_vectors)
            != len(texts)
        ):

            raise ValueError(
                f"Embedding generation failed "
                f"for {file_upload.name}."
            )

        # ----------------------------------------------------
        # ChromaDB
        # ----------------------------------------------------

        collection_name = (
            f"pdf_"
            f"{abs(hash(file_upload.name + pdf_id))}"
        )

        vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )

        vector_db.add_texts(
            texts=texts,
            metadatas=[
                chunk.metadata
                for chunk in chunks
            ],
            embeddings=embedding_vectors,
        )

        # ----------------------------------------------------
        # Store in Streamlit session
        # ----------------------------------------------------

        st.session_state["pdfs"][pdf_id] = {
            "name": file_upload.name,
            "vector_db": vector_db,
            "page_count": page_count,
            "file_bytes": file_bytes,
            "file_upload": file_upload,
            "collection_name": collection_name,
            "upload_timestamp": datetime.now(),
            "doc_count": len(chunks),
            "is_sample": is_sample,
        }

        st.session_state[
            "active_pdfs"
        ].append(pdf_id)

        logger.info(
            f"PDF stored successfully: "
            f"{file_upload.name}"
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


# ============================================================
# DELETE ONE PDF
# ============================================================

def delete_pdf(
    pdf_id: str
):

    if pdf_id not in (
        st.session_state["pdfs"]
    ):
        return

    pdf_data = (
        st.session_state["pdfs"][pdf_id]
    )

    vector_db = (
        pdf_data.get("vector_db")
    )

    if vector_db is not None:

        try:

            vector_db.delete_collection()

        except Exception as e:

            logger.error(
                f"Error deleting vector collection: {e}"
            )

    del st.session_state[
        "pdfs"
    ][pdf_id]

    if pdf_id in (
        st.session_state["active_pdfs"]
    ):

        st.session_state[
            "active_pdfs"
        ].remove(pdf_id)

    st.success(
        f"Deleted {pdf_data['name']}"
    )


# ============================================================
# DELETE ALL PDFs
# ============================================================

def delete_all_pdfs():

    for pdf_id in list(
        st.session_state["pdfs"].keys()
    ):

        delete_pdf(
            pdf_id
        )

    st.session_state[
        "pdfs"
    ] = {}

    st.session_state[
        "active_pdfs"
    ] = []


# ============================================================
# QUESTION PROCESSING
# ============================================================

def process_question_multi_pdf(
    question: str,
    pdfs_dict: Dict[str, Dict],
    selected_model: str,
) -> Tuple[str, List[Dict]]:

    logger.info(
        f"Processing question across "
        f"{len(pdfs_dict)} PDFs: {question}"
    )

    # --------------------------------------------------------
    # Direct filename questions
    # --------------------------------------------------------

    name_words = (
        "file name",
        "filename",
        "document name",
        "pdf name",
        "name of this document",
    )

    normalized_question = (
        question.lower().strip()
    )

    if any(
        word in normalized_question
        for word in name_words
    ):

        if len(pdfs_dict) == 1:

            pdf_id = next(
                iter(pdfs_dict)
            )

            pdf_data = (
                pdfs_dict[pdf_id]
            )

            name = pdf_data.get(
                "name",
                "Unknown file",
            )

            return (
                f"The uploaded document is "
                f"**{name}**.\n\n"
                f"Source: **{name}**",
                [
                    {
                        "pdf_name": name,
                        "pdf_id": pdf_id,
                        "page": "N/A",
                        "chunk_index": 0,
                    }
                ],
            )

        names = [
            pdf_data.get(
                "name",
                "Unknown file",
            )
            for pdf_data
            in pdfs_dict.values()
        ]

        answer = (
            "The uploaded documents are:\n\n"
            +
            "\n".join(
                f"- **{name}**"
                for name in names
            )
        )

        sources = [
            {
                "pdf_name":
                    pdf_data.get(
                        "name",
                        "Unknown file",
                    ),
                "pdf_id": pdf_id,
                "page": "N/A",
                "chunk_index": 0,
            }
            for pdf_id, pdf_data
            in pdfs_dict.items()
        ]

        return answer, sources

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = get_llm(
        selected_model
    )

    all_retrieved_docs = []

    # --------------------------------------------------------
    # Retrieve from every PDF
    # --------------------------------------------------------

    for pdf_id, pdf_data in (
        pdfs_dict.items()
    ):

        vector_db = (
            pdf_data["vector_db"]
        )

        try:

            docs = (
                vector_db
                .similarity_search(
                    question,
                    k=2,
                )
            )

            logger.info(
                f"Retrieved {len(docs)} chunks "
                f"from {pdf_data['name']}"
            )

            for doc in docs:

                doc.metadata[
                    "pdf_name"
                ] = pdf_data["name"]

                doc.metadata[
                    "pdf_id"
                ] = pdf_id

            all_retrieved_docs.extend(
                docs
            )

        except Exception as e:

            logger.error(
                f"Error retrieving from "
                f"{pdf_data['name']}: {e}"
            )

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not all_retrieved_docs:

        return (
            "I could not find relevant "
            "information in the uploaded PDF.",
            [],
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for doc in all_retrieved_docs[:4]:

        pdf_name = doc.metadata.get(
            "pdf_name",
            "Unknown file",
        )

        page_number = doc.metadata.get(
            "page",
            "Unknown page",
        )

        context_parts.append(
            f"[Source: {pdf_name}, "
            f"Page: {page_number}]\n"
            f"{doc.page_content}"
        )

    formatted_context = (
        "\n\n---\n\n".join(
            context_parts
        )
    )

    # --------------------------------------------------------
    # RAG Prompt
    # --------------------------------------------------------

    template = """
You are a secure document question-answering assistant.

Answer the user's question using ONLY
the information provided in the PDF context.

Rules:

1. Do not use outside knowledge.

2. Do not invent information.

3. If the answer is not present,
say:

"I could not find this information
in the uploaded document."

4. If multiple PDFs are provided,
use the correct PDF information.

5. Give a short and direct answer.

6. Mention the relevant source
filename and page number.

7. If the user asks for a comparison,
compare only information available
in the provided PDFs.

PDF CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    prompt = (
        ChatPromptTemplate
        .from_template(template)
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    try:

        response = chain.invoke(
            {
                "context":
                    formatted_context,

                "question":
                    question,
            }
        )

    except Exception as e:

        logger.error(
            f"Error generating response: {e}"
        )

        return (
            f"Sorry, I could not generate "
            f"an answer.\n\nError: {e}",
            [],
        )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    source_details = [
        {
            "pdf_name":
                doc.metadata.get(
                    "pdf_name",
                    "Unknown file",
                ),

            "pdf_id":
                doc.metadata.get(
                    "pdf_id"
                ),

            "page":
                doc.metadata.get(
                    "page",
                    "Unknown",
                ),

            "chunk_index":
                doc.metadata.get(
                    "chunk_index",
                    0,
                ),
        }

        for doc
        in all_retrieved_docs[:4]
    ]

    return (
        response,
        source_details,
    )


# ============================================================
# DELETE VECTOR DATABASE
# ============================================================

def delete_vector_db(
    vector_db: Optional[Chroma]
) -> None:

    if vector_db is not None:

        try:

            vector_db.delete_collection()

            st.session_state.pop(
                "pdf_pages",
                None
            )

            st.session_state.pop(
                "file_upload",
                None
            )

            st.session_state.pop(
                "vector_db",
                None
            )

            st.success(
                "Collection deleted successfully."
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Error deleting collection: {e}"
            )

    else:

        st.error(
            "No vector database found."
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.subheader(
        "🧠 Secure PDF RAG Assistant",
        divider="gray",
        anchor=False,
    )

    # --------------------------------------------------------
    # Session state
    # --------------------------------------------------------

    if "messages" not in st.session_state:

        st.session_state[
            "messages"
        ] = []

    if "pdfs" not in st.session_state:

        st.session_state[
            "pdfs"
        ] = {}

    if "active_pdfs" not in st.session_state:

        st.session_state[
            "active_pdfs"
        ] = []

    if "vector_db" not in st.session_state:

        st.session_state[
            "vector_db"
        ] = None

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if groq_api_key:

        available_models = (
            GROQ_MODEL,
        )

    else:

        try:

            # Lazy import:
            # Ollama is only needed locally.
            import ollama

            models_info = (
                ollama.list()
            )

            available_models = (
                extract_model_names(
                    models_info
                )
            )

        except Exception as e:

            logger.error(
                f"Could not connect to Ollama: {e}"
            )

            available_models = ()

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    col1, col2 = st.columns(
        [1.5, 2]
    )

    # --------------------------------------------------------
    # Model selector
    # --------------------------------------------------------

    if groq_api_key:

        selected_model = GROQ_MODEL

        col2.success(
            "☁️ Using Groq cloud model"
        )

    elif available_models:

        selected_model = (
            col2.selectbox(
                "Pick a local Ollama model ↓",
                available_models,
                key="model_select",
            )
        )

    else:

        selected_model = (
            OLLAMA_DEFAULT_MODEL
        )

        col2.warning(
            "Ollama is not available. "
            "Add GROQ_API_KEY for cloud deployment."
        )

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.divider()

        st.subheader(
            "📚 Loaded PDFs"
        )

        if st.session_state.get("pdfs"):

            total_pdfs = len(
                st.session_state["pdfs"]
            )

            total_chunks = sum(
                pdf["doc_count"]
                for pdf
                in st.session_state[
                    "pdfs"
                ].values()
            )

            st.metric(
                "Total PDFs",
                total_pdfs,
            )

            st.metric(
                "Total Chunks",
                total_chunks,
            )

            st.divider()

            for pdf_id in (
                st.session_state[
                    "active_pdfs"
                ]
            ):

                if pdf_id not in (
                    st.session_state[
                        "pdfs"
                    ]
                ):

                    continue

                pdf_data = (
                    st.session_state[
                        "pdfs"
                    ][pdf_id]
                )

                with st.expander(
                    f"📄 {pdf_data['name']}",
                    expanded=False,
                ):

                    st.caption(
                        f"Chunks: "
                        f"{pdf_data['doc_count']}"
                    )

                    st.caption(
                        f"Pages: "
                        f"{pdf_data['page_count']}"
                    )

                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{pdf_id}",
                    ):

                        delete_pdf(
                            pdf_id
                        )

                        st.rerun()

            st.divider()

            if st.button(
                "🗑️ Delete All PDFs"
            ):

                delete_all_pdfs()

                st.rerun()

        else:

            st.info(
                "No PDFs loaded yet."
            )

    # ========================================================
    # SAMPLE PDF
    # ========================================================

    use_sample = col1.toggle(
        "Use sample PDF (Scammer Agent Paper)",
        key="sample_checkbox",
    )

    # ========================================================
    # UPLOAD PDF
    # ========================================================

    if use_sample:

        sample_pdf_path = Path(
            "data/pdfs/sample/scammer-agent.pdf"
        )

        if sample_pdf_path.exists():

            sample_id = "sample_pdf"

            if sample_id not in (
                st.session_state[
                    "pdfs"
                ]
            ):

                with st.spinner(
                    "Processing sample PDF..."
                ):

                    with open(
                        sample_pdf_path,
                        "rb",
                    ) as f:

                        file_bytes = f.read()

                    class SampleFile:

                        def __init__(
                            self,
                            path,
                            content,
                        ):

                            self.name = (
                                path.name
                            )

                            self._content = (
                                content
                            )

                        def getvalue(self):

                            return self._content

                    sample_file = (
                        SampleFile(
                            sample_pdf_path,
                            file_bytes,
                        )
                    )

                    process_and_store_pdf(
                        sample_file,
                        sample_id,
                        is_sample=True,
                    )

        else:

            st.error(
                "Sample PDF not found."
            )

    else:

        file_uploads = (
            col1.file_uploader(
                "Upload PDF files ↓",
                type="pdf",
                accept_multiple_files=True,
                key="pdf_uploader",
            )
        )

        if file_uploads:

            for file_upload in file_uploads:

                pdf_id = (
                    generate_pdf_id(
                        file_upload
                    )
                )

                if pdf_id not in (
                    st.session_state[
                        "pdfs"
                    ]
                ):

                    with st.spinner(
                        f"Processing "
                        f"{file_upload.name}..."
                    ):

                        try:

                            process_and_store_pdf(
                                file_upload,
                                pdf_id,
                            )

                        except Exception as e:

                            st.error(
                                f"Error processing "
                                f"{file_upload.name}: {e}"
                            )

                            logger.error(
                                f"PDF processing error: {e}"
                            )

    # ========================================================
    # PDF VIEWER
    # ========================================================

    if (
        st.session_state.get("pdfs")
        and
        st.session_state.get("active_pdfs")
    ):

        zoom_level = col1.slider(
            "Zoom Level",
            min_value=100,
            max_value=1000,
            value=700,
            step=50,
            key="zoom_slider",
        )

        with col1:

            with st.container(
                height=410,
                border=True,
            ):

                for pdf_id in (
                    st.session_state[
                        "active_pdfs"
                    ]
                ):

                    if pdf_id not in (
                        st.session_state[
                            "pdfs"
                        ]
                    ):

                        continue

                    pdf_data = (
                        st.session_state[
                            "pdfs"
                        ][pdf_id]
                    )

                    st.markdown(
                        f"### 📄 "
                        f"{pdf_data['name']}"
                    )

                    st.caption(
                        f"Uploaded: "
                        f"{pdf_data['upload_timestamp'].strftime('%Y-%m-%d %H:%M')} "
                        f"| Chunks: "
                        f"{pdf_data['doc_count']} "
                        f"| Pages: "
                        f"{pdf_data['page_count']}"
                    )

                    if st.button(
                        "🗑️ Remove",
                        key=f"remove_{pdf_id}",
                    ):

                        delete_pdf(
                            pdf_id
                        )

                        st.rerun()

                    st.divider()

                    viewer_pdf = pymupdf.open(
                        stream=pdf_data[
                            "file_bytes"
                        ],
                        filetype="pdf",
                    )

                    for page_idx, page in enumerate(
                        viewer_pdf
                    ):

                        st.caption(
                            f"Page "
                            f"{page_idx + 1}"
                        )

                        pixmap = (
                            page.get_pixmap(
                                matrix=pymupdf.Matrix(
                                    1.2,
                                    1.2,
                                ),
                                alpha=False,
                            )
                        )

                        st.image(
                            pixmap.tobytes(
                                "png"
                            ),
                            width=zoom_level,
                        )

                    viewer_pdf.close()

                    st.markdown(
                        "---"
                    )

    else:

        col1.info(
            "Upload PDF files to view them here."
        )

    # ========================================================
    # DELETE COLLECTION
    # ========================================================

    delete_collection = col1.button(
        "⚠️ Delete collection",
        type="secondary",
        key="delete_button",
    )

    if delete_collection:

        delete_vector_db(
            st.session_state[
                "vector_db"
            ]
        )

    # ========================================================
    # CHAT
    # ========================================================

    with col2:

        message_container = (
            st.container(
                height=500,
                border=True,
            )
        )

        # ----------------------------------------------------
        # Chat history
        # ----------------------------------------------------

        for message in (
            st.session_state[
                "messages"
            ]
        ):

            avatar = (
                "🤖"
                if message["role"]
                == "assistant"
                else "😎"
            )

            with message_container.chat_message(
                message["role"],
                avatar=avatar,
            ):

                st.markdown(
                    message["content"]
                )

                if (
                    message["role"]
                    == "assistant"
                    and
                    "sources" in message
                ):

                    st.divider()

                    st.caption(
                        "📚 Sources:"
                    )

                    sources_by_pdf = {}

                    for src in (
                        message["sources"]
                    ):

                        pdf_name = src.get(
                            "pdf_name",
                            "Unknown",
                        )

                        if pdf_name not in (
                            sources_by_pdf
                        ):

                            sources_by_pdf[
                                pdf_name
                            ] = 0

                        sources_by_pdf[
                            pdf_name
                        ] += 1

                    for (
                        pdf_name,
                        count
                    ) in (
                        sources_by_pdf.items()
                    ):

                        st.markdown(
                            f"- **{pdf_name}** "
                            f"({count} chunks)"
                        )

        # ----------------------------------------------------
        # User question
        # ----------------------------------------------------

        prompt = st.chat_input(
            "Ask a question about your PDFs..."
        )

        if prompt:

            try:

                st.session_state[
                    "messages"
                ].append(
                    {
                        "role": "user",
                        "content": prompt,
                    }
                )

                with message_container.chat_message(
                    "user",
                    avatar="😎",
                ):

                    st.markdown(
                        prompt
                    )

                # ------------------------------------------------
                # Generate answer
                # ------------------------------------------------

                with message_container.chat_message(
                    "assistant",
                    avatar="🤖",
                ):

                    with st.spinner(
                        "Searching documents..."
                    ):

                        if st.session_state.get(
                            "pdfs"
                        ):

                            (
                                response,
                                sources,
                            ) = (
                                process_question_multi_pdf(
                                    prompt,
                                    st.session_state[
                                        "pdfs"
                                    ],
                                    selected_model,
                                )
                            )

                            st.markdown(
                                response
                            )

                            if sources:

                                st.divider()

                                st.caption(
                                    "📚 Sources:"
                                )

                                sources_by_pdf = {}

                                for src in sources:

                                    pdf_name = (
                                        src.get(
                                            "pdf_name",
                                            "Unknown",
                                        )
                                    )

                                    if pdf_name not in (
                                        sources_by_pdf
                                    ):

                                        sources_by_pdf[
                                            pdf_name
                                        ] = 0

                                    sources_by_pdf[
                                        pdf_name
                                    ] += 1

                                for (
                                    pdf_name,
                                    count,
                                ) in (
                                    sources_by_pdf.items()
                                ):

                                    st.markdown(
                                        f"- **{pdf_name}** "
                                        f"({count} chunks)"
                                    )

                        else:

                            st.warning(
                                "Please upload "
                                "PDF files first."
                            )

                            response = None
                            sources = None

                # ------------------------------------------------
                # Save response
                # ------------------------------------------------

                if response:

                    st.session_state[
                        "messages"
                    ].append(
                        {
                            "role": "assistant",
                            "content": response,
                            "sources": sources,
                        }
                    )

            except Exception as e:

                st.error(
                    str(e),
                    icon="⛔",
                )

                logger.error(
                    f"Error processing prompt: {e}"
                )

        else:

            if not st.session_state.get(
                "pdfs"
            ):

                st.warning(
                    "Upload PDF files or use "
                    "the sample PDF to begin."
                )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()