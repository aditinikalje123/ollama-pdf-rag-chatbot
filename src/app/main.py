"""
Secure PDF RAG Assistant - Lightweight Render Version

Features:
- Multiple PDF upload
- PDF text extraction using PyMuPDF
- Lightweight TF-IDF retrieval
- Keyword fallback retrieval
- Groq cloud LLM
- Source filename and page numbers
- PDF viewer
- Chat input at bottom
- Render-friendly
"""

import hashlib
import os
import re

import pymupdf
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

MAX_CONTEXT_CHUNKS = 6
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Secure PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "pdfs" not in st.session_state:
    st.session_state.pdfs = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_pdf_id" not in st.session_state:
    st.session_state.active_pdf_id = None


# ============================================================
# PDF ID
# ============================================================

def generate_pdf_id(filename, pdf_bytes):
    """Generate a unique ID for a PDF."""
    content = filename.encode("utf-8") + pdf_bytes
    return hashlib.md5(content).hexdigest()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """Clean extracted PDF text."""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""

    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(pdf_bytes, filename):
    """
    Extract text from PDF page by page.

    Each chunk keeps its filename and page number.
    """

    documents = []

    try:

        pdf = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text")
            text = clean_text(text)

            if not text:
                continue

            chunks = split_text(text)

            for chunk in chunks:

                documents.append(
                    {
                        "text": chunk,
                        "page": page_number,
                        "filename": filename,
                    }
                )

        pdf.close()

    except Exception as e:

        st.error(
            f"Error reading {filename}: {e}"
        )

    return documents


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):
    """Process and store an uploaded PDF."""

    pdf_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    pdf_id = generate_pdf_id(
        filename,
        pdf_bytes
    )

    if pdf_id in st.session_state.pdfs:
        return pdf_id

    documents = extract_pdf(
        pdf_bytes,
        filename
    )

    if not documents:

        st.warning(
            f"No readable text was found in "
            f"'{filename}'. This PDF may be "
            f"image/scanned based."
        )

        return None

    st.session_state.pdfs[pdf_id] = {
        "filename": filename,
        "bytes": pdf_bytes,
        "documents": documents,
    }

    return pdf_id


# ============================================================
# DELETE PDF
# ============================================================

def delete_pdf(pdf_id):

    if pdf_id in st.session_state.pdfs:
        del st.session_state.pdfs[pdf_id]

    if st.session_state.active_pdf_id == pdf_id:
        st.session_state.active_pdf_id = None


# ============================================================
# DELETE ALL
# ============================================================

def delete_all_pdfs():

    st.session_state.pdfs = {}
    st.session_state.active_pdf_id = None
    st.session_state.messages = []


# ============================================================
# QUESTION KEYWORDS
# ============================================================

def expand_question(question):
    """
    Add related words for conceptual questions.

    This helps questions such as:
    - What are the learnings?
    - What is the moral?
    - What is the main idea?
    """

    q = question.lower()

    extra_words = []

    if any(
        word in q
        for word in [
            "learning",
            "learnings",
            "lesson",
            "lessons",
            "moral",
            "message",
            "takeaway",
            "takeaways",
        ]
    ):
        extra_words.extend(
            [
                "lesson",
                "learning",
                "moral",
                "message",
                "teaches",
                "teaching",
                "important",
                "learn",
                "understand",
                "values",
                "experience",
                "mistake",
                "wisdom",
            ]
        )

    if any(
        word in q
        for word in [
            "summary",
            "summarize",
            "summarise",
            "overview",
            "main idea",
        ]
    ):
        extra_words.extend(
            [
                "summary",
                "main",
                "idea",
                "story",
                "important",
                "events",
                "conclusion",
            ]
        )

    if any(
        word in q
        for word in [
            "character",
            "characters",
        ]
    ):
        extra_words.extend(
            [
                "character",
                "person",
                "people",
                "behavior",
                "actions",
            ]
        )

    if any(
        word in q
        for word in [
            "conclusion",
            "ending",
            "end",
        ]
    ):
        extra_words.extend(
            [
                "conclusion",
                "ending",
                "result",
                "finally",
                "outcome",
            ]
        )

    return question + " " + " ".join(extra_words)


# ============================================================
# KEYWORD FALLBACK
# ============================================================

def keyword_score(question, text):
    """
    Lightweight keyword matching fallback.

    This is useful when TF-IDF similarity is weak.
    """

    question_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            question.lower()
        )
    )

    text_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            text.lower()
        )
    )

    if not question_words:
        return 0.0

    common_words = (
        question_words & text_words
    )

    return len(common_words) / len(
        question_words
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(question, selected_pdf_ids):
    """
    Retrieve relevant PDF chunks using:
    1. TF-IDF similarity
    2. Keyword matching fallback
    """

    all_documents = []

    for pdf_id in selected_pdf_ids:

        pdf_data = st.session_state.pdfs.get(
            pdf_id
        )

        if pdf_data:
            all_documents.extend(
                pdf_data["documents"]
            )

    if not all_documents:
        return []

    texts = [
        doc["text"]
        for doc in all_documents
    ]

    search_question = expand_question(
        question
    )

    tfidf_scores = [0.0] * len(
        all_documents
    )

    # --------------------------------------------------------
    # TF-IDF retrieval
    # --------------------------------------------------------

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
        )

        document_vectors = (
            vectorizer.fit_transform(texts)
        )

        question_vector = (
            vectorizer.transform(
                [search_question]
            )
        )

        tfidf_scores = cosine_similarity(
            question_vector,
            document_vectors
        )[0].tolist()

    except Exception:
        pass

    # --------------------------------------------------------
    # Combined scoring
    # --------------------------------------------------------

    ranked_documents = []

    for index, doc in enumerate(
        all_documents
    ):

        keyword = keyword_score(
            search_question,
            doc["text"]
        )

        tfidf = tfidf_scores[index]

        combined_score = (
            (tfidf * 0.65)
            + (keyword * 0.35)
        )

        new_doc = doc.copy()

        new_doc["score"] = (
            combined_score
        )

        ranked_documents.append(
            new_doc
        )

    ranked_documents.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Take relevant documents
    # --------------------------------------------------------

    results = [
        doc
        for doc in ranked_documents
        if doc["score"] > 0
    ]

    # If retrieval scores are weak,
    # still provide top chunks to the LLM.
    if not results:
        results = ranked_documents[
            :MAX_CONTEXT_CHUNKS
        ]

    return results[
        :MAX_CONTEXT_CHUNKS
    ]


# ============================================================
# GROQ
# ============================================================

def ask_groq(question, context):
    """Send the retrieved PDF context to Groq."""

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        return (
            "GROQ_API_KEY is not configured. "
            "Please add it to Render Environment Variables."
        )

    prompt = f"""
You are a PDF question-answering assistant.

The user has uploaded one or more documents.

Answer the user's question using ONLY the
PDF context provided below.

IMPORTANT RULES:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the question asks for learnings, lessons,
   morals, messages, or takeaways, infer them
   ONLY from the events and information in the
   supplied stories.
4. If there are two stories, discuss the learning
   from each story separately.
5. Give a clear and natural answer.
6. If the context genuinely does not contain
   enough information, say that the information
   could not be found in the uploaded PDF.

PDF CONTEXT:
{context}

USER QUESTION:
{question}
"""

    headers = {
        "Authorization": (
            f"Bearer {api_key}"
        ),
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer questions strictly "
                    "from the provided PDF context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.1,
        "max_tokens": 800,
    }

    try:

        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:

            try:

                error_data = response.json()

                error_message = (
                    error_data
                    .get("error", {})
                    .get(
                        "message",
                        response.text
                    )
                )

            except Exception:

                error_message = response.text

            return (
                f"Groq API error: "
                f"{error_message}"
            )

        data = response.json()

        return data[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]

    except requests.exceptions.Timeout:

        return (
            "The AI request timed out. "
            "Please try again."
        )

    except Exception as e:

        return (
            f"Error connecting to Groq: {e}"
        )


# ============================================================
# QUESTION PROCESSING
# ============================================================

def process_question(
    question,
    selected_pdf_ids
):
    """Process a user question."""

    if not selected_pdf_ids:

        return (
            "Please upload and select "
            "a PDF first.",
            []
        )

    question_lower = (
        question.lower()
    )

    # --------------------------------------------------------
    # Filename questions
    # --------------------------------------------------------

    filename_keywords = [
        "file name",
        "filename",
        "document name",
        "pdf name",
        "which file",
        "which document",
    ]

    if any(
        keyword in question_lower
        for keyword in filename_keywords
    ):

        names = [
            st.session_state.pdfs[
                pdf_id
            ]["filename"]
            for pdf_id in selected_pdf_ids
            if pdf_id in st.session_state.pdfs
        ]

        if names:

            return (
                "The uploaded PDF file(s) are:\n\n"
                + "\n".join(
                    f"- {name}"
                    for name in names
                ),
                [],
            )

    # --------------------------------------------------------
    # Retrieve chunks
    # --------------------------------------------------------

    retrieved_docs = retrieve_documents(
        question,
        selected_pdf_ids
    )

    if not retrieved_docs:

        return (
            "I could not find relevant "
            "information in the uploaded PDF.",
            []
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for doc in retrieved_docs:

        context_parts.append(
            f"""
Source: {doc['filename']}
Page: {doc['page']}

{doc['text']}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Ask Groq
    # --------------------------------------------------------

    answer = ask_groq(
        question,
        context
    )

    return answer, retrieved_docs


# ============================================================
# SOURCE DISPLAY
# ============================================================

def display_sources(documents):

    if not documents:
        return

    with st.expander(
        "📚 Sources used"
    ):

        seen = set()

        for doc in documents:

            key = (
                doc["filename"],
                doc["page"],
            )

            if key in seen:
                continue

            seen.add(key)

            st.write(
                f"📄 **{doc['filename']}** — "
                f"Page {doc['page']}"
            )


# ============================================================
# PDF VIEWER
# ============================================================

def display_pdf_viewer(pdf_data):

    if not pdf_data:
        return

    st.subheader(
        "📖 PDF Viewer"
    )

    try:

        pdf = pymupdf.open(
            stream=pdf_data["bytes"],
            filetype="pdf",
        )

        total_pages = len(pdf)

        page_number = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
        )

        page = pdf[
            page_number - 1
        ]

        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(
                1.2,
                1.2
            ),
            alpha=False,
        )

        image_bytes = pix.tobytes(
            "png"
        )

        st.image(
            image_bytes,
            caption=(
                f"{pdf_data['filename']} - "
                f"Page {page_number}"
            ),
            use_container_width=True,
        )

        pdf.close()

    except Exception as e:

        st.error(
            f"Could not display PDF: {e}"
        )


# ============================================================
# SIDEBAR
# ============================================================

def sidebar():

    st.sidebar.title(
        "📄 PDF Manager"
    )

    uploaded_files = (
        st.sidebar.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:
            process_pdf(
                uploaded_file
            )

    st.sidebar.divider()

    if st.session_state.pdfs:

        st.sidebar.subheader(
            "Uploaded PDFs"
        )

        pdf_ids = list(
            st.session_state.pdfs.keys()
        )

        for pdf_id in pdf_ids:

            pdf_data = (
                st.session_state.pdfs[
                    pdf_id
                ]
            )

            col1, col2 = (
                st.sidebar.columns(
                    [4, 1]
                )
            )

            with col1:

                if st.button(
                    pdf_data["filename"],
                    key=f"select_{pdf_id}",
                    use_container_width=True,
                ):

                    st.session_state.active_pdf_id = (
                        pdf_id
                    )

            with col2:

                if st.button(
                    "🗑️",
                    key=f"delete_{pdf_id}",
                ):

                    delete_pdf(
                        pdf_id
                    )

                    st.rerun()

        st.sidebar.divider()

        if st.sidebar.button(
            "🗑️ Delete all PDFs",
            use_container_width=True,
        ):

            delete_all_pdfs()
            st.rerun()

    else:

        st.sidebar.info(
            "Upload one or more PDFs to start."
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    sidebar()

    st.title(
        "🔐 Secure PDF RAG Assistant"
    )

    st.caption(
        "Ask questions about your uploaded "
        "company documents."
    )

    # --------------------------------------------------------
    # No PDFs
    # --------------------------------------------------------

    if not st.session_state.pdfs:

        st.info(
            "Upload a PDF from the sidebar "
            "to start asking questions."
        )

        st.markdown(
            """
### How it works

1. Upload a company PDF.
2. The application extracts the PDF text.
3. Relevant sections are retrieved.
4. The retrieved context is sent to Groq.
5. The AI answers using the PDF context.
"""
        )

        return

    # --------------------------------------------------------
    # PDF selection
    # --------------------------------------------------------

    pdf_ids = list(
        st.session_state.pdfs.keys()
    )

    selected_pdf_ids = st.multiselect(
        "Select PDFs to search",
        options=pdf_ids,
        default=pdf_ids,
        format_func=lambda pdf_id:
            st.session_state.pdfs[
                pdf_id
            ]["filename"],
    )

    if not selected_pdf_ids:

        st.warning(
            "Please select at least one PDF."
        )

        return

    # ========================================================
    # CHAT HISTORY
    # ========================================================

    st.subheader(
        "💬 Ask your PDF"
    )

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"]
                == "assistant"
                and message.get(
                    "sources"
                )
            ):

                display_sources(
                    message["sources"]
                )

    # ========================================================
    # CHAT INPUT
    # IMPORTANT: OUTSIDE COLUMNS
    # ========================================================

    question = st.chat_input(
        "Ask a question about the PDF..."
    )

    if question:

        # User message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        # Assistant message
        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching the PDF..."
            ):

                answer, sources = (
                    process_question(
                        question,
                        selected_pdf_ids,
                    )
                )

            st.markdown(
                answer
            )

            if sources:
                display_sources(
                    sources
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

    # ========================================================
    # PDF VIEWER
    # ========================================================

    st.divider()

    active_id = (
        st.session_state.active_pdf_id
    )

    if (
        active_id
        and active_id in st.session_state.pdfs
    ):

        display_pdf_viewer(
            st.session_state.pdfs[
                active_id
            ]
        )

    else:

        first_pdf_id = (
            selected_pdf_ids[0]
        )

        display_pdf_viewer(
            st.session_state.pdfs[
                first_pdf_id
            ]
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()