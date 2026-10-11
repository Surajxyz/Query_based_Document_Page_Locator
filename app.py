import os
import re
import json
import shutil
from functools import lru_cache

import fitz
import faiss
import nltk
import numpy as np
from nltk.corpus import stopwords

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash
)

from markupsafe import escape
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = "pdf-rag-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
FAISS_FOLDER = os.path.join(BASE_DIR, "faiss_store")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

model = SentenceTransformer(MODEL_NAME)

EMBEDDING_DIMENSION = 768

print("Embedding model loaded.")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_pdf_name(filename):
    """
    Convert PDF filename into a safe folder name.
    """

    name = os.path.splitext(filename)[0]

    name = secure_filename(name)

    if not name:
        name = "pdf_document"

    return name


def get_pdf_folder(pdf_name):
    """
    Return persistent FAISS folder for a PDF.
    """

    safe_name = clean_pdf_name(pdf_name)

    return os.path.join(
        FAISS_FOLDER,
        safe_name
    )


def tokenize_text(text):
    """
    Simple whitespace tokenizer.

    This allows the application to create approximately
    400-token chunks without requiring tiktoken.
    """

    return re.findall(r"\S+", text)


def get_query_stop_words():
    """Return NLTK's English stop-word list, downloading it if needed."""

    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def extract_important_words(query):
    """Return meaningful query words used to validate retrieved pages."""

    words = re.findall(r"[\w]+", query.lower())
    query_stop_words = get_query_stop_words()

    return {
        word
        for word in words
        if len(word) > 2 and word not in query_stop_words
    }


def page_contains_query_word(page_text, important_words):
    """Check whether a page contains at least one important query word."""

    page_words = set(re.findall(r"[\w]+", page_text.lower()))

    return bool(page_words & important_words)


def highlight_important_words(text, important_words):
    """Escape page text and highlight retained query words in it."""

    if not important_words:
        return str(escape(text))

    words_pattern = "|".join(
        re.escape(word)
        for word in sorted(important_words, key=len, reverse=True)
    )

    pattern = re.compile(
        rf"(?<!\w)({words_pattern})(?!\w)",
        re.IGNORECASE
    )

    highlighted_text = []
    last_end = 0

    for match in pattern.finditer(text):
        highlighted_text.append(
            str(escape(text[last_end:match.start()]))
        )
        highlighted_text.append(
            "<mark>"
            + str(escape(match.group(0)))
            + "</mark>"
        )
        last_end = match.end()

    highlighted_text.append(
        str(escape(text[last_end:]))
    )

    return "".join(highlighted_text)


def create_chunks(text, page_number, chunk_size=300):
    """
    Split page text into chunks of approximately 300 tokens.

    Every generated chunk receives the page number from
    the original PDF page.
    """

    tokens = tokenize_text(text)

    chunks = []

    for start in range(0, len(tokens), chunk_size):

        chunk_tokens = tokens[start:start + chunk_size]

        if not chunk_tokens:
            continue

        chunk_text = " ".join(chunk_tokens)

        chunks.append({
            "text": chunk_text,
            "page": page_number,
            "page_text": text.strip()
        })

    return chunks


# ============================================================
# PDF EXTRACTION
# ============================================================

@lru_cache(maxsize=1)
def get_paddle_ocr():
    """Create the PaddleOCR engine once, only when OCR is needed."""

    from paddleocr import PaddleOCR

    return PaddleOCR(
        use_angle_cls=True,
        lang="en",
        show_log=False
    )


def extract_page_ocr_text(page):
    """Render a PDF page and return its recognized text."""

    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(2, 2),
        alpha=False
    )
    image = np.frombuffer(
        pixmap.samples,
        dtype=np.uint8
    ).reshape(pixmap.height, pixmap.width, pixmap.n)
    image = image[:, :, ::-1].copy()

    results = get_paddle_ocr().ocr(image, cls=True)
    recognized_lines = []

    for line in results[0] or []:
        recognized_lines.append(line[1][0])

    return "\n".join(recognized_lines).strip()


def extract_pdf_chunks(pdf_path):
    """
    Extract PDF text page by page, using OCR when native text is sparse.

    Each page is chunked independently so that the page number
    attached to every chunk is exact.
    """

    all_chunks = []

    with fitz.open(pdf_path) as document:
        for page_index in range(len(document)):

            page_number = page_index + 1

            page = document[page_index]

            text = page.get_text("text") or ""

            if len(text.strip()) <= 200:
                ocr_text = extract_page_ocr_text(page)
                text_parts = [
                    part
                    for part in (text.strip(), ocr_text)
                    if part
                ]
                text = "\n".join(text_parts)

            if not text.strip():
                continue

            page_chunks = create_chunks(
                text=text,
                page_number=page_number,
                chunk_size=300
            )

            for chunk in page_chunks:

                chunk["chunk_id"] = len(all_chunks)

                all_chunks.append(chunk)

    return all_chunks


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_store(pdf_path, original_filename):
    """
    Create persistent FAISS index for one PDF.
    """

    chunks = extract_pdf_chunks(pdf_path)

    if not chunks:
        raise ValueError(
            "No readable text was found in the PDF."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Generating embeddings for {len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype(
        np.float32
    )

    # --------------------------------------------------------
    # FAISS cosine similarity
    #
    # Since vectors are normalized, inner product is equivalent
    # to cosine similarity.
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        EMBEDDING_DIMENSION
    )

    index.add(embeddings)

    # --------------------------------------------------------
    # Persistent folder
    # --------------------------------------------------------

    pdf_folder = get_pdf_folder(
        original_filename
    )

    os.makedirs(
        pdf_folder,
        exist_ok=True
    )

    index_path = os.path.join(
        pdf_folder,
        "index.faiss"
    )

    metadata_path = os.path.join(
        pdf_folder,
        "metadata.json"
    )

    # Save FAISS index
    faiss.write_index(
        index,
        index_path
    )

    # Save metadata
    metadata = {
        "pdf_name": original_filename,
        "chunk_size": 400,
        "embedding_model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    return {
        "pdf_name": original_filename,
        "chunks": len(chunks),
        "folder": pdf_folder
    }


# ============================================================
# LOAD FAISS STORE
# ============================================================

def load_faiss_store(pdf_name):
    """
    Load persistent FAISS index and metadata.
    """

    pdf_folder = get_pdf_folder(pdf_name)

    index_path = os.path.join(
        pdf_folder,
        "index.faiss"
    )

    metadata_path = os.path.join(
        pdf_folder,
        "metadata.json"
    )

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "FAISS index not found."
        )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            "Metadata not found."
        )

    index = faiss.read_index(
        index_path
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    return index, metadata


# ============================================================
# GET ALL INDEXED PDFS
# ============================================================

def get_indexed_pdfs():
    """
    Return all PDFs for which a persistent FAISS index exists.
    """

    pdfs = []

    if not os.path.exists(FAISS_FOLDER):
        return pdfs

    for folder_name in os.listdir(FAISS_FOLDER):

        folder_path = os.path.join(
            FAISS_FOLDER,
            folder_name
        )

        if not os.path.isdir(folder_path):
            continue

        metadata_path = os.path.join(
            folder_path,
            "metadata.json"
        )

        index_path = os.path.join(
            folder_path,
            "index.faiss"
        )

        if not (
            os.path.exists(metadata_path)
            and os.path.exists(index_path)
        ):
            continue

        try:

            with open(
                metadata_path,
                "r",
                encoding="utf-8"
            ) as file:

                metadata = json.load(file)

            pdfs.append(
                metadata["pdf_name"]
            )

        except Exception:
            continue

    return sorted(pdfs)


# ============================================================
# SEARCH PDF
# ============================================================

def search_pdf(pdf_name, query, top_k=3):
    """
    Search a selected PDF using FAISS.

    Returns matching page content along with exact page numbers.
    """

    index, metadata = load_faiss_store(
        pdf_name
    )

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        np.float32
    )

    actual_k = min(
        top_k,
        index.ntotal
    )

    important_words = extract_important_words(query)

    scores, indices = index.search(
        query_embedding,
        actual_k
    )

    results = []
    seen_pages = set()

    chunks = metadata["chunks"]

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id < 0:
            continue

        chunk = chunks[index_id]

        page_text = chunk.get(
            "page_text",
            chunk["text"]
        )

        if not important_words or not page_contains_query_word(
            page_text,
            important_words
        ):
            continue

        if chunk["page"] in seen_pages:
            continue

        seen_pages.add(chunk["page"])

        results.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "text": page_text,
            "highlighted_text": highlight_important_words(
                page_text,
                important_words
            ),
            "score": float(score)
        })

    return results


# ============================================================
# SIMPLE EXTRACTIVE ANSWER
# ============================================================

def generate_answer(query, results):
    """
    Create a simple answer from retrieved chunks.

    This version does not require an external LLM/API key.

    The retrieved source text is displayed so that the user can
    verify the answer directly against the PDF page.
    """

    if not results:
        return "Answer not found in the PDF."

    # Select strongest result
    best_result = results[0]

    answer = best_result["text"]

    return answer


# ============================================================
# HOME / PDF INSERTION PAGE
# ============================================================

@app.route("/")
def index():

    pdfs = get_indexed_pdfs()

    return render_template(
        "index.html",
        pdfs=pdfs
    )


# ============================================================
# UPLOAD PDF
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdf():

    if "pdf" not in request.files:

        flash(
            "No PDF file was selected.",
            "error"
        )

        return redirect(
            url_for("index")
        )

    file = request.files["pdf"]

    if file.filename == "":

        flash(
            "Please select a PDF.",
            "error"
        )

        return redirect(
            url_for("index")
        )

    if not file.filename.lower().endswith(".pdf"):

        flash(
            "Only PDF files are allowed.",
            "error"
        )

        return redirect(
            url_for("index")
        )

    filename = secure_filename(
        file.filename
    )

    pdf_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(pdf_path)

    try:

        result = create_faiss_store(
            pdf_path,
            filename
        )

        flash(
            f"PDF indexed successfully. "
            f"{result['chunks']} chunks created.",
            "success"
        )

    except Exception as e:

        flash(
            f"Error while processing PDF: {str(e)}",
            "error"
        )

    return redirect(
        url_for("index")
    )


# ============================================================
# QUERY PAGE
# ============================================================

@app.route("/query")
def query_page():

    pdfs = get_indexed_pdfs()

    return render_template(
        "query.html",
        pdfs=pdfs
    )


# ============================================================
# QUERY PDF
# ============================================================

@app.route(
    "/search",
    methods=["POST"]
)
def search():

    pdf_name = request.form.get(
        "pdf_name",
        ""
    ).strip()

    query = request.form.get(
        "query",
        ""
    ).strip()

    if not pdf_name:

        return render_template(
            "query.html",
            pdfs=get_indexed_pdfs(),
            error="Please select a PDF."
        )

    if not query:

        return render_template(
            "query.html",
            pdfs=get_indexed_pdfs(),
            error="Please enter a question."
        )

    try:

        results = search_pdf(
            pdf_name=pdf_name,
            query=query,
            top_k=3
        )

        answer = generate_answer(
            query,
            results
        )

        # ----------------------------------------------------
        # Unique pages
        # ----------------------------------------------------

        pages = sorted(
            list(
                set(
                    result["page"]
                    for result in results
                )
            )
        )

        return render_template(
            "query.html",
            pdfs=get_indexed_pdfs(),
            selected_pdf=pdf_name,
            query=query,
            answer=answer,
            results=results,
            pages=pages
        )

    except Exception as e:

        return render_template(
            "query.html",
            pdfs=get_indexed_pdfs(),
            error=str(e)
        )


# ============================================================
# API: LIST PDFS
# ============================================================

@app.route(
    "/api/pdfs",
    methods=["GET"]
)
def api_pdfs():

    return jsonify({
        "pdfs": get_indexed_pdfs()
    })


# ============================================================
# API: SEARCH
# ============================================================

@app.route(
    "/api/search",
    methods=["POST"]
)
def api_search():

    data = request.get_json()

    if not data:

        return jsonify({
            "error": "JSON body required."
        }), 400

    pdf_name = data.get(
        "pdf_name"
    )

    query = data.get(
        "query"
    )

    top_k = int(
        data.get(
            "top_k",
            5
        )
    )

    if not pdf_name:

        return jsonify({
            "error": "pdf_name is required."
        }), 400

    if not query:

        return jsonify({
            "error": "query is required."
        }), 400

    try:

        results = search_pdf(
            pdf_name,
            query,
            top_k
        )

        pages = sorted(
            list(
                set(
                    result["page"]
                    for result in results
                )
            )
        )

        answer = generate_answer(
            query,
            results
        )

        return jsonify({
            "pdf_name": pdf_name,
            "query": query,
            "answer": answer,
            "pages": pages,
            "results": results
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# DELETE PDF INDEX
# ============================================================

@app.route(
    "/delete/<path:pdf_name>",
    methods=["POST"]
)
def delete_pdf(pdf_name):

    pdf_folder = get_pdf_folder(
        pdf_name
    )

    if os.path.exists(pdf_folder):

        shutil.rmtree(
            pdf_folder
        )

    # Delete uploaded PDF
    uploaded_pdf = os.path.join(
        UPLOAD_FOLDER,
        secure_filename(pdf_name)
    )

    if os.path.exists(uploaded_pdf):

        os.remove(
            uploaded_pdf
        )

    flash(
        f"{pdf_name} deleted successfully.",
        "success"
    )

    return redirect(
        url_for("index")
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )