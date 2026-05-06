import streamlit as st
from config import UPLOAD_DIR
from ingestion import extract_text
from preprocessing import process_document
from embeddings import add_document_to_index, is_already_indexed
from search import search
from summarizer import summarize_chunks

#PAGE CONFIG
st.set_page_config(
    page_title="Doc Search & Summarize",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI-Powered Document Search & Summarization")
st.caption("Upload documents, ask questions, get answers backed by sources.")

#SIDEBAR: UPLOAD AND INDEX DOCUMENTS
with st.sidebar:
    st.header("📥 Upload Documents")

    # accept_multiple_files=True lets users batch-upload
    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process Documents"):
        for uploaded_file in uploaded_files:
            # Skip if already indexed (prevents duplicate vectors)
            if is_already_indexed(uploaded_file.name):
                st.info(f"⏭️ {uploaded_file.name} already indexed, skipping.")
                continue

            # Save the upload to disk so we have a permanent copy
            file_path = UPLOAD_DIR / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())

            # Process: extract, then chunk, then embed, then index
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    raw_text = extract_text(file_path)
                    chunks = process_document(raw_text)
                    num_added = add_document_to_index(uploaded_file.name, chunks)
                    st.success(f"✅ Added {num_added} chunks from {uploaded_file.name}")
                except Exception as e:
                    # Don't let one bad file kill the whole batch
                    st.error(f"❌ Failed on {uploaded_file.name}: {e}")

    st.divider()
    st.caption("💡 Tip: You can also drop files directly into `data/uploads/` "
               "if the file watcher is running.")
    
#MAIN: QUERY AND RESULTS
st.header("🔎 Ask a Question")
query = st.text_input(
    "Enter your question:",
    placeholder="e.g., What were the main findings of the study?",
)

if query:
    with st.spinner("Searching..."):
        results = search(query)

    if not results:
        st.warning("No documents indexed yet. Upload some on the left to get started.")
    else:
        # ----- Summary at top (the "answer") -----
        st.subheader("📝 Summary")
        with st.spinner("Generating summary..."):
            top_texts = [r["text"] for r in results]
            summary = summarize_chunks(top_texts)
        st.write(summary)

        # ----- Source chunks below (the "evidence") -----
        # Showing sources is crucial: it lets users verify the summary
        # isn't hallucinated and dig deeper if needed.
        st.subheader("🔍 Source Chunks")
        for i, r in enumerate(results, 1):
            with st.expander(
                f"{i}. {r['doc_name']}, chunk {r['chunk_index']} "
                f"(similarity: {r['score']:.3f})"
            ):
                st.write(r["text"])
    