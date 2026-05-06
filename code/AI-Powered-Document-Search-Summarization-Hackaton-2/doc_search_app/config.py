from pathlib import Path

BASE_DIR = Path(__file__).parent
# Where uploaded documents are stored on disk
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
# Where the FAISS index and metadata get persisted
INDEX_DIR = BASE_DIR / "data" / "index"
INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "metadata.pkl"
# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Models

# all-MiniLM-L6-v2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# distilbart-cnn-12-6
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"

# CHUNKING
CHUNK_SIZE = 300

CHUNK_OVERLAP = 50

# SEARCH
TOP_K = 5

# EMBEDDING DIMENSION
EMBEDDING_DIM = 384
