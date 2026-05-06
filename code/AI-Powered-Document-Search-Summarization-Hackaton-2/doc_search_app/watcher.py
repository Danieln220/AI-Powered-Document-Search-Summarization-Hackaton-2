#BONUS module: file watcher that automatically processes any new document
#dropped into the uploads folder.

import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config import UPLOAD_DIR
from ingestion import extract_text
from preprocessing import process_document
from embeddings import add_document_to_index, is_already_indexed

SUPPORTED = {".pdf", ".docx", ".txt"}


class UploadHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        if file_path.suffix.lower() not in SUPPORTED:
            return
        if is_already_indexed(file_path.name):
            print(f"[watcher] Already indexed, skipping: {file_path.name}")
            return
        print(f"[watcher] New file detected: {file_path.name}")
        try:
            raw_text = extract_text(file_path)
            chunks = process_document(raw_text)
            num_added = add_document_to_index(file_path.name, chunks)
            print(f"[watcher] Indexed {num_added} chunks from {file_path.name}")
        except Exception as e:
            print(f"[watcher] Failed to index {file_path.name}: {e}")


def main():
    observer = Observer()
    observer.schedule(UploadHandler(), str(UPLOAD_DIR), recursive=False)
    observer.start()
    print(f"[watcher] Watching {UPLOAD_DIR} for new documents. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
