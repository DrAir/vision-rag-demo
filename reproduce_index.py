import os
import shutil
from byaldi import RAGMultiModalModel

def reproduce():
    print("Loading RAG Model...")
    try:
        # Assuming MPS as per app.py
        rag_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", device="mps")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    images_dir = "temp_pdf_images"
    index_name = "user_pdf_index"

    print(f"Indexing images from {images_dir} (Run 1)...")
    try:
        rag_model.index(
            input_path=images_dir,
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True
        )
        print("Run 1 completed successfully.")
    except Exception as e:
        print(f"Error during Run 1: {e}")
        return

    print(f"Indexing images from {images_dir} (Run 2 - Same Model Instance)...")
    
    # ATTEMPTING FIX: Manually clear in-memory state
    print("Manually clearing in-memory state...")
    if hasattr(rag_model.model, 'doc_ids'):
        rag_model.model.doc_ids = set()
    if hasattr(rag_model.model, 'embed_id_to_doc_id'):
        rag_model.model.embed_id_to_doc_id = {}
    if hasattr(rag_model.model, 'indexed_embeddings'):
        rag_model.model.indexed_embeddings = []
    if hasattr(rag_model.model, 'doc_id_to_metadata'):
        rag_model.model.doc_id_to_metadata = {}
    if hasattr(rag_model.model, 'doc_ids_to_file_names'):
        rag_model.model.doc_ids_to_file_names = {}
    if hasattr(rag_model.model, 'collection'):
        rag_model.model.collection = {}
    
    # highest_doc_id is reset by overwrite=True in index(), but good to be safe if not
    
    try:
        rag_model.index(
            input_path=images_dir,
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True
        )
        print("Run 2 completed successfully.")
    except Exception as e:
        print(f"Error during Run 2: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce()
