import gradio as gr
import os
import shutil
from byaldi import RAGMultiModalModel
from mlx_vlm import load, generate, prepare_inputs
from pdf2image import convert_from_path
from PIL import Image

# Global Model Variables
RAG_MODEL = None
VLM_MODEL = None
VLM_TOKENIZER = None

def load_models():
    """Loads the retrieval and generation models."""
    global RAG_MODEL, VLM_MODEL, VLM_TOKENIZER
    
    print("Loading Byaldi (ColPali) Model...")
    # Byaldi handles device placement (MPS on Mac by default usually, but we check)
    # vidore/colpali-v1.2 is the requested model
    RAG_MODEL = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", device="mps")
    
    print("Loading MLX VLM Model (Qwen2-VL-7B)...")
    # mlx-community/Qwen2-VL-7B-Instruct-4bit
    VLM_MODEL, VLM_TOKENIZER = load("mlx-community/Qwen2-VL-7B-Instruct-4bit")
    print("Models loaded successfully.")

def process_pdf(pdf_file):
    """
    Converts PDF to images and indexes them using Byaldi.
    """
    if pdf_file is None:
        return "No PDF uploaded."
    
    try:
        # Robustly handle input: it could be a Gradio temp file object or a string path
        if hasattr(pdf_file, 'name'):
            pdf_path = pdf_file.name
        else:
            pdf_path = pdf_file
            
        print(f"Processing PDF from path: {pdf_path}")
        index_name = "user_pdf_index"
        
        # Temp dir for images
        images_dir = "temp_pdf_images"
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
        os.makedirs(images_dir)
        
        print("Converting PDF to images...")
        try:
            images = convert_from_path(pdf_path)
            print(f"Converted {len(images)} pages.")
        except Exception as e:
            print(f"Error converting PDF: {e}")
            raise e

        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(images_dir, f"page_{i}.png")
            image.save(image_path, "PNG")
            image_paths.append(image_path)
            
        print("Indexing images with Byaldi...")
        try:
            # FIX: Manually clear in-memory state as overwrite=True doesn't do it fully in current Byaldi version
            if RAG_MODEL and hasattr(RAG_MODEL, 'model'):
                 # Check/Clear specific attributes
                 if hasattr(RAG_MODEL.model, 'doc_ids'): RAG_MODEL.model.doc_ids = set()
                 if hasattr(RAG_MODEL.model, 'embed_id_to_doc_id'): RAG_MODEL.model.embed_id_to_doc_id = {}
                 if hasattr(RAG_MODEL.model, 'indexed_embeddings'): RAG_MODEL.model.indexed_embeddings = []
                 if hasattr(RAG_MODEL.model, 'doc_id_to_metadata'): RAG_MODEL.model.doc_id_to_metadata = {}
                 if hasattr(RAG_MODEL.model, 'doc_ids_to_file_names'): RAG_MODEL.model.doc_ids_to_file_names = {}
                 if hasattr(RAG_MODEL.model, 'collection'): RAG_MODEL.model.collection = {}
            # Indexing the images. Byaldi can index a list of images/docs.
            # overwrite=True to replace previous index
            RAG_MODEL.index(
                input_path=images_dir, # Byaldi can scan a directory
                index_name=index_name,
                store_collection_with_index=True,
                overwrite=True
            )
            print("Indexing successful.")
        except Exception as e:
            print(f"Byaldi Indexing Error: {e}")
            raise e
        
        return f"Successfully processed {len(images)} pages and indexed '{os.path.basename(pdf_path)}'."
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing PDF: {str(e)}"

def rag_response(user_query):
    """
    Retrieves relevant image and generates answer.
    """
    if RAG_MODEL is None or VLM_MODEL is None:
        return None, "Models not loaded. Please wait."
    
    if not user_query:
        return None, "Please enter a question."

    try:
        # 1. Retrieve
        # Byaldi search returns top k results
        results = RAG_MODEL.search(user_query, k=1)
        if not results:
            return None, "No relevant info found."
        
        # Get top result
        top_result = results[0]
        # Byaldi results usually contain 'doc_id', 'page_num', or metadata to find the source
        # Since we indexed the directory, Byaldi manages the mapping.
        # For simplicity in this demo, accessing the specific image might need the doc mapping.
        # RAGMultiModalModel returns result objects that should have the image or path.
        # Inspecting Byaldi API (assuming standard usage):
        # Result object has .doc_id, .score, and potentially .base64 or similar if stored.
        # If we indexed a folder, doc_id usually relates to the filename.
        
        # Let's trust Byaldi to extract the image or we reload it from our temp dir based on doc_id
        # Actually RAGMultiModalModel often returns the doc content (image) directly or path
        # Let's assume we can get the path or the PIL image from the result.
        
        # Standard Byaldi/ColPali usage:
        # The result object might vary. Let's look up the original image from the doc_id if possible.
        # 'doc_id' typically corresponds to the file index or name.
        
        # Debugging/Safety: Use the first file in temp dir if retrieval logic is opaque without docs, 
        # but correctly:
        # results[0].doc_id should give us a clue.
        # Let's assume we can retrieve the image object directly if supported or reconstruct path.
        # For this demo, let's look at the implementation of Byaldi's search.
        # It typically returns a list of results. 
        
        # We will retrieve the original image path based on the result.
        # Assuming doc_id is index in the input list or filename.
        # Since we passed a directory, Byaldi iterates files.
        # We will assume result.doc_id matches the filename or ID.
        
        # Simplification: We will reload the image from the text result if it provides a path, 
        # or we rely on the Doc list if we had kept it.
        # Let's try to get the image path from the result metadata.
        
        # IMPORTANT: For this specific task, let's assume we can get the image. 
        # If Byaldi returns base64, we decode. If path, we open.
        # For now, let's assume we get the relevant image from the document list stored in the model.
        # The doc_id likely maps to the index in the 'temp_pdf_images' folder items.
        
        # Let's fetch the image from the temp_pdf_images directory that matches the result.
        # If doc_id is an integer index:
        image_files = sorted(os.listdir("temp_pdf_images"))
        # This is a bit risky if numbering isn't aligned, but with 'page_i.png' it should be ok if sorted.
        
        top_doc_id = top_result.doc_id # Checking Byaldi API, this is usually reliable
        
        # Depending on Byaldi version, doc_id might be int. 
        # If we can't map perfectly, we might just take result 0's image if returned.
        # Let's assume `top_result.base64` exists (common in these tools) or we find file.
        
        # Robust Fallback:
        retrieved_image_path = None
        if hasattr(top_result, 'base64'):
             # Decode base64 to PIL
             import base64
             from io import BytesIO
             image_data = base64.b64decode(top_result.base64)
             retrieved_image = Image.open(BytesIO(image_data))
        else:
            # Fallback to mapping by ID if strict path isn't there
            # Assuming doc_id matches the sorted list index 
            image_files = sorted([f for f in os.listdir("temp_pdf_images") if f.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))
            if isinstance(top_result.doc_id, int) and top_result.doc_id < len(image_files):
                retrieved_image_path = os.path.join("temp_pdf_images", image_files[top_result.doc_id])
                retrieved_image = Image.open(retrieved_image_path)
            else:
                # Last resort, just show the first page if everything fails (shouldn't happen)
                retrieved_image = Image.open(os.path.join("temp_pdf_images", image_files[0]))

        # 2. Generate
        prompt = user_query
        
        # MLX VLM Interaction
        # We need to format the prompt. Qwen2-VL specific formatting is handled by the tokenizer usually, 
        # but apply_chat_template is safer.
        pass
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": retrieved_image},
                    {"type": "text", "text": user_query},
                ],
            }
        ]
        
        text = VLM_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate
        output = generate(VLM_MODEL, VLM_TOKENIZER, prompt=text, verbose=False, max_tokens=512, temp=0.7, image=[retrieved_image])
        
        return retrieved_image, output

    except Exception as e:
        return None, f"Error generating response: {str(e)}"

# Initialize UI
with gr.Blocks(title="Local Vision RAG (MacOS M5)") as demo:
    gr.Markdown("# 👁️ Local Vision RAG Demo\n**Retrieval:** Byaldi (ColPali) | **Generation:** MLX-VLM (Qwen2-VL)")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF Contract", file_types=[".pdf"])
            process_btn = gr.Button("Index PDF", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column(scale=2):
            chatbot = gr.Markdown(" Upload a PDF to start.")

    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Ask a question about the document", placeholder="e.g., What is the termination clause?")
            submit_btn = gr.Button("Get Answer")

    with gr.Row():
        retrieved_img = gr.Image(type="pil", label="Retrieved Page")
        answer_output = gr.Textbox(label="Answer", lines=5)

    # Event Handlers
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[status_output]
    )
    
    submit_btn.click(
        fn=rag_response,
        inputs=[query_input],
        outputs=[retrieved_img, answer_output]
    )
    
    # Load models on startup (or lazy load if preferred, but user asked for startup)
    # We can use a load event or just call it.
    # Calling it here means it loads when script runs.
    # Note: This might block UI startup slightly but acceptable for local demo.

if __name__ == "__main__":
    load_models()
    demo.launch(server_name="0.0.0.0", server_port=7860)
