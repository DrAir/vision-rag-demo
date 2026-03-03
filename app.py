import gradio as gr
import os
import shutil
import json
import urllib.request
import urllib.error
from datetime import datetime
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from pdf2image import convert_from_path
from PIL import Image

# Global Model Variables
VLM_MODEL = None
VLM_TOKENIZER = None

# Global Metadata Storage (for context in Q&A)
EXTRACTED_METADATA = {}

# Track current PDF filename
CURRENT_PDF_FILENAME = ""
CURRENT_INVOICE_FILENAME = ""

# Data file paths
METADATA_RECORDS_FILE = "metadata_records.json"
INVOICE_RECORDS_FILE = "invoice_records.json"

# Batch invoice processing
ALL_INVOICE_RESULTS = []  # List of dicts, 1 per page
INVOICE_PAGE_PATHS = []   # List of image paths for invoice pages only
CONFIRMED_INVOICES = set()  # Set of confirmed page indices

def load_models():
    """Loads the VLM model for All-Page Scan."""
    global VLM_MODEL, VLM_TOKENIZER
    
    print("Loading MLX VLM Model (Qwen3-VL-30B-A3B)...")
    VLM_MODEL, VLM_TOKENIZER = load("mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit")
    print("Model loaded successfully.")

def process_pdf(pdf_file):
    """
    Converts PDF to images for All-Page Scan.
    Returns: (status_text, gallery_images_list)
    """
    if pdf_file is None:
        return "No PDF uploaded.", []
    
    if VLM_MODEL is None:
        return "Model not loaded yet. Please wait.", []
    
    try:
        # Robustly handle input
        if hasattr(pdf_file, 'name'):
            pdf_path = pdf_file.name
        else:
            pdf_path = pdf_file
            
        print(f"Processing PDF from path: {pdf_path}")
        
        # Temp dir for images
        images_dir = "temp_pdf_images"
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
        os.makedirs(images_dir)
        
        # DPI settings
        TARGET_DPI = 150
        MAX_DIMENSION = 1500  # Max width or height
        
        print(f"Converting PDF to images (DPI={TARGET_DPI}, max={MAX_DIMENSION}px)...")
        try:
            images = convert_from_path(pdf_path, dpi=TARGET_DPI)
            print(f"Converted {len(images)} pages.")
        except Exception as e:
            print(f"Error converting PDF: {e}")
            raise e

        saved_paths = []
        for i, image in enumerate(images):
            # MEMORY OPTIMIZATION: Resize if still too large
            w, h = image.size
            if max(w, h) > MAX_DIMENSION:
                ratio = MAX_DIMENSION / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                image = image.resize(new_size, Image.LANCZOS)
                print(f"  Page {i}: Resized to {new_size[0]}x{new_size[1]}")
            
            image_path = os.path.join(images_dir, f"page_{i}.png")
            image.save(image_path, "PNG", optimize=True)
            saved_paths.append(os.path.abspath(image_path))
            
            # MEMORY OPTIMIZATION: Explicitly release image memory
            image.close()
            del image
        
        # Clear the list to free memory
        del images
        import gc
        gc.collect()
        
        # Build gallery for preview
        gallery = [(Image.open(p), f"Trang {i+1}") for i, p in enumerate(saved_paths)]
        
        num_pages = len(saved_paths)
        return f"✅ Đã chuyển đổi {num_pages} trang (DPI={TARGET_DPI}). Đang trích xuất metadata...", gallery
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing PDF: {str(e)}", []


def upload_and_process(pdf_file):
    """
    Auto-trigger pipeline: upload → index (convert to images) → extract metadata.
    Called automatically when user uploads a PDF.
    Returns: status, gallery, 9 metadata fields, 8 checkboxes (False), 8 notes (""), confirm_status ("")
    """
    global CURRENT_PDF_FILENAME
    
    # Reset values for checkboxes + notes (8 fields x 2 = 16) + confirm status
    reset_corrections = (False, "") * 8 + ("",)  # 8x(checkbox, note) + confirm_status
    
    # Track PDF filename
    if pdf_file is not None:
        if hasattr(pdf_file, 'name'):
            CURRENT_PDF_FILENAME = os.path.basename(pdf_file.name)
        else:
            CURRENT_PDF_FILENAME = os.path.basename(str(pdf_file))
    
    # Step 1: Convert PDF to images
    status, gallery = process_pdf(pdf_file)
    
    empty_meta = ("N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "")
    
    if "Error" in status or not gallery:
        return (status, gallery, *empty_meta, *reset_corrections)
    
    # Step 2: Auto-extract metadata
    metadata_results = extract_metadata()
    
    status = status.replace("Đang trích xuất metadata...", "Trích xuất metadata hoàn tất.")
    
    return (status, gallery, *metadata_results, *reset_corrections)

def rag_response(user_query):
    """
    ALL-PAGE SCAN: Iterates through ALL document pages and finds the best answer.
    No retrieval - VLM reads every page.
    """
    if VLM_MODEL is None:
        return [], "Models not loaded. Please wait."
    
    if not user_query:
        return [], "Please enter a question."

    try:
        images_dir = "temp_pdf_images"
        if not os.path.exists(images_dir):
            return [], "No document indexed. Please upload a PDF first."
        
        # Get all page images (exclude cropped temp files)
        image_files = sorted(
            [f for f in os.listdir(images_dir) if f.endswith('.png') and f.startswith('page_')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        
        if not image_files:
            return [], "No pages found. Please upload a PDF first."
        
        total_pages = len(image_files)
        print(f"\n=== ALL-PAGE SCAN: '{user_query}' ({total_pages} pages) ===")
        
        # Scan ALL pages and collect answers
        page_answers = []
        
        for i, filename in enumerate(image_files):
            img_path = os.path.abspath(os.path.join(images_dir, filename))
            page_num = i + 1
            
            print(f"Scanning page {page_num}/{total_pages}...")
            
            # Build metadata context if available
            metadata_context = ""
            if EXTRACTED_METADATA:
                context_lines = []
                if EXTRACTED_METADATA.get('signing_date') and EXTRACTED_METADATA.get('signing_date') != 'N/A':
                    context_lines.append(f"- Ngày ký hợp đồng: {EXTRACTED_METADATA['signing_date']}")
                if EXTRACTED_METADATA.get('duration') and EXTRACTED_METADATA.get('duration') != 'N/A':
                    context_lines.append(f"- Thời gian thực hiện: {EXTRACTED_METADATA['duration']}")
                if EXTRACTED_METADATA.get('contract_value') and EXTRACTED_METADATA.get('contract_value') != 'N/A':
                    context_lines.append(f"- Giá trị hợp đồng: {EXTRACTED_METADATA['contract_value']} VNĐ")
                if context_lines:
                    metadata_context = "\n\n**THÔNG TIN ĐÃ BIẾT:**\n" + "\n".join(context_lines) + "\n\nHãy sử dụng thông tin trên để tính toán nếu cần (ví dụ: ngày hết hạn = ngày ký + thời gian thực hiện).\n"
            
            # Ask VLM to check if this page has the answer
            check_prompt = f"""Nhìn vào trang tài liệu này và trả lời câu hỏi sau. 
Nếu trang này KHÔNG chứa thông tin liên quan, hãy trả lời chính xác: "KHÔNG CÓ THÔNG TIN"
Nếu trang này CÓ thông tin liên quan, hãy trả lời chi tiết.
{metadata_context}
Câu hỏi: {user_query}"""
            
            prompt = apply_chat_template(
                VLM_TOKENIZER,
                VLM_MODEL.config,
                check_prompt,
                num_images=1
            )
            
            result = generate(
                VLM_MODEL,
                VLM_TOKENIZER,
                prompt=prompt,
                image=img_path,
                max_tokens=512,
                temp=0.2,
                verbose=False
            )
            
            # Extract text
            if hasattr(result, 'text'):
                answer = result.text
            elif isinstance(result, str):
                answer = result
            else:
                answer = str(result)
            
            # Clean up
            answer = answer.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
            
            # Check if page has relevant info
            has_info = "KHÔNG CÓ THÔNG TIN" not in answer.upper() and len(answer) > 20
            
            page_answers.append({
                'page_num': page_num,
                'path': img_path,
                'answer': answer,
                'has_info': has_info
            })
            
            if has_info:
                print(f"  ✓ Page {page_num}: Found relevant info")
            else:
                print(f"  ✗ Page {page_num}: No relevant info")
        
        # Separate pages with and without info
        pages_with_info = [p for p in page_answers if p['has_info']]
        
        if not pages_with_info:
            # No page had relevant info, return first page with a note
            gallery = [(Image.open(p['path']), f"Page {p['page_num']}") for p in page_answers]
            return gallery, "❌ Không tìm thấy thông tin liên quan trong tài liệu này.\n\nVui lòng thử câu hỏi khác hoặc kiểm tra lại tài liệu."
        
        # ===== PHASE 2: VERIFICATION (Anti-Hallucination) =====
        print(f"\n=== VERIFICATION CHECK ({len(pages_with_info)} candidates) ===")
        
        verified_pages = []
        
        for p in pages_with_info:
            # Ask VLM to verify by quoting EXACT text from the image
            verify_prompt = f"""Kiểm tra xác nhận: Câu trả lời sau có THỰC SỰ dựa trên nội dung trang này không?

Câu hỏi: {user_query}
Câu trả lời đã cho: {p['answer'][:400]}

Hãy làm 2 việc:
1. TRÍCH DẪN CHÍNH XÁC một đoạn văn bản từ trang này (copy nguyên văn) chứng minh câu trả lời đúng
2. Nếu KHÔNG TÌM THẤY đoạn văn bản nào trong trang này liên quan, hãy trả lời: "KHÔNG XÁC NHẬN ĐƯỢC"

Trả lời theo format:
TRÍCH DẪN: [đoạn text chính xác từ trang]
hoặc
KHÔNG XÁC NHẬN ĐƯỢC"""
            
            prompt = apply_chat_template(
                VLM_TOKENIZER,
                VLM_MODEL.config,
                verify_prompt,
                num_images=1
            )
            
            result = generate(
                VLM_MODEL,
                VLM_TOKENIZER,
                prompt=prompt,
                image=p['path'],
                max_tokens=300,
                temp=0.1,
                verbose=False
            )
            
            # Extract verification result
            if hasattr(result, 'text'):
                verify_text = result.text
            elif isinstance(result, str):
                verify_text = result
            else:
                verify_text = str(result)
            
            verify_text = verify_text.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
            
            # Check if verification passed
            is_verified = (
                "KHÔNG XÁC NHẬN" not in verify_text.upper() and
                "TRÍCH DẪN:" in verify_text.upper() and
                len(verify_text) > 30
            )
            
            # Extract the quote if verified
            quote = ""
            if is_verified and "TRÍCH DẪN:" in verify_text.upper():
                import re
                quote_match = re.search(r'TRÍCH DẪN[:\s]*(.+?)(?:\n|$)', verify_text, re.IGNORECASE | re.DOTALL)
                if quote_match:
                    quote = quote_match.group(1).strip()[:200]
            
            p['verified'] = is_verified
            p['quote'] = quote
            
            if is_verified:
                verified_pages.append(p)
                print(f"  ✓ Page {p['page_num']}: VERIFIED - \"{quote[:50]}...\"")
            else:
                print(f"  ? Page {p['page_num']}: Could not verify quote (still included)")
        
        # ===== SOFT VERIFICATION: Include ALL pages with info =====
        # Use verification as ranking boost, NOT as hard filter
        # This prevents rejecting correct answers that can't produce exact quotes
        
        # Include ALL pages_with_info, not just verified ones
        for p in pages_with_info:
            if not p.get('verified'):
                p['verified'] = False
                if not p.get('quote'):
                    p['quote'] = ""
        
        # Use pages_with_info for classification (all pages, not just verified)
        all_candidate_pages = pages_with_info
        
        # ===== PHASE 3: SOURCE CLASSIFICATION =====
        print(f"\n=== SOURCE CLASSIFICATION ({len(all_candidate_pages)} candidates) ===")
        
        for p in all_candidate_pages:
            # Ask VLM to classify what type of document this page is
            classify_prompt = f"""Nhìn vào trang tài liệu này và xác định loại tài liệu.

Trả lời chính xác MỘT trong các loại sau:
- HỢP ĐỒNG CHÍNH (trang chứa nội dung hợp đồng gốc, điều khoản chính)
- PHỤ LỤC (phụ lục hợp đồng, bảng phụ)
- BIÊN BẢN (biên bản nghiệm thu, hoàn thiện, bàn giao)
- KHÁC (tài liệu khác)

Chỉ trả lời tên loại, không giải thích thêm."""
            
            prompt = apply_chat_template(
                VLM_TOKENIZER,
                VLM_MODEL.config,
                classify_prompt,
                num_images=1
            )
            
            result = generate(
                VLM_MODEL,
                VLM_TOKENIZER,
                prompt=prompt,
                image=p['path'],
                max_tokens=30,
                temp=0.1,
                verbose=False
            )
            
            if hasattr(result, 'text'):
                doc_type = result.text
            else:
                doc_type = str(result)
            
            doc_type = doc_type.replace('<|im_start|>', '').replace('<|im_end|>', '').strip().upper()
            
            # ALSO check the quote for document type keywords (more reliable)
            quote_lower = p.get('quote', '').lower()
            answer_lower = p.get('answer', '').lower()
            combined_text = quote_lower + " " + answer_lower
            
            # Quote-based detection (higher priority than VLM classification)
            if "biên bản" in combined_text or "nghiệm thu" in combined_text or "hoàn thiện" in combined_text or "bàn giao" in combined_text:
                p['doc_type'] = "BIÊN BẢN"
                p['priority'] = 3
            elif "phụ lục" in combined_text:
                p['doc_type'] = "PHỤ LỤC"
                p['priority'] = 2
            # Fall back to VLM classification
            elif "HỢP ĐỒNG" in doc_type and "PHỤ" not in doc_type:
                p['doc_type'] = "HỢP ĐỒNG CHÍNH"
                p['priority'] = 1
            elif "PHỤ LỤC" in doc_type:
                p['doc_type'] = "PHỤ LỤC"
                p['priority'] = 2
            elif "BIÊN BẢN" in doc_type:
                p['doc_type'] = "BIÊN BẢN"
                p['priority'] = 3
            else:
                p['doc_type'] = "KHÁC"
                p['priority'] = 4
            
            print(f"  Page {p['page_num']}: {p['doc_type']} (quote-based: {'biên bản' in combined_text})")
        
        # Sort by: 1) verified status (verified first), 2) priority (contract > appendix > record), 3) answer length
        all_candidate_pages.sort(key=lambda x: (not x.get('verified', False), x.get('priority', 99), -len(x['answer'])))
        
        # ===== PHASE 4: SMART ANSWER SYNTHESIS =====
        print(f"\n=== ANSWER SYNTHESIS ===")
        
        # If multiple sources, ask VLM to synthesize
        if len(all_candidate_pages) > 1:
            # Prepare context from top pages (prioritize verified + high priority)
            sources_context = ""
            for i, p in enumerate(all_candidate_pages[:3]):  # Max 3 sources
                verified_mark = "✓" if p.get('verified') else "?"
                sources_context += f"\n\n[Nguồn {i+1} {verified_mark} - {p['doc_type']} (Page {p['page_num']})]:\n{p['answer'][:400]}"
            
            synthesis_prompt = f"""Câu hỏi: {user_query}

Dưới đây là thông tin từ nhiều nguồn trong cùng một bộ tài liệu:
{sources_context}

Hãy tổng hợp và đưa ra câu trả lời CHÍNH XÁC NHẤT dựa trên các nguồn trên.
Ưu tiên thông tin từ HỢP ĐỒNG CHÍNH hơn BIÊN BẢN.
Nếu các nguồn có thông tin khác nhau, hãy ghi rõ sự khác biệt.

Trả lời ngắn gọn, chính xác."""
            
            prompt = apply_chat_template(
                VLM_TOKENIZER,
                VLM_MODEL.config,
                synthesis_prompt,
                num_images=1
            )
            
            # Use the primary source image for synthesis
            result = generate(
                VLM_MODEL,
                VLM_TOKENIZER,
                prompt=prompt,
                image=all_candidate_pages[0]['path'],
                max_tokens=600,
                temp=0.2,
                verbose=False
            )
            
            if hasattr(result, 'text'):
                synthesized = result.text
            else:
                synthesized = str(result)
            
            synthesized = synthesized.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
            
            final_answer = f"**📋 Tổng hợp từ {len(all_candidate_pages)} nguồn:**\n\n{synthesized}"
            
            # Add source breakdown
            final_answer += "\n\n---\n📚 **Chi tiết từng nguồn:**"
            for p in all_candidate_pages[:3]:
                verified_mark = "✓" if p.get('verified') else "?"
                final_answer += f"\n\n**{verified_mark} {p['doc_type']} (Page {p['page_num']}):**\n{p['answer'][:200]}..."
                if p.get('quote'):
                    final_answer += f"\n📝 \"{p['quote'][:80]}...\""
        else:
            # Single source
            best = all_candidate_pages[0]
            verified_mark = "✓" if best.get('verified') else "?"
            final_answer = f"**{verified_mark} Từ {best['doc_type']} (Page {best['page_num']}):**\n\n{best['answer']}"
            if best.get('quote'):
                final_answer += f"\n\n📝 **Trích dẫn:** \"{best['quote']}\""
        
        # Add scan summary - show ALL candidates with their status
        def get_icon(p):
            icons = {"HỢP ĐỒNG CHÍNH": "📄", "PHỤ LỤC": "📎", "BIÊN BẢN": "📋", "KHÁC": "📃"}
            return icons.get(p.get('doc_type', 'KHÁC'), "📃")
        
        rankings = "\n".join([f"- {get_icon(p)} Page {p['page_num']}: {p.get('doc_type', 'N/A')} {'✓ verified' if p.get('verified') else '? unverified'}" for p in all_candidate_pages])
        summary = f"\n\n---\n📊 **Tất cả nguồn (ưu tiên verified + hợp đồng chính):**\n{rankings}"
        final_answer += summary
        
        # Create gallery with doc type labels
        gallery = []
        for p in page_answers:
            if p.get('verified'):
                doc_type_short = p.get('doc_type', '')[:10]
                label = f"✓ Page {p['page_num']} ({doc_type_short})"
            elif p['has_info']:
                label = f"? Page {p['page_num']}"
            else:
                label = f"Page {p['page_num']}"
            gallery.append((Image.open(p['path']), label))
        
        # MEMORY OPTIMIZATION: Force garbage collection after heavy processing
        import gc
        gc.collect()
        
        return gallery, final_answer

    except Exception as e:
        import traceback
        traceback.print_exc()
        # MEMORY OPTIMIZATION: Clean up on error too
        import gc
        gc.collect()
        return [], f"Error: {str(e)}"


def rag_response_with_page(user_query, selected_page_idx):
    """
    Generate answer from a specific selected page.
    """
    if RAG_MODEL is None or VLM_MODEL is None:
        return "Models not loaded. Please wait."
    
    if not user_query:
        return "Please enter a question."
    
    if selected_page_idx is None:
        return "Please select a page from the gallery first."

    try:
        # Get all image files sorted by page number (exclude cropped temp files)
        image_files = sorted(
            [f for f in os.listdir("temp_pdf_images") if f.endswith('.png') and f.startswith('page_')], 
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        
        # Get the selected image path
        if selected_page_idx < len(image_files):
            img_path = os.path.abspath(os.path.join("temp_pdf_images", image_files[selected_page_idx]))
        else:
            return "Invalid page selection."
        
        print(f"\n=== Generating from selected Page {selected_page_idx + 1} ===")
        
        prompt = apply_chat_template(
            VLM_TOKENIZER,
            VLM_MODEL.config,
            user_query,
            num_images=1
        )
        
        result = generate(
            VLM_MODEL, 
            VLM_TOKENIZER, 
            prompt=prompt,
            image=img_path,
            max_tokens=1024,
            temp=0.3,
            verbose=False
        )
        
        # Extract text
        if hasattr(result, 'text'):
            output = result.text
        elif isinstance(result, str):
            output = result
        else:
            output = str(result)
        
        # Clean up special tokens
        output = output.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
        
        return f"**Từ Page {selected_page_idx + 1}:**\n\n{output}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"


def extract_metadata():
    """
    Automatically extract contract metadata from PDF pages using ALL-PAGE SCAN + VERIFICATION.
    Returns: contract_number, contract_name, signing_date, contract_value, duration, party_a, party_b, expiry_date, extraction_time
    """
    import time
    start_time = time.time()
    
    if VLM_MODEL is None:
        return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "Model not loaded"
    
    images_dir = "temp_pdf_images"
    if not os.path.exists(images_dir):
        return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "No PDF"
    
    # ALL-PAGE SCAN: Get ALL pages for thorough metadata extraction (exclude cropped temp files)
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.endswith('.png') and f.startswith('page_')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    
    if not image_files:
        return "N/A", "N/A", "N/A", "N/A"
    
    total_pages = len(image_files)
    print(f"\n=== METADATA EXTRACTION (ALL-PAGE SCAN + VERIFICATION: {total_pages} pages) ===")
    
    # Define metadata fields to extract with verification prompts
    # max_pages: limit how many pages to scan (None = all pages)
    # skip_verify: skip verification step for high-confidence fields
    # debug: show raw VLM output
    metadata_fields = [
        {
            "name": "contract_number",
            "prompt": "Tìm và trả lời chính xác SỐ HỢP ĐỒNG trong tài liệu này. Thường có dạng 'Số: XXX/...' hoặc 'HỢP ĐỒNG SỐ: XXX'. Chỉ trả lời số, không giải thích. Nếu không tìm thấy, trả lời 'N/A'.",
            "format": "int",
            "verify_prompt": "Trong trang này, có hiển thị SỐ HỢP ĐỒNG '{answer}' không? Trả lời YES hoặc NO.",
            "max_pages": 1,  # Only scan page 1 (header)
            "skip_verify": True,  # Skip verification - regex is strict enough
            "debug": False
        },
        {
            "name": "contract_name",
            "prompt": "Liệt kê TỪNG DÒNG văn bản bạn thấy trong hình này, mỗi dòng một hàng. Chỉ liệt kê, không giải thích.",
            "format": "text",
            "verify_prompt": "Trong hình này, có dòng chứa '{answer}' không? Trả lời YES hoặc NO.",
            "max_pages": 1,  # Only scan page 1 (header)
            "crop_region": (0, 0, 1.0, 0.5),  # Crop top 50% of image
            "skip_verify": False,  # Keep verification - prone to hallucination
            "debug": True  # DEBUG enabled
        },
        {
            "name": "signing_date", 
            "prompt": "Tìm dòng chứa 'Hôm nay, ngày' hoặc 'ký kết ngày' trong tài liệu và COPY NGUYÊN VĂN cả dòng đó. Ví dụ: 'Hôm nay, ngày 28 tháng 10 năm 2025' hoặc '...ký kết ngày 15 tháng 01 năm 2025...'. Chỉ trả lời dòng chứa ngày tháng, không giải thích. Nếu không tìm thấy, trả lời 'N/A'.",
            "format": "date",
            "verify_prompt": "Trong trang này, có hiển thị ngày '{answer}' (dưới bất kỳ định dạng nào như 'ngày X tháng Y năm Z' hoặc 'DD/MM/YYYY') không? Trả lời YES hoặc NO.",
            "max_pages": 2,  # Only scan first 2 pages
            "skip_verify": True,  # Skip verification - regex extraction is strict enough
            "debug": True  # DEBUG enabled
        },
        {
            "name": "contract_value",
            "prompt": "Tìm dòng chứa 'GIÁ TRỊ HỢP ĐỒNG' hoặc 'TỔNG GIÁ TRỊ' và COPY NGUYÊN VĂN số tiền. Ví dụ: 'Giá trị hợp đồng: 179.600.000 đồng'. Chỉ trả lời dòng chứa giá trị, không giải thích. Nếu không tìm thấy, trả lời 'N/A'.",
            "format": "money",
            "verify_prompt": "Trong trang này, có hiển thị GIÁ TRỊ HỢP ĐỒNG hoặc TỔNG GIÁ TRỊ kèm số tiền không? Trả lời YES hoặc NO.",
            "start_page": 2,  # Start from page 2
            "skip_last_percent": 0.5,  # Skip last 50% pages
            "skip_verify": False,  # Verify contract value against page
            "debug": False
        },
        {
            "name": "duration",
            "prompt": "Tìm dòng chứa 'Tổng thời gian thực hiện hợp đồng' hoặc 'Thời gian thực hiện' và COPY NGUYÊN VĂN cả dòng đó. Ví dụ: 'Tổng thời gian thực hiện hợp đồng: 30 ngày' hoặc 'Thời gian thực hiện: 12 tháng'. Chỉ trả lời dòng chứa thời gian, không giải thích. Nếu không tìm thấy, trả lời 'N/A'.",
            "format": "duration",
            "verify_prompt": "Trong trang này, có cụm từ 'Tổng thời gian thực hiện' hoặc 'Thời gian thực hiện' kèm số ngày/tháng/năm không? Trả lời YES hoặc NO.",
            "start_page": 2,  # Start from page 2
            "skip_last_percent": 0.5,  # Skip last 50% pages
            "skip_verify": True,  # Skip verification - extraction is reliable
            "debug": True  # DEBUG enabled
        },
        {
            "name": "party_a",
            "prompt": "Tìm dòng chứa 'Bên A:' và COPY NGUYÊN VĂN tên công ty/đơn vị phía sau. Ví dụ: 'Bên A: CÔNG TY CỔ PHẦN DỊCH VỤ'. Chỉ trả lời tên đơn vị, không giải thích. Nếu không tìm thấy, trả lời 'N/A'.",
            "format": "party",
            "verify_prompt": "Trong trang này, có cụm từ 'Bên A:' kèm tên công ty/đơn vị không? Trả lời YES hoặc NO.",
            "max_pages": 2,  # Only scan first 2 pages
            "skip_verify": True,  # Skip verification - regex is strict
            "debug": True  # DEBUG enabled
        },
        {
            "name": "party_b",
            "prompt": "Tìm dòng chứa 'Bên B:' và COPY NGUYÊN VĂN tên công ty/đơn vị phía sau. Ví dụ: 'Bên B: CÔNG TY TNHH ABC'. Chỉ trả lời tên đơn vị, không giải thích. Nếu không tìm thấy, trả lời 'N/A'.",
            "format": "party",
            "verify_prompt": "Trong trang này, có cụm từ 'Bên B:' kèm tên công ty/đơn vị không? Trả lời YES hoặc NO.",
            "max_pages": 2,  # Only scan first 2 pages
            "skip_verify": True,  # Skip verification - regex is strict
            "debug": True  # DEBUG enabled
        }
    ]
    
    results = {}
    import re
    import json
    
    print("\n=== EXTRACTING METADATA ===")
    
    # ========== BATCH EXTRACTION: Page 1 Header ==========
    print("  [BATCH] Page 1: contract_number, contract_name...")
    page1_path = os.path.abspath(os.path.join(images_dir, image_files[0]))
    
    # Crop top 50% for better focus on header
    from PIL import Image
    img = Image.open(page1_path)
    w, h = img.size
    cropped = img.crop((0, 0, w, int(h * 0.5)))
    cropped_path = os.path.join(images_dir, "batch_page1_cropped.png")
    cropped.save(cropped_path)
    img.close()
    cropped.close()
    
    batch_prompt_page1 = """Đọc trang hợp đồng này và trả lời theo format sau:
SỐ HỢP ĐỒNG: [chỉ số, ví dụ: 368]
GÓI THẦU: [tên gói thầu nếu có]
HẠNG MỤC: [tên hạng mục nếu có]

Chỉ trả lời đúng format, mỗi thông tin một dòng. Nếu không tìm thấy, ghi N/A."""

    batch_prompt = apply_chat_template(
        VLM_TOKENIZER,
        VLM_MODEL.config,
        batch_prompt_page1,
        num_images=1
    )
    
    batch_result = generate(
        VLM_MODEL,
        VLM_TOKENIZER,
        prompt=batch_prompt,
        image=cropped_path,
        max_tokens=200,
        temp=0.1,
        verbose=False
    )
    
    if hasattr(batch_result, 'text'):
        batch_answer = batch_result.text
    else:
        batch_answer = str(batch_result)
    
    batch_answer = batch_answer.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
    print(f"    [DEBUG] Batch page 1 raw: {batch_answer}")
    
    # Parse batch response
    # Extract contract number
    num_match = re.search(r'SỐ\s*HỢP\s*ĐỒNG[:\s]*(\d+)', batch_answer, re.IGNORECASE)
    if num_match:
        results['contract_number'] = num_match.group(1)
        print(f"    ✓ contract_number: {results['contract_number']} (batch)")
    else:
        results['contract_number'] = "N/A"
    
    # Extract contract name from Gói thầu or Hạng mục
    name_match = re.search(r'(?:GÓI\s*THẦU|HẠNG\s*MỤC)[:\s]*(.+?)(?:\n|$)', batch_answer, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        # Clean quotes
        name = re.sub(r'^[\s"\'""\'\'\`]+|[\s"\'""\'\'\`]+$', '', name)
        if name and name.upper() != 'N/A' and len(name) > 5:
            results['contract_name'] = name
            print(f"    ✓ contract_name: {results['contract_name']} (batch)")
        else:
            results['contract_name'] = "N/A"
    else:
        results['contract_name'] = "N/A"
    
    # Fallback: If contract_name is N/A, try to extract from "HỢP ĐỒNG" line in raw page text
    # This will be handled in the individual field extraction with enhanced regex
    
    # Process remaining fields (not batch-extracted)
    for field in metadata_fields:
        # Skip fields already extracted via batch
        if field['name'] in results and results[field['name']] != "N/A":
            continue
        
        # Calculate page range to scan
        start_page = field.get('start_page', 1)  # Default: start from page 1
        skip_last_percent = field.get('skip_last_percent', 0)  # Default: don't skip
        max_pages = field.get('max_pages')
        
        # Calculate start and end indices
        start_idx = start_page - 1  # Convert to 0-indexed
        if skip_last_percent > 0:
            end_idx = int(total_pages * (1 - skip_last_percent))
        elif max_pages:
            end_idx = min(max_pages, total_pages)
        else:
            end_idx = total_pages
        
        pages_to_scan = image_files[start_idx:end_idx]
        print(f"  Extracting: {field['name']} (pages {start_page}-{end_idx}, {len(pages_to_scan)} page(s))...")
        candidates = []
        
        # PHASE 1: Collect candidates from limited pages
        for page_idx, filename in enumerate(pages_to_scan):
            img_path = os.path.abspath(os.path.join(images_dir, filename))
            page_num = start_idx + page_idx + 1  # Actual page number
            
            # Handle image cropping if specified
            actual_img_path = img_path
            crop_region = field.get('crop_region')
            if crop_region:
                from PIL import Image
                img = Image.open(img_path)
                w, h = img.size
                x1, y1, x2, y2 = crop_region  # Ratios (0-1)
                crop_box = (int(w * x1), int(h * y1), int(w * x2), int(h * y2))
                cropped = img.crop(crop_box)
                # Save to temp file
                cropped_path = os.path.join(images_dir, f"cropped_{filename}")
                cropped.save(cropped_path)
                actual_img_path = cropped_path
                img.close()
                cropped.close()
            
            prompt = apply_chat_template(
                VLM_TOKENIZER,
                VLM_MODEL.config,
                field['prompt'],
                num_images=1
            )
            
            result = generate(
                VLM_MODEL,
                VLM_TOKENIZER,
                prompt=prompt,
                image=actual_img_path,
                max_tokens=100,
                temp=0.1,
                verbose=False
            )
            
            if hasattr(result, 'text'):
                answer = result.text
            else:
                answer = str(result)
            
            answer = answer.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
            
            # DEBUG: Show VLM output if debug is enabled for this field
            if field.get('debug'):
                print(f"      [DEBUG] Page {page_num} raw answer: '{answer}'")
            
            # Check if answer is valid and extract formatted value
            if answer and answer.upper() != "N/A" and len(answer) < 300:
                formatted_answer = None
                
                if field['format'] == 'int':
                    numbers = re.findall(r'\d+', answer.replace(',', '').replace('.', ''))
                    if numbers:
                        formatted_answer = numbers[0]
                elif field['format'] == 'money':
                    # Smart money extraction: find actual monetary patterns
                    # Match patterns like 288,000,000 or 288.000.000 or 179600000
                    money_matches = re.findall(r'\b(\d{1,3}(?:[.,]\d{3})+)\b', answer)
                    if money_matches:
                        # Take the largest number found (most likely the total value)
                        best_value = 0
                        for m in money_matches:
                            raw = re.sub(r'[^\d]', '', m)
                            val = int(raw) if raw else 0
                            if val > best_value:
                                best_value = val
                        if best_value >= 10000:  # At least 10,000
                            formatted_answer = "{:,}".format(best_value)
                    else:
                        # Fallback: look for standalone large numbers
                        num_matches = re.findall(r'\b(\d{5,})\b', answer)
                        if num_matches:
                            best_value = max(int(n) for n in num_matches)
                            if best_value >= 10000:
                                formatted_answer = "{:,}".format(best_value)
                elif field['format'] == 'date':
                    # Try format: ngày X tháng Y năm Z
                    hom_nay_match = re.search(r'[Hh]ôm\s*nay[,\s]*ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})', answer, re.IGNORECASE)
                    if hom_nay_match:
                        day, month, year = hom_nay_match.groups()
                        formatted_answer = f"{day}/{month}/{year}"
                    else:
                        # Try format: DD/MM/YYYY or D/M/YYYY
                        slash_match = re.search(r'ngày\s*(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', answer, re.IGNORECASE)
                        if slash_match:
                            day, month, year = slash_match.groups()
                            formatted_answer = f"{day}/{month}/{year}"
                    
                    # Fallback: "ký kết ngày X tháng Y năm Z"
                    if not formatted_answer:
                        ky_ket_match = re.search(r'ký\s*kết\s*ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})', answer, re.IGNORECASE)
                        if ky_ket_match:
                            day, month, year = ky_ket_match.groups()
                            formatted_answer = f"{day}/{month}/{year}"
                            print(f"      [DEBUG] Fallback: Used 'ký kết ngày' extract: {formatted_answer}")
                elif field['format'] == 'duration':
                    # Match "Tổng thời gian thực hiện hợp đồng" or "Thời gian thực hiện"
                    duration_match = re.search(r'(?:[Tt]ổng\s*)?[Tt]hời\s*gian\s*thực\s*hiện(?:\s*hợp\s*đồng)?[:\s]*(\d+)\s*(?:\([^)]*\))?\s*(ngày|tháng|năm)', answer, re.IGNORECASE)
                    if duration_match:
                        number, unit = duration_match.groups()
                        formatted_answer = f"{number} {unit}"  # Include unit
                elif field['format'] == 'text':
                    # Extract contract name from "Gói thầu:" or "Hạng mục:" line
                    name_match = re.search(r'(?:Gói\s*thầu|Hạng\s*mục)[:\s]*(.+?)$', answer, re.IGNORECASE | re.MULTILINE)
                    if name_match:
                        # Remove all quote types using regex
                        cleaned = re.sub(r'^[\s"\'""\'\'\`]+|[\s"\'""\'\'\`]+$', '', name_match.group(1))
                        if len(cleaned) > 5 and len(cleaned) < 300:
                            formatted_answer = cleaned
                    
                    # Fallback: If not found, extract text after "HỢP ĐỒNG" until "Số:"
                    if not formatted_answer:
                        # Match "HỢP ĐỒNG" followed by text on same/next lines, before "Số:"
                        hd_match = re.search(r'HỢP\s*ĐỒNG\s*(.+?)(?=\n*Số[:\s]|\n*Căn\s*cứ|$)', answer, re.IGNORECASE | re.DOTALL)
                        if hd_match:
                            # Join multiple lines and clean
                            hd_text = hd_match.group(1).strip()
                            # Replace newlines with spaces
                            hd_text = re.sub(r'\s*\n\s*', ' ', hd_text)
                            # Remove quotes
                            hd_text = re.sub(r'^[\s"\'""\'\'\`]+|[\s"\'""\'\'\`]+$', '', hd_text)
                            if len(hd_text) > 5 and len(hd_text) < 300:
                                formatted_answer = hd_text
                                print(f"      [DEBUG] Fallback: Used 'HỢP ĐỒNG' extract: {formatted_answer}")
                elif field['format'] == 'party':
                    # Extract party name - VLM returns company name directly
                    # Try with "Bên A/B:" prefix first
                    party_match = re.search(r'[Bb]ên\s*[AB][:\s]+(.+?)(?:\n|$)', answer, re.IGNORECASE)
                    if party_match:
                        cleaned = party_match.group(1).strip()
                    else:
                        # If no prefix, use the whole answer as party name
                        cleaned = answer.strip()
                    
                    # Remove quotes and extra whitespace
                    cleaned = re.sub(r'^[\s"\'""\'\'\'\`]+|[\s"\'""\'\'\'\`]+$', '', cleaned)
                    if len(cleaned) > 3 and len(cleaned) < 200 and cleaned.upper() != 'N/A':
                        formatted_answer = cleaned
                
                if formatted_answer:
                    candidates.append({
                        'page_num': page_num,
                        'raw_answer': answer,
                        'formatted': formatted_answer,
                        'img_path': img_path
                    })
        
        # PHASE 2: Verify candidates (prioritize earlier pages = main contract)
        # PHASE 2: Verify candidates (or skip verification)
        skip_verify = field.get('skip_verify', False)
        
        if skip_verify and candidates:
            # Skip verification - take first candidate
            best_answer = candidates[0]['formatted']
            found_on_page = candidates[0]['page_num']
            print(f"    Found {len(candidates)} candidates, skipping verification...")
            print(f"    ✓ {field['name']}: {best_answer} (page {found_on_page}, no verify)")
        else:
            print(f"    Found {len(candidates)} candidates, verifying...")
            
            best_answer = "N/A"
            found_on_page = None
            
            for candidate in candidates:
                verify_prompt_text = field['verify_prompt'].replace('{answer}', candidate['formatted'])
                
                verify_prompt = apply_chat_template(
                    VLM_TOKENIZER,
                    VLM_MODEL.config,
                    verify_prompt_text,
                    num_images=1
                )
                
                verify_result = generate(
                    VLM_MODEL,
                    VLM_TOKENIZER,
                    prompt=verify_prompt,
                    image=candidate['img_path'],
                    max_tokens=20,
                    temp=0.1,
                    verbose=False
                )
                
                if hasattr(verify_result, 'text'):
                    verify_answer = verify_result.text
                else:
                    verify_answer = str(verify_result)
                
                verify_answer = verify_answer.replace('<|im_start|>', '').replace('<|im_end|>', '').strip().upper()
                
                # DEBUG: Show verification response
                if field.get('debug'):
                    print(f"      [DEBUG] Verify response: '{verify_answer}'")
                
                is_verified = 'YES' in verify_answer
                
                if is_verified:
                    print(f"      ✓ Page {candidate['page_num']}: VERIFIED - {candidate['formatted']}")
                    best_answer = candidate['formatted']
                    found_on_page = candidate['page_num']
                    break  # Take first verified (earlier page = main contract)
                else:
                    print(f"      ✗ Page {candidate['page_num']}: NOT VERIFIED (hallucination)")
            
            if found_on_page:
                print(f"    ✓ {field['name']}: {best_answer} (verified on page {found_on_page})")
            else:
                print(f"    ✗ {field['name']}: N/A (no verified answer in {total_pages} pages)")
        
        results[field['name']] = best_answer
    
    # ========== FALLBACK: contract_value from appendix (last 30% pages) ==========
    if results.get('contract_value', 'N/A') == 'N/A':
        print(f"  [FALLBACK] Scanning last 30% pages for contract_value (tổng tiền/tổng giá trị)...")
        last_start = int(total_pages * 0.7)  # Last 30%
        fallback_pages = image_files[last_start:]
        
        for page_idx, filename in enumerate(fallback_pages):
            img_path = os.path.abspath(os.path.join(images_dir, filename))
            page_num = last_start + page_idx + 1
            
            fallback_prompt = "Tìm dòng chứa 'TỔNG TIỀN' hoặc 'TỔNG GIÁ TRỊ' hoặc 'TỔNG CỘNG' và COPY NGUYÊN VĂN số tiền. Ví dụ: 'Tổng tiền: 179.600.000 đồng'. Chỉ trả lời dòng chứa giá trị, không giải thích. Nếu không tìm thấy, trả lời 'N/A'."
            
            prompt = apply_chat_template(
                VLM_TOKENIZER,
                VLM_MODEL.config,
                fallback_prompt,
                num_images=1
            )
            
            result = generate(
                VLM_MODEL,
                VLM_TOKENIZER,
                prompt=prompt,
                image=img_path,
                max_tokens=100,
                temp=0.1,
                verbose=False
            )
            
            if hasattr(result, 'text'):
                answer = result.text
            else:
                answer = str(result)
            
            answer = answer.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
            print(f"      [DEBUG] Fallback page {page_num} raw: '{answer}'")
            
            if 'N/A' not in answer.upper() and len(answer) > 5:
                # Smart money extraction: find actual monetary patterns
                import re
                money_matches = re.findall(r'\b(\d{1,3}(?:[.,]\d{3})+)\b', answer)
                if money_matches:
                    best_value = 0
                    for m in money_matches:
                        raw = re.sub(r'[^\d]', '', m)
                        val = int(raw) if raw else 0
                        if val > best_value:
                            best_value = val
                    if best_value >= 10000:
                        results['contract_value'] = "{:,}".format(best_value)
                        print(f"    ✓ contract_value (fallback): {results['contract_value']} (page {page_num})")
                        break
                else:
                    num_matches = re.findall(r'\b(\d{5,})\b', answer)
                    if num_matches:
                        best_value = max(int(n) for n in num_matches)
                        if best_value >= 10000:
                            results['contract_value'] = "{:,}".format(best_value)
                            print(f"    ✓ contract_value (fallback): {results['contract_value']} (page {page_num})")
                            break
    
    # ========== FALLBACK: duration from appendix (last 30% pages) ==========
    if results.get('duration', 'N/A') == 'N/A':
        print(f"  [FALLBACK] Scanning last 30% pages for duration (thời gian thực hiện/duy trì)...")
        last_start = int(total_pages * 0.7)  # Last 30%
        fallback_pages = image_files[last_start:]
        
        for page_idx, filename in enumerate(fallback_pages):
            img_path = os.path.abspath(os.path.join(images_dir, filename))
            page_num = last_start + page_idx + 1
            
            fallback_prompt = "Tìm dòng chứa 'thời gian thực hiện' hoặc 'thời gian duy trì' hoặc 'số năm duy trì' hoặc 'số tháng duy trì' hoặc 'số ngày duy trì' và COPY NGUYÊN VĂN cả dòng đó. Ví dụ: 'Thời gian thực hiện: 12 tháng' hoặc 'Số năm duy trì: 3 năm'. Chỉ trả lời dòng chứa thời gian, không giải thích. Nếu không tìm thấy, trả lời 'N/A'."
            
            prompt = apply_chat_template(
                VLM_TOKENIZER,
                VLM_MODEL.config,
                fallback_prompt,
                num_images=1
            )
            
            result = generate(
                VLM_MODEL,
                VLM_TOKENIZER,
                prompt=prompt,
                image=img_path,
                max_tokens=100,
                temp=0.1,
                verbose=False
            )
            
            if hasattr(result, 'text'):
                answer = result.text
            else:
                answer = str(result)
            
            answer = answer.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
            print(f"      [DEBUG] Fallback page {page_num} raw: '{answer}'")
            
            if 'N/A' not in answer.upper() and len(answer) > 5:
                import re
                # Match duration patterns: thời gian thực hiện, duy trì, etc.
                duration_match = re.search(r'(\d+)\s*(?:\([^)]*\))?\s*(ngày|tháng|năm)', answer, re.IGNORECASE)
                if duration_match:
                    number, unit = duration_match.groups()
                    results['duration'] = f"{number} {unit}"
                    print(f"    ✓ duration (fallback): {results['duration']} (page {page_num})")
                    break
    
    # ========== COMPUTE EXPIRY DATE ==========
    expiry_date = "N/A"
    signing_date = results.get('signing_date', 'N/A')
    duration = results.get('duration', 'N/A')
    
    if signing_date != 'N/A' and duration != 'N/A':
        try:
            from datetime import datetime, timedelta
            import re
            
            # Parse signing date (DD/MM/YYYY)
            date_parts = signing_date.split('/')
            if len(date_parts) == 3:
                day, month, year = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
                start_date = datetime(year, month, day)
                
                # Parse duration (e.g., "365 ngày", "12 tháng", "1 năm")
                duration_match = re.search(r'(\d+)\s*(ngày|tháng|năm)', duration, re.IGNORECASE)
                if duration_match:
                    num = int(duration_match.group(1))
                    unit = duration_match.group(2).lower()
                    
                    if unit == 'ngày':
                        end_date = start_date + timedelta(days=num)
                    elif unit == 'tháng':
                        end_date = start_date + timedelta(days=num * 30)
                    elif unit == 'năm':
                        end_date = start_date + timedelta(days=num * 365)
                    else:
                        end_date = None
                    
                    if end_date:
                        expiry_date = end_date.strftime('%d/%m/%Y')
                        print(f"  ✓ Computed expiry_date: {signing_date} + {duration} = {expiry_date}")
        except Exception as e:
            print(f"  ✗ Could not compute expiry_date: {e}")
    
    results['expiry_date'] = expiry_date
    
    # Save to global for Q&A context
    global EXTRACTED_METADATA
    EXTRACTED_METADATA = results.copy()
    print(f"  [Metadata saved to global context: {list(EXTRACTED_METADATA.keys())}]")
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Calculate extraction time
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    extraction_time = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    print(f"\n⏱️ Metadata extraction completed in {extraction_time}")
    
    return (
        results.get('contract_number', 'N/A'),
        results.get('contract_name', 'N/A'),
        results.get('signing_date', 'N/A'),
        results.get('contract_value', 'N/A'),
        results.get('duration', 'N/A'),
        results.get('party_a', 'N/A'),
        results.get('party_b', 'N/A'),
        results.get('expiry_date', 'N/A'),
        extraction_time
    )


def confirm_metadata(
    contract_number, contract_name, signing_date,
    contract_value, duration, party_a, party_b, expiry_date,
    wrong_contract_number, note_contract_number,
    wrong_contract_name, note_contract_name,
    wrong_signing_date, note_signing_date,
    wrong_contract_value, note_contract_value,
    wrong_duration, note_duration,
    wrong_party_a, note_party_a,
    wrong_party_b, note_party_b,
    wrong_expiry_date, note_expiry_date
):
    """
    Save confirmed metadata to JSON data file for statistics.
    Each field has its own wrong-flag and correction note.
    """
    try:
        # Field definitions for iteration
        field_defs = [
            ("contract_number", contract_number, wrong_contract_number, note_contract_number, "Số hợp đồng"),
            ("contract_name", contract_name, wrong_contract_name, note_contract_name, "Tên hợp đồng"),
            ("signing_date", signing_date, wrong_signing_date, note_signing_date, "Ngày ký"),
            ("contract_value", contract_value, wrong_contract_value, note_contract_value, "Giá trị HĐ"),
            ("duration", duration, wrong_duration, note_duration, "Thời gian TH"),
            ("party_a", party_a, wrong_party_a, note_party_a, "Bên A"),
            ("party_b", party_b, wrong_party_b, note_party_b, "Bên B"),
            ("expiry_date", expiry_date, wrong_expiry_date, note_expiry_date, "Ngày hết hạn"),
        ]
        
        metadata = {}
        corrections = {}
        wrong_field_names = []
        
        for key, value, is_wrong, note, label in field_defs:
            metadata[key] = value
            if is_wrong:
                wrong_field_names.append(label)
                corrections[key] = {
                    "is_wrong": True,
                    "note": note or ""
                }
        
        # Build record
        record = {
            "timestamp": datetime.now().isoformat(),
            "pdf_filename": CURRENT_PDF_FILENAME or "unknown",
            "metadata": metadata,
            "corrections": {
                "has_corrections": len(wrong_field_names) > 0,
                "wrong_fields": wrong_field_names,
                "details": corrections
            }
        }
        
        # Load existing records or create new
        records = []
        if os.path.exists(METADATA_RECORDS_FILE):
            with open(METADATA_RECORDS_FILE, 'r', encoding='utf-8') as f:
                records = json.load(f)
        
        records.append(record)
        
        # Save
        with open(METADATA_RECORDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        # Status message
        total = len(records)
        if wrong_field_names:
            fields_str = ", ".join(wrong_field_names)
            return f"✅ Đã lưu (bản ghi #{total}). ⚠️ Sai: {fields_str}"
        else:
            return f"✅ Đã xác nhận & lưu thành công (bản ghi #{total})."
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Lỗi khi lưu: {str(e)}"



def extract_single_invoice(page_path, page_num):
    """Extract metadata from a single invoice page. Returns 10 fields."""
    import re
    
    batch_prompt = """Đọc hoá đơn này và trả lời theo format sau (mỗi thông tin một dòng):
SỐ HOÁ ĐƠN: [số hoá đơn - nằm ngay dưới dòng "Ký hiệu", chỉ gồm số]
NGÀY HOÁ ĐƠN: [ngày/tháng/năm]
LOẠI XĂNG DẦU: [tên loại xăng/dầu]
SỐ LƯỢNG: [số lít - chỉ gồm số]
ĐƠN GIÁ: [đơn giá/lít - chỉ gồm số]
THÀNH TIỀN TRƯỚC THUẾ: [số tiền - chỉ gồm số]
THUẾ SUẤT: [%]
TIỀN THUẾ: [số tiền thuế - chỉ gồm số]
TỔNG CỘNG: [tổng số tiền - chỉ gồm số]
BIỂN SỐ XE: [biển số xe nếu có]

Lưu ý: SỐ HOÁ ĐƠN nằm ở dòng ngay dưới "Ký hiệu:" và chỉ chứa chữ số.
Các trường số tiền chỉ ghi số, không ghi chữ hay ký tự lạ.
Nếu không tìm thấy, ghi N/A. Chỉ trả lời đúng format."""
    
    prompt = apply_chat_template(VLM_TOKENIZER, VLM_MODEL.config, batch_prompt, num_images=1)
    result = generate(VLM_MODEL, VLM_TOKENIZER, prompt=prompt, image=page_path, max_tokens=500, temp=0.1, verbose=False)
    
    answer = result.text if hasattr(result, 'text') else str(result)
    answer = answer.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
    print(f"  [Page {page_num}] Raw:\n{answer}")
    
    field_patterns = [
        ("invoice_number", r'SỐ\s*HOÁ\s*ĐƠN[:\s]*(.+?)(?:\n|$)'),
        ("invoice_date", r'NGÀY\s*HOÁ\s*ĐƠN[:\s]*(.+?)(?:\n|$)'),
        ("fuel_type", r'LOẠI\s*XĂNG\s*DẦU[:\s]*(.+?)(?:\n|$)'),
        ("quantity", r'SỐ\s*LƯỢNG[:\s]*(.+?)(?:\n|$)'),
        ("unit_price", r'ĐƠN\s*GIÁ[:\s]*(.+?)(?:\n|$)'),
        ("amount_before_tax", r'THÀNH\s*TIỀN\s*TRƯỚC\s*THUẾ[:\s]*(.+?)(?:\n|$)'),
        ("tax_rate", r'THUẾ\s*SUẤT[:\s]*(.+?)(?:\n|$)'),
        ("tax_amount", r'TIỀN\s*THUẾ[:\s]*(.+?)(?:\n|$)'),
        ("total_amount", r'TỔNG\s*CỘNG[:\s]*(.+?)(?:\n|$)'),
        ("vehicle_plate", r'BIỂN\s*SỐ\s*XE[:\s]*(.+?)(?:\n|$)'),
    ]
    
    results = {}
    for key, pattern in field_patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            value = re.sub(r'^[\s"\'\'\'\'`]+|[\s"\'\'\'\'`]+$', '', value)
            # Anti-hallucination: detect repeating patterns
            if len(value) > 80:
                found_repeat = False
                for seg_len in range(15, min(80, len(value)//2)):
                    segment = value[:seg_len]
                    count = value.count(segment)
                    if count >= 2:
                        value = segment.rstrip(' -–,.')
                        print(f"    [anti-repeat] {key}: truncated (repeated {count}x)")
                        found_repeat = True
                        break
                if not found_repeat and len(value) > 200:
                    value = value[:200].rstrip()
            results[key] = value if value and value.upper() != 'N/A' else 'N/A'
        else:
            results[key] = 'N/A'
    
    # Post-processing: validate numeric fields — re-extract if invalid
    reextract_fields = []
    
    # invoice_number must be digits only
    inv_num = results.get('invoice_number', 'N/A')
    if inv_num and inv_num != 'N/A':
        if re.search(r'[^0-9]', inv_num):
            print(f"    [validate] invoice_number '{inv_num}' has non-digits → re-extract")
            reextract_fields.append(('invoice_number', 'SỐ HOÁ ĐƠN (chỉ gồm số, nằm ngay dưới dòng Ký hiệu)'))
    
    # Money/quantity fields: only digits, dots, commas allowed
    money_prompts = {
        'quantity': 'SỐ LƯỢNG (chỉ gồm số)',
        'unit_price': 'ĐƠN GIÁ (chỉ gồm số)',
        'amount_before_tax': 'THÀNH TIỀN TRƯỚC THUẾ (chỉ gồm số)',
        'tax_amount': 'TIỀN THUẾ (chỉ gồm số)',
        'total_amount': 'TỔNG CỘNG (chỉ gồm số)',
    }
    for field, hint in money_prompts.items():
        value = results.get(field, 'N/A')
        if value and value != 'N/A':
            if re.search(r'[^0-9.,\s]', value):
                print(f"    [validate] {field} '{value}' has invalid chars → re-extract")
                reextract_fields.append((field, hint))
    
    # Re-extract failed fields with targeted prompt
    if reextract_fields:
        field_lines = "\n".join([hint for _, hint in reextract_fields])
        retry_prompt_text = f"""Đọc lại hoá đơn này thật kỹ. Chỉ trả lời các trường sau, mỗi trường một dòng:
{field_lines}

CHÚ Ý:
- SỐ HOÁ ĐƠN nằm ở dòng ngay dưới "Ký hiệu:" và CHỈ CHỨA CHỮ SỐ (0-9).
- Các trường số tiền CHỈ CHỨA SỐ, dấu chấm và dấu phẩy. KHÔNG có chữ cái hay ký tự lạ.
- Format: TÊN TRƯỜNG: giá trị"""
        
        print(f"    [re-extract] Retrying {len(reextract_fields)} fields: {[f for f,_ in reextract_fields]}")
        
        retry_prompt = apply_chat_template(VLM_TOKENIZER, VLM_MODEL.config, retry_prompt_text, num_images=1)
        retry_result = generate(VLM_MODEL, VLM_TOKENIZER, prompt=retry_prompt, image=page_path, max_tokens=200, temp=0.05, verbose=False)
        retry_answer = retry_result.text if hasattr(retry_result, 'text') else str(retry_result)
        retry_answer = retry_answer.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
        print(f"    [re-extract] Raw:\n{retry_answer}")
        
        retry_patterns = {
            'invoice_number': r'SỐ\s*HOÁ\s*ĐƠN[^:]*[:\s]*(.+?)(?:\n|$)',
            'quantity': r'SỐ\s*LƯỢNG[^:]*[:\s]*(.+?)(?:\n|$)',
            'unit_price': r'ĐƠN\s*GIÁ[^:]*[:\s]*(.+?)(?:\n|$)',
            'amount_before_tax': r'THÀNH\s*TIỀN[^:]*[:\s]*(.+?)(?:\n|$)',
            'tax_amount': r'TIỀN\s*THUẾ[^:]*[:\s]*(.+?)(?:\n|$)',
            'total_amount': r'TỔNG\s*CỘNG[^:]*[:\s]*(.+?)(?:\n|$)',
        }
        
        for field, _ in reextract_fields:
            pattern = retry_patterns.get(field)
            if pattern:
                match = re.search(pattern, retry_answer, re.IGNORECASE)
                if match:
                    new_val = match.group(1).strip().strip('"\'` ')
                    if field == 'invoice_number':
                        new_clean = re.sub(r'[^0-9]', '', new_val)
                        if new_clean:
                            print(f"    [re-extract] {field}: '{results[field]}' → '{new_clean}' ✅")
                            results[field] = new_clean
                        else:
                            print(f"    [re-extract] {field}: retry also failed")
                    else:
                        if not re.search(r'[^0-9.,\s]', new_val):
                            print(f"    [re-extract] {field}: '{results[field]}' → '{new_val.strip()}' ✅")
                            results[field] = new_val.strip()
                        else:
                            fallback = re.sub(r'[^0-9.,]', '', new_val)
                            if fallback:
                                print(f"    [re-extract] {field}: still noisy, fallback → '{fallback}'")
                                results[field] = fallback
                            else:
                                print(f"    [re-extract] {field}: retry failed")
                else:
                    print(f"    [re-extract] {field}: not found in retry")
    
    # Post-processing: normalize fuel type
    KNOWN_FUELS = {
        "E5 RON 92": ["e5", "ron 92", "e5 ron 92", "xăng e5", "xang e5", "e5ron92", "ron92"],
        "RON 95-III": ["ron 95", "ron95", "xăng 95", "xang 95", "95-iii", "95 iii", "ron 95-iii"],
        "RON 95-IV": ["95-iv", "95 iv", "ron 95-iv"],
        "RON 95-V": ["95-v", "ron 95-v"],
        "DO 0.05S-II": ["diesel", "do ", "do 0.05", "dầu diesel", "d.o", "0.05s", "0,05s"],
        "DO 0.001S-V": ["0.001s", "0,001s"],
        "Dầu hỏa": ["dầu hỏa", "kerosene", "dau hoa"],
        "Dầu mazut": ["mazut", "fo ", "fuel oil"],
    }
    
    fuel = results.get('fuel_type', 'N/A')
    if fuel and fuel != 'N/A':
        fuel_lower = fuel.lower().strip()
        fuel_clean = re.sub(r'^(xăng|xang|dầu|dau)\s*', '', fuel_lower)
        matched = None
        for standard_name, keywords in KNOWN_FUELS.items():
            for kw in keywords:
                if kw in fuel_lower or kw in fuel_clean:
                    matched = standard_name
                    break
            if matched:
                break
        if matched:
            if matched != fuel:
                print(f"    [post] fuel normalized: '{fuel}' → '{matched}'")
                results['fuel_type'] = matched
        else:
            print(f"    [post] ⚠️ unknown fuel type: '{fuel}'")
    
    return results


def check_is_invoice(page_path):
    """Check if a page is a valid invoice by looking for 'HOÁ ĐƠN GIÁ TRỊ GIA TĂNG' in top 1/4."""
    from PIL import Image
    import tempfile
    
    try:
        img = Image.open(page_path)
        w, h = img.size
        # Crop top 1/4
        top_quarter = img.crop((0, 0, w, h // 4))
        
        # Save cropped image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            top_quarter.save(tmp.name)
            crop_path = tmp.name
        
        check_prompt = """Trong ảnh này có tiêu đề "HOÁ ĐƠN GIÁ TRỊ GIA TĂNG" không?
Chỉ trả lời CÓ hoặc KHÔNG."""
        
        prompt = apply_chat_template(VLM_TOKENIZER, VLM_MODEL.config, check_prompt, num_images=1)
        result = generate(VLM_MODEL, VLM_TOKENIZER, prompt=prompt, image=crop_path, max_tokens=20, temp=0.05, verbose=False)
        answer = result.text if hasattr(result, 'text') else str(result)
        answer = answer.replace('<|im_start|>', '').replace('<|im_end|>', '').strip().upper()
        
        # Clean up temp file
        os.remove(crop_path)
        
        is_invoice = 'CÓ' in answer or 'CO' in answer
        print(f"    [title check] '{answer}' → {'✅ Invoice' if is_invoice else '❌ Not invoice'}")
        return is_invoice
        
    except Exception as e:
        print(f"    [title check] Error: {e} → assuming invoice")
        return True  # On error, assume it's an invoice


def extract_all_invoices():
    """Extract metadata from ALL pages (batch processing). Each page = 1 invoice."""
    global ALL_INVOICE_RESULTS, CONFIRMED_INVOICES, INVOICE_PAGE_PATHS
    import time
    start_time = time.time()
    
    ALL_INVOICE_RESULTS = []
    CONFIRMED_INVOICES = set()
    INVOICE_PAGE_PATHS = []
    
    if VLM_MODEL is None:
        return "Model not loaded", 0
    
    images_dir = "temp_pdf_images"
    if not os.path.exists(images_dir):
        return "No PDF", 0
    
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.endswith('.png') and f.startswith('page_')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    
    if not image_files:
        return "No pages", 0
    
    total = len(image_files)
    skipped = 0
    print(f"\n=== BATCH INVOICE EXTRACTION: {total} pages ===")
    
    for i, img_file in enumerate(image_files):
        page_path = os.path.abspath(os.path.join(images_dir, img_file))
        print(f"\n--- Page {i+1}/{total} ---")
        
        # Check if page is a valid invoice
        if not check_is_invoice(page_path):
            print(f"    ⏭️ Skipped page {i+1} (not an invoice)")
            skipped += 1
            continue
        
        result = extract_single_invoice(page_path, i+1)
        ALL_INVOICE_RESULTS.append(result)
        INVOICE_PAGE_PATHS.append(page_path)
    
    # Force garbage collection
    import gc
    gc.collect()
    
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    extraction_time = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    extracted = len(ALL_INVOICE_RESULTS)
    print(f"\n⏱️ Batch completed: {extracted} invoices extracted, {skipped} skipped, in {extraction_time}")
    
    return extraction_time, extracted
def load_invoice_page(page_idx):
    """Load metadata for a specific page. Returns tuple of 10 fields + reset corrections."""
    page_idx = int(page_idx) - 1  # Convert 1-based to 0-based
    
    # Reset values for corrections (10 fields)
    reset_corrections = tuple([False, ""] * 10)
    
    if page_idx < 0 or page_idx >= len(ALL_INVOICE_RESULTS):
        empty = tuple(["N/A"] * 10)
        return (*empty, "", *reset_corrections, "")
    
    r = ALL_INVOICE_RESULTS[page_idx]
    
    # Confirm status for this page
    confirmed = page_idx in CONFIRMED_INVOICES
    confirm_text = f"✅ Đã xác nhận trang {page_idx+1}" if confirmed else ""
    batch_status = f"Đã xác nhận {len(CONFIRMED_INVOICES)}/{len(ALL_INVOICE_RESULTS)} hoá đơn"
    
    return (
        r.get('invoice_number', 'N/A'),
        r.get('invoice_date', 'N/A'),
        r.get('fuel_type', 'N/A'),
        r.get('quantity', 'N/A'),
        r.get('unit_price', 'N/A'),
        r.get('amount_before_tax', 'N/A'),
        r.get('tax_rate', 'N/A'),
        r.get('tax_amount', 'N/A'),
        r.get('total_amount', 'N/A'),
        r.get('vehicle_plate', 'N/A'),
        batch_status,
        *reset_corrections,
        confirm_text
    )


def upload_and_process_invoices(pdf_file):
    """Upload PDF, convert to images, extract all invoice pages. Returns 10 fields."""
    global CURRENT_INVOICE_FILENAME, ALL_INVOICE_RESULTS, CONFIRMED_INVOICES, INVOICE_PAGE_PATHS
    
    ALL_INVOICE_RESULTS = []
    CONFIRMED_INVOICES = set()
    INVOICE_PAGE_PATHS = []
    
    reset_corrections = tuple([False, ""] * 10)
    empty_meta = tuple(["N/A"] * 10)
    
    if pdf_file is not None:
        if hasattr(pdf_file, 'name'):
            CURRENT_INVOICE_FILENAME = os.path.basename(pdf_file.name)
        else:
            CURRENT_INVOICE_FILENAME = os.path.basename(str(pdf_file))
    
    status, gallery = process_pdf(pdf_file)
    
    if "Error" in status or not gallery:
        return (status, gallery, *empty_meta, "0/0", *reset_corrections, "",
                gr.update(maximum=1, value=1, label="Hoá đơn 1/1"))
    
    total_pages = len(gallery)
    extraction_time, count = extract_all_invoices()
    
    # Build gallery with only invoice pages
    invoice_gallery = INVOICE_PAGE_PATHS if INVOICE_PAGE_PATHS else gallery
    
    status = f"✅ Đã trích xuất {count}/{total_pages} hoá đơn ({extraction_time})"
    
    # Load page 1
    if ALL_INVOICE_RESULTS:
        page_data = load_invoice_page(1)
        return (status, invoice_gallery, *page_data,
                gr.update(maximum=count, value=1, label=f"Hoá đơn 1/{count}"))
    else:
        return (status, invoice_gallery, *empty_meta, f"0/{total_pages}", *reset_corrections, "",
                gr.update(maximum=1, value=1, label="Hoá đơn 1/1"))
def confirm_current_invoice(
    page_idx,
    invoice_number, invoice_date, fuel_type, quantity,
    unit_price, amount_before_tax, tax_rate, tax_amount,
    total_amount, vehicle_plate,
    w_invoice_number, n_invoice_number,
    w_invoice_date, n_invoice_date,
    w_fuel_type, n_fuel_type,
    w_quantity, n_quantity,
    w_unit_price, n_unit_price,
    w_amount_before_tax, n_amount_before_tax,
    w_tax_rate, n_tax_rate,
    w_tax_amount, n_tax_amount,
    w_total_amount, n_total_amount,
    w_vehicle_plate, n_vehicle_plate
):
    """Save confirmed invoice metadata for current page to JSON."""
    global CONFIRMED_INVOICES
    try:
        page_idx = int(page_idx)
        
        field_defs = [
            ("invoice_number", invoice_number, w_invoice_number, n_invoice_number, "Số HĐ"),
            ("invoice_date", invoice_date, w_invoice_date, n_invoice_date, "Ngày HĐ"),
            ("fuel_type", fuel_type, w_fuel_type, n_fuel_type, "Loại XD"),
            ("quantity", quantity, w_quantity, n_quantity, "Số lượng"),
            ("unit_price", unit_price, w_unit_price, n_unit_price, "Đơn giá"),
            ("amount_before_tax", amount_before_tax, w_amount_before_tax, n_amount_before_tax, "Trước thuế"),
            ("tax_rate", tax_rate, w_tax_rate, n_tax_rate, "Thuế suất"),
            ("tax_amount", tax_amount, w_tax_amount, n_tax_amount, "Tiền thuế"),
            ("total_amount", total_amount, w_total_amount, n_total_amount, "Tổng cộng"),
            ("vehicle_plate", vehicle_plate, w_vehicle_plate, n_vehicle_plate, "Biển số"),
        ]
        
        metadata = {}
        corrections = {}
        wrong_field_names = []
        
        for key, value, is_wrong, note, label in field_defs:
            metadata[key] = value
            if is_wrong:
                wrong_field_names.append(label)
                corrections[key] = {"is_wrong": True, "note": note or ""}
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "pdf_filename": CURRENT_INVOICE_FILENAME or "unknown",
            "page_number": page_idx,
            "type": "fuel_invoice",
            "metadata": metadata,
            "corrections": {
                "has_corrections": len(wrong_field_names) > 0,
                "wrong_fields": wrong_field_names,
                "details": corrections
            }
        }
        
        records = []
        if os.path.exists(INVOICE_RECORDS_FILE):
            with open(INVOICE_RECORDS_FILE, 'r', encoding='utf-8') as f:
                records = json.load(f)
        
        records.append(record)
        
        with open(INVOICE_RECORDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        # Mark as confirmed
        CONFIRMED_INVOICES.add(page_idx - 1)
        
        total_records = len(records)
        confirmed_count = len(CONFIRMED_INVOICES)
        total_invoices = len(ALL_INVOICE_RESULTS)
        
        status = f"✅ Trang {page_idx} đã lưu (bản ghi #{total_records}). Xác nhận {confirmed_count}/{total_invoices}"
        if wrong_field_names:
            status += f" | ⚠️ Sai: {', '.join(wrong_field_names)}"
        
        # Auto-advance to next page
        next_page = page_idx + 1
        if next_page <= total_invoices:
            next_data = load_invoice_page(next_page)
            # next_data = (10 fields, batch_status, 20 corrections, confirm_text)
            # We return: status, 10 fields, batch_status, 20 corrections, slider_update, gallery_update
            return (
                status,
                *next_data[:-1],  # All except confirm_text (31 values)
                gr.update(value=next_page),  # slider
                gr.update(selected_index=next_page - 1)  # gallery
            )
        else:
            # Already at last page, stay
            reset_corrections = tuple([False, ""] * 10)
            return (
                status + " | 🎉 Đã hoàn thành tất cả!",
                *[gr.update()] * 10,  # keep current fields
                f"✅ Hoàn thành {confirmed_count}/{total_invoices}",
                *reset_corrections,
                gr.update(),  # slider stays
                gr.update()   # gallery stays
            )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        reset_corrections = tuple([False, ""] * 10)
        return (
            f"❌ Lỗi khi lưu: {str(e)}",
            *[gr.update()] * 10,
            gr.update(),
            *reset_corrections,
            gr.update(),
            gr.update()
        )
def generate_dashboard():
    """Generate extraction accuracy dashboard from invoice_records.json."""
    from collections import Counter
    
    if not os.path.exists(INVOICE_RECORDS_FILE):
        return "⚠️ Chưa có dữ liệu. Hãy xác nhận hoá đơn trước.", "", ""
    
    try:
        with open(INVOICE_RECORDS_FILE, 'r', encoding='utf-8') as f:
            records = json.load(f)
    except Exception:
        return "❌ Lỗi đọc file dữ liệu.", "", ""
    
    if not records:
        return "⚠️ Chưa có bản ghi nào.", "", ""
    
    total = len(records)
    correct = sum(1 for r in records if not r['corrections']['has_corrections'])
    wrong = total - correct
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    # Per-field error count
    field_labels = {
        'Số HĐ': 'invoice_number', 'Ngày HĐ': 'invoice_date',
        'Loại XD': 'fuel_type', 'Số lượng': 'quantity',
        'Đơn giá': 'unit_price', 'Trước thuế': 'amount_before_tax',
        'Thuế suất': 'tax_rate', 'Tiền thuế': 'tax_amount',
        'Tổng cộng': 'total_amount', 'Biển số': 'vehicle_plate'
    }
    
    field_errors = Counter()
    for r in records:
        for field in r['corrections']['wrong_fields']:
            field_errors[field] += 1
    
    # Overview markdown
    bar_correct = '🟩' * int(accuracy / 5)
    bar_wrong = '🟥' * (20 - int(accuracy / 5))
    overview = f"""## 📊 Tổng quan trích xuất hoá đơn

| Chỉ số | Giá trị |
|--------|--------|
| 📄 Tổng hoá đơn | **{total}** |
| ✅ Đúng hoàn toàn | **{correct}** ({accuracy:.1f}%) |
| ⚠️ Có lỗi | **{wrong}** ({100-accuracy:.1f}%) |
| 📊 Tổng lỗi | **{sum(field_errors.values())}** trường sai |

### Độ chính xác tổng thể
{bar_correct}{bar_wrong} **{accuracy:.1f}%**
"""
    
    # Per-field accuracy table
    field_table = "## 📋 Độ chính xác theo trường\n\n"
    field_table += "| Trường | Số lỗi | Tỷ lệ đúng | Biểu đồ |\n"
    field_table += "|--------|--------|-----------|---------|\n"
    
    for label in field_labels:
        errors = field_errors.get(label, 0)
        field_acc = ((total - errors) / total) * 100
        bar_len = int(field_acc / 10)
        bar = '█' * bar_len + '░' * (10 - bar_len)
        icon = '✅' if errors == 0 else ('⚠️' if errors <= 5 else '❌')
        field_table += f"| {icon} {label} | {errors}/{total} | {field_acc:.0f}% | `{bar}` |\n"
    
    # Most common errors detail
    error_detail = "## 🔍 Chi tiết lỗi thường gặp\n\n"
    if field_errors:
        for label, count in field_errors.most_common():
            pct = (count / total) * 100
            error_detail += f"### {label} — {count} lỗi ({pct:.0f}%)\n"
            # Show examples
            examples = []
            for r in records:
                if label in r['corrections']['wrong_fields']:
                    detail = r['corrections']['details']
                    field_key = field_labels.get(label, '')
                    if field_key in detail:
                        extracted = r['metadata'].get(field_key, 'N/A')
                        note = detail[field_key].get('note', '')
                        if note:
                            examples.append(f"  - `{extracted}` → `{note}` (trang {r.get('page_number', '?')})")
                    if len(examples) >= 3:
                        break
            if examples:
                error_detail += "\n".join(examples) + "\n\n"
            else:
                error_detail += "\n"
    else:
        error_detail += "🎉 Không có lỗi nào!\n"
    
    return overview, field_table, error_detail


# Initialize UI
with gr.Blocks(title="Local Vision RAG (MacOS M5)", css="""
    .gallery-item { cursor: pointer; }
    .pdf-preview-gallery { min-height: 500px; }
    .metadata-section { padding: 8px 0; }
    #invoice-meta-col .gr-group { gap: 2px !important; }
    #invoice-meta-col .gr-block { padding: 4px 8px !important; min-height: 0 !important; }
    #invoice-meta-col .gr-box { padding: 2px !important; min-height: 0 !important; }
    #invoice-meta-col .row { gap: 4px !important; }
    #invoice-meta-col textarea, #invoice-meta-col input { padding: 4px 6px !important; font-size: 13px !important; }
    #invoice-meta-col label { font-size: 11px !important; margin-bottom: 0 !important; }
""") as demo:
    gr.Markdown("# 👁️ Local Vision RAG Demo\n**Generation:** MLX-VLM (Qwen3-VL-30B-A3B-3bit) | Upload PDF để tự động trích xuất metadata")
    
    with gr.Tabs():
        # ==================== TAB 1: HỢP ĐỒNG ====================
        with gr.Tab("📄 Hợp đồng"):
            with gr.Row(equal_height=False):
                # LEFT: Upload + PDF Preview
                with gr.Column(scale=3):
                    pdf_input = gr.File(label="📤 Upload PDF Hợp đồng", file_types=[".pdf"])
                    status_output = gr.Textbox(label="Trạng thái", interactive=False)
                    gr.Markdown("### 📄 Xem trước tài liệu")
                    pdf_preview_gallery = gr.Gallery(
                        label="PDF Preview",
                        show_label=False,
                        columns=2,
                        height=600,
                        allow_preview=True,
                        preview=True,
                        elem_classes=["pdf-preview-gallery"]
                    )
                
                # RIGHT: Contract Metadata
                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Metadata (tự động trích xuất)")
                    extract_btn = gr.Button("🔍 Trích xuất lại Metadata", variant="secondary", size="sm")
                    
                    with gr.Group():
                        with gr.Row():
                            contract_number_output = gr.Textbox(label="Số hợp đồng", interactive=False, scale=3)
                            wrong_contract_number = gr.Checkbox(label="Sai?", value=False, scale=1)
                            note_contract_number = gr.Textbox(label="Ghi chú", placeholder="Giá trị đúng...", interactive=True, scale=2)
                        
                        with gr.Row():
                            contract_name_output = gr.Textbox(label="Tên HĐ / Gói thầu", interactive=False, scale=3)
                            wrong_contract_name = gr.Checkbox(label="Sai?", value=False, scale=1)
                            note_contract_name = gr.Textbox(label="Ghi chú", placeholder="Giá trị đúng...", interactive=True, scale=2)
                        
                        with gr.Row():
                            signing_date_output = gr.Textbox(label="Ngày ký", interactive=False, scale=3)
                            wrong_signing_date = gr.Checkbox(label="Sai?", value=False, scale=1)
                            note_signing_date = gr.Textbox(label="Ghi chú", placeholder="Giá trị đúng...", interactive=True, scale=2)
                        
                        with gr.Row():
                            contract_value_output = gr.Textbox(label="Giá trị HĐ (VNĐ)", interactive=False, scale=3)
                            wrong_contract_value = gr.Checkbox(label="Sai?", value=False, scale=1)
                            note_contract_value = gr.Textbox(label="Ghi chú", placeholder="Giá trị đúng...", interactive=True, scale=2)
                        
                        with gr.Row():
                            duration_output = gr.Textbox(label="Thời gian TH", interactive=False, scale=3)
                            wrong_duration = gr.Checkbox(label="Sai?", value=False, scale=1)
                            note_duration = gr.Textbox(label="Ghi chú", placeholder="Giá trị đúng...", interactive=True, scale=2)
                        
                        with gr.Row():
                            party_a_output = gr.Textbox(label="Bên A", interactive=False, scale=3)
                            wrong_party_a = gr.Checkbox(label="Sai?", value=False, scale=1)
                            note_party_a = gr.Textbox(label="Ghi chú", placeholder="Giá trị đúng...", interactive=True, scale=2)
                        
                        with gr.Row():
                            party_b_output = gr.Textbox(label="Bên B", interactive=False, scale=3)
                            wrong_party_b = gr.Checkbox(label="Sai?", value=False, scale=1)
                            note_party_b = gr.Textbox(label="Ghi chú", placeholder="Giá trị đúng...", interactive=True, scale=2)
                        
                        with gr.Row():
                            expiry_date_output = gr.Textbox(label="Ngày hết hạn", interactive=False, scale=3)
                            wrong_expiry_date = gr.Checkbox(label="Sai?", value=False, scale=1)
                            note_expiry_date = gr.Textbox(label="Ghi chú", placeholder="Giá trị đúng...", interactive=True, scale=2)
                        
                        extraction_time_output = gr.Textbox(label="⏱️ Thời gian trích xuất", interactive=False)
                    
                    confirm_btn = gr.Button("✅ Xác nhận & Lưu Metadata", variant="primary")
                    confirm_status_output = gr.Textbox(label="💾 Trạng thái lưu", interactive=False)
        
        # ==================== TAB 2: HOÁ ĐƠN XĂNG DẦU ====================
        with gr.Tab("⛽ Hoá đơn xăng dầu"):
            with gr.Row(equal_height=False):
                # LEFT: Upload + Preview
                with gr.Column(scale=3):
                    inv_pdf_input = gr.File(label="📤 Upload PDF Hoá đơn (nhiều trang)", file_types=[".pdf", ".jpg", ".jpeg", ".png"])
                    inv_status_output = gr.Textbox(label="Trạng thái", interactive=False)
                    gr.Markdown("### 📄 Xem trước hoá đơn")
                    inv_preview_gallery = gr.Gallery(
                        label="Invoice Preview",
                        show_label=False,
                        columns=2,
                        height=600,
                        allow_preview=True,
                        preview=True,
                        elem_classes=["pdf-preview-gallery"]
                    )
                
                # RIGHT: Invoice Metadata
                with gr.Column(scale=2, elem_id="invoice-meta-col"):
                    gr.Markdown("### ⛽ Metadata Hoá đơn")
                    
                    with gr.Row():
                        inv_page_slider = gr.Slider(minimum=1, maximum=1, step=1, value=1, label="Hoá đơn 1/1", scale=3)
                        inv_batch_status = gr.Textbox(label="Tiến độ", interactive=False, scale=2)
                    
                    with gr.Group():
                        with gr.Row():
                            inv_number = gr.Textbox(label="Số hoá đơn", interactive=False, scale=3, lines=1, max_lines=1)
                            w_inv_number = gr.Checkbox(label="Sai?", value=False, scale=1)
                            n_inv_number = gr.Textbox(label="Sửa", placeholder="Giá trị đúng...", interactive=True, scale=2, lines=1, max_lines=1)
                        with gr.Row():
                            inv_date = gr.Textbox(label="Ngày hoá đơn", interactive=False, scale=3, lines=1, max_lines=1)
                            w_inv_date = gr.Checkbox(label="Sai?", value=False, scale=1)
                            n_inv_date = gr.Textbox(label="Sửa", placeholder="Giá trị đúng...", interactive=True, scale=2, lines=1, max_lines=1)
                        with gr.Row():
                            inv_fuel_type = gr.Textbox(label="Loại xăng/dầu", interactive=False, scale=3, lines=1, max_lines=1)
                            w_fuel_type = gr.Checkbox(label="Sai?", value=False, scale=1)
                            n_fuel_type = gr.Textbox(label="Sửa", placeholder="Giá trị đúng...", interactive=True, scale=2, lines=1, max_lines=1)
                        with gr.Row():
                            inv_quantity = gr.Textbox(label="Số lượng (lít)", interactive=False, scale=3, lines=1, max_lines=1)
                            w_quantity = gr.Checkbox(label="Sai?", value=False, scale=1)
                            n_quantity = gr.Textbox(label="Sửa", placeholder="Giá trị đúng...", interactive=True, scale=2, lines=1, max_lines=1)
                        with gr.Row():
                            inv_unit_price = gr.Textbox(label="Đơn giá", interactive=False, scale=3, lines=1, max_lines=1)
                            w_unit_price = gr.Checkbox(label="Sai?", value=False, scale=1)
                            n_unit_price = gr.Textbox(label="Sửa", placeholder="Giá trị đúng...", interactive=True, scale=2, lines=1, max_lines=1)
                        with gr.Row():
                            inv_amount_before_tax = gr.Textbox(label="Trước thuế", interactive=False, scale=3, lines=1, max_lines=1)
                            w_amount_before_tax = gr.Checkbox(label="Sai?", value=False, scale=1)
                            n_amount_before_tax = gr.Textbox(label="Sửa", placeholder="Giá trị đúng...", interactive=True, scale=2, lines=1, max_lines=1)
                        with gr.Row():
                            inv_tax_rate = gr.Textbox(label="Thuế suất (%)", interactive=False, scale=3, lines=1, max_lines=1)
                            w_tax_rate = gr.Checkbox(label="Sai?", value=False, scale=1)
                            n_tax_rate = gr.Textbox(label="Sửa", placeholder="Giá trị đúng...", interactive=True, scale=2, lines=1, max_lines=1)
                        with gr.Row():
                            inv_tax_amount = gr.Textbox(label="Tiền thuế", interactive=False, scale=3, lines=1, max_lines=1)
                            w_tax_amount = gr.Checkbox(label="Sai?", value=False, scale=1)
                            n_tax_amount = gr.Textbox(label="Sửa", placeholder="Giá trị đúng...", interactive=True, scale=2, lines=1, max_lines=1)
                        with gr.Row():
                            inv_total_amount = gr.Textbox(label="Tổng cộng", interactive=False, scale=3, lines=1, max_lines=1)
                            w_total_amount = gr.Checkbox(label="Sai?", value=False, scale=1)
                            n_total_amount = gr.Textbox(label="Sửa", placeholder="Giá trị đúng...", interactive=True, scale=2, lines=1, max_lines=1)
                        with gr.Row():
                            inv_vehicle_plate = gr.Textbox(label="Biển số xe", interactive=False, scale=3, lines=1, max_lines=1)
                            w_vehicle_plate = gr.Checkbox(label="Sai?", value=False, scale=1)
                            n_vehicle_plate = gr.Textbox(label="Sửa", placeholder="Giá trị đúng...", interactive=True, scale=2, lines=1, max_lines=1)
                    
                    
                    inv_confirm_btn = gr.Button("✅ Xác nhận & Lưu Hoá đơn này", variant="primary")
                    inv_confirm_status = gr.Textbox(label="💾 Trạng thái lưu", interactive=False)
        
        # ==================== TAB 3: DASHBOARD ====================
        with gr.Tab("📊 Dashboard"):
            dashboard_refresh_btn = gr.Button("🔄 Cập nhật Dashboard", variant="primary")
            with gr.Row():
                with gr.Column(scale=1):
                    dashboard_overview = gr.Markdown(value="_Nhấn 'Cập nhật Dashboard' để xem thống kê_")
                with gr.Column(scale=1):
                    dashboard_fields = gr.Markdown(value="")
            dashboard_errors = gr.Markdown(value="")
    
    gr.Markdown("---")
    
    # ===== Q&A (full width, outside tabs) =====
    with gr.Row():
        query_input = gr.Textbox(label="Đặt câu hỏi về tài liệu", placeholder="Ví dụ: ngày ký hợp đồng? giá trị bảo lãnh?", scale=3)
        submit_btn = gr.Button("Trả lời", variant="primary", scale=1)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Trang liên quan (click để xem lại từ trang đó)")
            retrieved_gallery = gr.Gallery(
                label="Retrieved Pages",
                show_label=False,
                columns=3,
                height="auto",
                allow_preview=True,
                preview=True
            )
            selected_page = gr.Number(label="Selected Page Index", visible=False)
            regenerate_btn = gr.Button("🔄 Trả lời lại từ trang đã chọn", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 💬 Câu trả lời")
            answer_output = gr.Markdown(label="Answer")

    # ===== EVENT HANDLERS: CONTRACT TAB =====
    
    pdf_input.change(
        fn=upload_and_process,
        inputs=[pdf_input],
        outputs=[
            status_output, pdf_preview_gallery,
            contract_number_output, contract_name_output,
            signing_date_output, contract_value_output,
            duration_output, party_a_output, party_b_output,
            expiry_date_output, extraction_time_output,
            wrong_contract_number, note_contract_number,
            wrong_contract_name, note_contract_name,
            wrong_signing_date, note_signing_date,
            wrong_contract_value, note_contract_value,
            wrong_duration, note_duration,
            wrong_party_a, note_party_a,
            wrong_party_b, note_party_b,
            wrong_expiry_date, note_expiry_date,
            confirm_status_output
        ]
    )
    
    extract_btn.click(
        fn=extract_metadata,
        inputs=[],
        outputs=[
            contract_number_output, contract_name_output,
            signing_date_output, contract_value_output,
            duration_output, party_a_output, party_b_output,
            expiry_date_output, extraction_time_output
        ]
    )
    
    confirm_btn.click(
        fn=confirm_metadata,
        inputs=[
            contract_number_output, contract_name_output,
            signing_date_output, contract_value_output,
            duration_output, party_a_output, party_b_output,
            expiry_date_output,
            wrong_contract_number, note_contract_number,
            wrong_contract_name, note_contract_name,
            wrong_signing_date, note_signing_date,
            wrong_contract_value, note_contract_value,
            wrong_duration, note_duration,
            wrong_party_a, note_party_a,
            wrong_party_b, note_party_b,
            wrong_expiry_date, note_expiry_date
        ],
        outputs=[confirm_status_output]
    )
    
    # ===== EVENT HANDLERS: INVOICE TAB =====
    
    # Upload PDF → batch extract all invoices
    inv_pdf_input.change(
        fn=upload_and_process_invoices,
        inputs=[inv_pdf_input],
        outputs=[
            inv_status_output, inv_preview_gallery,
            inv_number, inv_date, inv_fuel_type, inv_quantity,
            inv_unit_price, inv_amount_before_tax, inv_tax_rate,
            inv_tax_amount, inv_total_amount, inv_vehicle_plate,
            inv_batch_status,
            w_inv_number, n_inv_number,
            w_inv_date, n_inv_date,
            w_fuel_type, n_fuel_type,
            w_quantity, n_quantity,
            w_unit_price, n_unit_price,
            w_amount_before_tax, n_amount_before_tax,
            w_tax_rate, n_tax_rate,
            w_tax_amount, n_tax_amount,
            w_total_amount, n_total_amount,
            w_vehicle_plate, n_vehicle_plate,
            inv_confirm_status,
            inv_page_slider
        ]
    )
    
    # Page slider → load that page's metadata
    inv_page_slider.change(
        fn=load_invoice_page,
        inputs=[inv_page_slider],
        outputs=[
            inv_number, inv_date, inv_fuel_type, inv_quantity,
            inv_unit_price, inv_amount_before_tax, inv_tax_rate,
            inv_tax_amount, inv_total_amount, inv_vehicle_plate,
            inv_batch_status,
            w_inv_number, n_inv_number,
            w_inv_date, n_inv_date,
            w_fuel_type, n_fuel_type,
            w_quantity, n_quantity,
            w_unit_price, n_unit_price,
            w_amount_before_tax, n_amount_before_tax,
            w_tax_rate, n_tax_rate,
            w_tax_amount, n_tax_amount,
            w_total_amount, n_total_amount,
            w_vehicle_plate, n_vehicle_plate,
            inv_confirm_status
        ]
    )
    
    # Confirm current page → auto-advance to next
    inv_confirm_btn.click(
        fn=confirm_current_invoice,
        inputs=[
            inv_page_slider,
            inv_number, inv_date, inv_fuel_type, inv_quantity,
            inv_unit_price, inv_amount_before_tax, inv_tax_rate,
            inv_tax_amount, inv_total_amount, inv_vehicle_plate,
            w_inv_number, n_inv_number,
            w_inv_date, n_inv_date,
            w_fuel_type, n_fuel_type,
            w_quantity, n_quantity,
            w_unit_price, n_unit_price,
            w_amount_before_tax, n_amount_before_tax,
            w_tax_rate, n_tax_rate,
            w_tax_amount, n_tax_amount,
            w_total_amount, n_total_amount,
            w_vehicle_plate, n_vehicle_plate
        ],
        outputs=[
            inv_confirm_status,
            inv_number, inv_date, inv_fuel_type, inv_quantity,
            inv_unit_price, inv_amount_before_tax, inv_tax_rate,
            inv_tax_amount, inv_total_amount, inv_vehicle_plate,
            inv_batch_status,
            w_inv_number, n_inv_number,
            w_inv_date, n_inv_date,
            w_fuel_type, n_fuel_type,
            w_quantity, n_quantity,
            w_unit_price, n_unit_price,
            w_amount_before_tax, n_amount_before_tax,
            w_tax_rate, n_tax_rate,
            w_tax_amount, n_tax_amount,
            w_total_amount, n_total_amount,
            w_vehicle_plate, n_vehicle_plate,
            inv_page_slider,
            inv_preview_gallery
        ]
    )
    
    # ===== EVENT HANDLERS: DASHBOARD TAB =====
    dashboard_refresh_btn.click(
        fn=generate_dashboard,
        inputs=[],
        outputs=[dashboard_overview, dashboard_fields, dashboard_errors]
    )
    
    # ===== Q&A HANDLERS =====
    submit_btn.click(
        fn=rag_response,
        inputs=[query_input],
        outputs=[retrieved_gallery, answer_output]
    )
    
    def on_gallery_select(evt: gr.SelectData):
        return evt.index
    
    retrieved_gallery.select(
        fn=on_gallery_select,
        outputs=[selected_page]
    )
    
    regenerate_btn.click(
        fn=rag_response_with_page,
        inputs=[query_input, selected_page],
        outputs=[answer_output]
    )

if __name__ == "__main__":
    load_models()
    demo.launch(server_name="0.0.0.0", server_port=7860)
