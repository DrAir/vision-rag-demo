import gradio as gr
import os
import shutil
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from pdf2image import convert_from_path
from PIL import Image

# Global Model Variables
VLM_MODEL = None
VLM_TOKENIZER = None

# Global Metadata Storage (for context in Q&A)
EXTRACTED_METADATA = {}

def load_models():
    """Loads the VLM model for All-Page Scan."""
    global VLM_MODEL, VLM_TOKENIZER
    
    print("Loading MLX VLM Model (Qwen3-VL-8B)...")
    VLM_MODEL, VLM_TOKENIZER = load("mlx-community/Qwen3-VL-8B-Instruct-4bit")
    print("Model loaded successfully.")

def process_pdf(pdf_file):
    """
    Converts PDF to images for All-Page Scan.
    """
    if pdf_file is None:
        return "No PDF uploaded."
    
    if VLM_MODEL is None:
        return "Model not loaded yet. Please wait."
    
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
            
            # MEMORY OPTIMIZATION: Explicitly release image memory
            image.close()
            del image
        
        # Clear the list to free memory
        del images
        import gc
        gc.collect()
        
        return f"✅ Converted {len(os.listdir(images_dir))} pages (DPI={TARGET_DPI}). Ready for All-Page Scan."
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing PDF: {str(e)}"

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
    Returns: contract_number (int), signing_date (str), contract_value (int), duration (str)
    """
    if VLM_MODEL is None:
        return "N/A", "N/A", "N/A", "N/A"
    
    images_dir = "temp_pdf_images"
    if not os.path.exists(images_dir):
        return "N/A", "N/A", "N/A", "N/A"
    
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
            "verify_prompt": "Trong trang này, có cụm từ 'Hôm nay, ngày' hoặc 'ký kết ngày' theo sau là ngày tháng năm không? Trả lời YES hoặc NO.",
            "max_pages": 2,  # Only scan first 2 pages
            "skip_verify": False,  # Keep verification
            "debug": True  # DEBUG enabled
        },
        {
            "name": "contract_value",
            "prompt": "Tìm dòng chứa 'GIÁ TRỊ HỢP ĐỒNG' hoặc 'TỔNG GIÁ TRỊ' và COPY NGUYÊN VĂN số tiền. Ví dụ: 'Giá trị hợp đồng: 179.600.000 đồng'. Chỉ trả lời dòng chứa giá trị, không giải thích. Nếu không tìm thấy, trả lời 'N/A'.",
            "format": "money",
            "verify_prompt": "Trong trang này, có hiển thị GIÁ TRỊ HỢP ĐỒNG hoặc TỔNG GIÁ TRỊ kèm số tiền không? Trả lời YES hoặc NO.",
            "start_page": 2,  # Start from page 2
            "skip_last_percent": 0.5,  # Skip last 50% pages
            "skip_verify": True,  # Skip verification - money regex is strict
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
                    raw_number = re.sub(r'[^\d]', '', answer)
                    if raw_number and len(raw_number) >= 5:
                        formatted_answer = "{:,}".format(int(raw_number))
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
                # Parse money value
                import re
                raw_number = re.sub(r'[^\d]', '', answer)
                if raw_number and len(raw_number) >= 5:
                    results['contract_value'] = "{:,}".format(int(raw_number))
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
    
    return (
        results.get('contract_number', 'N/A'),
        results.get('contract_name', 'N/A'),
        results.get('signing_date', 'N/A'),
        results.get('contract_value', 'N/A'),
        results.get('duration', 'N/A'),
        results.get('party_a', 'N/A'),
        results.get('party_b', 'N/A'),
        results.get('expiry_date', 'N/A')
    )


# Initialize UI
with gr.Blocks(title="Local Vision RAG (MacOS M5)", css="""
    .gallery-item { cursor: pointer; }
""") as demo:
    gr.Markdown("# 👁️ Local Vision RAG Demo\n**Retrieval:** Byaldi (ColPali) | **Generation:** MLX-VLM (Qwen3-VL)")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF Contract", file_types=[".pdf"])
            process_btn = gr.Button("Index PDF", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)
    
    # Metadata Extraction Section
    gr.Markdown("---")
    gr.Markdown("### 📋 Metadata Extraction")
    with gr.Row():
        extract_btn = gr.Button("🔍 Extract Metadata", variant="secondary")
    with gr.Row():
        contract_number_output = gr.Textbox(label="Số hợp đồng", interactive=False, scale=1)
        contract_name_output = gr.Textbox(label="Tên hợp đồng", interactive=False, scale=2)
        signing_date_output = gr.Textbox(label="Ngày ký hợp đồng", interactive=False, scale=1)
        contract_value_output = gr.Textbox(label="Giá trị hợp đồng (VNĐ)", interactive=False, scale=1)
        duration_output = gr.Textbox(label="Thời gian thực hiện", interactive=False, scale=1)
    with gr.Row():
        party_a_output = gr.Textbox(label="Bên A", interactive=False, scale=1)
        party_b_output = gr.Textbox(label="Bên B", interactive=False, scale=1)
        expiry_date_output = gr.Textbox(label="Ngày hết hạn", interactive=False, scale=1)
    
    gr.Markdown("---")
    
    with gr.Row():
        query_input = gr.Textbox(label="Ask a question about the document", placeholder="e.g., ngày ký hợp đồng?", scale=3)
        submit_btn = gr.Button("Get Answer", variant="primary", scale=1)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Retrieved Pages (click to regenerate from that page)")
            retrieved_gallery = gr.Gallery(
                label="Retrieved Pages",
                show_label=False,
                columns=3,
                height="auto",
                allow_preview=True,
                preview=True
            )
            selected_page = gr.Number(label="Selected Page Index", visible=False)
            regenerate_btn = gr.Button("🔄 Regenerate from Selected Page", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 💬 Answer")
            answer_output = gr.Markdown(label="Answer")

    # Event Handlers
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[status_output]
    )
    
    submit_btn.click(
        fn=rag_response,
        inputs=[query_input],
        outputs=[retrieved_gallery, answer_output]
    )
    
    # Gallery selection handler
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
    
    # Metadata extraction handler
    extract_btn.click(
        fn=extract_metadata,
        inputs=[],
        outputs=[contract_number_output, contract_name_output, signing_date_output, contract_value_output, duration_output, party_a_output, party_b_output, expiry_date_output]
    )

if __name__ == "__main__":
    load_models()
    demo.launch(server_name="0.0.0.0", server_port=7860)
