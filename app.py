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

def load_models():
    """Loads the VLM model for All-Page Scan."""
    global VLM_MODEL, VLM_TOKENIZER
    
    print("Loading MLX VLM Model (Qwen2-VL-7B)...")
    VLM_MODEL, VLM_TOKENIZER = load("mlx-community/Qwen2-VL-7B-Instruct-4bit")
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
        
        # Get all page images
        image_files = sorted(
            [f for f in os.listdir(images_dir) if f.endswith('.png')],
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
            
            # Ask VLM to check if this page has the answer
            check_prompt = f"""Nhìn vào trang tài liệu này và trả lời câu hỏi sau. 
Nếu trang này KHÔNG chứa thông tin liên quan, hãy trả lời chính xác: "KHÔNG CÓ THÔNG TIN"
Nếu trang này CÓ thông tin liên quan, hãy trả lời chi tiết.

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
        # Get all image files sorted by page number
        image_files = sorted(
            [f for f in os.listdir("temp_pdf_images") if f.endswith('.png')], 
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
    
    # ALL-PAGE SCAN: Get ALL pages for thorough metadata extraction
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.endswith('.png')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    
    if not image_files:
        return "N/A", "N/A", "N/A", "N/A"
    
    total_pages = len(image_files)
    print(f"\n=== METADATA EXTRACTION (ALL-PAGE SCAN + VERIFICATION: {total_pages} pages) ===")
    
    # Define metadata fields to extract with verification prompts
    # max_pages: limit how many pages to scan (None = all pages)
    # debug: show raw VLM output
    metadata_fields = [
        {
            "name": "contract_number",
            "prompt": "Tìm và trả lời chính xác SỐ HỢP ĐỒNG trong tài liệu này. Thường có dạng 'Số: XXX/...' hoặc 'HỢP ĐỒNG SỐ: XXX'. Chỉ trả lời số, không giải thích. Nếu không tìm thấy, trả lời 'N/A'.",
            "format": "int",
            "verify_prompt": "Trong trang này, có hiển thị SỐ HỢP ĐỒNG '{answer}' không? Trả lời YES hoặc NO.",
            "max_pages": 1,  # Only scan page 1 (header)
            "debug": False
        },
        {
            "name": "contract_name",
            "prompt": "Đọc TIÊU ĐỀ của hợp đồng này. Tiêu đề thường nằm ngay sau dòng 'HỢP ĐỒNG' và trước dòng 'Căn cứ'. Ví dụ: 'Thuê dịch vụ Hệ thống đánh giá chất lượng dịch vụ'. COPY NGUYÊN VĂN tiêu đề, không giải thích. Nếu không tìm thấy, trả lời 'N/A'.",
            "format": "text",
            "verify_prompt": "Trong trang này, có dòng TIÊU ĐỀ HỢP ĐỒNG chứa '{answer}' không? Trả lời YES hoặc NO.",
            "max_pages": 1,  # Only scan page 1 (header)
            "debug": True  # DEBUG enabled
        },
        {
            "name": "signing_date", 
            "prompt": "Tìm dòng chứa 'Hôm nay, ngày' trong tài liệu và COPY NGUYÊN VĂN cả dòng đó. Ví dụ: 'Hôm nay, ngày 28 tháng 10 năm 2025'. Chỉ trả lời dòng chứa 'Hôm nay', không giải thích. Nếu không tìm thấy, trả lời 'N/A'.",
            "format": "date",
            "verify_prompt": "Trong trang này, có cụm từ 'Hôm nay, ngày' theo sau là ngày tháng năm không? Trả lời YES hoặc NO.",
            "max_pages": 2,  # Only scan first 2 pages
            "debug": False
        },
        {
            "name": "contract_value",
            "prompt": "Tìm dòng chứa 'GIÁ TRỊ HỢP ĐỒNG' hoặc 'TỔNG GIÁ TRỊ' và COPY NGUYÊN VĂN số tiền. Ví dụ: 'Giá trị hợp đồng: 179.600.000 đồng'. Chỉ trả lời dòng chứa giá trị, không giải thích. Nếu không tìm thấy, trả lời 'N/A'.",
            "format": "money",
            "verify_prompt": "Trong trang này, có hiển thị GIÁ TRỊ HỢP ĐỒNG hoặc TỔNG GIÁ TRỊ kèm số tiền không? Trả lời YES hoặc NO.",
            "start_page": 2,  # Start from page 2
            "skip_last_percent": 0.2,  # Skip last 20% pages
            "debug": False
        },
        {
            "name": "duration",
            "prompt": "Tìm dòng chứa 'Tổng thời gian thực hiện hợp đồng' và COPY NGUYÊN VĂN cả dòng đó. Ví dụ: 'Tổng thời gian thực hiện hợp đồng: 30 ngày'. Chỉ trả lời dòng chứa 'Tổng thời gian', không giải thích. Nếu không tìm thấy, trả lời 'N/A'.",
            "format": "duration",
            "verify_prompt": "Trong trang này, có cụm từ 'Tổng thời gian thực hiện hợp đồng' kèm số ngày hoặc năm không? Trả lời YES hoặc NO.",
            "start_page": 2,  # Start from page 2
            "skip_last_percent": 0.2,  # Skip last 20% pages
            "debug": True  # DEBUG enabled
        }
    ]
    
    results = {}
    import re
    
    print("\n=== EXTRACTING METADATA ===")
    
    for field in metadata_fields:
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
                    hom_nay_match = re.search(r'[Hh]ôm\s*nay[,\s]*ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})', answer, re.IGNORECASE)
                    if hom_nay_match:
                        day, month, year = hom_nay_match.groups()
                        formatted_answer = f"{day}/{month}/{year}"
                elif field['format'] == 'duration':
                    # Match ngày, tháng, or năm
                    duration_match = re.search(r'[Tt]ổng\s*thời\s*gian\s*thực\s*hiện\s*hợp\s*đồng[:\s]*(\d+)\s*(?:\([^)]*\))?\s*(ngày|tháng|năm)', answer, re.IGNORECASE)
                    if duration_match:
                        number, unit = duration_match.groups()
                        formatted_answer = f"{number} {unit}"  # Include unit
                elif field['format'] == 'text':
                    # Accept raw answer as-is (for contract title)
                    if len(answer) > 5 and len(answer) < 200:
                        formatted_answer = answer.strip()
                
                if formatted_answer:
                    candidates.append({
                        'page_num': page_num,
                        'raw_answer': answer,
                        'formatted': formatted_answer,
                        'img_path': img_path
                    })
        
        # PHASE 2: Verify candidates (prioritize earlier pages = main contract)
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
            
            is_verified = 'YES' in verify_answer
            
            if is_verified:
                print(f"      ✓ Page {candidate['page_num']}: VERIFIED - {candidate['formatted']}")
                best_answer = candidate['formatted']
                found_on_page = candidate['page_num']
                break  # Take first verified (earlier page = main contract)
            else:
                print(f"      ✗ Page {candidate['page_num']}: NOT VERIFIED (hallucination)")
        
        results[field['name']] = best_answer
        if found_on_page:
            print(f"    ✓ {field['name']}: {best_answer} (verified on page {found_on_page})")
        else:
            print(f"    ✗ {field['name']}: N/A (no verified answer in {total_pages} pages)")
    
    # Force garbage collection
    import gc
    gc.collect()
    
    return (
        results.get('contract_number', 'N/A'),
        results.get('contract_name', 'N/A'),
        results.get('signing_date', 'N/A'),
        results.get('contract_value', 'N/A'),
        results.get('duration', 'N/A')
    )


# Initialize UI
with gr.Blocks(title="Local Vision RAG (MacOS M5)", css="""
    .gallery-item { cursor: pointer; }
""") as demo:
    gr.Markdown("# 👁️ Local Vision RAG Demo\n**Retrieval:** Byaldi (ColPali) | **Generation:** MLX-VLM (Qwen2-VL)")
    
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
        outputs=[contract_number_output, contract_name_output, signing_date_output, contract_value_output, duration_output]
    )

if __name__ == "__main__":
    load_models()
    demo.launch(server_name="0.0.0.0", server_port=7860)
