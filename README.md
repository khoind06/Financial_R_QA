# Financial Report Question Answering (FinReport-QA)

**Financial Report Question Answering (FinReport-QA)** là hệ thống AI hỗ trợ **Hỏi - Đáp, Trích xuất & Suy luận Chỉ số Báo cáo Tài chính tự động** dành cho các doanh nghiệp niêm yết tại Việt Nam.

Hệ thống cho phép đọc hiểu, tự động phân tích và trích xuất bảng biểu từ **1.973 Báo cáo Tài chính (2015–2025)** của **100 công ty niêm yết**, từ đó đưa ra câu trả lời chính xác, minh bạch cùng các đoạn mã tính toán Pandas có thể kiểm định 100%.

---

## 🌟 Các Tính năng Cốt lõi & Kiến trúc Hệ thống

1. **Đánh chỉ mục Báo cáo Tài chính Tốc độ cao (`build_index`)**:
   - Tự động parse và lưu trữ **146.246 bảng biểu HTML/Text** cùng **1.535.824 dòng dữ liệu** vào cơ sở dữ liệu SQLite (`artifacts/tables.sqlite3`).
2. **Khối dữ liệu Tài chính Chuẩn hóa (`build_panel`)**:
   - Xây dựng Panel dữ liệu tài chính đa chiều (100 mã cổ phiếu × 884 năm-công ty) bao gồm các bảng: *Cân đối kế toán (CDKT)*, *Kết quả kinh doanh (KQKD)* và *Lưu chuyển tiền tệ (LCTT)*.
   - Tích hợp **Fuzzy Text Label Matching** giúp trích xuất chuẩn xác các chỉ tiêu tài chính kể cả khi bị lỗi OCR hoặc mất mã số dòng.
3. **Động cơ Phân luồng & Suy luận Đa tuyến (`solve`)**:
   - **Tra cứu Trực tiếp (Direct Solver)**: Tra cứu nhanh các chỉ số đơn lẻ bằng kỹ thuật Reranking kết hợp với LLM.
   - **Công thức Tài chính (Hard Formula Solver)**: Tự động tổng hợp và tính toán các tỷ suất, chênh lệch tăng/giảm giữa các năm.
   - **Phân tích Thuyết minh BCTC (Note Solver)**: Trích xuất thông tin chuyên sâu trong phần Thuyết minh (chi tiết nợ xấu, trái phiếu, phải thu,...).
   - **Mẫu Chuỗi thời gian (Template Solver)**: Xử lý các câu hỏi so sánh và thống kê (Max, Mean, Sum, Count, Argmax).
4. **Động cơ Linh hoạt LLM Cloud & Local (`local_llm.py`)**:
   - Tương thích linh hoạt với cả **Cloud API miễn phí** (Groq `qwen-2.5-32b` / `llama-3.3-70b`, Google Gemini, OpenAI) lẫn **Local LLM** (Ollama, llama.cpp).
   - Tự động xử lý lỗi Rate Limit (`429 Too Many Requests`) để đảm bảo quá trình chạy không bị gián đoạn.
5. **Tính toán Chính xác & Minh bạch tuyệt đối**:
   - LLM chỉ đóng vai trò trích xuất và định vị dữ liệu grounded, toàn bộ phép toán số học được thực thi hoàn toàn bằng Python/Pandas để đảm bảo không bị sai số hay bịa số (Hallucination).

---

## 📂 Sơ đồ Cấu trúc Dự án

```text
Financial-Report-QA/
├── data/
│   └── vifinqa/
│       ├── code_stock.csv           # Danh sách 100 mã cổ phiếu & tên doanh nghiệp
│       ├── financial_statements/    # Kho 1.973 tệp báo cáo tài chính (.txt)
│       └── questions/
│           └── questions.jsonl      # Tập câu hỏi đánh giá và truy vấn
├── src/
│   └── road2ai_vifinqa/
│       ├── paths.py                 # Quản lý đường dẫn hệ thống
│       ├── corpus.py                # Quản lý & truy xuất dữ liệu kho báo cáo
│       ├── text.py                  # Chuẩn hóa tiếng Việt & xử lý số học/tỷ lệ scale
│       ├── html_tables.py           # Parse bảng biểu HTML trong OCR
│       ├── build_index.py           # Đánh chỉ mục SQLite (tables.sqlite3)
│       ├── build_panel.py           # Tạo khối dữ liệu panel chuẩn hóa
│       ├── local_llm.py             # Động cơ LLM (Hỗ trợ Groq/Gemini API & Ollama)
│       ├── retrieval.py             # Tìm kiếm & truy xuất tài liệu/bảng biểu
│       ├── easy_solver.py           # Solver cho câu hỏi tra cứu
│       ├── hard_solver.py           # Solver cho câu hỏi tính toán công thức
│       ├── hard_note_solver.py      # Solver cho câu hỏi thuyết minh BCTC
│       ├── panel_solver.py          # Solver dữ liệu chuỗi thời gian
│       ├── template_solver.py       # Solver theo mẫu câu hỏi
│       ├── solve.py                 # Trình điều khiển thực thi toàn bộ hệ thống
│       └── submission.py            # Đóng gói & kiểm định kết quả
├── tools/                           # Các công cụ kiểm định & chẩn đoán
├── artifacts/                       # Cơ sở dữ liệu SQLite & dữ liệu panel
├── runs/                            # Lịch sử các lượt chạy & checkpoint
├── submission.zip                   # Kết quả đóng gói hoàn chỉnh
├── pyproject.toml                   # Cấu hình dự án
└── README.md                        # Tài liệu hướng dẫn sử dụng
```

---

## 🚀 Hướng dẫn Cài đặt & Vận hành

### 1. Yêu cầu Môi trường
- **Python**: `>= 3.11`
- **Hệ điều hành**: Windows, Linux, macOS

### 2. Cài đặt Dự án

Mở terminal tại thư mục gốc của dự án:

```powershell
# Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# Cài đặt dự án ở chế độ editable
python -m pip install -e .
```

### 3. Thực thi Hệ thống

#### **Cách 1: Sử dụng Cloud API miễn phí (Nhanh nhất & Chính xác nhất)**

Đăng ký API Key miễn phí tại [Groq Cloud](https://console.groq.com/) hoặc [Google AI Studio](https://aistudio.google.com/), sau đó thiết lập biến môi trường và chạy:

```powershell
# Đặt API Key của bạn (ví dụ với Groq)
$env:LLM_API_KEY="gsk_dien_api_key_cua_ban_tai_day"
$env:LLM_MODEL="qwen-2.5-32b"

# Bước 1: Đánh chỉ mục báo cáo tài chính
python -m road2ai_vifinqa.build_index --force

# Bước 2: Tạo Panel dữ liệu tài chính
python -m road2ai_vifinqa.build_panel --force

# Bước 3: Chạy giải câu hỏi & tạo kết quả
python -m road2ai_vifinqa.solve --iteration 1 --publish
```

#### **Cách 2: Chạy hoàn toàn Local qua Ollama**

Đảm bảo dịch vụ Ollama đang chạy (`ollama list` có `qwen2.5:latest`):

```powershell
$env:USE_OLLAMA="1"

python -m road2ai_vifinqa.build_index --force
python -m road2ai_vifinqa.build_panel --force
python -m road2ai_vifinqa.solve --iteration 1 --publish
```

---

## 📊 Kết quả Output & Kiểm định

Sau khi quá trình thực thi hoàn tất:
- File kết quả tổng hợp [`submission.zip`](file:///d:/Road2AI/submission.zip) sẽ được tạo tại thư mục gốc.
- Mọi câu hỏi đều được tự động replay mã `pandas_query` để kiểm định tính hợp lệ 100% trước khi xuất bản.
- Lịch sử suy luận và checkpoint được ghi nhận tại thư mục `runs/iteration_1/`.
