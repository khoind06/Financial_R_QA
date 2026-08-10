import json
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download

# 1. Tự động xác định đường dẫn thư mục data/vifinqa trong dự án ROAD2AI
# Path(__file__).resolve().parents[2] sẽ lấy thư mục gốc ROAD2AI
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "vifinqa"

print(f"Đang tải dữ liệu về thư mục: {DATA_DIR}")

# 2. Tải toàn bộ dataset từ Hugging Face về ROAD2AI/data/vifinqa
snapshot_download(
    repo_id="AIGuruTinix/ViFinQA",
    repo_type="dataset",
    local_dir=DATA_DIR
)
print("-> Tải thành công!")

# 3. Đọc file CSV mã chứng khoán
stock_file = DATA_DIR / "code_stock.csv"
if stock_file.exists():
    df_stocks = pd.read_csv(stock_file)
    print("\n--- Danh sách mã chứng khoán ---")
    print(df_stocks.head())

# 4. Quét toàn bộ file trong thư mục financial_statements
financial_dir = DATA_DIR / "financial_statements"
financial_files = [f for f in financial_dir.rglob("*") if f.is_file()]
print(f"\nTổng số file BCTC tìm thấy: {len(financial_files)}")

# 5. Quét toàn bộ file trong thư mục questions
questions_dir = DATA_DIR / "questions"
question_files = [f for f in questions_dir.rglob("*") if f.is_file()]
print(f"Tổng số file câu hỏi tìm thấy: {len(question_files)}")

# 6. Đọc thử 1 file BCTC mẫu
if financial_files:
    with open(financial_files[0], "r", encoding="utf-8") as f:
        sample_bctc = json.load(f)
        print(f"\nĐã đọc thành công file: {financial_files[0].name}")