# ViFinQA competition solution

Pipeline tái lập cho cuộc thi **R2AI2026 Financial Table Retrieval &
Text-to-Pandas**. Kết quả nộp cuối nằm tại `submission.zip`; toàn bộ dữ liệu
trung gian, checkpoint theo câu hỏi, log suy luận và báo cáo kiểm định được giữ
lại để có thể kiểm tra hoặc tiếp tục chạy.

## Phương pháp

- Lập chỉ mục 146.246 bảng trong báo cáo tài chính vào
  `artifacts/tables.sqlite3` và chuẩn hoá các chỉ tiêu phổ biến vào
  `artifacts/financial_panel.json`.
- Định tuyến 1.012 câu hỏi qua các bộ giải deterministic theo loại câu hỏi;
  những trường hợp cần hiểu ngôn ngữ dùng Qwen3-8B cục bộ ở nhiệt độ 0.
- Mỗi đáp án dẫn chiếu tài liệu/bảng tồn tại trong kho chính thức, có một CSV
  riêng và một `pandas_query` có thể chạy lại để sinh ra `answer`.
- Không fine-tune hay huấn luyện có giám sát. Snapshot model, nguồn model,
  prompt/response, cache và checkpoint đều được lưu trong workspace.

## Môi trường đã dùng

- Python 3.12, pandas 3.0.3, NumPy 2.5.1, Windows 11.
- Dataset ViFinQA revision:
  `0450088ab22ec946f04f097586967ca405955b3b`.
- Mã tham chiếu `DSKT-NOWJ/ViFinQA` commit:
  `9a046de2f2daea4d2be0a05d4a5f3f1220e6922a`.
- Model:
  `artifacts/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`.
- Nguồn model:
  `Qwen/Qwen3-8B-GGUF@main:Qwen3-8B-Q4_K_M.gguf`.
- Runtime cục bộ: `tools/llama.cpp/runtime/llama-server.exe`.

## Chạy lại

Các lệnh dưới đây chạy từ thư mục gốc dự án trong PowerShell:

```powershell
python -m pip install -e .
$env:PYTHONPATH = "src"
$env:PYTHONIOENCODING = "utf-8"
$env:VIFINQA_MODEL = (Resolve-Path "artifacts/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf").Path
$env:VIFINQA_MODEL_SOURCE = "Qwen/Qwen3-8B-GGUF@main:Qwen3-8B-Q4_K_M.gguf"

python -m road2ai_vifinqa.build_index
python -m road2ai_vifinqa.build_panel
python -m road2ai_vifinqa.solve --iteration 3 --run-dir runs/iteration_3 --ids 1-1012 --fail-fast --publish
python tools/release_audit.py --zip submission.zip --model "$env:VIFINQA_MODEL" --run-dir runs/iteration_3 --report runs/iteration_3/final_release_audit.json --replays 3
```

Lệnh `solve` tự tiếp tục từ checkpoint. Nếu cần giải lại độc lập từ đầu, dùng
`--no-resume` với một run mới (ví dụ `--iteration 4 --run-dir
runs/reproduction_clean`) để không ghi đè bản phát hành đã kiểm định.

## Kết quả và nhật ký

- `runs/iteration_1`, `runs/iteration_2`, `runs/iteration_3`: lịch sử ba vòng
  giải, kiểm tra và cải thiện liên tiếp; đây không phải ba vòng tự động của một
  lệnh `solve`.
- `runs/iteration_3/cache`: 1.012 checkpoint `.pkl` dùng để tiếp tục an toàn;
  chỉ nạp pickle do chính workspace tin cậy này tạo ra.
- `runs/iteration_3/checkpoints`: 1.012 bản tóm tắt JSON để kiểm tra thủ công.
- `runs/iteration_3/llm`: log suy luận cục bộ cho các câu dùng mô hình.
- `runs/iteration_3/manifest.json`: cấu hình và thống kê của vòng cuối.
- `runs/iteration_3/release_audit.json`: kiểm định độc lập ZIP vòng 3.
- `runs/iteration_3/final_release_audit.json`: kiểm định lại file đã xuất bản
  tại thư mục gốc.
- `submission.zip`: tệp duy nhất cần tải lên dashboard cuộc thi.

Kiểm định phát hành yêu cầu đủ 1.012 ID đúng thứ tự, đúng schema, đúng câu hỏi
gốc, mọi tài liệu/bảng/CSV đều tồn tại, không có CSV mồ côi, và mỗi
`pandas_query` phải chạy lại độc lập ba lần với kết quả hữu hạn, deterministic
và khớp `answer`.

Kết quả vòng cuối: 1.012 dòng, 1.012 CSV, 1.013 thành viên ZIP, ba lượt replay
mới và 0 lỗi. SHA-256 của `submission.zip` là
`786adf2f5840562e698d3e39cbd53b056534d7bbba5c03f79b659d5dc2c79f2d`;
SHA-256 của model là
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.

Kiểm định này bảo đảm tính toàn vẹn, khả năng chạy lại và tính hợp lệ của hồ
sơ nộp. Vì đáp án chuẩn của bộ kiểm thử không được công bố, nó không thể bảo
đảm trước điểm số trên leaderboard ẩn.

Lưu ý: `.gitignore` loại trừ các snapshot và sản phẩm dung lượng lớn
(`data/source`, `external`, `artifacts`, `runs`, `submission.zip`). Vì vậy một
Git clone sạch không tự chứa các payload này; khi bàn giao cần sao chép cả các
thư mục trên hoặc tái tạo chúng từ đúng snapshot nguồn.
