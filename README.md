# UIT-DSC-2026
Một repo với pipeline cơ bản cho kỳ thi UIT-DSC-2026.

# LegalIR pipeline

Pipeline offline tối giản cho UIT DSC 2026 Task 1:

```text
Điều-first hierarchical chunking-> BM25 heading/body + Vietnamese_Embedding_v2/FAISS -> weighted RRF
-> top-n average theo document -> unique top-5 -> submission.zip
```

`prepare` nhận diện phần/văn bản đính kèm (`QUY CHẾ`, `QUY ĐỊNH`, `PHỤ LỤC`),
`Chương`, `Mục` và `Điều`. Một Điều tối đa 240 từ là một chunk; Điều dài mới
được nhóm theo các Khoản liên tiếp, và Khoản quá dài mới tách theo Điểm/đoạn
với overlap 40 từ. Regex Điều yêu cầu dấu `.`/`:` sau số hoặc số nằm cuối dòng,
nhờ đó tham chiếu bị wrap như `Điều 2 Quyết định này...` không trở thành heading.
Mỗi chunk lặp lại đầy đủ đường dẫn tên văn bản → phần đính kèm → Chương → Mục
→ Điều → khoảng Khoản. Đuôi hành chính bắt đầu bằng `Nơi nhận:` không được index.

## Cài đặt

```powershell
python -m pip install -r codebase/requirements-legalir.txt
```

Trên Kaggle, Colab, chỉ cần cài đặt các requirement như bm25, faiss-cpu

## Chạy

Từ thư mục gốc project:

```powershell
# Tạo chunks 
python codebase/legalir.py prepare

# Tách tập train.json thành 70/30 train-validation
python codebase/legalir.py split

# Tạo index cho bm25, sử dụng bm25 cho phần heading và text trong chunk
python codebase/legalir.py build-bm25

# Tạo index cho dense retrieval 
python codebase/legalir.py build-dense --model AITeamVN/Vietnamese_Embedding_v2

# Đánh giá kết quả khi không có reranker (Cần tinh chỉnh các hệ số, --help để biết rõ hơn)
# Mặc định nó chỉ chấm R@5, nếu quá 5 đáp án thì mặc định 0 điểm
python codebase/legalir.py evaluate

# Tạo submission không cần reranker
python codebase/legalir.py predict
```

`build-dense` saves progress every 2,048 rows and resumes automatically when rerun
with the same chunks, model, revision, and maximum length. Use
`--checkpoint-size` to change the interval.

Kết quả mặc định nằm trong `codebase/artifacts/`. `predict` tạo cả
`submission.json` và `submission.zip`; file ZIP chỉ chứa `submission.json` ở root.

Các tham số cần ablation trước:

```powershell
python codebase/legalir.py evaluate --top-n 2 --heading-weight 1 --body-weight 4 --dense-weight 5 --rrf-k 10
```

- `--top-n`: số chunk tốt nhất được lấy trung bình thành document score.
- Ba `--*-weight`: trọng số các ranking trong weighted RRF.
- `--rrf-k`: hằng số RRF.
- `--top-chunks`: số chunk mỗi retriever trả về trước khi gộp document.

## Index

`index_manifest.json` lưu checksum chunk, cấu hình BM25, model/revision, chiều vector
và max length. Evaluate/predict từ chối chạy nếu index không khớp `chunks.jsonl`.

## Qwen3-0.6B zero-shot / Unsloth QLoRA reranker

Reranker chấm độc lập từng cặp câu hỏi–chunk trên thang 0–5. 

Điểm cuối của document kết hợp hai tín hiệu đã chuẩn hóa:

```text
qwen_normalized      = max(expected_grade_của_các_chunk) / 5
retrieval_normalized = min-max(document_score) trong từng câu hỏi
```
Sử dụng weighting cho từng điểm số final score, với mặc định tỉ lệ 6/4:

```text
final_score          = 0.6 * qwen_normalized + 0.4 * retrieval_normalized
```

Nếu mọi retrieval score bằng nhau thì tất cả nhận `retrieval_normalized = 1`,
nên tín hiệu retrieval không thay đổi thứ hạng. File pair giữ thêm
`retrieval_score`; sau thay đổi này cần tạo lại train/validation pairs từ candidate.
Có thể đổi tỷ lệ cho cả checkpoint metric và inference bằng
`--qwen-weight`/`--retrieval-weight`; hai weight được tự chia cho tổng nên `6 4`
và `0.6 0.4` là tương đương.

`legalir.py candidates` xuất document cùng các chunk đã chọn. Trong mỗi document
`D`, chunk được xếp hạng riêng bằng BM25 heading, BM25 body và dense cosine/IP,
rồi hợp nhất bằng RRF score. Mỗi document giữ tối đa hai chunk có score `>= 0.4`. Các CLI tương ứng là
`--chunk-heading-weight`, `--chunk-body-weight`, `--chunk-dense-weight`,
`--chunk-rrf-k`, `--chunk-threshold` và `--chunks-per-document` (chỉ nhận 1 hoặc
2). Với tập có gold answer, document đúng bị retrieval bỏ sót vẫn được thêm vào
candidate với `retrieved=false` để tạo positive training pair mà không làm tăng
Recall giả.

## training (đang test - không cần chạy)

Train bằng cách để model học phân biệt đâu là document với chunk đúng, trả lời được câu hỏi (positive) và document với chunk không trả lời được câu hỏi (negative)

Mặc định ta đã tải dependency của legalir.txt theo hướng dẫn trên, ta tiến hành tải dependency như sau:
```powershell
# Tải Dependency
python -m pip install -r codebase/requirements-qlora.txt
```

Các bước train:
```text
chuẩn bị train/test -> tạo câu trả lời mẫu bằng retrieve không rerank -> tạo các cặp positive/negative -> training
```

```powershell 
# Split Train-Test
python codebase/legalir.py split

# Tạo ra các đáp án mẫu, mặc định chọn 20 documents cho mỗi question
python codebase/legalir.py candidates --questions codebase/artifacts/split/train.json --output codebase/artifacts/train_candidates.json --top-documents 20
python codebase/legalir.py candidates --questions codebase/artifacts/split/validation.json --output codebase/artifacts/validation_candidates.json --top-documents 20

# Tạo ra các cặp positive-negative để training
python codebase/train_qlora.py prepare --questions codebase/artifacts/split/train.json --candidates codebase/artifacts/train_candidates.json --output codebase/artifacts/train_pairs.jsonl
python codebase/train_qlora.py prepare --questions codebase/artifacts/split/validation.json --candidates codebase/artifacts/validation_candidates.json --output codebase/artifacts/validation_pairs.jsonl

# chạy training
accelerate launch --num_processes 2 codebase/train_qlora.py train --train-pairs codebase/artifacts/train_pairs.jsonl --validation-pairs codebase/artifacts/validation_pairs.jsonl
```

`train_qlora.py prepare` không chọn chunk lần nữa: document thuộc gold nhận nhãn
5, mọi document còn lại nhận nhãn 0, áp dụng cho toàn bộ chunk trong file
candidate. Sau khi đổi weight/threshold, phải chạy lại `legalir.py candidates`
rồi tạo lại pair; mỗi pair giữ `retrieved` để Recall@5 không tính positive được
chèn vào để huấn luyện.

So sánh cùng prompt và cách tính xác suất grade. Không truyền `--adapter` là
zero-shot; truyền adapter là kết quả fine-tune:

```powershell
# rerank không fine-tune, dùng để validation
python codebase/train_qlora.py rerank --questions codebase/artifacts/split/validation.json --candidates codebase/artifacts/validation_candidates.json --output codebase/artifacts/qwen3-0.6b-validation.json

# rerank không fine-tune, dùng để submisison
python codebase/train_qlora.py rerank --questions public-official.json --candidates codebase/artifacts/public_candidates.json --output codebase/artifacts/qwen3-0.6b-public.json --submission codebase/artifacts/submission.json

# rerank có fine-tune, dùng cho evaluation
python codebase/train_qlora.py rerank --questions codebase/artifacts/split/validation.json --candidates codebase/artifacts/validation_candidates.json --adapter codebase/artifacts/qlora/final_adapter --output codebase/artifacts/finetuned_rerank.json

# rerank có fine-tune, dùng cho submission:
python codebase/train_qlora.py rerank --questions public-official.json --candidates codebase/artifacts/public_candidates.json --adapter codebase/artifacts/qlora/final_adapter --output codebase/artifacts/qwen3-0.6b-public.json --submission codebase/artifacts/submission.json

# merge qlora và qwen để tạo thành 1 model cuối
python codebase/train_qlora.py merge --adapter codebase/artifacts/qlora/final_adapter --output codebase/artifacts/qwen3-0.6b-legalir-merged
```

Ví dụ ưu tiên Qwen hơn retrieval:

```powershell
python codebase/train_qlora.py rerank --questions codebase/artifacts/split/validation.json --candidates codebase/artifacts/validation_candidates.json --output codebase/artifacts/rerank.json --qwen-weight 0.7 --retrieval-weight 0.3
```

Mặc định Unsloth QLoRA dùng 4-bit, BF16 khi GPU hỗ trợ, nếu không thì FP16; LoRA
rank 64, alpha 16, dropout 0 (đường tối ưu của Unsloth), learning rate `2e-4`,
sequence length 1024 và Unsloth gradient checkpointing. Validation giữ đủ top-20
candidate và chọn checkpoint có macro Recall@5 cao nhất. `rerank` nạp model/adapter
bằng Unsloth, bật fast inference, rồi ghi score, phân phối grade và
Recall/Precision tại top 1..5. Mặc định zero-shot là `Qwen/Qwen3-0.6B` và tắt
thinking mode; thêm `--submission PATH` để ghi cả JSON và ZIP nộp bài.
