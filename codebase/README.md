# LegalIR pipeline

Pipeline offline tối giản cho UIT DSC 2026 Task 1:

```text
Điều-first hierarchical chunking -> BM25 heading/body + Vietnamese_Embedding_v2/FAISS -> weighted RRF
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

Trên Kaggle, cài requirement trước khi import NumPy/SciPy; nếu đã import hoặc vừa
thay phiên bản các gói binary này, restart session một lần sau khi cài.

Trên Kaggle không bật Internet, attach hai model dưới dạng Kaggle Model/Dataset và
truyền đường dẫn local vào `--model`, ví dụ `/kaggle/input/vietnamese-embedding-v2`
cho `legalir.py` và `/kaggle/input/qwen3-0.6b` cho `train_qlora.py`. Pipeline không
gọi API. `--device` của retrieval và `device_map` của reranker đều tự chọn phần
cứng; `--batch-size` vẫn mở để tận dụng GPU Kaggle đang cấp.

## Chạy

Từ thư mục gốc project:

```powershell
python codebase/legalir.py prepare
python codebase/legalir.py split
python codebase/legalir.py build-bm25
python codebase/legalir.py build-dense --model AITeamVN/Vietnamese_Embedding_v2
python codebase/legalir.py evaluate
python codebase/legalir.py predict
```

`build-dense` saves progress every 2,048 rows and resumes automatically when rerun
with the same chunks, model, revision, and maximum length. Use
`--checkpoint-size` to change the interval.

Kết quả mặc định nằm trong `codebase/artifacts/`. `predict` tạo cả
`submission.json` và `submission.zip`; file ZIP chỉ chứa `submission.json` ở root.

Các tham số cần ablation trước:

```powershell
python codebase/legalir.py evaluate --top-n 2 --heading-weight 1.5 --body-weight 1.0 --dense-weight 1.0 --rrf-k 60
```

- `--top-n`: số chunk tốt nhất được lấy trung bình thành document score.
- Ba `--*-weight`: trọng số các ranking trong weighted RRF.
- `--rrf-k`: hằng số RRF.
- `--top-chunks`: số chunk mỗi retriever trả về trước khi gộp document.

## Kiểm thử

```powershell
python -m unittest discover -s codebase -p "test_*.py"
```

`index_manifest.json` lưu checksum chunk, cấu hình BM25, model/revision, chiều vector
và max length. Evaluate/predict từ chối chạy nếu index không khớp `chunks.jsonl`.

## Qwen3-0.6B zero-shot / Unsloth QLoRA reranker

Reranker chấm độc lập từng cặp câu hỏi–chunk trên thang 0–5. Vì `train.json`
chỉ cho biết document đúng, dữ liệu fine-tune dùng nhãn `5` cho positive và `0`
cho hard negative; các mức 1–4 chỉ được mô hình zero-shot suy ra.

Điểm cuối của document kết hợp hai tín hiệu đã chuẩn hóa:

```text
qwen_normalized      = max(expected_grade_của_các_chunk) / 5
retrieval_normalized = min-max(document_score) trong từng câu hỏi
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
rồi hợp nhất theo: `0.5/(4+r_heading) + 2/(4+r_body) + 4/(4+r_dense)`. Mỗi
document giữ tối đa hai chunk có score `>= 0.4`. Các CLI tương ứng là
`--chunk-heading-weight`, `--chunk-body-weight`, `--chunk-dense-weight`,
`--chunk-rrf-k`, `--chunk-threshold` và `--chunks-per-document` (chỉ nhận 1 hoặc
2). Với tập có gold answer, document đúng bị retrieval bỏ sót vẫn được thêm vào
candidate với `retrieved=false` để tạo positive training pair mà không làm tăng
Recall giả.

```powershell
python -m pip install -r codebase/requirements-legalir.txt
python -m pip install -r codebase/requirements-qlora.txt

python codebase/legalir.py split
python codebase/legalir.py candidates --questions codebase/artifacts/split/train.json --output codebase/artifacts/train_candidates.json --top-documents 20
python codebase/legalir.py candidates --questions codebase/artifacts/split/validation.json --output codebase/artifacts/validation_candidates.json --top-documents 20

python codebase/train_qlora.py prepare --questions codebase/artifacts/split/train.json --candidates codebase/artifacts/train_candidates.json --output codebase/artifacts/train_pairs.jsonl
python codebase/train_qlora.py prepare --questions codebase/artifacts/split/validation.json --candidates codebase/artifacts/validation_candidates.json --output codebase/artifacts/validation_pairs.jsonl

accelerate launch --num_processes 2 codebase/train_qlora.py train --train-pairs codebase/artifacts/train_pairs.jsonl --validation-pairs codebase/artifacts/validation_pairs.jsonl
```

`train_qlora.py prepare` không chọn chunk lần nữa: document thuộc gold nhận nhãn
5, mọi document còn lại nhận nhãn 0, áp dụng cho toàn bộ chunk trong file
candidate. Sau khi đổi weight/threshold, phải chạy lại `legalir.py candidates`
rồi tạo lại pair; mỗi pair giữ `retrieved` để Recall@5 không tính positive được
chèn vào để huấn luyện.

Trên Kaggle, bắt đầu bằng session mới, cài requirements một lần rồi restart
session nếu pip yêu cầu. Không force-reinstall riêng Torch/Transformers/PEFT vì
Unsloth quản lý bộ phiên bản tương thích. Pipeline text không cần `torchvision`
hoặc `torchaudio`.

So sánh cùng prompt và cách tính xác suất grade. Không truyền `--adapter` là
zero-shot; truyền adapter là kết quả fine-tune:

```powershell
python codebase/train_qlora.py rerank --questions codebase/artifacts/split/validation.json --candidates codebase/artifacts/validation_candidates.json --output codebase/artifacts/qwen3-0.6b-validation.json
python codebase/legalir.py candidates --questions public-official.json --output codebase/artifacts/public_candidates.json --top-documents 20
python codebase/train_qlora.py rerank --questions public-official.json --candidates codebase/artifacts/public_candidates.json --output codebase/artifacts/qwen3-0.6b-public.json --submission codebase/artifacts/submission.json
python codebase/train_qlora.py rerank --questions codebase/artifacts/split/validation.json --candidates codebase/artifacts/validation_candidates.json --adapter codebase/artifacts/qlora/final_adapter --output codebase/artifacts/finetuned_rerank.json
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
