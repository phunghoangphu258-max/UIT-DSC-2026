# Project Context

## Snapshot

- **Phạm vi:** UIT Data Science Challenge 2026 - Task 1, Legal Information Retrieval (LegalIR) cho tiếng Việt.
- **Xác minh lần cuối:** 2026-08-23 (Asia/Bangkok).
- **Revision/branch:** Không áp dụng. Thư mục hiện tại chưa phải Git repository.
- **Trạng thái hiện tại:** Có pipeline BM25/Vietnamese_Embedding_v2 weighted-RRF, evaluator/submission, candidate export và Qwen3-0.6B zero-shot/QLoRA reranker; index đã có nhưng runtime/model weights chưa được cài trong workspace này.
- **Nguồn yêu cầu:** `DSC2026_Task1_LegalIR_Data_Overview.docx`, đặc biệt các mục 1, 2, 4 và 5.

## Project at a Glance

- **Mục tiêu:** Với một câu hỏi pháp luật tiếng Việt, trả về ID của các văn bản hành chính/pháp lý chứa thông tin cần thiết để trả lời câu hỏi.
- **Input logic:** `question_id` cùng chuỗi `question`.
- **Output logic:** Mỗi `question_id` ánh xạ tới một danh sách tối đa 5 `document_id`.
- **Ưu tiên tối ưu:** Tối đa hóa macro Recall trước; macro Precision chỉ dùng để phân hạng khi Recall bằng nhau.
- **Kiến trúc đang có:** `codebase/legalir.py` phụ trách chunk/index/retrieval/evaluate/submit; `codebase/train_qlora.py` nhận fused candidates, tạo hard-negative pairs, train QLoRA, zero-shot Qwen3-0.6B và rerank; các CLI kiểm tra/crawler vẫn phụ trách data quality.
- **Luồng runtime mục tiêu:** nạp/chuẩn hóa corpus -> lập chỉ mục -> truy xuất ứng viên -> rerank -> chọn tối đa 5 ID -> kiểm tra submission -> đóng gói ZIP.

## Contract của bài toán

Ký hiệu tập văn bản đúng cho câu hỏi `i` là `R_i`, tập dự đoán là `P_i`:

- `Recall_i = |R_i ∩ P_i| / |R_i|`.
- `Precision_i = |R_i ∩ P_i| / |P_i|`; nếu không trả văn bản nào thì `Precision_i = 0`.
- Điểm cuối là trung bình theo câu hỏi (macro average), không phải gộp toàn bộ document prediction rồi tính micro average.
- Thứ tự xếp hạng là lexicographic: Recall là chỉ số chính, Precision là chỉ số phụ khi Recall bằng nhau.
- Mỗi câu hỏi được đánh giá tối đa 5 ID. Nếu một câu có hơn 5 ID thì cả Recall và Precision của câu đó bằng 0; câu này vẫn được đưa vào trung bình toàn bộ bài.
- Tài liệu mô tả metric theo **tập ID**. Không có bằng chứng rằng thứ tự trong danh sách trực tiếp tham gia công thức, nhưng pipeline vẫn nên giữ thứ tự score giảm dần để phục vụ cắt top-k và khả năng evaluator có xử lý thứ hạng nội bộ.

## Data Inventory

| Asset | Nội dung đã xác minh | Vai trò |
|---|---|---|
| `DSC2026_Task1_LegalIR_Data_Overview.docx` | Mô tả task, data schema, metric, submission và giới hạn 5 ID | Product/evaluation contract |
| `train.json` | 7.000 câu hỏi có nhãn | Huấn luyện, validation và phân tích lỗi |
| `public-official.json` | 1.000 câu hỏi; trường `answer` đều là `null` | Public-test inference input |
| `selected-contexts/selected-contexts/` | 8.532 tệp `context_<id>.json` đã giải nén | Corpus truy xuất |
| `PROJECT_CONTEXT.md` | Bản đồ dự án hiện tại | Context bền vững cho các lượt triển khai sau |
| `codebase/inspect_non_passage.py` | Báo cáo context thiếu/null/rỗng/whitespace `passage` và train references | Data-quality inspection CLI |
| `codebase/crawl_non_passage.py` | Phục hồi passage từ `link` bằng browser, có resume và opt-in `--in-place` | Data recovery CLI |
| `codebase/legalir.py` | Structural chunking, document-disjoint split, BM25/BGE-M3 weighted RRF, metrics và submission | First-stage retrieval CLI |
| `codebase/train_qlora.py` | Pointwise grade-token reranking, QLoRA training, baseline/fine-tuned evaluation và model merge | Second-stage reranker CLI |
| `codebase/requirements-qlora.txt` | Unsloth/TensorBoard and retrieval dependencies | Kaggle QLoRA environment |

Các tệp `warmup.json` và `private-official.json` được tài liệu nhắc đến theo timeline cuộc thi nhưng **không có trong workspace tại thời điểm xác minh**.

## Schema và đặc điểm dữ liệu

### `train.json`

Top-level là JSON object; key là `question_id` dạng chuỗi:

```json
{
  "86666": {
    "question": "Thời hạn cấp đăng ký xe máy của người nước ngoài làm việc tại Việt Nam là bao lâu?",
    "answer": ["280282"]
  }
}
```

Các thống kê đã quét toàn bộ file:

- 7.000 câu hỏi, không có câu hỏi rỗng.
- 7.637 lượt nhãn, tương ứng 3.105 `document_id` duy nhất.
- Số nhãn/câu: 6.447 câu có 1 nhãn; 485 có 2; 53 có 3; 14 có 4; 1 có 5.
- Tất cả document ID xuất hiện trong nhãn train đều tồn tại trong corpus.
- Có 13 nhóm câu hỏi trùng nguyên văn (13 dòng dư); 3 nhóm có danh sách đáp án khác nhau. Khi chia train/validation phải group theo nội dung câu hỏi đã chuẩn hóa để tránh leakage, đồng thời xem các nhóm khác nhãn là label ambiguity/noise thay vì tự động hợp nhất.

### `public-official.json`

Top-level và schema giống train, nhưng toàn bộ 1.000 giá trị `answer` là `null`:

```json
{
  "38096": {
    "question": "...",
    "answer": null
  }
}
```

`answer: null` là placeholder của test set, không phải nhãn âm và không được dùng để huấn luyện hoặc tính metric cục bộ.

### `selected-contexts/selected-contexts/`

Mỗi JSON context có contract:

```json
{
  "id": 100686,
  "name": "Nghi-dinh-...",
  "link": "https://thuvienphapluat.vn/...",
  "passage": "Nội dung văn bản pháp luật..."
}
```

Các thống kê đã quét toàn bộ thư mục corpus:

- 8.532 context và 8.532 ID duy nhất.
- Tên file `context_<id>.json` khớp `id` trong object ở toàn bộ corpus.
- `id` trong context là **số nguyên**, trong khi ID trong `train.json`, key câu hỏi và submission là **chuỗi**.
- 1.125 context không có key `name`; 294 context trong số này được tham chiếu bởi nhãn train.
- 20 context có `passage` rỗng; cả 20 đồng thời không có `name` nhưng có `link`; 6 context trong số này được tham chiếu bởi 11 dòng train.
- Độ dài `passage` có median khoảng 23.110 ký tự, mean khoảng 41.455 và max khoảng 5.983.358 ký tự. Không nên giả định một văn bản luôn vừa trong context window của model.
- Archive khoảng 97 MB nén và 489 MB giải nén; ingestion nên đọc tuần tự từ ZIP hoặc cache thành index, không mở/parse lại toàn bộ corpus cho mỗi query.

## ID, text và serialization invariants

1. Chuẩn hóa mọi `question_id` và `document_id` sang `str` ngay tại data boundary.
2. Chỉ dùng ID tồn tại trong corpus; không sinh ID tự do.
3. Deduplicate prediction trước khi cắt top-k; mỗi câu phải có `0..5` ID duy nhất.
4. Giữ một bản raw của `passage`, `name`, `link`; tạo trường normalized riêng cho retrieval. Không ghi đè dữ liệu nguồn.
5. Chuẩn hóa có kiểm soát các chuỗi `\r`, `\n`, NBSP và khoảng trắng do văn bản crawl, nhưng không làm mất số điều/khoản, ký hiệu văn bản, ngày tháng hoặc dấu tiếng Việt.
6. `name` là optional. Retrieval không được crash hoặc loại document chỉ vì thiếu `name`.
7. Với `passage` rỗng, dùng slug từ `link` như tín hiệu fallback; không coi document là vô hiệu vì có 6 trường hợp như vậy nằm trong gold labels của train.
8. Submission phải serialize ID dưới dạng chuỗi, kể cả khi corpus lưu `id` dưới dạng số.

## Architecture Map

### Hiện trạng đã xác minh

```text
DOCX task specification
        |
        +--> train.json -------------------- labeled questions
        +--> public-official.json ---------- unlabeled public questions
        +--> selected-contexts/ ------------ legal-document corpus

legalir.py
  -> chunks.jsonl
  -> BM25 heading/body + BGE-M3 FAISS
  -> weighted RRF document candidates
  -> official metrics or submission

train_qlora.py
  -> fused top-20 candidates + gold document labels
  -> query-aware best-chunk selection + 0/5 pointwise pairs
  -> Qwen3-0.6B zero-shot grade logits or QLoRA adapter
  -> reranked unique top-1..5 + macro Recall/Precision
```

### Kiến trúc triển khai đề xuất (chưa tồn tại trong repo)

```text
Corpus loader
  -> raw + normalized document records
  -> lexical and/or dense index

Question loader
  -> query normalization
  -> candidate retrieval
  -> optional reranking
  -> score-sorted unique top-k (k <= 5)
  -> submission validator
  -> submission.json -> submission.zip

Train labels
  -> group-aware split
  -> offline macro Recall / macro Precision
  -> threshold, fusion and top-k tuning
```

Ranh giới module nên tách theo trách nhiệm, không theo từng experiment:

- **Data contract:** đọc train/test/context, normalize ID, xử lý field thiếu.
- **Corpus/index:** tiền xử lý và cache index có version gắn với checksum/config.
- **Retriever:** trả danh sách `(document_id, score)`; không tự serialize submission.
- **Reranker/fusion:** nhận candidates, không đọc trực tiếp toàn corpus.
- **Evaluation:** triển khai đúng macro metric và luật `>5 => 0`.
- **Submission:** đảm bảo coverage, type, uniqueness, giới hạn ID và cấu trúc ZIP.
- **Experiment config:** lưu seed, split, tokenizer/model/index version, top-k và fusion weights.

## Key Execution and Data Flows

### 1. Ingestion/index build

1. Duyệt `selected-contexts/selected-contexts/`, chỉ đọc các tệp kết thúc bằng `.json`.
2. Validate tên file, schema, `id` duy nhất và filename-ID consistency.
3. Chuyển `id` sang chuỗi nội bộ.
4. Giữ raw fields; tạo retrieval text từ `name + passage`, fallback sang URL slug nếu cần.
5. Với văn bản rất dài, chunk theo cấu trúc pháp lý/độ dài có overlap; mọi chunk phải giữ parent `document_id`.
6. Lập index và ghi metadata đủ để tái tạo index.

### 2. Train/validation

1. Nạp `train.json`, validate mỗi answer có 1..5 ID và ID có trong corpus.
2. Group các câu trùng sau normalize trước khi split.
3. Retrieve/rerank ở mức chunk nếu cần nhưng aggregate score về document trước khi đánh giá.
4. Deduplicate và thử `k = 1..5` hoặc threshold trên validation.
5. Chọn cấu hình tốt nhất theo tuple `(macro_recall, macro_precision)`, không gộp hai metric bằng trọng số tùy ý.

### 3. Public inference/submission

1. Nạp đủ 1.000 câu từ `public-official.json` và bỏ qua placeholder `answer: null`.
2. Sinh tối đa 5 document ID hợp lệ cho từng question ID.
3. Validate toàn bộ key, kiểu chuỗi, uniqueness và corpus membership.
4. Ghi duy nhất `submission.json` vào root của `submission.zip`.

Định dạng yêu cầu:

```json
{
  "147194": {
    "answer": ["177504", "740"]
  }
}
```

## Stack and Commands

| Area | Trạng thái | Evidence |
|---|---|---|
| Data format | JSON + ZIP + DOCX | Các asset ở root |
| Ngôn ngữ/runtime | Python | `codebase/*.py` |
| Package manager | pip requirements | `codebase/requirements*.txt` |
| Index/retrieval engine | bm25s + sentence-transformers BGE-M3 + FAISS | `codebase/legalir.py` |
| Reranker | Unsloth-optimized Transformers + PEFT QLoRA + bitsandbytes | `codebase/train_qlora.py` |
| Build/train/eval command | argparse CLIs; Accelerate for 2-GPU train | `codebase/README.md` |
| Test framework | Python `unittest` (chỉ cho CLI hiện tại) | `codebase/test_inspect_non_passage.py` |

CLI kiểm tra data quality hiện có:

```text
python codebase/inspect_non_passage.py
python codebase/inspect_non_passage.py --json
python -m pip install -r codebase/requirements.txt
python codebase/crawl_non_passage.py
python -m unittest discover -s codebase -p "test_*.py"
```

Crawler dùng browser thật vì request HTTP trực tiếp tới nguồn bị Cloudflare chặn. Mặc định crawler mở Microsoft Edge, chạy tuần tự, ghi vào `recovered-contexts/` và chỉ sửa corpus gốc khi gọi với `--in-place`. Nếu Cloudflare Turnstile không tự hoàn tất sau 10 giây, crawler fail fast và yêu cầu nguồn/API được cấp phép; không có logic né CAPTCHA.

Không ghi các command giả như thể chúng đã hoạt động. Khi bắt đầu code, README/manifest phải trở thành nguồn chuẩn cho setup, index, train, evaluate, predict và package submission.

## Implementation Guide

| Change type | Primary touchpoints | Pattern phải giữ | Tests/checks bắt buộc |
|---|---|---|---|
| Data loader | Module data/schema mới | ID nội bộ là string; field optional; raw + normalized tách biệt | Full-scan schema, unique ID, filename-ID match, gold coverage |
| Text preprocessing | Module normalization/chunking mới | Bảo toàn citation/legal structure và parent document ID | Vietnamese diacritics, điều/khoản, whitespace, empty passage cases |
| Retriever/index | Module index + retrieve mới | Output `(doc_id, score)` deterministic cho cùng config | Recall@1..5, latency, reproducible index metadata |
| Reranking/fusion | Module rerank mới | Aggregate chunk -> document trước top-k | Ablation against lexical baseline; no duplicate IDs |
| Evaluator | Module metrics mới | Macro per-query; Precision empty=0; over-5 query=0 | Golden unit cases cho exact match, empty, partial, duplicate, >5 |
| Submission builder | Module submit mới | Đủ question IDs; string IDs; max 5; ZIP chỉ có 1 JSON | Schema + corpus membership + ZIP member audit |
| Experiment split/config | Config + experiment runner mới | Group duplicate question texts; seed/version cố định | Leakage check; config snapshot; repeatability |

## Baseline Strategy

Baseline nhỏ dưới đây đã được triển khai; index/model experiment vẫn chưa chạy trong workspace:

1. Lexical retrieval (BM25 hoặc tương đương) trên `name + passage`, có word n-gram và/hoặc character n-gram để chịu lỗi tách từ tiếng Việt.
2. Chunk các văn bản dài nhưng aggregate về document ID.
3. Đánh giá group-aware validation với đúng macro Recall/Precision.
4. Tuning top-k trong 1..5 theo ưu tiên Recall rồi mới Precision.
5. Sau khi có baseline đáng tin cậy mới thêm dense retrieval, query expansion hoặc cross-encoder reranking và luôn so sánh bằng ablation.

Đây là hướng đề xuất, không phải stack đã được quyết định bởi tài liệu hoặc repo.

## Conventions and Invariants

- Không biến task thành question answering/generation; output được chấm là document ID.
- Không dùng `public-official.json` như dữ liệu có nhãn.
- Không đánh giá micro metric thay cho macro metric.
- Không để chunk ID lọt ra submission; chỉ parent document ID hợp lệ được nộp.
- Không vượt quá 5 ID dù nội bộ retrieval trả nhiều candidates hơn.
- Mọi prediction phải deterministic khi seed/config/index không đổi.
- Luôn lưu cấu hình preprocessing/index cùng kết quả experiment để tránh so sánh sai phiên bản.
- Không đưa credential, token hoặc giá trị bí mật vào context/config được commit.
- Khi được giao **task code**, chỉ làm code trong folder "./codebase"

## Risks and Unknowns

### Đã xác minh

- Workspace chưa là Git repository; không có diff/history chuẩn để phân biệt thay đổi theo revision.
- QLoRA runtime/model weights chưa được cài hoặc tải trong workspace, nên chỉ compile, CLI và unit-test phần data path; GPU training/reranking phải xác minh trên Kaggle T4.
- Nhãn train chỉ ở document-level. Pair builder chọn chunk có lexical overlap tốt nhất rồi gán positive `5`; đây là weak supervision và có thể tạo label noise nếu chunk trả lời dùng từ khác câu hỏi.
- `name` thiếu ở 1.125 context; `passage` rỗng ở 20 context; 6 context rỗng passage có mặt trong gold train.
- Corpus có văn bản cực dài, nên full-document embedding/context stuffing sẽ gặp giới hạn bộ nhớ hoặc context window.
- Train có duplicate question text và một số duplicate có nhãn không đồng nhất.
- Public set không có nhãn thật; chỉ có placeholder `null`.

### Cần xác nhận khi organizer cung cấp thêm artifact

- Official evaluator có canonicalize ID, loại duplicate prediction hay coi duplicate là phần tử riêng.
- Thứ tự prediction có ảnh hưởng nào ngoài việc chọn top 5 hay không.
- Có bắt buộc submission chứa đúng và đủ toàn bộ question ID hay evaluator tự xử lý key thiếu/thừa.
- Encoding/ZIP metadata cụ thể ngoài yêu cầu `submission.zip` chứa duy nhất `submission.json`.
- Domain shift và schema cuối cùng của `private-official.json`.
- License/quy định dùng external model, external corpus, internet hoặc API trong cuộc thi.

Cho tới khi có evaluator/rules đầy đủ, validator nội bộ nên dùng cách hiểu nghiêm ngặt nhất: đúng toàn bộ question keys, ID dạng chuỗi, unique, thuộc corpus, tối đa 5 và ZIP không có member thừa.

## Refresh Notes

- Đã đọc cấu trúc và nội dung nghiệp vụ của file DOCX; không sửa file nguồn.
- Đã quét toàn bộ `train.json`, `public-official.json` và 8.532 JSON trong `selected-contexts/selected-contexts/` để xác minh schema, counts, ID coverage và anomaly chính.
- Các CLI kiểm tra non-passage và crawler Playwright vẫn tách biệt khỏi retrieval/model training.
- Đã chuyển zero-shot mặc định sang `Qwen/Qwen3-0.6B`, tắt thinking mode, nạp FP16 trực tiếp khi không có adapter và tạo cả JSON/ZIP từ `--submission`.
- Reranker dùng cùng prompt và next-token grade probabilities cho zero-shot/fine-tuned; QLoRA adapter vẫn dùng NF4 double quantization.
- Ngày 2026-08-23, 5 unit tests không cần dependency ML chạy thành công; full suite và validation runtime chưa chạy vì môi trường thiếu NumPy/PyTorch/Transformers và việc tải dependency không được cấp quyền.
- Render trực quan DOCX không thực hiện được do runtime hiện tại không có LibreOffice/`soffice`; phần task contract được kiểm tra bằng cấu trúc OOXML và nội dung text/table.
- Cần refresh context sau khi thêm source code, manifest, evaluator chính thức, warmup/private data hoặc thay đổi submission rules.
