# status ที่เกณฑ์ใหม่ (ค่าเริ่มต้นของ API) — questions.jsonl ผ่าน POST /v1/retrieve

- เกณฑ์: RETRIEVE_MIN_SCORE 0.55 (dense top_score, reranker ปิด) · RERANK_MIN_SCORE 0.55 (top_relevance, RERANKER=bge-v2-m3 บน cuda)
- **เกณฑ์เลือกจากครึ่ง calibrate** (stratified ตาม type, seed 20261123; ดู questions_eval.md ข้อ B) — ตัวเลขบนชุดเต็มรวมครึ่ง calibrate ที่ใช้เลือกเกณฑ์ด้วย จึง**ดีเกินจริงเล็กน้อย**โดยหลักการ · คอลัมน์ครึ่ง test คือตัวเลขที่ไม่วัดวน · ค่าเริ่มต้น 0.55 ปัดจากเกณฑ์ที่เลือกได้ (dense 0.5483, reranker 0.5513) จึงต่างจากตาราง B ของ questions_eval.md เล็กน้อย
- รอบนี้ชุดเต็ม**ไม่ได้ดีกว่า**ครึ่ง test ในช่อง status_baseline/question, status_baseline/unanswerable, status_reranker/unanswerable (ครึ่ง test เล็ก — unanswerable 19, chitchat 6 — สัดส่วนแกว่งได้มาก) ใช้ทั้งสองคอลัมน์ประกอบกัน
- ตรวจแล้ว: status ทุก request ตรงกับกติกาที่เกณฑ์ค่าเริ่มต้น และ top_relevance มี/null ตรงโหมด
- วัดเมื่อ: status_baseline 2026-11-23 13:25:30 · status_reranker 2026-11-23 13:27:54

| config (กติกา) | type | ชุดเต็ม: no_relevant_context | ครึ่ง test: no_relevant_context |
|---|---|---:|---:|
| status_baseline (top_score < RETRIEVE_MIN_SCORE) | คำถามจริง (ปฏิเสธผิด) | 10/238 (4.2%) | 3/119 (2.5%) |
| status_baseline (top_score < RETRIEVE_MIN_SCORE) | unanswerable (ปฏิเสธถูก) | 22/38 (57.9%) | 12/19 (63.2%) |
| status_baseline (top_score < RETRIEVE_MIN_SCORE) | chitchat (ปฏิเสธถูก) | 8/12 (66.7%) | 4/6 (66.7%) |
| status_reranker (top_relevance < RERANK_MIN_SCORE) | คำถามจริง (ปฏิเสธผิด) | 10/238 (4.2%) | 5/119 (4.2%) |
| status_reranker (top_relevance < RERANK_MIN_SCORE) | unanswerable (ปฏิเสธถูก) | 29/38 (76.3%) | 16/19 (84.2%) |
| status_reranker (top_relevance < RERANK_MIN_SCORE) | chitchat (ปฏิเสธถูก) | 12/12 (100.0%) | 6/6 (100.0%) |

- คำถามจริงที่ได้ no_relevant_context ใน status_baseline: q005, q016, q039, q040, q055, q066, q071, q199, q223, q237
- คำถามจริงที่ได้ no_relevant_context ใน status_reranker: q005, q006, q007, q008, q012, q040, q084, q104, q149, q207
