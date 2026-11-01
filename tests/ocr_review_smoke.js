// Smoke test ของหน้าตรวจ (scripts/ocr_review_template.html) ด้วย DOM จำลองแบบเบา — ไม่ต้องมี jsdom/เบราว์เซอร์
// ใช้: node tests/ocr_review_smoke.js <review.html>   พิมพ์ JSON สรุปผลออก stdout
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync(process.argv[2], 'utf8');
const dataText = /<script id="data" type="application\/json">([\s\S]*?)<\/script>/.exec(html)[1];
const code = /<script>([\s\S]*?)<\/script>/.exec(html)[1];

const storage = new Map();
function run() {
  const els = {};
  const el = id => (els[id] = els[id] || {
    id, innerHTML: '', textContent: id === 'data' ? dataText : '', style: {}, value: '', checked: false, tagName: 'DIV',
    onclick: null, onchange: null, onload: null, files: [], click() { if (this.onclick) this.onclick(); },
  });
  const downloads = [];
  const document = {
    getElementById: el,
    createElement: () => ({ click() { downloads.push(this); }, set href(v) {}, set download(v) { this._name = v; } }),
    addEventListener: (ev, fn) => { document._keys = fn; },
  };
  class Blob { constructor(parts) { this.text = parts.join(''); } }
  const ctx = {
    document, Blob, JSON, Math, Object, Set, Array, String,
    localStorage: { getItem: k => (storage.has(k) ? storage.get(k) : null), setItem: (k, v) => storage.set(k, v) },
    URL: { createObjectURL: b => { downloads.push({ blob: b }); return 'blob:x'; } },
    window: { confirm: () => true }, alert: () => {}, console,
  };
  vm.createContext(ctx);
  vm.runInContext(code, ctx);
  return { els, downloads, document };
}

const out = {};
let s = run();
out.countInitial = s.els.count.textContent;
out.firstCardHasKind = /tag /.test(s.els.card.innerHTML);
out.hasPageOptions = /<option/.test(s.els.fPage.innerHTML);
// ตัดสินรายการแรกด้วย "ไม่มีในภาพ"
s.els.bAbsent.onclick();
out.countAfterOne = s.els.count.textContent;
// รายการถัดไป: พิมพ์ข้อความแก้เองแล้วบันทึก
s.els.truth.value = 'ข้อความที่แก้';
s.els.bSave.onclick();
out.countAfterTwo = s.els.count.textContent;
// ดาวน์โหลด
s.els.btnDownload.onclick();
const blob = s.downloads.find(d => d.blob).blob;
const dl = JSON.parse(blob.text);
out.download = { version: dl.version, total: dl.total, done: dl.done, verdicts: Object.values(dl.judgments).map(v => v.verdict) };
// "โหลดหน้าใหม่": ความคืบหน้าต้องยังอยู่
s = run();
out.countAfterReload = s.els.count.textContent;
// ตัวกรอง "เฉพาะที่ยังไม่ทำ" ต้องข้ามที่ทำแล้ว
s.els.fUndone.checked = true;
s.els.fUndone.onchange();
out.posUndone = s.els.pos.textContent;
console.log(JSON.stringify(out));
