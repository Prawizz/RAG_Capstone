"""vector store: แยก collection ตาม visibility, index เฉพาะ unit ok, chunk + named vectors, และสิทธิ์ 2 ด่าน

test ปกติใช้ Qdrant ในกระบวนการ (``VectorStore.in_memory``) — semantics ของ filter เหมือนของจริง
แต่ไม่มี payload index ส่วน test ที่ใช้ server จริงจาก docker อยู่ท้ายไฟล์ (mark ``qdrant``)

test เรื่องสิทธิ์ (ที่ใช้ fixture ``store`` / ``search_mode``) รันซ้ำทั้ง 4 โหมดค้น:
dense · dense+sparse_bge · dense+sparse_bm25 · all
"""

import hashlib

import pytest

from app.access import AclStore, Box, BoxRole, FileAcl, Principal, Visibility
from app.chunking import make_chunks
from app.config import Settings
from app.embedding import HashEmbedder, SparseVector
from app.models import (
    Document,
    ExtractionMethod,
    QualityFlag,
    QualityReport,
    Unit,
    UnitStatus,
    UnitType,
)
from app.vectorstore import (
    EMPTY_RESULT,
    SEARCH_MODES,
    AclUnknownFileError,
    CollectionSchemaError,
    IndexedChunk,
    Query,
    SearchModeError,
    VectorSizeError,
    VectorStore,
    embed_chunks,
    embed_query,
    fetch_limit,
    point_id,
)

VECTOR_SIZE = 8
SETTINGS = Settings(
    vector_size=VECTOR_SIZE, collection_public="test_public", collection_private="test_private"
)

# เวกเตอร์กำหนดเองให้ผลค้นคาดเดาได้ — test นี้วัดการกรองสิทธิ์ ไม่ได้วัดคุณภาพ embedding
# sparse ของคำถามกับของ chunk เป็นตัวเดียวกันหมด (index 1 น้ำหนัก 1) จึงไม่เปลี่ยนว่าใครมีสิทธิ์เห็นอะไร
SPARSE_ONE = SparseVector((1,), (1.0,))
QUERY = Query(dense=[1.0] + [0.0] * (VECTOR_SIZE - 1), sparse_bge=SPARSE_ONE, sparse_bm25=SPARSE_ONE)


def vec(n: float = 1.0) -> list[float]:
    return [n] + [0.0] * (VECTOR_SIZE - 1)


def make_unit(
    index: int,
    text: str,
    *,
    status: UnitStatus = UnitStatus.OK,
    flags=(),
    reason=None,
) -> Unit:
    return Unit(
        index=index,
        type=UnitType.PAGE,
        position={"page": index + 1},
        label=f"หน้า {index + 1}",
        status=status,
        reason=reason,
        method=ExtractionMethod.TEXT_LAYER,
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
        quality=QualityReport(flags=list(flags)),
    )


def make_doc(file_id: str, units: list[Unit], filename: str = "เอกสาร.pdf") -> Document:
    return Document(
        file_id=file_id,
        content_hash="0" * 64,
        filename=filename,
        detected_type="pdf",
        size_bytes=1234,
        units=units,
        pipeline_version="test",
        normalizer_version="test",
    )


def join(acl: AclStore, principal: Principal) -> Principal:
    """ลงทะเบียนสมาชิกภาพใน AclStore ตาม ``principal.box_ids`` (ด่าน 2 ถามสมาชิกภาพสดจาก AclStore)"""
    for box_id in principal.box_ids:
        if acl.get_box(box_id) is not None:
            acl.set_member(box_id, principal.user_id, BoxRole.VIEWER)
    return principal


@pytest.fixture
def acl() -> AclStore:
    store = AclStore()
    store.register_box(Box(box_id="pub", visibility=Visibility.PUBLIC))
    store.register_box(
        Box(box_id="hr", visibility=Visibility.PRIVATE, departments=frozenset({"hr"}))
    )
    store.register_file(FileAcl(file_id="f_pub", box_id="pub"))
    store.register_file(FileAcl(file_id="f_hr", box_id="hr"))
    return store


@pytest.fixture(params=list(SEARCH_MODES))
def search_mode(request) -> str:
    return request.param


def new_store(acl: AclStore, mode: str) -> VectorStore:
    return VectorStore.in_memory(acl, SETTINGS.model_copy(update={"search_mode": mode}))


@pytest.fixture
def store(acl, search_mode) -> VectorStore:
    return new_store(acl, search_mode)


@pytest.fixture
def alice(acl) -> Principal:
    return join(acl, Principal(
        user_id="alice",
        departments=frozenset({"hr"}),
        security_level=5,
        box_ids=frozenset({"pub", "hr"}),
    ))


def indexed(unit: Unit, dense=None, target: int = 1000) -> list[IndexedChunk]:
    """chunk ของ unit พร้อมเวกเตอร์กำหนดเอง (dense ตัวเดียวกันทุก chunk, sparse = SPARSE_ONE)"""
    return [IndexedChunk(c, dense or vec(), SPARSE_ONE, SPARSE_ONE) for c in make_chunks(unit, target)]


def index_one(store: VectorStore, file_id: str, text: str = "เนื้อหาทดสอบภาษาไทย", dense=None):
    doc = make_doc(file_id, [make_unit(0, text)])
    return store.upsert_chunks(doc, indexed(doc.units[0], dense))


# ---------- collection แยกตาม visibility ----------


def test_public_and_private_go_to_separate_collections(store, acl):
    assert index_one(store, "f_pub").collection == "test_public"
    assert index_one(store, "f_hr").collection == "test_private"


def test_collection_is_chosen_from_the_store_not_from_input(store, acl, alice):
    """visibility มาจาก Box ใน AclStore เท่านั้น — ผู้ใช้ชี้ collection เองไม่ได้"""
    index_one(store, "f_hr", "ข้อมูลลับของ HR")
    assert store.client.count("test_public", exact=True).count == 0
    assert store.client.count("test_private", exact=True).count == 1
    hits = store.search(QUERY, alice, "hr").hits
    assert [h.file_id for h in hits] == ["f_hr"]


def test_public_box_search_never_returns_private_points(store, acl, alice):
    index_one(store, "f_pub", "คู่มือพนักงาน")
    index_one(store, "f_hr", "เงินเดือน")
    assert [h.file_id for h in store.search(QUERY, alice, "pub").hits] == ["f_pub"]


# ---------- index เฉพาะ unit ที่ใช้ได้ ----------


def test_only_ok_units_are_indexed(store, acl):
    doc = make_doc(
        "f_pub",
        [
            make_unit(0, "หน้าที่อ่านได้"),
            make_unit(1, "", status=UnitStatus.NEEDS_OCR, reason="หน้าเป็นภาพ"),
            make_unit(2, "พัง", status=UnitStatus.FAILED, reason="อ่านไม่ได้"),
            make_unit(3, "   ", flags=[QualityFlag.EMPTY]),
        ],
    )
    report = store.upsert_document(doc, HashEmbedder(VECTOR_SIZE))
    assert (report.indexed, report.skipped_not_ok, report.skipped_empty) == (1, 2, 1)
    assert store.client.count("test_public", exact=True).count == 1


def test_cannot_index_a_file_with_unknown_acl(store):
    """เนื้อหาที่ไม่รู้ว่าใครเห็นได้ ห้ามเข้า index (fail closed)"""
    doc = make_doc("ยังไม่ลงทะเบียน", [make_unit(0, "ข้อความ")])
    with pytest.raises(AclUnknownFileError):
        store.upsert_chunks(doc, indexed(doc.units[0]))
    with pytest.raises(AclUnknownFileError):
        store.upsert_document(doc, HashEmbedder(VECTOR_SIZE))


def test_vector_size_mismatch_is_rejected(store, acl):
    doc = make_doc("f_pub", [make_unit(0, "ข้อความ")])
    with pytest.raises(VectorSizeError):
        store.upsert_chunks(doc, indexed(doc.units[0], dense=[1.0, 2.0]))


def test_query_vector_size_mismatch_is_rejected(store, acl, alice):
    with pytest.raises(VectorSizeError):
        store.search(Query(dense=[1.0, 2.0], sparse_bge=SPARSE_ONE, sparse_bm25=SPARSE_ONE), alice, "pub")


def test_reingest_overwrites_instead_of_duplicating(store, acl):
    index_one(store, "f_pub", "รอบแรก")
    index_one(store, "f_pub", "รอบสอง")
    assert store.client.count("test_public", exact=True).count == 1
    assert point_id("f_pub", 0, 0) == point_id("f_pub", 0, 0) != point_id("f_pub", 0, 1)


def test_delete_file_removes_its_points(store, acl, alice):
    index_one(store, "f_pub")
    store.delete_file("f_pub")
    assert store.client.count("test_public", exact=True).count == 0
    assert store.search(QUERY, alice, "pub") == EMPTY_RESULT


def test_payload_carries_the_contract_fields(store, acl, alice):
    index_one(store, "f_pub", "ข้อความอ้างอิงได้")
    hit = store.search(QUERY, alice, "pub").hits[0]
    assert hit.label == "หน้า 1"
    assert hit.position == {"page": 1}
    assert (hit.unit_index, hit.unit_type, hit.status) == (0, "page", "ok")
    assert hit.method == "text_layer" and hit.filename == "เอกสาร.pdf"
    assert hit.text == "ข้อความอ้างอิงได้"
    assert (hit.chunk_index, hit.char_start, hit.char_end) == (0, 0, len("ข้อความอ้างอิงได้"))
    assert hit.sentences == ((0, len("ข้อความอ้างอิงได้")),) and hit.context == ""


# ---------- ด่าน 1: pre-filter ----------


def test_member_of_another_department_gets_nothing(store, acl):
    """เป็นสมาชิก box แต่คนละแผนก — pre-filter ต้องกรองออก"""
    index_one(store, "f_hr", "ข้อมูล HR")
    bob = join(acl, Principal(
        user_id="bob",
        departments=frozenset({"finance"}),
        security_level=5,
        box_ids=frozenset({"pub", "hr"}),
    ))
    assert store.search(QUERY, bob, "hr") == EMPTY_RESULT


def test_security_level_below_the_box_gets_nothing(store, acl):
    acl.register_box(
        Box(box_id="secret", visibility=Visibility.PRIVATE, min_security_level=4)
    )
    acl.register_file(FileAcl(file_id="f_secret", box_id="secret"))
    index_one(store, "f_secret", "เอกสารชั้นความลับ")
    junior = join(acl, Principal(user_id="j", security_level=1, box_ids=frozenset({"secret"})))
    senior = join(acl, Principal(user_id="s", security_level=4, box_ids=frozenset({"secret"})))
    assert store.search(QUERY, junior, "secret") == EMPTY_RESULT
    assert len(store.search(QUERY, senior, "secret").hits) == 1


def test_principal_without_any_department(store, acl):
    """แผนกว่าง -> MatchAny ของ filter ได้ list ว่าง ต้องไม่ทำให้เงื่อนไข should พังทั้งก้อน

    box ที่ไม่จำกัดแผนกต้องยังค้นได้ ส่วน box ที่จำกัดต้องไม่ได้อะไร
    """
    index_one(store, "f_pub", "ประกาศทั่วไป")
    index_one(store, "f_hr", "ข้อมูล HR")
    nobody = join(acl, Principal(user_id="n", security_level=5, box_ids=frozenset({"pub", "hr"})))
    assert [h.file_id for h in store.search(QUERY, nobody, "pub").hits] == ["f_pub"]
    assert store.search(QUERY, nobody, "hr") == EMPTY_RESULT


def test_file_level_narrowing_applies_at_search_time(store, acl):
    acl.register_file(FileAcl(file_id="f_hr2", box_id="hr", min_security_level=9))
    index_one(store, "f_hr2", "ไฟล์ที่ยกระดับเฉพาะตัว")
    mid = join(acl, Principal(
        user_id="m", departments=frozenset({"hr"}), security_level=5, box_ids=frozenset({"hr"})
    ))
    assert store.search(QUERY, mid, "hr") == EMPTY_RESULT


# ---------- ด่าน 2: re-check กับ ACL store ----------


def test_stale_payload_is_caught_by_the_second_gate(store, acl, alice):
    """สิทธิ์เปลี่ยนแต่ยังไม่ re-index — pre-filter ปล่อยผ่าน ด่าน 2 ต้องจับได้"""
    index_one(store, "f_hr", "ข้อมูลที่สิทธิ์เปลี่ยนทีหลัง")
    assert len(store.search(QUERY, alice, "hr").hits) == 1

    acl.register_box(  # ยกระดับ box แต่ payload ใน index ยังเป็นค่าเดิม
        Box(box_id="hr", visibility=Visibility.PRIVATE, departments=frozenset({"hr"}),
            min_security_level=9)
    )
    result = store.search(QUERY, alice, "hr")
    assert result.hits == ()
    assert result.dropped_by_recheck == 1  # สัญญาณว่า index ค้างเก่า ต้อง re-index


def test_revoked_file_is_caught_by_the_second_gate(store, acl, alice):
    index_one(store, "f_hr", "ไฟล์ที่ถูกถอนสิทธิ์")
    acl.remove_file("f_hr")
    result = store.search(QUERY, alice, "hr")
    assert result.hits == () and result.dropped_by_recheck == 1


# ---------- ไม่มีสิทธิ์ ต้องเหมือนหาไม่เจอ (กฎเหล็กข้อ 9) ----------


def test_non_member_box_is_indistinguishable_from_empty(store, acl):
    index_one(store, "f_hr", "ข้อมูลลับ")
    outsider = join(acl, Principal(user_id="o", departments=frozenset({"hr"}), security_level=5,
                         box_ids=frozenset({"pub"})))
    denied = store.search(QUERY, outsider, "hr")       # ไม่ได้เป็นสมาชิก
    nothing = store.search(QUERY, outsider, "pub")     # เป็นสมาชิกแต่ไม่มีข้อมูล
    assert denied == nothing == EMPTY_RESULT
    assert denied.dropped_by_recheck == nothing.dropped_by_recheck == 0


def test_unknown_box_is_indistinguishable_from_empty(store, acl, alice):
    assert store.search(QUERY, alice, "ไม่มีกล่องนี้") == EMPTY_RESULT


def test_no_leak_across_the_whole_matrix(store, acl):
    """ไล่ทุกคู่ (principal x box) — ไม่มีใครได้ hit ของไฟล์ที่ตัวเองไม่มีสิทธิ์"""
    index_one(store, "f_pub", "สาธารณะ")
    index_one(store, "f_hr", "ลับ")
    principals = [
        join(acl, Principal(user_id="hr_user", departments=frozenset({"hr"}), security_level=5,
                  box_ids=frozenset({"pub", "hr"}))),
        join(acl, Principal(user_id="fin_user", departments=frozenset({"finance"}), security_level=5,
                  box_ids=frozenset({"pub", "hr"}))),
        join(acl, Principal(user_id="outsider", departments=frozenset({"hr"}), security_level=5,
                  box_ids=frozenset())),
        join(acl, Principal(user_id="no_dept", security_level=5, box_ids=frozenset({"pub", "hr"}))),
    ]
    leaks = []
    for principal in principals:
        for box_id in ("pub", "hr", "ไม่มีกล่องนี้"):
            for hit in store.search(QUERY, principal, box_id).hits:
                if not acl.can_access(principal, hit.file_id):
                    leaks.append(f"{principal.user_id} เห็น {hit.file_id} ใน box {box_id}")
    assert not leaks, "สิทธิ์รั่ว:\n" + "\n".join(leaks)


# ---------- Qdrant จริงจาก docker (pytest -m qdrant) ----------


@pytest.mark.qdrant
def test_against_real_qdrant_server(acl, alice):
    """ยืนยันกับ server จริง เพราะ local mode ไม่มี payload index และไม่ใช่โค้ดชุดเดียวกัน

    ใช้ชื่อ collection แยกและลบทิ้งท้าย test
    """
    from qdrant_client import QdrantClient

    settings = Settings(
        vector_size=VECTOR_SIZE,
        collection_public="pytest_tmp_public",
        collection_private="pytest_tmp_private",
    )
    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=int(settings.qdrant_timeout))
        client.get_collections()
    except Exception as e:  # ไม่มี docker ขึ้นอยู่ -> ข้าม ไม่ใช่ fail
        pytest.skip(f"ไม่มี Qdrant ที่ {settings.qdrant_url}: {e}")

    for name in (settings.collection_public, settings.collection_private):
        if client.collection_exists(name):
            client.delete_collection(name)
    try:
        store = VectorStore(client, acl, settings)
        store.ensure_collections()
        store.ensure_collections()  # เรียกซ้ำได้
        index_one(store, "f_pub", "คู่มือพนักงาน")
        index_one(store, "f_hr", "เงินเดือน")

        assert [h.file_id for h in store.search(QUERY, alice, "pub").hits] == ["f_pub"]
        assert [h.file_id for h in store.search(QUERY, alice, "hr").hits] == ["f_hr"]
        outsider = join(acl, Principal(user_id="o", departments=frozenset({"hr"}), security_level=5,
                             box_ids=frozenset({"pub"})))
        assert store.search(QUERY, outsider, "hr") == EMPTY_RESULT
        store.delete_file("f_pub")
        assert store.search(QUERY, alice, "pub") == EMPTY_RESULT
    finally:
        for name in (settings.collection_public, settings.collection_private):
            if client.collection_exists(name):
                client.delete_collection(name)


# ---------- ด่าน 2 ต้องเช็ค box และ visibility ด้วย ----------


def _both_boxes_member(acl: AclStore) -> Principal:
    return join(acl, Principal(user_id="m", security_level=5, box_ids=frozenset({"pub", "priv"})))


@pytest.fixture
def moving_acl() -> AclStore:
    store = AclStore()
    for box_id, vis in (("pub", Visibility.PUBLIC), ("pub2", Visibility.PUBLIC), ("priv", Visibility.PRIVATE)):
        store.register_box(Box(box_id=box_id, visibility=vis))
    return store


def test_file_moved_public_to_private_does_not_leak_from_the_public_search(moving_acl, search_mode):
    """probe: index ใน pub -> ย้าย FileAcl ไป priv โดยไม่ re-index -> user ที่อยู่ทั้งสอง box ค้นใน pub ต้องไม่ได้ f1"""
    moving_acl.register_file(FileAcl(file_id="f1", box_id="pub"))
    store = new_store(moving_acl, search_mode)
    index_one(store, "f1", "เนื้อหาที่กลายเป็นความลับ")
    user = _both_boxes_member(moving_acl)
    assert len(store.search(QUERY, user, "pub").hits) == 1

    moving_acl.register_file(FileAcl(file_id="f1", box_id="priv"))
    result = store.search(QUERY, user, "pub")
    assert result.hits == () and result.dropped_by_recheck == 1
    assert store.search(QUERY, user, "priv").hits == ()  # ยังไม่ re-index -> ใน priv ก็ไม่เจอ


def test_file_moved_private_to_public_does_not_appear_in_the_private_search(moving_acl, search_mode):
    moving_acl.register_file(FileAcl(file_id="f1", box_id="priv"))
    store = new_store(moving_acl, search_mode)
    index_one(store, "f1", "เนื้อหาเดิมของกล่องส่วนตัว")
    user = _both_boxes_member(moving_acl)
    assert len(store.search(QUERY, user, "priv").hits) == 1

    moving_acl.register_file(FileAcl(file_id="f1", box_id="pub"))
    result = store.search(QUERY, user, "priv")
    assert result.hits == () and result.dropped_by_recheck == 1


def test_file_moved_between_boxes_of_the_same_visibility_is_dropped_by_box_check(moving_acl, search_mode):
    """visibility เท่ากัน (public -> public) แต่ box ไม่ใช่ box ที่ค้น -> ทิ้งด้วยข้อ box_id"""
    moving_acl.register_file(FileAcl(file_id="f1", box_id="pub"))
    store = new_store(moving_acl, search_mode)
    index_one(store, "f1")
    user = join(moving_acl, Principal(user_id="m", box_ids=frozenset({"pub", "pub2"})))
    moving_acl.register_file(FileAcl(file_id="f1", box_id="pub2"))
    result = store.search(QUERY, user, "pub")
    assert result.hits == () and result.dropped_by_recheck == 1


# ---------- refresh_acl_payload ----------


def test_refresh_updates_payload_in_place_when_visibility_is_unchanged(moving_acl, search_mode):
    moving_acl.register_file(FileAcl(file_id="f1", box_id="pub"))
    store = new_store(moving_acl, search_mode)
    index_one(store, "f1", "ย้ายไปอีกกล่องสาธารณะ")
    user = join(moving_acl, Principal(user_id="m", box_ids=frozenset({"pub", "pub2"})))
    moving_acl.register_file(FileAcl(file_id="f1", box_id="pub2"))

    report = store.refresh_acl_payload("f1")
    assert (report.updated, report.removed, report.needs_reindex) == (1, 0, False)
    assert report.collection == SETTINGS.collection_public
    assert store.search(QUERY, user, "pub") == EMPTY_RESULT  # ไม่ใช่ box นี้แล้ว (ด่าน 1 กรองเอง ไม่นับ dropped)
    moved = store.search(QUERY, user, "pub2")
    assert [h.file_id for h in moved.hits] == ["f1"] and moved.dropped_by_recheck == 0  # ไม่ต้อง embed ใหม่


def test_refresh_after_box_tightened_makes_the_prefilter_agree_with_the_store(store, acl, alice):
    index_one(store, "f_hr", "ข้อมูลที่ box ถูกยกระดับ")
    changed = acl.register_box(
        Box(box_id="hr", visibility=Visibility.PRIVATE, departments=frozenset({"hr"}), min_security_level=9)
    )
    assert changed == ["f_hr"]
    assert store.search(QUERY, alice, "hr").dropped_by_recheck == 1  # ก่อน refresh: ด่าน 2 จับ

    report = store.refresh_acl_payload("f_hr")
    assert (report.updated, report.needs_reindex) == (1, False)
    after = store.search(QUERY, alice, "hr")
    assert after.hits == () and after.dropped_by_recheck == 0  # หลัง refresh: ด่าน 1 กรองเอง


def test_refresh_with_changed_visibility_deletes_old_points_and_asks_for_reindex(moving_acl, search_mode):
    moving_acl.register_file(FileAcl(file_id="f1", box_id="pub"))
    store = new_store(moving_acl, search_mode)
    index_one(store, "f1", "ย้ายข้าม visibility")
    user = _both_boxes_member(moving_acl)
    moving_acl.register_file(FileAcl(file_id="f1", box_id="priv"))

    report = store.refresh_acl_payload("f1")
    assert report.needs_reindex and report.removed == 1 and report.updated == 0
    assert report.removed_from == (SETTINGS.collection_public,)
    assert report.collection == SETTINGS.collection_private
    # ไม่ได้ย้าย point ไป private เงียบๆ: ไม่มีที่ไหนเจอจนกว่าจะ re-index
    assert store.client.count(SETTINGS.collection_public).count == 0
    assert store.client.count(SETTINGS.collection_private).count == 0
    assert store.search(QUERY, user, "pub") == EMPTY_RESULT and store.search(QUERY, user, "priv") == EMPTY_RESULT

    index_one(store, "f1", "ย้ายข้าม visibility")  # re-index แล้วเจอใน priv เท่านั้น
    assert [h.file_id for h in store.search(QUERY, user, "priv").hits] == ["f1"]
    assert store.search(QUERY, user, "pub") == EMPTY_RESULT


def test_refresh_leaves_other_files_alone(store, acl, alice):
    index_one(store, "f_pub")
    index_one(store, "f_hr")
    store.refresh_acl_payload("f_hr")
    assert [h.file_id for h in store.search(QUERY, alice, "pub").hits] == ["f_pub"]
    assert [h.file_id for h in store.search(QUERY, alice, "hr").hits] == ["f_hr"]


def test_refresh_of_a_file_with_no_points_is_a_noop(store, acl):
    report = store.refresh_acl_payload("f_pub")
    assert (report.updated, report.removed, report.needs_reindex) == (0, 0, False)


def test_refresh_unknown_file_is_rejected_and_nothing_is_deleted(store, acl, alice):
    index_one(store, "f_hr")
    acl.remove_file("f_hr")
    with pytest.raises(AclUnknownFileError):
        store.refresh_acl_payload("f_hr")
    assert store.client.count(SETTINGS.collection_private).count == 1


# ---------- ดึงเกิน limit เผื่อด่าน 2 ----------


def test_fetch_limit_overfetches_up_to_the_cap():
    assert [fetch_limit(n) for n in (1, 10, 25, 30, 50, 100)] == [2, 20, 50, 50, 50, 100]


def test_search_overfetches_so_rechecked_hits_do_not_shrink_the_result(store, acl, alice):
    """top 2 ตามคะแนนเป็นไฟล์ที่ถูกถอนสิทธิ์ (ผ่านด่าน 1 ตกด่าน 2) — ถ้าดึงแค่ limit จะได้ผลไม่ครบ"""

    def direction(k: float) -> list[float]:
        return [1.0, k] + [0.0] * (VECTOR_SIZE - 2)  # k น้อย = ใกล้ QUERY มาก = คะแนนสูง

    for i, k in enumerate((0.0, 0.1, 1.0, 2.0, 3.0)):
        fid = ("stale" if i < 2 else "ok") + str(i)
        acl.register_file(FileAcl(file_id=fid, box_id="hr"))
        index_one(store, fid, f"เนื้อหา {fid}", dense=direction(k))
    acl.remove_file("stale0")
    acl.remove_file("stale1")

    result = store.search(QUERY, alice, "hr", limit=3)
    assert sorted(h.file_id for h in result.hits) == ["ok2", "ok3", "ok4"]  # ลำดับขึ้นกับโหมด (RRF) จึงเทียบเป็นชุด
    assert result.dropped_by_recheck == 2


def test_search_never_returns_more_than_limit(store, acl, alice):
    for i in range(5):
        acl.register_file(FileAcl(file_id=f"x{i}", box_id="hr"))
        index_one(store, f"x{i}")
    assert len(store.search(QUERY, alice, "hr", limit=2).hits) == 2
    assert len(store.search(QUERY, alice, "hr", limit=5).hits) == 5


@pytest.mark.qdrant
def test_moved_file_recheck_and_refresh_against_real_qdrant_server(moving_acl):
    """ด่าน 2 เช็ค box/visibility + refresh_acl_payload (set_payload แบบ filter, count, delete) กับ server จริง"""
    from qdrant_client import QdrantClient

    settings = Settings(
        vector_size=VECTOR_SIZE,
        collection_public="pytest_tmp2_public",
        collection_private="pytest_tmp2_private",
    )
    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=int(settings.qdrant_timeout))
        client.get_collections()
    except Exception as e:  # ไม่มี docker ขึ้นอยู่ -> ข้าม ไม่ใช่ fail
        pytest.skip(f"ไม่มี Qdrant ที่ {settings.qdrant_url}: {e}")

    for name in (settings.collection_public, settings.collection_private):
        if client.collection_exists(name):
            client.delete_collection(name)
    try:
        store = VectorStore(client, moving_acl, settings)
        store.ensure_collections()
        moving_acl.register_file(FileAcl(file_id="f1", box_id="pub"))
        index_one(store, "f1", "เนื้อหาที่ย้ายกล่อง")
        user = join(moving_acl, Principal(user_id="m", box_ids=frozenset({"pub", "pub2", "priv"})))

        moving_acl.register_file(FileAcl(file_id="f1", box_id="priv"))
        leaked = store.search(QUERY, user, "pub")
        assert leaked.hits == () and leaked.dropped_by_recheck == 1

        moving_acl.register_file(FileAcl(file_id="f1", box_id="pub2"))
        report = store.refresh_acl_payload("f1")
        assert (report.updated, report.removed, report.needs_reindex) == (1, 0, False)
        assert [h.file_id for h in store.search(QUERY, user, "pub2").hits] == ["f1"]

        moving_acl.register_file(FileAcl(file_id="f1", box_id="priv"))
        report = store.refresh_acl_payload("f1")
        assert report.needs_reindex and report.removed == 1
        assert client.count(settings.collection_public, exact=True).count == 0
    finally:
        for name in (settings.collection_public, settings.collection_private):
            if client.collection_exists(name):
                client.delete_collection(name)


# ---------- chunk + named vectors + โหมดค้น ----------


def long_text(n_lines: int) -> str:
    return "\n".join(f"บรรทัดที่ {i} ของเอกสารทดสอบ มีข้อความยาวพอให้ตัดเป็นหลาย chunk ได้" for i in range(n_lines))


def count(store: VectorStore, name: str = "test_public") -> int:
    return store.client.count(name, exact=True).count


def test_collection_has_three_named_vectors_with_idf_on_bm25(store):
    params = store.client.get_collection("test_public").config.params
    assert set(params.vectors) == {"dense"} and params.vectors["dense"].size == VECTOR_SIZE
    assert set(params.sparse_vectors) == {"sparse_bge", "sparse_bm25"}
    assert params.sparse_vectors["sparse_bm25"].modifier is not None
    assert params.sparse_vectors["sparse_bge"].modifier is None


def test_existing_collection_with_old_schema_is_rejected_not_patched(acl):
    from qdrant_client import QdrantClient, models

    client = QdrantClient(":memory:")
    client.create_collection("test_public", vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE))
    with pytest.raises(CollectionSchemaError):
        VectorStore(client, acl, SETTINGS, supports_payload_indexes=False).ensure_collections()


def test_one_unit_becomes_many_chunks_with_stable_point_ids(store, acl):
    doc = make_doc("f_pub", [make_unit(0, long_text(30))])
    small = SETTINGS.model_copy(update={"chunk_target_chars": 300})
    store.settings = small
    report = store.upsert_document(doc, HashEmbedder(VECTOR_SIZE))
    assert report.indexed > 3 and count(store) == report.indexed
    points, _ = store.client.scroll("test_public", limit=100, with_payload=True)
    assert {p.id for p in points} == {point_id("f_pub", 0, i) for i in range(report.indexed)}
    assert sorted(p.payload["chunk_index"] for p in points) == list(range(report.indexed))


def test_upsert_removes_old_chunks_when_the_chunk_count_shrinks(store, acl):
    """3 chunk -> เขียนใหม่เหลือ 1 chunk: chunk เก่าที่ id ไม่ซ้ำกับชุดใหม่ต้องไม่ค้าง"""
    store.settings = SETTINGS.model_copy(update={"chunk_target_chars": 300})
    store.upsert_document(make_doc("f_pub", [make_unit(0, long_text(30))]), HashEmbedder(VECTOR_SIZE))
    before = count(store)
    assert before >= 3
    store.upsert_document(make_doc("f_pub", [make_unit(0, "สั้นนิดเดียว")]), HashEmbedder(VECTOR_SIZE))
    assert count(store) == 1
    points, _ = store.client.scroll("test_public", limit=10, with_payload=True)
    assert points[0].payload["text"] == "สั้นนิดเดียว"


def test_upsert_removes_old_points_of_units_that_disappear(store, acl):
    store.upsert_document(make_doc("f_pub", [make_unit(0, "หน้าแรก"), make_unit(1, "หน้าสอง")]), HashEmbedder(VECTOR_SIZE))
    assert count(store) == 2
    store.upsert_document(make_doc("f_pub", [make_unit(0, "หน้าแรก")]), HashEmbedder(VECTOR_SIZE))
    assert count(store) == 1


def test_upsert_with_no_chunks_still_clears_the_old_ones(store, acl):
    index_one(store, "f_pub", "ของเดิม")
    doc = make_doc("f_pub", [make_unit(0, "", status=UnitStatus.NEEDS_OCR, reason="หน้าเป็นภาพ")])
    report = store.upsert_document(doc, HashEmbedder(VECTOR_SIZE))
    assert report.indexed == 0 and count(store) == 0


def test_upsert_clears_the_other_collection_when_visibility_moves(moving_acl, search_mode):
    moving_acl.register_file(FileAcl(file_id="f1", box_id="pub"))
    store = new_store(moving_acl, search_mode)
    index_one(store, "f1", "เดิมอยู่ public")
    moving_acl.register_file(FileAcl(file_id="f1", box_id="priv"))
    index_one(store, "f1", "ย้ายมา private")
    assert count(store, "test_public") == 0 and count(store, "test_private") == 1


def test_failed_precheck_does_not_delete_existing_points(store, acl):
    """มิติผิด -> ล้มก่อนลบ: ของเดิมต้องยังอยู่"""
    index_one(store, "f_pub", "ของเดิม")
    doc = make_doc("f_pub", [make_unit(0, "ใหม่")])
    with pytest.raises(VectorSizeError):
        store.upsert_chunks(doc, indexed(doc.units[0], dense=[1.0]))
    assert count(store) == 1


def test_chunk_of_a_non_ok_unit_cannot_be_indexed(store, acl):
    ok_doc = make_doc("f_pub", [make_unit(0, "ข้อความ")])
    chunks = indexed(ok_doc.units[0])
    bad_doc = make_doc("f_pub", [make_unit(0, "", status=UnitStatus.FAILED, reason="พัง")])
    with pytest.raises(ValueError):
        store.upsert_chunks(bad_doc, chunks)


def test_payload_keeps_sentence_offsets_for_citing_and_context_outside_the_text(store, acl, alice):
    text = "[แถวแรกของตาราง: ชื่อ | ยอด]\nรายการแรก ยอด 100\nรายการสอง ยอด 200\nรายการสาม ยอด 300"
    unit = make_unit(0, text)
    doc = make_doc("f_pub", [unit])
    store.settings = SETTINGS.model_copy(update={"chunk_target_chars": 40})
    store.upsert_document(doc, HashEmbedder(VECTOR_SIZE))
    hits = store.search(embed_query("รายการ ยอด", HashEmbedder(VECTOR_SIZE)), alice, "pub", limit=10).hits
    assert len(hits) >= 2
    for hit in hits:
        assert unit.text[hit.char_start:hit.char_end] == hit.text  # offset ใช้ cite ตรงกับ Unit.text เดิม
        for start, end in hit.sentences:
            assert hit.char_start <= start < end <= hit.char_end and unit.text[start:end].strip()
    later = [h for h in hits if h.chunk_index > 0]
    assert later and all(h.context == "[แถวแรกของตาราง: ชื่อ | ยอด]" for h in later)
    assert all("แถวแรกของตาราง" not in h.text for h in later)  # context อยู่นอก text / offset


def test_embed_chunks_prepends_context_but_bm25_and_dense_see_the_same_text(store):
    unit = make_unit(0, "[ป้ายก่อนหน้า: รายได้]\nแถวหนึ่ง 1\nแถวสอง 2")
    chunks = make_chunks(unit, 30)
    assert len(chunks) >= 2
    embedder = HashEmbedder(VECTOR_SIZE)
    items = embed_chunks(chunks, embedder)
    later = next(i for i in items if i.chunk.context)
    assert later.chunk.embed_text.startswith("[ป้ายก่อนหน้า: รายได้]\n")
    assert later.dense == embedder.embed_documents([later.chunk.embed_text])[0].dense
    assert later.sparse_bm25  # ตัดคำแล้วมี token (รวมคำจาก context)


# ---------- โหมดค้น ----------


def lexical_corpus(store: VectorStore, acl: AclStore) -> HashEmbedder:
    embedder = HashEmbedder(VECTOR_SIZE)
    for fid, text in (("f_a", "เงินปันผลต่อหุ้นของบริษัท"), ("f_b", "นโยบายวันลาพักร้อนของพนักงาน"), ("f_c", "ตารางสรุปจำนวนสาขา")):
        acl.register_file(FileAcl(file_id=fid, box_id="pub"))
        store.upsert_document(make_doc(fid, [make_unit(0, text)]), embedder)
    return embedder


@pytest.mark.parametrize("mode", ["dense+sparse_bm25", "all"])
def test_bm25_branch_finds_the_exact_word_even_when_dense_vectors_are_uninformative(acl, mode):
    """dense ของทุก chunk ชี้ทางเดียวกัน (ไม่มีข้อมูล) -> มีแต่ sparse_bm25 ที่แยก f_a ออกมาได้"""
    store = new_store(acl, mode)
    embedder = HashEmbedder(VECTOR_SIZE)
    for fid, text in (("f_a", "เงินปันผลต่อหุ้นของบริษัท"), ("f_b", "นโยบายวันลาพักร้อนของพนักงาน"), ("f_c", "ตารางสรุปจำนวนสาขา")):
        acl.register_file(FileAcl(file_id=fid, box_id="pub"))
        doc = make_doc(fid, [make_unit(0, text)])
        items = [IndexedChunk(c, vec(), SparseVector(), e.sparse) for c, e in zip(make_chunks(doc.units[0], 1000), embedder.embed_documents([text]))]
        store.upsert_chunks(doc, items)
    user = join(acl, Principal(user_id="u", box_ids=frozenset({"pub"})))
    query = Query(dense=vec(), sparse_bge=SPARSE_ONE, sparse_bm25=embedder.embed_query("เงินปันผล").sparse)
    assert store.search(query, user, "pub", limit=3, mode=mode).hits[0].file_id == "f_a"


def test_sparse_bge_branch_uses_its_own_vector(acl):
    store = new_store(acl, "dense+sparse_bge")
    for fid, idx in (("f_a", 7), ("f_b", 8)):
        acl.register_file(FileAcl(file_id=fid, box_id="pub"))
        doc = make_doc(fid, [make_unit(0, f"ข้อความของ {fid}")])
        store.upsert_chunks(doc, [IndexedChunk(c, vec(), SparseVector((idx,), (1.0,)), SparseVector()) for c in make_chunks(doc.units[0], 1000)])
    user = join(acl, Principal(user_id="u", box_ids=frozenset({"pub"})))
    query = Query(dense=vec(), sparse_bge=SparseVector((8,), (1.0,)))
    assert store.search(query, user, "pub", mode="dense+sparse_bge").hits[0].file_id == "f_b"


def test_explicit_mode_overrides_the_setting(store, acl, alice):
    index_one(store, "f_pub", "คู่มือ")
    for mode in SEARCH_MODES:
        assert [h.file_id for h in store.search(QUERY, alice, "pub", mode=mode).hits] == ["f_pub"]


def test_unknown_mode_and_dense_only_query_in_sparse_mode_are_errors(store, acl, alice):
    with pytest.raises(SearchModeError):
        store.search(QUERY, alice, "pub", mode="bm25")
    with pytest.raises(SearchModeError):
        store.search(QUERY.dense, alice, "pub", mode="all")
    assert store.search(QUERY.dense, alice, "pub", mode="dense") == EMPTY_RESULT  # รายการ dense ล้วนใช้ได้เฉพาะโหมด dense


def test_empty_sparse_branch_is_skipped_instead_of_failing(acl, alice):
    store = new_store(acl, "all")
    index_one(store, "f_pub", "คู่มือ")
    q = Query(dense=QUERY.dense)  # คำถามที่ตัดคำแล้วไม่เหลือ token
    assert [h.file_id for h in store.search(q, alice, "pub").hits] == ["f_pub"]


def test_acl_filter_applies_to_every_branch_in_fusion_mode(acl):
    """ไฟล์ที่ไม่มีสิทธิ์ต้องไม่โผล่แม้ได้คะแนนสูงสุดจาก sparse branch"""
    store = new_store(acl, "all")
    embedder = HashEmbedder(VECTOR_SIZE)
    store.upsert_document(make_doc("f_hr", [make_unit(0, "เงินเดือนผู้บริหาร")]), embedder)
    store.upsert_document(make_doc("f_pub", [make_unit(0, "ประกาศทั่วไป")]), embedder)
    bob = join(acl, Principal(user_id="b", departments=frozenset({"finance"}), security_level=5, box_ids=frozenset({"pub", "hr"})))
    assert store.search(embed_query("เงินเดือนผู้บริหาร", embedder), bob, "hr") == EMPTY_RESULT


def test_modes_return_no_more_than_limit_and_overfetch_still_applies(store, acl, alice):
    for i in range(4):
        acl.register_file(FileAcl(file_id=f"y{i}", box_id="hr"))
        index_one(store, f"y{i}", f"ข้อความ {i}")
    assert len(store.search(QUERY, alice, "hr", limit=2).hits) == 2


# ---------- สมาชิกกล่อง: ถอดแล้วมีผลทันทีที่ด่าน 2 ----------


def test_removed_member_is_dropped_by_the_second_gate_immediately(store, acl):
    """Principal ที่สร้างไว้ก่อน (box_ids ยังมี hr -> pre-filter ปล่อยผ่าน) ต้องโดนด่าน 2 ตัดทันทีหลังถอดสมาชิก"""
    acl.set_member("hr", "carol", BoxRole.VIEWER)
    carol = acl.principal_for("carol", {"hr"}, 5)
    index_one(store, "f_hr", "ข้อมูล HR ที่สมาชิกเห็น")
    assert [h.file_id for h in store.search(QUERY, carol, "hr").hits] == ["f_hr"]

    acl.remove_member("hr", "carol")
    stale = store.search(QUERY, carol, "hr")
    assert stale.hits == () and stale.dropped_by_recheck == 1
    assert store.search(QUERY, acl.principal_for("carol", {"hr"}, 5), "hr") == EMPTY_RESULT  # request ถัดไป


def test_membership_alone_does_not_pass_the_search_acl(store, acl):
    """สมาชิก (แม้ owner) คนละแผนก ยังค้นไม่เจอ — สมาชิกภาพเป็นแค่ชั้นแรก"""
    acl.set_member("hr", "dan", BoxRole.OWNER)
    index_one(store, "f_hr", "ข้อมูล HR")
    assert store.search(QUERY, acl.principal_for("dan", {"finance"}, 9), "hr") == EMPTY_RESULT
