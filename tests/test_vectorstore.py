"""vector store: แยก collection ตาม visibility, index เฉพาะ unit ok, และสิทธิ์ 2 ด่าน

test ปกติใช้ Qdrant ในกระบวนการ (``VectorStore.in_memory``) — semantics ของ filter เหมือนของจริง
แต่ไม่มี payload index ส่วน test ที่ใช้ server จริงจาก docker อยู่ท้ายไฟล์ (mark ``qdrant``)
"""

import hashlib

import pytest

from app.access import AclStore, Box, FileAcl, Principal, Visibility
from app.config import Settings
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
    AclUnknownFileError,
    VectorSizeError,
    VectorStore,
    point_id,
)

VECTOR_SIZE = 8
SETTINGS = Settings(
    vector_size=VECTOR_SIZE, collection_public="test_public", collection_private="test_private"
)

# เวกเตอร์กำหนดเองให้ผลค้นคาดเดาได้ — test นี้วัดการกรองสิทธิ์ ไม่ได้วัดคุณภาพ embedding
QUERY = [1.0] + [0.0] * (VECTOR_SIZE - 1)


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


@pytest.fixture
def store(acl) -> VectorStore:
    return VectorStore.in_memory(acl, SETTINGS)


@pytest.fixture
def alice() -> Principal:
    return Principal(
        user_id="alice",
        departments=frozenset({"hr"}),
        security_level=5,
        box_ids=frozenset({"pub", "hr"}),
    )


def index_one(store: VectorStore, file_id: str, text: str = "เนื้อหาทดสอบภาษาไทย"):
    doc = make_doc(file_id, [make_unit(0, text)])
    return store.upsert_units(doc, {0: vec()})


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
    report = store.upsert_units(doc, {i: vec() for i in range(4)})
    assert (report.indexed, report.skipped_not_ok, report.skipped_empty) == (1, 2, 1)
    assert store.client.count("test_public", exact=True).count == 1


def test_cannot_index_a_file_with_unknown_acl(store):
    """เนื้อหาที่ไม่รู้ว่าใครเห็นได้ ห้ามเข้า index (fail closed)"""
    doc = make_doc("ยังไม่ลงทะเบียน", [make_unit(0, "ข้อความ")])
    with pytest.raises(AclUnknownFileError):
        store.upsert_units(doc, {0: vec()})


def test_vector_size_mismatch_is_rejected(store, acl):
    doc = make_doc("f_pub", [make_unit(0, "ข้อความ")])
    with pytest.raises(VectorSizeError):
        store.upsert_units(doc, {0: [1.0, 2.0]})


def test_query_vector_size_mismatch_is_rejected(store, acl, alice):
    with pytest.raises(VectorSizeError):
        store.search([1.0, 2.0], alice, "pub")


def test_reingest_overwrites_instead_of_duplicating(store, acl):
    index_one(store, "f_pub", "รอบแรก")
    index_one(store, "f_pub", "รอบสอง")
    assert store.client.count("test_public", exact=True).count == 1
    assert point_id("f_pub", 0) == point_id("f_pub", 0)


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


# ---------- ด่าน 1: pre-filter ----------


def test_member_of_another_department_gets_nothing(store, acl):
    """เป็นสมาชิก box แต่คนละแผนก — pre-filter ต้องกรองออก"""
    index_one(store, "f_hr", "ข้อมูล HR")
    bob = Principal(
        user_id="bob",
        departments=frozenset({"finance"}),
        security_level=5,
        box_ids=frozenset({"pub", "hr"}),
    )
    assert store.search(QUERY, bob, "hr") == EMPTY_RESULT


def test_security_level_below_the_box_gets_nothing(store, acl):
    acl.register_box(
        Box(box_id="secret", visibility=Visibility.PRIVATE, min_security_level=4)
    )
    acl.register_file(FileAcl(file_id="f_secret", box_id="secret"))
    index_one(store, "f_secret", "เอกสารชั้นความลับ")
    junior = Principal(user_id="j", security_level=1, box_ids=frozenset({"secret"}))
    senior = Principal(user_id="s", security_level=4, box_ids=frozenset({"secret"}))
    assert store.search(QUERY, junior, "secret") == EMPTY_RESULT
    assert len(store.search(QUERY, senior, "secret").hits) == 1


def test_principal_without_any_department(store, acl):
    """แผนกว่าง -> MatchAny ของ filter ได้ list ว่าง ต้องไม่ทำให้เงื่อนไข should พังทั้งก้อน

    box ที่ไม่จำกัดแผนกต้องยังค้นได้ ส่วน box ที่จำกัดต้องไม่ได้อะไร
    """
    index_one(store, "f_pub", "ประกาศทั่วไป")
    index_one(store, "f_hr", "ข้อมูล HR")
    nobody = Principal(user_id="n", security_level=5, box_ids=frozenset({"pub", "hr"}))
    assert [h.file_id for h in store.search(QUERY, nobody, "pub").hits] == ["f_pub"]
    assert store.search(QUERY, nobody, "hr") == EMPTY_RESULT


def test_file_level_narrowing_applies_at_search_time(store, acl):
    acl.register_file(FileAcl(file_id="f_hr2", box_id="hr", min_security_level=9))
    index_one(store, "f_hr2", "ไฟล์ที่ยกระดับเฉพาะตัว")
    mid = Principal(
        user_id="m", departments=frozenset({"hr"}), security_level=5, box_ids=frozenset({"hr"})
    )
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
    outsider = Principal(user_id="o", departments=frozenset({"hr"}), security_level=5,
                         box_ids=frozenset({"pub"}))
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
        Principal(user_id="hr_user", departments=frozenset({"hr"}), security_level=5,
                  box_ids=frozenset({"pub", "hr"})),
        Principal(user_id="fin_user", departments=frozenset({"finance"}), security_level=5,
                  box_ids=frozenset({"pub", "hr"})),
        Principal(user_id="outsider", departments=frozenset({"hr"}), security_level=5,
                  box_ids=frozenset()),
        Principal(user_id="no_dept", security_level=5, box_ids=frozenset({"pub", "hr"})),
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
        client = QdrantClient(url=settings.qdrant_url, timeout=5)
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
        outsider = Principal(user_id="o", departments=frozenset({"hr"}), security_level=5,
                             box_ids=frozenset({"pub"}))
        assert store.search(QUERY, outsider, "hr") == EMPTY_RESULT
        store.delete_file("f_pub")
        assert store.search(QUERY, alice, "pub") == EMPTY_RESULT
    finally:
        for name in (settings.collection_public, settings.collection_private):
            if client.collection_exists(name):
                client.delete_collection(name)
