"""ACL 4 ชั้น: box · แผนก · ระดับความปลอดภัย · สิทธิ์ระดับไฟล์ (กฎเหล็กข้อ 9)

เมทริกซ์ของจริงอยู่ใน ``eval/datasets/access_control.jsonl`` (คนเขียน ไม่ได้สร้างจากโค้ดนี้)
"""

import pytest

from app.access import (
    AclError,
    AclStore,
    AclUnknownBoxError,
    AclWideningError,
    Box,
    BoxRole,
    FileAcl,
    Principal,
    Visibility,
    effective_acl,
)
from tests.acl_testbed import load_testbed

# ---------- เมทริกซ์จาก testbed ----------


@pytest.fixture(scope="module")
def testbed():
    return load_testbed()


def test_testbed_covers_every_layer(testbed):
    """กันไม่ให้ testbed หดจนไม่ได้ทดสอบชั้นใดชั้นหนึ่ง"""
    denied_layers = {c.layer for c in testbed.cases if c.expected == "deny"}
    assert denied_layers == {"box", "department", "security_level", "unknown"}
    assert any(c.expected == "allow" for c in testbed.cases)


def test_every_case_matches_expected(testbed):
    """ทุกกรณีในเมทริกซ์ต้องตัดสินตรงกับที่คนเขียนไว้ และตกด้วยชั้นที่ถูก"""
    wrong = []
    for case in testbed.cases:
        decision = testbed.store.decide(testbed.principals[case.user_id], case.file_id)
        if decision.allowed != case.should_allow or decision.denied_layer != case.layer:
            wrong.append(
                f"{case}: ได้ allowed={decision.allowed} layer={decision.denied_layer} "
                f"(คาด allowed={case.should_allow} layer={case.layer}) {case.note}"
            )
    assert not wrong, "ผลไม่ตรงกับเมทริกซ์:\n" + "\n".join(wrong)


def test_no_leaks_across_full_matrix(testbed):
    """ทุกคู่ (principal x ไฟล์ทั้งหมด) ต้องไม่มีใครเห็นไฟล์ที่ไม่ได้อยู่ใน box ของตัวเอง

    กว้างกว่าเมทริกซ์ที่เขียนมือ: ไล่ทุกคู่ ไม่ใช่เฉพาะที่ระบุไว้
    """
    leaks = []
    for user_id, principal in testbed.principals.items():
        for file_id, file in testbed.files.items():
            if testbed.store.can_access(principal, file_id) and file.box_id not in principal.box_ids:
                leaks.append(f"{user_id} เห็น {file_id} ของ box {file.box_id} ที่ไม่ได้เป็นสมาชิก")
    assert not leaks, "สิทธิ์รั่ว:\n" + "\n".join(leaks)


def test_same_level_different_department_is_denied(testbed):
    """ระดับเท่ากัน เป็นสมาชิก box เดียวกัน แต่คนละแผนก -> ชั้นแผนกต้องกัน"""
    alice, bob = testbed.principals["alice"], testbed.principals["bob"]
    assert alice.security_level == bob.security_level
    assert "shared_private" in alice.box_ids and "shared_private" in bob.box_ids
    assert testbed.store.can_access(alice, "f_shared_policy")
    assert not testbed.store.can_access(bob, "f_shared_policy")


def test_unknown_file_denied_like_no_permission(testbed):
    """ไฟล์ไม่รู้จักต้องถูกปฏิเสธ และ ``can_access`` ไม่บอกต่างจากกรณีไม่มีสิทธิ์"""
    alice = testbed.principals["alice"]
    assert testbed.store.can_access(alice, "ไม่มีไฟล์นี้") is False
    assert testbed.store.can_access(alice, "f_fin_report") is False  # มีไฟล์แต่ไม่มีสิทธิ์


def test_boxes_for_lists_only_member_boxes(testbed):
    dave = testbed.principals["dave"]
    assert testbed.store.boxes_for(dave) == []
    carol_boxes = [b.box_id for b in testbed.store.boxes_for(testbed.principals["carol"])]
    assert carol_boxes == ["hr_private", "hr_public"]  # เรียงตาม box_id


# ---------- สิทธิ์ระดับไฟล์ทำให้แคบลงได้อย่างเดียว ----------

PUBLIC_BOX = Box(box_id="b", visibility=Visibility.PUBLIC, departments=frozenset({"hr"}), min_security_level=1)


def test_file_may_narrow_departments():
    acl = effective_acl(PUBLIC_BOX, FileAcl(file_id="f", box_id="b", departments=frozenset()))
    assert acl.departments == frozenset()  # set ว่าง = ไม่มีใครเข้าได้ (fail closed)


def test_file_may_raise_security_level():
    acl = effective_acl(PUBLIC_BOX, FileAcl(file_id="f", box_id="b", min_security_level=5))
    assert acl.min_security_level == 5


def test_file_cannot_add_a_department_outside_the_box():
    with pytest.raises(AclWideningError, match="finance"):
        effective_acl(PUBLIC_BOX, FileAcl(file_id="f", box_id="b", departments=frozenset({"hr", "finance"})))


def test_file_cannot_lower_security_level():
    with pytest.raises(AclWideningError):
        effective_acl(PUBLIC_BOX, FileAcl(file_id="f", box_id="b", min_security_level=0))


def test_file_inherits_when_unset():
    acl = effective_acl(PUBLIC_BOX, FileAcl(file_id="f", box_id="b"))
    assert acl.departments == frozenset({"hr"}) and acl.min_security_level == 1


def test_file_cannot_widen_back_to_all_departments():
    """ "ทุกแผนก" เขียนได้แค่ด้วย None ซึ่งแปลว่าสืบทอด ไฟล์จึงขยายกลับไม่ได้โดยโครงสร้าง"""
    acl = effective_acl(PUBLIC_BOX, FileAcl(file_id="f", box_id="b", departments=None))
    assert acl.departments == PUBLIC_BOX.departments


def test_widening_is_rejected_at_registration_not_at_query():
    """ACL ที่ตั้งผิดต้องดังตอนลงทะเบียน ไม่ใช่เงียบไว้แล้วไปรั่วตอนค้น"""
    store = AclStore()
    store.register_box(PUBLIC_BOX)
    with pytest.raises(AclWideningError):
        store.register_file(FileAcl(file_id="f", box_id="b", min_security_level=0))
    assert store.get_effective("f") is None  # ไม่ถูกบันทึกไว้เลย


# ---------- visibility เป็นของ box และเปลี่ยนไม่ได้ ----------


def test_box_is_frozen():
    with pytest.raises(Exception):  # pydantic ValidationError (frozen)
        PUBLIC_BOX.visibility = Visibility.PRIVATE


def test_box_cannot_change_visibility_by_reregistering():
    store = AclStore()
    store.register_box(Box(box_id="b", visibility=Visibility.PRIVATE))
    with pytest.raises(AclError, match="visibility"):
        store.register_box(Box(box_id="b", visibility=Visibility.PUBLIC))


def test_file_must_belong_to_a_known_box():
    store = AclStore()
    with pytest.raises(AclUnknownBoxError):
        store.register_file(FileAcl(file_id="f", box_id="ไม่มีกล่องนี้"))


def test_file_box_mismatch_is_an_error():
    with pytest.raises(AclError):
        effective_acl(PUBLIC_BOX, FileAcl(file_id="f", box_id="other"))


# ---------- fail closed ----------


def test_empty_department_set_denies_everyone():
    """set ว่างต้องแปลว่า 'ไม่มีใคร' ไม่ใช่ 'ทุกคน' — config ที่ถูกลบค่าต้องปิดกั้น"""
    store = AclStore()
    store.register_box(Box(box_id="b", visibility=Visibility.PRIVATE, departments=frozenset()))
    store.register_file(FileAcl(file_id="f", box_id="b"))
    principal = Principal(user_id="u", departments=frozenset({"hr"}), box_ids=frozenset({"b"}))
    assert not store.can_access(principal, "f")


def test_none_department_allows_any_member():
    store = AclStore()
    store.register_box(Box(box_id="b", visibility=Visibility.PRIVATE, departments=None))
    store.register_file(FileAcl(file_id="f", box_id="b"))
    store.set_member("b", "u", BoxRole.VIEWER)
    principal = Principal(user_id="u", departments=frozenset({"อะไรก็ได้"}), box_ids=frozenset({"b"}))
    assert store.can_access(principal, "f")


def test_principal_with_no_boxes_sees_nothing(testbed):
    dave = testbed.principals["dave"]
    assert testbed.store.accessible_files(dave, testbed.files) == []


def test_removing_a_file_revokes_access(testbed):
    store = load_testbed().store  # สำเนาของตัวเอง ไม่แตะ fixture ที่ใช้ร่วม
    alice = testbed.principals["alice"]
    assert store.can_access(alice, "f_hr_salary")
    store.remove_file("f_hr_salary")
    assert not store.can_access(alice, "f_hr_salary")


def test_changing_box_recomputes_files_in_it():
    """ยกระดับความปลอดภัยของ box แล้วไฟล์ที่สืบทอดต้องแคบตามทันที"""
    store = AclStore()
    store.register_box(Box(box_id="b", visibility=Visibility.PRIVATE, min_security_level=0))
    store.register_file(FileAcl(file_id="f", box_id="b"))
    store.set_member("b", "u", BoxRole.VIEWER)
    principal = Principal(user_id="u", security_level=1, box_ids=frozenset({"b"}))
    assert store.can_access(principal, "f")
    store.register_box(Box(box_id="b", visibility=Visibility.PRIVATE, min_security_level=5))
    assert not store.can_access(principal, "f")


# ---------- register_box ต้อง atomic ----------


def _box_b(departments):
    return Box(box_id="b", visibility=Visibility.PRIVATE, departments=frozenset(departments))


def test_register_box_that_would_widen_a_file_is_rejected_without_touching_state():
    """probe: box {hr,it}, ไฟล์ {it} -> register_box({hr}) ต้อง error และ state ต้องเหมือนเดิมทุกจุด"""
    store = AclStore()
    store.register_box(_box_b({"hr", "it"}))
    store.register_file(FileAcl(file_id="g", box_id="b", departments=frozenset({"it"})))
    store.set_member("b", "i", BoxRole.VIEWER)
    it_user = Principal(user_id="i", departments=frozenset({"it"}), box_ids=frozenset({"b"}))
    before = store.get_effective("g")

    with pytest.raises(AclWideningError):
        store.register_box(_box_b({"hr"}))

    assert store.get_box("b").departments == frozenset({"hr", "it"})
    assert store.get_effective("g") == before
    assert store.can_access(it_user, "g")


def test_register_box_rejects_all_if_any_file_fails_even_when_others_are_fine():
    store = AclStore()
    store.register_box(_box_b({"hr", "it"}))
    store.register_file(FileAcl(file_id="a_ok", box_id="b", departments=frozenset({"hr"})))
    store.register_file(FileAcl(file_id="z_bad", box_id="b", departments=frozenset({"it"})))
    before = {fid: store.get_effective(fid) for fid in ("a_ok", "z_bad")}
    with pytest.raises(AclWideningError):
        store.register_box(_box_b({"hr"}))
    assert {fid: store.get_effective(fid) for fid in ("a_ok", "z_bad")} == before
    assert store.get_box("b").departments == frozenset({"hr", "it"})


def test_register_box_visibility_change_is_rejected_without_touching_state():
    store = AclStore()
    store.register_box(Box(box_id="b", visibility=Visibility.PRIVATE, min_security_level=1))
    store.register_file(FileAcl(file_id="f", box_id="b"))
    before = store.get_effective("f")
    with pytest.raises(AclError, match="visibility"):
        store.register_box(Box(box_id="b", visibility=Visibility.PUBLIC, min_security_level=9))
    assert store.get_box("b").min_security_level == 1 and store.get_effective("f") == before


def test_register_box_returns_the_files_whose_effective_acl_changed():
    store = AclStore()
    store.register_box(Box(box_id="b", visibility=Visibility.PRIVATE, min_security_level=0))
    store.register_box(Box(box_id="other", visibility=Visibility.PRIVATE))
    store.register_file(FileAcl(file_id="inherits", box_id="b"))
    store.register_file(FileAcl(file_id="already_high", box_id="b", min_security_level=7))
    store.register_file(FileAcl(file_id="elsewhere", box_id="other"))

    changed = store.register_box(Box(box_id="b", visibility=Visibility.PRIVATE, min_security_level=5))
    assert changed == ["inherits"]  # already_high ยังเป็น 7, elsewhere อยู่คนละ box
    assert store.register_box(Box(box_id="b", visibility=Visibility.PRIVATE, min_security_level=5)) == []
    assert store.register_box(Box(box_id="brand_new", visibility=Visibility.PUBLIC)) == []


# ---------- สมาชิกกล่อง (owner/admin/viewer) ----------


def _box_with_owner(store=None, owner="own"):
    store = store or AclStore()
    store.create_box(Box(box_id="b", visibility=Visibility.PRIVATE, departments=frozenset({"hr"}),
                         min_security_level=2, name="กล่อง HR"), owner)
    store.register_file(FileAcl(file_id="f", box_id="b"))
    return store


def test_principal_box_ids_come_only_from_membership():
    store = _box_with_owner()
    store.register_box(Box(box_id="other", visibility=Visibility.PUBLIC))
    store.set_member("b", "u", BoxRole.VIEWER)
    assert store.principal_for("u", {"hr"}, 3).box_ids == frozenset({"b"})
    assert store.principal_for("stranger", {"hr"}, 9).box_ids == frozenset()
    assert store.principal_for("own").box_ids == frozenset({"b"})


def test_membership_does_not_skip_department_or_security_level():
    """เป็นสมาชิก (แม้เป็น owner) ยังต้องผ่านแผนกและระดับความลับ"""
    store = _box_with_owner()
    store.set_member("b", "fin", BoxRole.ADMIN)
    assert store.decide(store.principal_for("own", {"hr"}, 1), "f").denied_layer == "security_level"
    assert store.decide(store.principal_for("fin", {"finance"}, 5), "f").denied_layer == "department"
    assert store.can_access(store.principal_for("own", {"hr"}, 2), "f")


def test_removed_member_is_denied_even_with_an_old_principal():
    """ด่าน 2 ถามสมาชิกภาพสด — Principal ที่สร้างก่อนถอด (box_ids ยังมี b) ต้องโดนตัด"""
    store = _box_with_owner()
    store.set_member("b", "u", BoxRole.VIEWER)
    old = store.principal_for("u", {"hr"}, 5)
    assert store.can_access(old, "f")
    assert store.remove_member("b", "u") is True
    assert "b" in old.box_ids
    assert store.decide(old, "f").denied_layer == "box"
    assert store.principal_for("u", {"hr"}, 5).box_ids == frozenset()


def test_principal_claiming_a_box_without_membership_is_denied():
    store = _box_with_owner()
    forged = Principal(user_id="x", departments=frozenset({"hr"}), security_level=9, box_ids=frozenset({"b"}))
    assert not store.can_access(forged, "f")


def test_last_owner_cannot_be_removed_or_demoted():
    from app.access import AclLastOwnerError

    store = _box_with_owner()
    with pytest.raises(AclLastOwnerError):
        store.remove_member("b", "own")
    with pytest.raises(AclLastOwnerError):
        store.set_member("b", "own", BoxRole.ADMIN)
    assert store.role_of("own", "b") == BoxRole.OWNER

    store.set_member("b", "own2", BoxRole.OWNER)  # มี owner สองคน -> ลด/ถอดคนหนึ่งได้
    store.set_member("b", "own", BoxRole.ADMIN)
    with pytest.raises(AclLastOwnerError):
        store.remove_member("b", "own2")
    assert store.members("b") == {"own": BoxRole.ADMIN, "own2": BoxRole.OWNER}


def test_create_box_rejects_existing_id_and_keeps_visibility():
    from app.access import AclBoxExistsError

    store = _box_with_owner()
    with pytest.raises(AclBoxExistsError):
        store.create_box(Box(box_id="b", visibility=Visibility.PUBLIC), "attacker")
    assert store.get_box("b").visibility == Visibility.PRIVATE
    assert store.role_of("attacker", "b") is None
    with pytest.raises(AclError):
        store.create_box(Box(box_id="c", visibility=Visibility.PUBLIC), "")


def test_member_operations_on_unknown_box_and_non_member():
    store = _box_with_owner()
    with pytest.raises(AclUnknownBoxError):
        store.set_member("nope", "u", BoxRole.VIEWER)
    with pytest.raises(AclUnknownBoxError):
        store.remove_member("nope", "u")
    assert store.remove_member("b", "never-joined") is False
    assert store.role_of("own", "nope") is None and store.members("nope") == {}


def test_box_registered_without_owner_is_visible_to_nobody():
    store = AclStore()
    store.register_box(Box(box_id="b", visibility=Visibility.PUBLIC))
    store.register_file(FileAcl(file_id="f", box_id="b"))
    anyone = Principal(user_id="u", security_level=9, box_ids=frozenset({"b"}))
    assert not store.can_access(anyone, "f") and store.principal_for("u").box_ids == frozenset()


def test_membership_backend_is_swappable():
    """AclStore ใช้แค่ interface MembershipBackend — backend อื่น (เช่นฐานข้อมูล) ได้กติกาเดียวกัน"""

    class Recording:
        def __init__(self):
            self.rows, self.calls = {}, []

        def get_role(self, box_id, user_id):
            self.calls.append("get")
            return self.rows.get((box_id, user_id))

        def set_role(self, box_id, user_id, role):
            self.rows[(box_id, user_id)] = role

        def remove(self, box_id, user_id):
            self.rows.pop((box_id, user_id), None)

        def members(self, box_id):
            return {u: r for (b, u), r in self.rows.items() if b == box_id}

        def boxes_of(self, user_id):
            return {b: r for (b, u), r in self.rows.items() if u == user_id}

    backend = Recording()
    store = _box_with_owner(AclStore(memberships=backend))
    assert backend.rows == {("b", "own"): BoxRole.OWNER}
    with pytest.raises(Exception):
        store.remove_member("b", "own")
    assert store.can_access(store.principal_for("own", {"hr"}, 2), "f") and "get" in backend.calls
