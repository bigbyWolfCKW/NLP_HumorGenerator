"""Select one formal assignment or the union of all member assignments."""

MEMBERS = ("member_1", "member_2", "member_3", "member_4")
SCOPES = (*MEMBERS, "all")


def includes_owner(scope, owner):
    return owner in MEMBERS if scope == "all" else owner == scope
