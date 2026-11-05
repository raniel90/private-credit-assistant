"""Server-owned personas for the interview demo; not production authentication."""
from dataclasses import dataclass
from uuid import uuid4

from core import Principal


@dataclass(frozen=True)
class DemoProfile:
    principal: Principal
    name: str
    role: str
    organization: str
    description: str
    token: str

    def public(self):
        return {"id": self.principal.user_id, "name": self.name, "role": self.role,
                "organization": self.organization, "tenant_id": self.principal.tenant_id,
                "groups": sorted(self.principal.groups), "description": self.description}


PROFILES = {
    "alice": DemoProfile(Principal("alice", "fund", frozenset({"risk"})),
                         "Alice", "Risk officer", "Alpha Capital",
                         "Risk policy and loan assessment.", "alice-demo-token"),
    "bob": DemoProfile(Principal("bob", "fund", frozenset({"credit"})),
                       "Bob", "Credit analyst", "Alpha Capital",
                       "Credit documents. Loan assessment is restricted.", "bob-demo-token"),
    "carol": DemoProfile(Principal("carol", "other_fund", frozenset({"risk"})),
                         "Carol", "Risk officer", "External Fund",
                         "Separate organization. No Alpha documents or loan access.", "carol-demo-token"),
}
DEMO_USERS = {profile.token: profile.principal for profile in PROFILES.values()}


def access_record(principal, decision, reason, **extra):
    return {"request_id": uuid4().hex, "user_id": principal.user_id if principal else None,
            "tenant_id": principal.tenant_id if principal else None,
            "groups": sorted(principal.groups) if principal else [],
            "decision": decision, "reason": reason, **extra}
