from dataclasses import dataclass
from fastapi import Header


@dataclass(frozen=True)
class UserContext:
    user_id: str
    role: str
    department: str


def build_user_context(
    x_user_id: str | None = None,
    x_role: str | None = None,
    x_department: str | None = None,
) -> UserContext:
    return UserContext(
        user_id=(x_user_id or "guest").strip() or "guest",
        role=(x_role or "employee").strip() or "employee",
        department=(x_department or "general").strip() or "general",
    )


async def get_current_user(
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_department: str | None = Header(default=None),
) -> UserContext:
    return build_user_context(x_user_id, x_role, x_department)
