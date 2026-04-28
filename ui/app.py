import uuid
import time
import os
import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api")

st.set_page_config(page_title="book-see-rag", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stChatInput"] {
        position: fixed;
        left: clamp(1rem, 22vw, 23rem);
        right: 1.5rem;
        bottom: 0;
        z-index: 999;
        padding: 0.75rem 0 1rem;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0), #ffffff 32%);
    }

    section.main > div {
        padding-bottom: 7rem;
    }

    @media (max-width: 900px) {
        div[data-testid="stChatInput"] {
            left: 1rem;
            right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state 初始化 ──────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "chat_loaded_for" not in st.session_state:
    st.session_state.chat_loaded_for = None
if "user_id" not in st.session_state:
    st.session_state.user_id = "guest"
if "role" not in st.session_state:
    st.session_state.role = "employee"
if "department" not in st.session_state:
    st.session_state.department = "general"
if "session_scope_loaded_for" not in st.session_state:
    st.session_state.session_scope_loaded_for = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "comparison" not in st.session_state:
    st.session_state.comparison = None


# ── 工具函数 ──────────────────────────────────────────────────
def api(method: str, path: str, **kwargs):
    try:
        headers = kwargs.pop("headers", {})
        headers = {
            **headers,
            "X-User-Id": st.session_state.user_id,
            "X-Role": st.session_state.role,
            "X-Department": st.session_state.department,
        }
        r = httpx.request(method, f"{API_BASE}{path}", timeout=120, headers=headers, **kwargs)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API 错误 {e.response.status_code}: {e.response.text}")
    except Exception as e:
        st.error(f"请求失败：{e}")
    return None


def poll_task(task_id: str, placeholder) -> dict | None:
    for _ in range(120):
        result = api("GET", f"/ingest/{task_id}")
        if not result:
            return None
        status = result["status"]
        detail = result.get("detail") or {}
        if status == "done":
            return result
        if status == "failed":
            placeholder.error(f"处理失败：{detail}")
            return None
        placeholder.info(f"处理中... ({detail.get('step', '...')})")
        time.sleep(2)
    placeholder.error("处理超时")
    return None


def load_chat_history() -> None:
    session_id = st.session_state.session_id
    if st.session_state.chat_loaded_for == session_id:
        return

    history = api("GET", f"/chat/sessions/{session_id}") or []
    st.session_state.qa_history = history
    st.session_state.chat_loaded_for = session_id


def load_session_scope(docs: list[dict]) -> None:
    session_id = st.session_state.session_id
    if st.session_state.session_scope_loaded_for == session_id:
        return
    scope = api("GET", f"/chat/sessions/{session_id}/scope") or {"doc_ids": [], "kb_ids": []}
    allowed_doc_ids = {doc["doc_id"] for doc in docs}
    for doc_id in scope.get("doc_ids", []):
        if doc_id in allowed_doc_ids:
            st.session_state[doc_id] = True
    st.session_state.session_scope_loaded_for = session_id


def render_sources(sources: list[str]) -> None:
    if not sources:
        return
    with st.expander("引用内容", expanded=False):
        for idx, src in enumerate(sources, 1):
            filename = src.get("filename", "未知文档")
            page = src.get("page") or "?"
            st.markdown(f"**[{idx}] {filename} · 第 {page} 页**")
            st.caption(src.get("content", ""))


def parse_csv_input(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def render_active_scope(docs: list[dict], kb_options: list[dict], selected_doc_ids: list[str]) -> None:
    kb_name_by_id = {kb["kb_id"]: kb["name"] for kb in kb_options}
    doc_map = {doc["doc_id"]: doc for doc in docs}
    active_docs = [doc_map[doc_id] for doc_id in selected_doc_ids if doc_id in doc_map]
    active_kb_ids = sorted({doc["kb_id"] for doc in active_docs})

    if active_docs:
        kb_text = "、".join(kb_name_by_id.get(kb_id, kb_id) for kb_id in active_kb_ids)
        st.caption(f"当前会话范围：{len(active_docs)} 份文档，知识库：{kb_text}")
        with st.expander("查看当前会话范围", expanded=False):
            for doc in active_docs:
                st.markdown(f"- {doc['filename']} [{kb_name_by_id.get(doc['kb_id'], doc['kb_id'])}]")
    else:
        visible_kbs = "、".join(kb["name"] for kb in kb_options) if kb_options else "当前可访问知识库"
        st.caption(f"当前会话范围：未固定到具体文档，将在你有权限访问的范围内检索。可访问知识库：{visible_kbs}")


def render_scope_badge(docs: list[dict], kb_options: list[dict], selected_doc_ids: list[str]) -> None:
    kb_name_by_id = {kb["kb_id"]: kb["name"] for kb in kb_options}
    doc_map = {doc["doc_id"]: doc for doc in docs}
    active_docs = [doc_map[doc_id] for doc_id in selected_doc_ids if doc_id in doc_map]
    active_kb_ids = sorted({doc["kb_id"] for doc in active_docs})

    if active_docs:
        doc_count = len(active_docs)
        kb_count = len(active_kb_ids)
        kb_text = " / ".join(kb_name_by_id.get(kb_id, kb_id) for kb_id in active_kb_ids)
        st.info(f"当前回答范围：{doc_count} 份文档，{kb_count} 个知识库 | {kb_text}")
    else:
        visible_kbs = " / ".join(kb["name"] for kb in kb_options) if kb_options else "当前可访问知识库"
        st.info(f"当前回答范围：未锁定具体文档，将在有权限的范围内检索 | {visible_kbs}")


def render_message_scope(scope: dict, docs: list[dict], kb_options: list[dict]) -> None:
    doc_ids = scope.get("doc_ids", []) if scope else []
    kb_ids = scope.get("kb_ids", []) if scope else []
    if not doc_ids and not kb_ids:
        return
    kb_name_by_id = {kb["kb_id"]: kb["name"] for kb in kb_options}
    doc_map = {doc["doc_id"]: doc for doc in docs}
    active_docs = [doc_map[doc_id] for doc_id in doc_ids if doc_id in doc_map]
    active_kb_ids = kb_ids or sorted({doc["kb_id"] for doc in active_docs})
    kb_text = " / ".join(kb_name_by_id.get(kb_id, kb_id) for kb_id in active_kb_ids) if active_kb_ids else "未指定知识库"
    st.caption(f"本条回答范围：{len(active_docs)} 份文档 | {kb_text}")


def apply_scope_to_selection(scope: dict, docs: list[dict]) -> None:
    allowed_doc_ids = {doc["doc_id"] for doc in docs}
    target_doc_ids = set(scope.get("doc_ids", [])) & allowed_doc_ids
    for doc in docs:
        st.session_state[doc["doc_id"]] = doc["doc_id"] in target_doc_ids


def build_scope_summary(scope: dict, docs: list[dict], kb_options: list[dict]) -> str:
    kb_name_by_id = {kb["kb_id"]: kb["name"] for kb in kb_options}
    doc_map = {doc["doc_id"]: doc for doc in docs}
    active_docs = [doc_map[doc_id] for doc_id in scope.get("doc_ids", []) if doc_id in doc_map]
    active_kb_ids = scope.get("kb_ids") or sorted({doc["kb_id"] for doc in active_docs})
    kb_text = " / ".join(kb_name_by_id.get(kb_id, kb_id) for kb_id in active_kb_ids) if active_kb_ids else "未指定知识库"
    return f"{len(active_docs)} 份文档 | {kb_text}"


def run_chat_turn(user_input: str, doc_filter: list[str] | None, comparison_payload: dict | None = None) -> None:
    user_msg = {"role": "user", "content": user_input, "sources": []}
    st.session_state.qa_history.append(user_msg)
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("检索并生成回答..."):
            result = api("POST", "/chat", json={
                "session_id": st.session_state.session_id,
                "message": user_input,
                "doc_ids": doc_filter,
            })
        if result:
            assistant_msg = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "scope": result.get("scope", {"doc_ids": [], "kb_ids": []}),
            }
            st.write(assistant_msg["content"])
            render_message_scope(assistant_msg["scope"], docs, kb_options)
            render_sources(assistant_msg["sources"])
            st.session_state.qa_history.append(assistant_msg)
            if comparison_payload:
                st.session_state.comparison = {
                    "question": user_input,
                    "previous_answer": comparison_payload["previous_answer"],
                    "previous_scope": comparison_payload["previous_scope"],
                    "current_answer": assistant_msg["content"],
                    "current_scope": assistant_msg["scope"],
                }
        else:
            st.session_state.qa_history.pop()


# ── 侧边栏：文档管理 ──────────────────────────────────────────
with st.sidebar:
    st.title("📚 book-see-rag")
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    st.subheader("当前用户")
    st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id, autocomplete="off")
    st.session_state.role = st.selectbox("Role", ["employee", "hr_admin"], index=["employee", "hr_admin"].index(st.session_state.role) if st.session_state.role in ["employee", "hr_admin"] else 0)
    st.session_state.department = st.selectbox("Department", ["general", "rd", "hr"], index=["general", "rd", "hr"].index(st.session_state.department) if st.session_state.department in ["general", "rd", "hr"] else 0)
    st.divider()

    kb_options = api("GET", "/knowledge-bases") or []
    if st.session_state.role == "hr_admin":
        with st.expander("新增知识库", expanded=False):
            new_kb_id = st.text_input("kb_id", placeholder="例如：kb_finance", key="new_kb_id", autocomplete="off")
            new_kb_name = st.text_input("名称", placeholder="例如：财务知识库", key="new_kb_name", autocomplete="off")
            new_kb_visibility = st.selectbox("可见范围", ["public", "department", "private"], key="new_kb_visibility")
            new_kb_departments = st.text_input("允许部门", placeholder="多个用逗号分隔，如：rd,hr", key="new_kb_departments", autocomplete="off")
            new_kb_roles = st.text_input("允许角色", placeholder="多个用逗号分隔，如：hr_admin", key="new_kb_roles", autocomplete="off")
            new_kb_user_ids = st.text_input("允许用户", placeholder="多个用逗号分隔，如：alice,bob", key="new_kb_user_ids", autocomplete="off")
            if st.button("创建知识库", key="create_kb_btn"):
                result = api("POST", "/knowledge-bases", json={
                    "kb_id": new_kb_id,
                    "name": new_kb_name,
                    "visibility": new_kb_visibility,
                    "departments": parse_csv_input(new_kb_departments),
                    "roles": parse_csv_input(new_kb_roles),
                    "user_ids": parse_csv_input(new_kb_user_ids),
                })
                if result:
                    st.success(f"已创建知识库：{result['name']} ({result['kb_id']})")
                    st.rerun()

    kb_labels = {kb["name"]: kb["kb_id"] for kb in kb_options}
    selected_kb_name = st.selectbox("上传到知识库", list(kb_labels.keys()) or ["公共知识库"])
    selected_kb_id = kb_labels.get(selected_kb_name, "kb_public")

    st.subheader("上传文档")
    uploaded = st.file_uploader(
        "支持 PDF / DOCX / TXT / MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("开始摄入", disabled=not uploaded):
        for f in uploaded:
            placeholder = st.empty()
            placeholder.info(f"上传 {f.name}...")
            result = api("POST", "/ingest", files={"file": (f.name, f.getvalue())}, data={"kb_id": selected_kb_id})
            if result:
                done = poll_task(result["task_id"], placeholder)
                if done:
                    placeholder.success(f"✅ {f.name}：{done['detail'].get('chunks', '?')} chunks")

    st.divider()
    st.subheader("已有文档")
    docs = api("GET", "/documents") or []
    selected_doc_ids = []
    if docs:
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            checked = col1.checkbox(f"{doc['filename']} [{doc['kb_id']}]", key=doc["doc_id"])
            if checked:
                selected_doc_ids.append(doc["doc_id"])
            if col2.button("🗑", key=f"del_{doc['doc_id']}"):
                api("DELETE", f"/documents/{doc['doc_id']}")
                st.rerun()
    else:
        st.caption("暂无文档，请先上传")

    st.divider()
    if st.button("🗑 清除对话记忆"):
        api("DELETE", f"/chat/sessions/{st.session_state.session_id}")
        st.session_state.qa_history = []
        st.session_state.session_scope_loaded_for = None
        st.success("记忆已清除")

load_session_scope(docs)
selected_doc_ids = [doc["doc_id"] for doc in docs if st.session_state.get(doc["doc_id"], False)]
doc_filter = selected_doc_ids if selected_doc_ids else None

# ── 主区域 ────────────────────────────────────────────────────
tab_qa, tab_summary, tab_extract = st.tabs(["💬 问答", "📝 摘要", "🔍 知识提取"])

# ── 问答 Tab ──────────────────────────────────────────────────
with tab_qa:
    load_chat_history()
    st.subheader("文档问答")
    st.caption("像聊天工具一样连续追问。若左侧勾选了文档，问答会限定在所选文档范围内。")
    render_active_scope(docs, kb_options, selected_doc_ids)
    render_scope_badge(docs, kb_options, selected_doc_ids)

    if st.session_state.comparison:
        comp = st.session_state.comparison
        st.markdown("**答案对比**")
        st.caption(f"同一问题在不同范围下的回答差异：{comp['question']}")
        col_prev, col_curr = st.columns(2)
        with col_prev:
            st.markdown("**之前的回答**")
            st.caption(f"范围：{build_scope_summary(comp['previous_scope'], docs, kb_options)}")
            st.write(comp["previous_answer"])
        with col_curr:
            st.markdown("**当前重问结果**")
            st.caption(f"范围：{build_scope_summary(comp['current_scope'], docs, kb_options)}")
            st.write(comp["current_answer"])
        if st.button("清除答案对比", key="clear_comparison"):
            st.session_state.comparison = None
            st.rerun()

    last_user_content = None
    for idx, msg in enumerate(st.session_state.qa_history):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "user":
                last_user_content = msg["content"]
            if msg["role"] == "assistant":
                render_message_scope(msg.get("scope", {}), docs, kb_options)
                scope = msg.get("scope", {})
                if (scope.get("doc_ids") or scope.get("kb_ids")) and st.button(
                    "恢复这条回答的范围",
                    key=f"restore_scope_{idx}_{msg['content'][:20]}",
                ):
                    apply_scope_to_selection(scope, docs)
                    st.rerun()
                if last_user_content and st.button(
                    "按此范围重问",
                    key=f"rerun_scope_{idx}_{msg['content'][:20]}",
                ):
                    apply_scope_to_selection(scope, docs)
                    st.session_state.pending_prompt = {
                        "prompt": last_user_content,
                        "comparison": {
                            "previous_answer": msg["content"],
                            "previous_scope": scope,
                        },
                    }
                    st.rerun()
            render_sources(msg.get("sources", []))

    if st.session_state.pending_prompt:
        payload = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        run_chat_turn(payload["prompt"], doc_filter, comparison_payload=payload.get("comparison"))

    elif user_input := st.chat_input("输入问题，例如：这份文档的核心结论是什么？"):
        run_chat_turn(user_input, doc_filter)

# ── 摘要 Tab ──────────────────────────────────────────────────
with tab_summary:
    st.subheader("文档摘要")
    topic = st.text_input("摘要主题（可选）", placeholder="留空则摘要整体内容", key="sum_topic", autocomplete="off")
    if st.button("生成摘要", key="sum_btn"):
        with st.spinner("生成中..."):
            result = api("POST", "/query", json={
                "question": topic or "文档的主要内容",
                "mode": "summary",
                "doc_ids": doc_filter,
            })
        if result:
            st.markdown(result["result"])

# ── 知识提取 Tab ──────────────────────────────────────────────
with tab_extract:
    st.subheader("结构化知识提取")
    ext_query = st.text_input("提取主题", placeholder="主要知识点", key="ext_q", autocomplete="off")
    if st.button("提取知识", key="ext_btn"):
        with st.spinner("提取中..."):
            result = api("POST", "/query", json={
                "question": ext_query or "主要知识点",
                "mode": "extraction",
                "doc_ids": doc_filter,
            })
        if result and isinstance(result["result"], dict):
            data = result["result"]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**实体**")
                for e in data.get("entities", []):
                    st.markdown(f"- {e}")
                st.markdown("**关系**")
                for r in data.get("relationships", []):
                    st.markdown(f"- {r}")
            with col2:
                st.markdown("**关键事实**")
                for f in data.get("key_facts", []):
                    st.markdown(f"- {f}")
