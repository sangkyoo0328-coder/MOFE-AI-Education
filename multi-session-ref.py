"""
멀티세션 RAG 챗봇 — Supabase 세션/벡터 저장, OpenAI 임베딩·gpt-4o-mini, Streamlit UI.
실행: cd 7.MultiService/code && uv run streamlit run multi-session-ref.py
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# 경로·환경
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOG_DIR = REPO_ROOT / "logs"
LOGO_PATH = REPO_ROOT / "logo.png"

load_dotenv(dotenv_path=ENV_PATH, override=False)

# ---------------------------------------------------------------------------
# 로깅 (ref.txt: ERROR/WARNING만, HTTP 로그 억제)
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_path = LOG_DIR / f"chatbot_{datetime.now():%Y%m%d}.log"
_root = logging.getLogger()
_root.handlers.clear()
_root.setLevel(logging.WARNING)
_fh = logging.FileHandler(_log_path, encoding="utf-8")
_fh.setLevel(logging.WARNING)
_ch = logging.StreamHandler()
_ch.setLevel(logging.WARNING)
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
_fh.setFormatter(_fmt)
_ch.setFormatter(_fmt)
_root.addHandler(_fh)
_root.addHandler(_ch)
for noisy in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger(noisy).propagate = False

log = logging.getLogger("multi_session_ref")


def remove_separators(text: str) -> str:
    if not text:
        return text
    out = re.sub(r"~~[^~]*~~", "", text)
    out = re.sub(r"(?m)^\s*(---+|===+|___+)\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def get_env_status() -> dict[str, bool]:
    import os

    return {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "SUPABASE_URL": bool(os.getenv("SUPABASE_URL")),
        "SUPABASE_ANON_KEY": bool(os.getenv("SUPABASE_ANON_KEY")),
    }


def get_supabase() -> Client | None:
    import os

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def get_llm(model_name: str, temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        streaming=True,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def list_sessions(supabase: Client) -> list[dict[str, Any]]:
    r = (
        supabase.table("chat_sessions")
        .select("id,title,updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    return r.data or []


def fetch_session_messages(supabase: Client, session_id: str) -> list[dict[str, str]]:
    r = supabase.table("chat_sessions").select("messages").eq("id", session_id).single().execute()
    if not r.data:
        return []
    raw = r.data.get("messages")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for m in raw:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and "content" in m:
            out.append({"role": m["role"], "content": str(m["content"])})
    return out


def ensure_chat_session(supabase: Client, session_id: str) -> None:
    r = supabase.table("chat_sessions").select("id").eq("id", session_id).execute()
    if r.data:
        return
    supabase.table("chat_sessions").insert(
        {"id": session_id, "title": "새 세션", "messages": []}
    ).execute()


def save_session_to_db(
    supabase: Client,
    session_id: str,
    messages: list[dict[str, str]],
    title: str | None = None,
) -> None:
    ensure_chat_session(supabase, session_id)
    payload: dict[str, Any] = {
        "messages": messages,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if title is not None:
        payload["title"] = title
    supabase.table("chat_sessions").update(payload).eq("id", session_id).execute()


def generate_session_title(openai_client: OpenAI, first_q: str, first_a: str) -> str:
    sys = (
        "첫 사용자 질문과 어시스턴트 답변을 바탕으로 세션 제목을 한 줄로 짧게(40자 이내) 지어 주세요. "
        "제목만 출력하고 따옴표나 접두사는 붙이지 마세요."
    )
    user = f"질문:\n{first_q}\n\n답변:\n{first_a[:2000]}"
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    t = (resp.choices[0].message.content or "").strip()
    t = re.sub(r"^[\"']|[\"']$", "", t)
    return (t[:80] or "새 세션").strip()


def generate_followup_questions(openai_client: OpenAI, question: str, answer: str) -> str:
    sys = (
        "다음 대화를 보고 사용자가 이어서 물어볼 만한 질문을 정확히 3개를 한국어로만 작성하세요. "
        "각 줄에 번호(1. 2. 3.)만 붙이고 다른 설명은 하지 마세요."
    )
    user = f"사용자 질문:\n{question}\n\n어시스턴트 답변:\n{answer[:4000]}"
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=0.5,
    )
    body = (resp.choices[0].message.content or "").strip()
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{body}"


def retrieve_documents(
    supabase: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    query: str,
    k: int = 10,
) -> list[Document]:
    q_emb = embeddings.embed_query(query)
    try:
        r = supabase.rpc(
            "match_vector_documents",
            {
                "query_embedding": q_emb,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
    except Exception as e:
        log.warning("RPC match_vector_documents failed: %s", e)
        return []

    rows = r.data or []
    docs: list[Document] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        content = row.get("content") or ""
        meta = {
            "file_name": row.get("file_name"),
            "similarity": row.get("similarity"),
        }
        md = row.get("metadata")
        if isinstance(md, dict):
            meta.update(md)
        docs.append(Document(page_content=str(content), metadata=meta))
    return docs


def insert_vector_batch(
    supabase: Client,
    session_id: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    supabase.table("vector_documents").insert(rows).execute()


def process_pdf_files(
    supabase: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    file_paths: Iterable[tuple[str, Path]],
) -> list[str]:
    ensure_chat_session(supabase, session_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    processed: list[str] = []
    batch: list[dict[str, Any]] = []
    batch_size = 10

    for display_name, path in file_paths:
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        for d in pages:
            d.metadata["file_name"] = display_name
        chunks = splitter.split_documents(pages)
        texts = [c.page_content for c in chunks]
        if not texts:
            continue
        for i in range(0, len(texts), batch_size):
            sub_docs = chunks[i : i + batch_size]
            sub_texts = [c.page_content for c in sub_docs]
            embs = embeddings.embed_documents(sub_texts)
            for doc, emb in zip(sub_docs, embs, strict=False):
                fn = doc.metadata.get("file_name") or display_name
                if not fn:
                    fn = "unknown.pdf"
                batch.append(
                    {
                        "session_id": session_id,
                        "content": doc.page_content,
                        "embedding": emb,
                        "file_name": str(fn),
                        "metadata": {k: v for k, v in doc.metadata.items() if k != "file_name"},
                    }
                )
            if len(batch) >= batch_size:
                insert_vector_batch(supabase, session_id, batch)
                batch = []
        processed.append(display_name)

    if batch:
        insert_vector_batch(supabase, session_id, batch)
    return processed


def duplicate_session_snapshot(
    supabase: Client,
    source_session_id: str,
    messages: list[dict[str, str]],
    title: str,
) -> str:
    new_id = str(uuid.uuid4())
    supabase.table("chat_sessions").insert(
        {"id": new_id, "title": title, "messages": messages}
    ).execute()
    r = (
        supabase.table("vector_documents")
        .select("content,embedding,file_name,metadata")
        .eq("session_id", source_session_id)
        .execute()
    )
    rows = r.data or []
    batch: list[dict[str, Any]] = []
    for row in rows:
        emb = row.get("embedding")
        if isinstance(emb, str):
            try:
                emb_out = json.loads(emb)
            except json.JSONDecodeError:
                emb_out = emb
        else:
            emb_out = emb
        fn = row.get("file_name") or "unknown.pdf"
        batch.append(
            {
                "session_id": new_id,
                "content": row.get("content") or "",
                "embedding": emb_out,
                "file_name": str(fn),
                "metadata": row.get("metadata") or {},
            }
        )
        if len(batch) >= 50:
            insert_vector_batch(supabase, new_id, batch)
            batch = []
    if batch:
        insert_vector_batch(supabase, new_id, batch)
    return new_id


def delete_session(supabase: Client, session_id: str) -> None:
    supabase.table("vector_documents").delete().eq("session_id", session_id).execute()
    supabase.table("chat_sessions").delete().eq("id", session_id).execute()


def list_vector_filenames(supabase: Client, session_id: str) -> list[str]:
    r = (
        supabase.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    names: set[str] = set()
    for row in r.data or []:
        if isinstance(row, dict) and row.get("file_name"):
            names.add(str(row["file_name"]))
    return sorted(names)


def render_header() -> None:
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
            <div style="text-align:center;">
              <span style="font-size:4rem !important; font-weight:800; color:#1f77b4 !important;">멀티세션 RAG</span>
              <span style="font-size:4rem !important; font-weight:800; color:#ffd700 !important;">챗봇</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
        h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
        h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
        div[data-testid="stChatMessage"] { font-size: 1rem; }
        button[kind="secondary"], button[kind="primary"] {
            background-color: #ff69b4 !important;
            color: #111 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = str(uuid.uuid4())
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "title_auto_done" not in st.session_state:
        st.session_state.title_auto_done = False
    if "_session_id_map" not in st.session_state:
        st.session_state._session_id_map = {}


def reset_screen() -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.processed_files = []
    st.session_state.active_session_id = str(uuid.uuid4())
    st.session_state.title_auto_done = False
    st.session_state.pop("session_select_widget", None)


def apply_loaded_messages(msgs: list[dict[str, str]]) -> None:
    st.session_state.chat_history = msgs
    st.session_state.conversation_memory = msgs[-50:]
    st.session_state.title_auto_done = True


def on_session_select_change() -> None:
    supabase = get_supabase()
    if not supabase:
        return
    label = st.session_state.get("session_select_widget")
    if not label:
        return
    sid = st.session_state.get("_session_id_map", {}).get(label)
    if not sid:
        return
    try:
        msgs = fetch_session_messages(supabase, sid)
        st.session_state.active_session_id = sid
        apply_loaded_messages(msgs)
        st.session_state.processed_files = list_vector_filenames(supabase, sid)
    except Exception as e:
        log.warning("Session select load failed: %s", e)
        st.session_state["_load_error"] = str(e)


def main() -> None:
    st.set_page_config(page_title="멀티세션 RAG 챗봇", page_icon="📚", layout="wide")
    inject_css()
    init_state()

    env_ok = get_env_status()
    missing = [k for k, v in env_ok.items() if not v]
    supabase = get_supabase() if not missing else None

    render_header()

    with st.sidebar:
        st.markdown("### 멀티세션 RAG 챗봇")
        if missing:
            st.warning("다음 환경 변수가 .env에 없습니다: " + ", ".join(missing))
            st.caption(f".env 경로: {ENV_PATH}")
        else:
            st.caption("Supabase·OpenAI 연결됨")

        st.markdown("#### LLM 모델")
        st.radio(
            "모델",
            options=["gpt-4o-mini"],
            index=0,
            disabled=True,
            help="요구사항에 따라 gpt-4o-mini 고정입니다.",
        )

        sessions: list[dict[str, Any]] = []
        if supabase:
            try:
                sessions = list_sessions(supabase)
            except Exception as e:
                log.warning("list_sessions: %s", e)
                st.error(f"세션 목록 로드 실패: {e}")

        labels: list[str] = []
        id_map: dict[str, str] = {}
        for s in sessions:
            sid = str(s["id"])
            title = str(s.get("title") or "제목 없음")
            lab = f"{title}"
            if lab in id_map:
                lab = f"{title} ({sid[:8]})"
            labels.append(lab)
            id_map[lab] = sid
        st.session_state._session_id_map = id_map

        st.markdown("#### 세션 관리")

        chosen: str | None
        if labels:
            chosen = st.selectbox(
                "저장된 세션",
                options=labels,
                index=None,
                placeholder="목록에서 세션을 선택하세요",
                key="session_select_widget",
                on_change=on_session_select_change,
            )
        else:
            st.caption("(저장된 세션이 없습니다)")
            chosen = None

        c_save, c_load = st.columns(2)
        with c_save:
            save_clicked = st.button("세션저장", use_container_width=True)
        with c_load:
            load_clicked = st.button("세션로드", use_container_width=True)

        c_del, c_reset = st.columns(2)
        with c_del:
            del_clicked = st.button("세션삭제", use_container_width=True)
        with c_reset:
            reset_clicked = st.button("화면초기화", use_container_width=True)

        vectordb_clicked = st.button("vectordb", use_container_width=True)

        st.markdown("#### PDF 업로드")
        uploads = st.file_uploader(
            "PDF (다중 선택)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        process_clicked = st.button("파일 처리하기", use_container_width=True)

        st.markdown("#### 현재 설정")
        st.text(
            "모델: gpt-4o-mini\n"
            f"활성 세션 ID: {st.session_state.active_session_id[:8]}…\n"
            f"대화 메시지 수: {len(st.session_state.chat_history)}\n"
            f"처리된 파일(로컬 목록): {len(st.session_state.processed_files)}"
        )

    err_banner = st.session_state.pop("_load_error", None)
    if err_banner:
        st.error(err_banner)

    if missing:
        st.info("OPENAI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY 를 설정한 뒤 다시 실행해 주세요.")
        return

    assert supabase is not None

    openai_sdk = OpenAI()

    # --- 사이드바 액션 ---
    if reset_clicked:
        reset_screen()
        st.success("화면을 초기화했습니다.")
        st.rerun()

    if vectordb_clicked:
        try:
            names = list_vector_filenames(supabase, st.session_state.active_session_id)
            if names:
                st.info("현재 세션 벡터 DB 파일명:\n" + "\n".join(f"- {n}" for n in names))
            else:
                st.info("현재 세션에 저장된 벡터 문서가 없습니다.")
        except Exception as e:
            st.error(str(e))
            log.warning("vectordb list: %s", e)

    if del_clicked:
        if not labels or not chosen:
            st.warning("삭제할 세션을 목록에서 먼저 선택해 주세요.")
        else:
            sid = id_map.get(chosen)
            if sid:
                try:
                    delete_session(supabase, sid)
                    if st.session_state.active_session_id == sid:
                        reset_screen()
                    st.success("세션을 삭제했습니다.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
                    log.warning("delete_session: %s", e)

    if load_clicked:
        if not labels or not chosen:
            st.warning("로드할 세션을 목록에서 먼저 선택해 주세요.")
        else:
            sid = id_map.get(chosen)
            if sid:
                try:
                    msgs = fetch_session_messages(supabase, sid)
                    st.session_state.active_session_id = sid
                    apply_loaded_messages(msgs)
                    st.session_state.processed_files = list_vector_filenames(supabase, sid)
                    st.success("세션을 불러왔습니다.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
                    log.warning("load: %s", e)

    if save_clicked:
        hist = st.session_state.chat_history
        if len(hist) < 2:
            st.warning("첫 질문과 답변이 있어야 세션을 저장할 수 있습니다.")
        else:
            first_q = next((m["content"] for m in hist if m["role"] == "user"), "")
            first_a = next((m["content"] for m in hist if m["role"] == "assistant"), "")
            try:
                title = generate_session_title(openai_sdk, first_q, first_a)
                new_id = duplicate_session_snapshot(
                    supabase,
                    st.session_state.active_session_id,
                    hist,
                    title,
                )
                st.success(f"새 세션으로 저장했습니다: {title} ({new_id[:8]}…)")
                st.rerun()
            except Exception as e:
                st.error(str(e))
                log.warning("save snapshot: %s", e)

    # --- PDF 처리 ---
    if process_clicked and uploads:
        tmp_paths: list[tuple[str, Path]] = []
        try:
            for uf in uploads:
                suffix = Path(uf.name).suffix or ".pdf"
                p = Path(tempfile.gettempdir()) / f"multi_session_ref_{uuid.uuid4()}{suffix}"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(uf.getvalue())
                tmp_paths.append((uf.name, p))
            emb = get_embeddings()
            with st.spinner("PDF 분할·임베딩·Supabase 저장 중…"):
                names = process_pdf_files(
                    supabase,
                    emb,
                    st.session_state.active_session_id,
                    tmp_paths,
                )
            st.session_state.processed_files = sorted(
                set(st.session_state.processed_files) | set(names)
            )
            ensure_chat_session(supabase, st.session_state.active_session_id)
            save_session_to_db(supabase, st.session_state.active_session_id, st.session_state.chat_history)
            st.success("파일 처리 및 세션 자동 저장을 완료했습니다.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
            log.warning("pdf process: %s", e)
    elif process_clicked and not uploads:
        st.warning("PDF 파일을 선택해 주세요.")

    # --- 채팅 영역 ---
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(remove_separators(m["content"]), unsafe_allow_html=False)

    user_input = st.chat_input("메시지를 입력하세요…")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory = st.session_state.chat_history[-50:]

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    llm = get_llm("gpt-4o-mini")
    emb_model = get_embeddings()

    docs = retrieve_documents(
        supabase,
        emb_model,
        st.session_state.active_session_id,
        user_input,
        k=10,
    )
    context = "\n\n".join(
        f"[파일: {d.metadata.get('file_name', '')}]\n{d.page_content}" for d in docs
    )
    mem = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.conversation_memory[:-1])

    system_parts = [
        "당신은 친절한 한국어 어시스턴트입니다. 존댓말로 답변하세요.",
        "답변은 # ## ### 헤딩으로 구조화하세요.",
        "구분선(---, ===, ___)와 취소선(~~)은 사용하지 마세요.",
        "참조·출처 표시 문구는 넣지 마세요.",
    ]
    if context.strip():
        system_parts.append("다음은 검색된 문서 발췌입니다. 이를 활용해 답변하세요:\n" + context)
    else:
        system_parts.append("검색된 문서가 없으면 일반 지식으로 답변하세요.")

    lc_messages_list: list[BaseMessage] = [SystemMessage(content="\n\n".join(system_parts))]
    if mem.strip():
        lc_messages_list.append(SystemMessage(content="최근 대화 맥락:\n" + mem))
    for m in st.session_state.conversation_memory:
        if m["role"] == "user":
            lc_messages_list.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages_list.append(AIMessage(content=m["content"]))

    assistant_text = ""
    with st.chat_message("assistant"):
        ph = st.empty()
        try:
            ensure_chat_session(supabase, st.session_state.active_session_id)
            for chunk in llm.stream(lc_messages_list):
                if chunk.content:
                    assistant_text += chunk.content
                    ph.markdown(remove_separators(assistant_text) + "▌")
            # 후속 질문 3개 (비스트림)
            extras = generate_followup_questions(openai_sdk, user_input, assistant_text)
            assistant_text += extras
            ph.markdown(remove_separators(assistant_text))
        except Exception as e:
            ph.error(f"응답 생성 중 오류: {e}")
            log.warning("stream: %s", e)
            assistant_text = f"오류가 발생했습니다: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
    st.session_state.conversation_memory = st.session_state.chat_history[-50:]

    try:
        save_session_to_db(supabase, st.session_state.active_session_id, st.session_state.chat_history)
        if (
            not st.session_state.title_auto_done
            and len(st.session_state.chat_history) >= 2
        ):
            hq = st.session_state.chat_history[0]["content"]
            ha = st.session_state.chat_history[1]["content"]
            t = generate_session_title(openai_sdk, hq, ha)
            save_session_to_db(
                supabase,
                st.session_state.active_session_id,
                st.session_state.chat_history,
                title=t,
            )
            st.session_state.title_auto_done = True
    except Exception as e:
        log.warning("autosave: %s", e)

    st.rerun()


if __name__ == "__main__":
    main()
