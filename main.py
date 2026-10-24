"""뜨개메이트(KnitMate) - Streamlit UI

실행: streamlit run main.py
"""
from __future__ import annotations

import json
import uuid

import pandas as pd
import streamlit as st

from knitmate import config
from knitmate.core import storage
from knitmate.core.chart import render_chart
from knitmate.core.formatting import korean_from_spec, korean_markdown
from knitmate.core.pdf_reader import read_upload
from knitmate.core.stitch_counter import ValidationReport, compare_progress, validate
from knitmate.schemas import PatternSpec

st.set_page_config(page_title="뜨개메이트 KnitMate", page_icon="🧶", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #fffaf3; }
    h1, h2, h3 { color: #3b2f2f; }
    .km-card { background: #ffffff; border: 1px solid #ecdccb; border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; }
    .km-row { font-size: 1.15rem; line-height: 1.6; }
    .km-badge { display: inline-block; background: #f6e3d4; color: #8a3d26; border-radius: 999px; padding: 2px 10px; font-size: 0.8rem; margin-right: 6px; }
    .km-muted { color: #8b7b6f; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLES = {
    "미니 레이스 코스터 (대바늘 · 영문)": "lace_coaster_knit_en",
    "원형 코스터 (코바늘 · 영문 US)": "round_coaster_crochet_en",
    "고무뜨기 넥워머 (대바늘 · 한글)": "rib_cowl_knit_ko",
}
EXAMPLE_QUESTIONS = [
    "한길긴뜨기 어떻게 떠?",
    "ssk랑 k2tog 차이가 뭐야?",
    "게이지가 도안은 22코인데 나는 19코야. 도안이 88코면 몇 코 잡아?",
    "원형코(매직링) 만드는 영상 추천해줘",
]


# ---------------- 세션 & 캐시 ----------------

def init_state():
    ss = st.session_state
    ss.setdefault("thread_id", uuid.uuid4().hex)
    ss.setdefault("chat", [])
    ss.setdefault("conv", None)
    ss.setdefault("project_id", None)
    ss.setdefault("craft_pref", "any")


@st.cache_resource(show_spinner=False)
def get_coach_graph():
    from knitmate.agents.coach_graph import build_coach_graph

    return build_coach_graph()


@st.cache_resource(show_spinner="지식 인덱스를 준비하는 중...")
def get_index_mode() -> str:
    from knitmate.core.rag import load_index

    return load_index().mode


# ---------------- 사이드바 ----------------

def sidebar():
    with st.sidebar:
        st.markdown("## 🧶 뜨개메이트")
        st.caption("도안 번역 · 기호 차트 · 코 수 검증 · 기법 코칭 · 진행 체크")

        st.markdown("#### 연결 상태")
        if config.llm_available():
            st.success(f"{config.provider_label()} 연결 설정 완료", icon="✅")
        else:
            st.error(f".env 누락: {', '.join(config.missing_llm_vars())}\n\n샘플 도안 데모와 진행 체크만 동작합니다.", icon="⚠️")
        try:
            mode = get_index_mode()
        except Exception as e:  # noqa: BLE001
            mode = f"오류: {e}"
        st.caption(f"📚 RAG 임베딩: {'API 임베딩' if mode == 'azure' else '로컬 TF-IDF (폴백)' if mode == 'tfidf' else mode}")
        st.caption(f"🎬 영상: {'YouTube API + 큐레이션' if config.youtube_available() else '큐레이션 + 검색 링크'}")

        st.divider()
        st.markdown("#### 내 설정")
        pref = st.radio("주로 하는 뜨개", ["any", "knit", "crochet"], horizontal=True,
                        format_func=lambda x: {"any": "둘 다", "knit": "대바늘", "crochet": "코바늘"}[x], key="craft_pref")

        projects = storage.list_projects()
        st.markdown("#### 내 프로젝트")
        if projects:
            ids = [None] + [p["id"] for p in projects]
            names = {p["id"]: f"{p['name']} · {p['current_row']}단" for p in projects}
            current = st.session_state.project_id if st.session_state.project_id in ids else None
            sel = st.selectbox("진행 중인 프로젝트", ids, index=ids.index(current),
                               format_func=lambda i: "선택 안 함" if i is None else names[i])
            st.session_state.project_id = sel
        else:
            st.caption("아직 프로젝트가 없어요. 도안을 변환한 뒤 '프로젝트 시작'을 눌러 보세요.")

        st.divider()
        if st.button("🗑️ 대화 새로 시작", width="stretch"):
            st.session_state.chat = []
            st.session_state.thread_id = uuid.uuid4().hex
            st.rerun()
        if st.button("🔄 지식 인덱스 다시 만들기", width="stretch"):
            from knitmate.core.rag import build_index

            with st.spinner("인덱싱 중..."):
                info = build_index(verbose=False)
            get_index_mode.clear()
            st.toast(f"문서 {info['count']}개 인덱싱 완료 ({info['mode']})")
        return pref


# ---------------- 탭 1: 도안 변환 ----------------

def tab_convert():
    st.markdown("### 📄 도안 → 한국어 도안 + 기호 차트")
    st.caption("PDF(텍스트/스캔), 이미지, 텍스트 도안을 올리면 번역·구조화·코 수 검증·차트 생성을 여러 에이전트가 나눠서 처리합니다.")

    c1, c2 = st.columns([3, 2])
    with c1:
        source = st.radio("도안 입력 방법", ["파일 업로드", "텍스트 붙여넣기", "샘플 도안"], horizontal=True)
        raw_text, images, filename, sample_key = "", [], "", None
        if source == "파일 업로드":
            up = st.file_uploader("도안 파일", type=["pdf", "txt", "md", "png", "jpg", "jpeg", "webp"])
            if up:
                info = read_upload(up.name, up.getvalue())
                raw_text, images, filename = info["text"], info["images"], up.name
                kind = {"pdf_text": "텍스트 PDF", "pdf_scan": "스캔/이미지 PDF → 비전 판독", "image": "이미지 → 비전 판독", "text": "텍스트"}[info["kind"]]
                st.caption(f"인식 결과: {kind} · {info.get('pages', 1)}쪽")
                if raw_text:
                    with st.expander("추출된 원문 보기"):
                        st.text(raw_text[:4000])
        elif source == "텍스트 붙여넣기":
            raw_text = st.text_area("도안 텍스트", height=220, placeholder="CO 20 sts.\nRow 1 (RS): *K2, p2; rep from * to end.\n...")
        else:
            label = st.selectbox("샘플 도안 (직접 제작한 무료 도안)", list(SAMPLES))
            sample_key = SAMPLES[label]
            raw_text = (config.SAMPLE_DIR / f"{sample_key}.txt").read_text(encoding="utf-8")
            filename = f"{sample_key}.txt"
            with st.expander("샘플 원문 보기", expanded=False):
                st.text(raw_text)
    with c2:
        craft_hint = st.radio("공예 종류", ["auto", "knit", "crochet"], horizontal=True,
                              format_func=lambda x: {"auto": "자동 판단", "knit": "대바늘", "crochet": "코바늘"}[x])
        st.markdown(
            "<div class='km-card km-muted'>처리 흐름<br>① 추출 에이전트 → ② 번역 에이전트 ∥ 파서 에이전트<br>"
            "③ 코 수 검증 Tool (오류 시 파서에게 피드백, 최대 2회 재시도)<br>④ 차트 렌더 Tool</div>",
            unsafe_allow_html=True,
        )
        run = st.button("🧶 변환 시작", type="primary", width="stretch", disabled=not (raw_text or images))
        demo = False
        if sample_key and (config.SAMPLE_DIR / f"{sample_key}.spec.json").exists():
            demo = st.button("⚡ 저장된 결과로 빠른 데모 (LLM 없이)", width="stretch")

    if run:
        if not config.llm_available():
            st.error("LLM 설정이 없어 변환할 수 없습니다. 샘플 도안의 '빠른 데모'를 사용하거나 .env를 설정하세요.")
        else:
            from knitmate.agents.pattern_graph import run_pattern_pipeline

            with st.status("에이전트들이 도안을 처리하는 중...", expanded=True) as status:
                try:
                    result = run_pattern_pipeline(raw_text, images, craft_hint, filename)
                    for line in result.get("log", []):
                        st.write(line)
                    st.session_state.conv = {
                        "korean": result.get("korean") or korean_from_spec(result["spec"]),
                        "spec": result["spec"],
                        "report": result["report"],
                        "chart_png": result.get("chart_png"),
                        "log": result.get("log", []),
                    }
                    status.update(label="변환 완료!", state="complete", expanded=False)
                except Exception as e:  # noqa: BLE001
                    status.update(label="변환 실패", state="error")
                    st.exception(e)
    if demo:
        spec = json.loads((config.SAMPLE_DIR / f"{sample_key}.spec.json").read_text(encoding="utf-8"))
        report = validate(PatternSpec(**spec))
        st.session_state.conv = {
            "korean": korean_from_spec(spec),
            "spec": spec,
            "report": report.to_dict(),
            "chart_png": render_chart(report, spec["craft"], spec["title"]),
            "log": ["⚡ 데모 모드: 미리 구조화한 도안을 코 수 검증 Tool과 차트 렌더 Tool로 처리했습니다."],
        }

    conv = st.session_state.conv
    if not conv:
        return
    render_conversion(conv)


def render_conversion(conv: dict):
    korean, spec, report = conv["korean"], conv["spec"], conv["report"]
    st.divider()
    n_err, n_warn = len(report["errors"]), len(report["warnings"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("공예", "대바늘" if spec["craft"] == "knit" else "코바늘")
    m2.metric("총 단 수", len(report["rows"]))
    m3.metric("시작 → 마지막 코", f"{spec['cast_on']} → {report['final_count']}")
    m4.metric("코 수 검증", "통과 ✓" if n_err == 0 else f"확인 필요 {n_err}건")

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"#### {korean.get('title_ko', spec['title'])}")
        if korean.get("materials_ko"):
            st.markdown("**준비물** · " + " / ".join(korean["materials_ko"]))
        if korean.get("gauge_ko"):
            st.markdown(f"**게이지** · {korean['gauge_ko']}")
        if korean.get("abbreviations"):
            with st.expander("약어표", expanded=False):
                st.table(pd.DataFrame(korean["abbreviations"]).rename(columns={"abbr": "약어", "ko": "한국어"}))
        st.markdown(korean.get("pattern_md", ""))
        for n in korean.get("translator_notes", []):
            st.caption(f"📝 {n}")
    with right:
        if conv.get("chart_png"):
            st.image(conv["chart_png"], caption="자동 생성 기호 차트 (아래에서 위로, 화살표 방향으로 읽기)", width="stretch")

    if n_err:
        with st.expander(f"⚠️ 코 수 확인이 필요한 단 {n_err}건", expanded=True):
            for e in report["errors"]:
                st.write("• " + e)
            st.caption("원 도안의 오타이거나 해석이 애매한 부분일 수 있어요. 해당 단을 원문과 비교해 보세요.")
    if n_warn:
        with st.expander(f"참고 사항 {n_warn}건"):
            for w in report["warnings"]:
                st.write("• " + w)

    with st.expander("단별 코 수 검증표", expanded=False):
        df = pd.DataFrame([{
            "단": r["row"], "면": r["side"], "지시": r["instruction_ko"], "시작 코": r["start_count"],
            "완료 코": r["produced"], "도안 표기": str(r["stated"]) if r["stated"] is not None else "",
            "검증": {"ok": "✓", "warn": "△", "error": "✗"}[r["status"]], "메모": " ".join(r["messages"]),
        } for r in report["rows"]])
        st.dataframe(df, hide_index=True, width="stretch")
    with st.expander("에이전트 처리 기록"):
        for line in conv.get("log", []):
            st.write(line)
        st.caption("구조화 분석(reasoning): " + spec.get("reasoning", "")[:600])

    d1, d2, d3, d4 = st.columns([1, 1, 1, 2])
    md = korean_markdown(korean, report)
    d1.download_button("⬇️ 한국어 도안(.md)", md, file_name=f"{spec['title']}_한글도안.md", width="stretch")
    if conv.get("chart_png"):
        d2.download_button("⬇️ 차트(.png)", conv["chart_png"], file_name=f"{spec['title']}_차트.png", mime="image/png", width="stretch")
    d3.download_button("⬇️ 구조 데이터(.json)", json.dumps(spec, ensure_ascii=False, indent=1), file_name=f"{spec['title']}.json", width="stretch")
    with d4:
        with st.form("start_project", border=False):
            name = st.text_input("프로젝트 이름", value=spec["title"], label_visibility="collapsed")
            if st.form_submit_button("📏 이 도안으로 프로젝트 시작", type="primary", width="stretch"):
                pid = storage.create_project(name, spec["craft"], spec, report, md)
                st.session_state.project_id = pid
                st.success("프로젝트를 만들었어요! '진행 체크' 탭에서 단마다 코 수를 확인해 보세요.")


# ---------------- 탭 2: 뜨개 코치 (채팅) ----------------

def tab_coach(pref: str):
    st.markdown("### 💬 뜨개 코치")
    st.caption("기법 질문, 영상 추천, 도안 해설, 게이지 계산, 진행 중 코 수 확인까지. 질문에 맞는 전문 에이전트가 답합니다.")
    if not config.llm_available():
        st.warning("LLM 설정(.env)이 있어야 코치와 대화할 수 있어요.")
        return

    cols = st.columns(len(EXAMPLE_QUESTIONS))
    clicked = None
    for c, q in zip(cols, EXAMPLE_QUESTIONS):
        if c.button(q, width="stretch"):
            clicked = q

    box = st.container(height=520)
    with box:
        for m in st.session_state.chat:
            with st.chat_message(m["role"], avatar="🧶" if m["role"] == "assistant" else None):
                if m.get("agent"):
                    st.markdown(f"<span class='km-badge'>{m['agent']}</span>", unsafe_allow_html=True)
                st.markdown(m["content"])
                render_extras(m)

    q = st.chat_input("예: 7단 끝났는데 42코야 / 코막음 어떻게 해?") or clicked
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with box:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant", avatar="🧶"):
                with st.spinner("코치가 자료를 찾아보는 중..."):
                    from knitmate.agents.coach_graph import ask_coach

                    try:
                        res = ask_coach(get_coach_graph(), st.session_state.thread_id, q, pref, st.session_state.project_id)
                    except Exception as e:  # noqa: BLE001
                        res = {"answer": f"죄송해요, 답변 중 오류가 났어요: {e}", "agent": "", "videos": [], "sources": [], "tools_used": []}
        st.session_state.chat.append({"role": "assistant", "content": res["answer"], "agent": res["agent"],
                                      "videos": res["videos"], "sources": res["sources"], "tools": res["tools_used"]})
        st.rerun()


def render_extras(m: dict):
    vids = m.get("videos") or []
    if vids:
        st.markdown("**🎬 영상**")
        for v in vids:
            tag = {"curated": "추천", "youtube_api": "YouTube", "search_link": "검색"}.get(v.get("source"), "")
            st.markdown(f"- [{v['title']}]({v['url']}) <span class='km-muted'>· {v.get('channel', '')} · {tag}</span>", unsafe_allow_html=True)
    if m.get("sources"):
        st.caption("📚 근거: " + " · ".join(m["sources"][:4]))
    if m.get("tools"):
        st.caption("🛠️ 사용한 도구: " + ", ".join(dict.fromkeys(m["tools"])))


# ---------------- 탭 3: 진행 체크 ----------------

def tab_progress():
    st.markdown("### 📏 진행 체크")
    pid = st.session_state.project_id
    if not pid:
        st.info("사이드바에서 프로젝트를 고르거나, '도안 변환' 탭에서 도안을 변환한 뒤 프로젝트를 시작하세요.")
        return
    p = storage.get_project(pid)
    if not p:
        st.session_state.project_id = None
        st.rerun()
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"), icon="✅")
    report = ValidationReport.from_dict(p["report"])
    total = len(report.rows)
    cur = min(p["current_row"], total + 1)

    st.markdown(f"#### {p['name']} <span class='km-badge'>{'대바늘' if p['craft'] == 'knit' else '코바늘'}</span>", unsafe_allow_html=True)
    st.progress(min((cur - 1) / max(total, 1), 1.0), text=f"{min(cur - 1, total)} / {total}단 완료")

    if cur > total:
        st.balloons()
        st.success("🎉 도안의 모든 단을 다 떴어요! 마무리 작업만 남았어요.")
        fin = p["spec"].get("finishing_ko")
        if fin:
            st.markdown(f"**마무리**: {fin}")
        if st.button("◀ 마지막 단으로 돌아가기"):
            storage.set_current_row(pid, total)
            st.rerun()
        return

    r = report.get_row(cur)
    side = {"RS": "겉면", "WS": "안면"}.get(r.side, "원형" if r.kind == "round" else "")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f"<div class='km-card'><span class='km-badge'>지금 뜰 단</span><span class='km-badge'>{cur}단 {side}</span>"
            f"<div class='km-row'>{r.instruction_ko}</div>"
            f"<div class='km-muted'>시작 {r.start_count}코 → 완료하면 <b>{r.produced}코</b></div></div>",
            unsafe_allow_html=True,
        )
        nav1, nav2, nav3 = st.columns(3)
        if nav1.button("◀ 이전 단", width="stretch", disabled=cur <= 1):
            storage.set_current_row(pid, cur - 1)
            st.rerun()
        if nav2.button("✓ 다 떴어요 (다음 단)", type="primary", width="stretch"):
            storage.add_log(pid, cur, None, r.produced, "done", "코 수 확인 없이 완료")
            storage.set_current_row(pid, cur + 1)
            st.rerun()
        with nav3.popover("단 이동", width="stretch"):
            jump = st.number_input("이동할 단", min_value=1, max_value=total, value=cur, key=f"jump_{pid}")
            if st.button("이동", key=f"jump_btn_{pid}"):
                storage.set_current_row(pid, int(jump))
                st.rerun()

        with st.form("count_check"):
            st.markdown("**코 수 체크** — 이 단을 다 뜨고 센 코 수를 입력하세요.")
            actual = st.number_input("센 코 수", min_value=0, value=int(r.produced), step=1)
            submitted = st.form_submit_button("코 수 확인")
        if submitted:
            res = compare_progress(report, cur, int(actual))
            storage.add_log(pid, cur, int(actual), res.get("expected"), res["status"], "")
            if res["status"] == "ok":
                storage.set_current_row(pid, cur + 1)
                st.session_state.flash = res["message"] + f" {cur + 1}단으로 넘어왔어요."
                st.rerun()
            else:
                st.error(res["message"])
                show_diagnosis(p, cur, int(actual), res["expected"])

        upcoming = [x for x in report.rows if cur < x.row <= cur + 3]
        if upcoming:
            st.markdown("**다음 단 미리 보기**")
            for x in upcoming:
                st.markdown(f"- {x.row}단: {x.instruction_ko} <span class='km-muted'>({x.produced}코)</span>", unsafe_allow_html=True)
    with right:
        png = render_chart(report, p["craft"], p["name"], highlight_row=cur)
        st.image(png, caption=f"현재 {cur}단 강조 표시", width="stretch")

    logs = storage.get_logs(pid)
    if logs:
        with st.expander("기록", expanded=False):
            st.dataframe(pd.DataFrame(logs).rename(columns={
                "row": "단", "actual_count": "센 코 수", "expected_count": "기대 코 수", "status": "결과", "note": "메모", "created_at": "시각"}),
                hide_index=True, width="stretch")
    with st.expander("프로젝트 관리"):
        if st.button("이 프로젝트 삭제", type="secondary"):
            storage.delete_project(pid)
            st.session_state.project_id = None
            st.rerun()


def show_diagnosis(p: dict, row: int, actual: int, expected: int):
    from knitmate.agents.diagnosis import diagnose, rule_based_diagnosis

    with st.spinner("진단 에이전트가 원인을 찾는 중..."):
        try:
            d = diagnose(p, row, actual, expected) if config.llm_available() else rule_based_diagnosis(p, row, actual, expected)
        except Exception:  # noqa: BLE001
            d = rule_based_diagnosis(p, row, actual, expected)
    st.markdown(f"<div class='km-card'><b>🔍 진단</b><br>{d.summary}</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("**가능성 높은 원인**")
        for c in d.likely_causes:
            st.markdown(f"- {c}")
    with b:
        st.markdown("**이렇게 해 보세요**")
        for h in d.how_to_fix:
            st.markdown(f"- {h}")
    if d.check_rows:
        st.caption("다시 확인해 볼 단: " + ", ".join(f"{x}단" for x in d.check_rows))


# ---------------- 탭 4: 안내 ----------------

def tab_about():
    st.markdown(
        """
### 뜨개메이트는 이렇게 동작해요

| 에이전트 / 도구 | 하는 일 | 사용 기술 |
|---|---|---|
| 추출 에이전트 | PDF 텍스트 추출, 스캔본·사진은 비전 모델로 판독 | PyMuPDF, GPT-4o Vision |
| 번역 에이전트 | 한국어 도안 작성, 용어 체계(US/UK) 판단 | 역할 프롬프트, Few-shot, Structured Output |
| 파서 에이전트 | 도안을 단별 구조(JSON)로 변환 | CoT(reasoning 필드), Structured Output |
| 코 수 검증 Tool | 단마다 사용·생성 코 수 계산, 도안 표기와 대조 → 오류를 파서에 피드백 | 결정적 코드, 자기 수정 루프 |
| 차트 렌더 Tool | 검증된 구조로 기호 차트 생성 | matplotlib |
| 라우터 | 질문 의도 분류 후 전문 에이전트 연결 | Structured Output |
| 기법 코치 | 근거 문서 검색 후 설명, 영상 추천 | ReAct, RAG(FAISS), Tool Calling |
| 진행 코치 · 진단 | 프로젝트 단·코 수 기록, 불일치 원인 진단 | SQLite 메모리, Tool Calling |
| 계산 도우미 | 게이지 환산, 폭→코 수, 실 볼 수 | Tool Calling |

**안전장치**
- 코 수·게이지 계산은 LLM이 아니라 도구가 합니다.
- 영상 링크는 큐레이션 목록 / YouTube API / 검색 링크에서만 나오며, 그 밖의 YouTube 주소는 답변에서 자동 제거됩니다.
- 외부 API(임베딩, YouTube)가 없어도 로컬 TF-IDF 검색과 검색 링크로 동작합니다.
"""
    )


def main():
    init_state()
    pref = sidebar()
    st.markdown("# 🧶 뜨개메이트 <span class='km-muted'>KnitMate</span>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["📄 도안 변환", "💬 뜨개 코치", "📏 진행 체크", "ℹ️ 안내"])
    with t1:
        tab_convert()
    with t2:
        tab_coach(pref)
    with t3:
        tab_progress()
    with t4:
        tab_about()


main()
