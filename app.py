
import os
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 환경변수(.env) 로드
load_dotenv()

# --- 페이지 설정 ---
st.set_page_config(page_title="다중 PDF RAG 챗봇", layout="wide")
st.title("📄 Multi-PDF Q&A System (FAISS + Streamlit)")

# --- 세션 상태 초기화 ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def extract_text_from_pdfs(uploaded_files):
    """업로드된 여러 PDF 파일에서 텍스트를 추출합니다."""
    documents = []
    for uploaded_file in uploaded_files:
        # 메모리 상의 바이트 스트림을 PyMuPDF로 로드
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        full_text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text:
                full_text += f"\n--- [{uploaded_file.name} - Page {page_num + 1}] ---\n" + page_text

        if full_text.strip():
            documents.append(full_text)
    return "\n\n".join(documents)


# --- 사이드바: PDF 업로드 및 파라미터 제어 ---
with st.sidebar:
    st.header("⚙️ 설정 및 파일 업로드")

    # OpenAI API Key 입력 (환경변수에 없는 경우 대비)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = st.text_input("OpenAI API Key 입력", type="password")

    uploaded_files = st.file_uploader(
        "PDF 파일 선택 (여러 개 가능)", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    # 검색 정밀도 제어
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=2000, value=600, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=300, value=80, step=20)
    top_k = st.slider("검색할 참조 문서 수 (Top-k)", min_value=1, max_value=15, value=5)

    if st.button("문서 색인화 (벡터 생성)", use_container_width=True):
        if not uploaded_files:
            st.warning("먼저 PDF 파일을 업로드해 주세요.")
        elif not openai_api_key:
            st.error("OpenAI API Key가 필요합니다.")
        else:
            with st.spinner("PDF 텍스트 추출 및 임베딩 생성 중..."):
                raw_text = extract_text_from_pdfs(uploaded_files)

                if not raw_text.strip():
                    st.error("PDF에서 텍스트를 추출하지 못했습니다. (이미지 기반 PDF인지 확인하세요)")
                else:
                    # 텍스트 분할
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    chunks = text_splitter.split_text(raw_text)

                    # FAISS 벡터 스토어 생성
                    embeddings = OpenAIEmbeddings(
                        model="text-embedding-3-large", 
                        openai_api_key=openai_api_key
                    )
                    st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
                    st.success(f"색인 완료: 총 {len(chunks)}개 청크 생성됨")


# --- 메인 Q&A 인터페이스 ---
# 이전 대화 히스토리 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 질문 입력
if query := st.chat_input("업로드한 PDF에 대해 질문하세요..."):
    # 사용자 메시지 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 챗봇 응답 로직
    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            warning_msg = "사이드바에서 PDF 파일을 업로드하고 [문서 색인화] 버튼을 먼저 눌러주세요."
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    else:
        with st.chat_message("assistant"):
            with st.spinner("문서 검색 및 답변 생성 중..."):
                # 1. 유사도 기반 관련 컨텍스트 검색
                docs_with_scores = st.session_state.vectorstore.similarity_search_with_score(query, k=top_k)
                context = "\n\n".join([doc.page_content for doc, _ in docs_with_scores])

                # 2. LLM 및 프롬프트 체인 설정
                llm = ChatOpenAI(
                    model="gpt-4o", 
                    temperature=0, 
                    openai_api_key=openai_api_key
                )

                prompt = ChatPromptTemplate.from_template("""당신은 제공된 문서를 바탕으로 사용자의 질문에 정확하고 명확하게 답변하는 AI 어시스턴트입니다.
반드시 아래 [배경 지식]에 명시된 내용을 근거로 답변하세요. 배경 지식으로 확인할 수 없는 내용은 추측하지 말고 솔직하게 알 수 없다고 말하세요.

[배경 지식]
{context}

[사용자 질문]
{question}

[답변]""")

                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({"context": context, "question": query})

                # 답변 출력
                st.markdown(response)

                # 3. 신뢰성 검증을 위한 참조 컨텍스트 아코디언 제공
                with st.expander("🔍 검색된 참조 원문 청크 확인"):
                    for idx, (doc, score) in enumerate(docs_with_scores):
                        st.markdown(f"**청크 #{idx + 1} (유사도 거리: {score:.4f})**")
                        st.text(doc.page_content)
                        st.divider()

            # 어시스턴트 메시지 저장
            st.session_state.messages.append({"role": "assistant", "content": response})
