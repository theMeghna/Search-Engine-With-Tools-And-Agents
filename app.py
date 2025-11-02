import streamlit as st
import traceback
import time
from time import sleep
from typing import Optional
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler
import os

# -------- CONFIG --------
# Put your GROQ key into Streamlit Secrets (Manage app → Secrets)
GROQ_KEY = st.secrets.get("GROQ_API_KEY", None)
if not GROQ_KEY:
    st.error("Missing GROQ_API_KEY in Streamlit Secrets. Add it in Advanced settings.")
    st.stop()

# LLM
llm = ChatGroq(groq_api_key=GROQ_KEY, model_name="llama-3.1-8b-instant")

# Tools (wrappers)
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Safe DuckDuckGo with exponential backoff + fallback
class SafeDuckDuckGoSearch(DuckDuckGoSearchRun):
    def _run(self, query: str, **kwargs) -> str:
        # Try exponential backoff up to n attempts
        attempts = 4
        base_wait = 1.5
        last_exc: Optional[Exception] = None
        for i in range(attempts):
            try:
                return super()._run(query, **kwargs)
            except Exception as e:
                last_exc = e
                text = str(e).lower()
                # if rate limit, backoff and retry; otherwise return friendly error
                if "ratelimit" in text or "202" in text or "rate limit" in text:
                    wait = base_wait * (2 ** i)
                    st.warning(f"⚠️ DuckDuckGo rate limit; retrying in {wait:.1f}s...")
                    sleep(wait)
                    continue
                else:
                    # non-rate errors -> return friendly message
                    return f"⚠️ DuckDuckGo error: {e}"
        # after retries, return fallback string
        return f"❌ DuckDuckGo unavailable after retries. Last error: {last_exc}"

search_tool = SafeDuckDuckGoSearch(name="duckduckgo_search")

# -------- Streamlit UI --------
st.set_page_config(page_title="LangChain Search Chatbot", page_icon="🔍", layout="wide")
st.title("🔍 LangChain — Wikipedia + arXiv + Web (robust)")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi — I can check Wikipedia, arXiv, and the web. Ask me anything."}
    ]

# show chat history
for m in st.session_state["messages"]:
    st.chat_message(m["role"]).write(m["content"])

prompt = st.chat_input("Ask me anything (e.g., 'What is Generative AI?')")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # small UI indicator
    with st.spinner("Searching Wikipedia, arXiv and the Web..."):
        try:
            # 1) Wikipedia (fast)
            try:
                wiki_raw = wiki_tool.run(prompt)
            except Exception as e:
                wiki_raw = f"⚠️ Wikipedia failed: {e}"

            # 2) arXiv
            try:
                arxiv_raw = arxiv_tool.run(prompt)
            except Exception as e:
                arxiv_raw = f"⚠️ arXiv failed: {e}"

            # 3) DuckDuckGo (may rate-limit)
            try:
                web_raw = search_tool.run(prompt)
            except Exception as e:
                web_raw = f"⚠️ DuckDuckGo failed: {e}"

            # Aggregate — remove duplicates (simple approach)
            parts = []
            if wiki_raw and "failed" not in str(wiki_raw).lower():
                parts.append("WIKIPEDIA RESULT:\n" + str(wiki_raw).strip())
            if arxiv_raw and "failed" not in str(arxiv_raw).lower():
                parts.append("ARXIV RESULT:\n" + str(arxiv_raw).strip())
            if web_raw and "failed" not in str(web_raw).lower():
                parts.append("WEB RESULT:\n" + str(web_raw).strip())

            if not parts:
                aggregated = "No external results available (all tools failed or returned no results)."
            else:
                # join with clear separators
                aggregated = "\n\n---\n\n".join(parts)

            # Ask the LLM to summarize and prioritize (concise, human-friendly)
            summary_prompt = f"""
You are an assistant. Summarize the following search outputs into one clear answer.
- Prefer concise, human-friendly language.
- When information conflicts, state that and give both sources.
- Provide a short 1-2 line summary and then 2-4 bullet points of important details with source tags (Wikipedia / arXiv / Web).
Do not include raw duplicated blocks.

Search outputs:
{aggregated}
"""

            # Use llm.invoke to get content
            try:
                summary_out = llm.invoke(summary_prompt).content
            except Exception as e:
                # fallback to plain aggregated if LLM fails
                summary_out = f"⚠️ LLM summarization failed: {e}\n\nRaw aggregated results:\n{aggregated}"

            # push to session and display formatted sections
            st.session_state["messages"].append({"role": "assistant", "content": summary_out})
            st.chat_message("assistant").write(summary_out)

        except Exception as e:
            # top-level safeguard
            st.error("An unexpected error happened.")
            st.code(traceback.format_exc())
            st.session_state["messages"].append({"role": "assistant", "content": f"Error: {e}"})
