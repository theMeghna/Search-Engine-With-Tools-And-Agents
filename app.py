import streamlit as st
import traceback
import time
import os
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from time import sleep

# -----------------------------------------------------------
# ✅ Safe DuckDuckGo wrapper with rate-limit protection
# -----------------------------------------------------------
class SafeDuckDuckGoSearch(DuckDuckGoSearchRun):
    def _run(self, query: str, **kwargs) -> str:
        """Retry automatically when hitting DuckDuckGo rate limits."""
        for attempt in range(3):
            try:
                return super()._run(query, **kwargs)
            except Exception as e:
                if "Ratelimit" in str(e) or "202" in str(e):
                    st.warning("⚠️ DuckDuckGo rate limit reached. Retrying...")
                    sleep(3)
                else:
                    return f"⚠️ DuckDuckGo Error: {e}"
        return "❌ DuckDuckGo rate limit still active. Try again later."

# -----------------------------------------------------------
# 🔑 API Key and LLM setup
# -----------------------------------------------------------
api_key = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
)

# -----------------------------------------------------------
# 🧠 Tools setup
# -----------------------------------------------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = SafeDuckDuckGoSearch(name="duckduckgo_search")

tools = [arxiv, wiki, search]

# -----------------------------------------------------------
# 💬 Streamlit App UI
# -----------------------------------------------------------
st.set_page_config(page_title="🔍 LangChain Search Chatbot", page_icon="🤖")
st.title("🔍 LangChain Chatbot with Web Search")
st.sidebar.title("⚙️ Settings")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi! I'm a chatbot that can search ArXiv, Wikipedia, and the Web. How can I help you?",
        }
    ]

# Display previous messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# -----------------------------------------------------------
# 💭 Handle new user input
# -----------------------------------------------------------
if prompt := st.chat_input("Ask me anything (e.g., 'What is Generative AI?')"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=3,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        try:
            response = search_agent.run(prompt, callbacks=[st_cb])
        except Exception as e:
            response = f"⚠️ Error: {e}\n\n{traceback.format_exc()}"

        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.write(response)

    time.sleep(1)  # small delay to prevent rate-limit spam
