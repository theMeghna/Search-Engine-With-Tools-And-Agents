import streamlit as st
import traceback
import os
from time import sleep
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler

# Load API key securely
api_key = st.secrets["GROQ_API_KEY"]

# Initialize tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Safer DuckDuckGo wrapper
class SafeDuckDuckGoSearch(DuckDuckGoSearchRun):
    def _run(self, query: str, **kwargs) -> str:
        try:
            return super()._run(query, **kwargs)
        except Exception as e:
            sleep(2)
            return f"⚠️ DuckDuckGo rate limit reached. Try again later. ({e})"

search = SafeDuckDuckGoSearch(name="Web Search")

# Streamlit UI
st.title("🔍 LangChain Chat with Search")
st.sidebar.title("Settings")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a smart search assistant. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

        tools = [arxiv, wiki, search]

        # Increased iterations + safety timeout
        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            max_iterations=6,  # Increased
            handle_parsing_errors=True,
            verbose=False,
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.code(traceback.format_exc())
