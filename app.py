import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# --- Initialize Tools ---
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="duckduckgo_search")

# --- Streamlit UI ---
st.title("🔍 Langchain: Smart Research & Web Chatbot")
st.sidebar.title("Settings")

api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I can search the web, Wikipedia, and Arxiv. What do you want to explore?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask me something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    # Combine all 3 tools
    tools = [arxiv, wiki, search]

    # Let agent choose tool intelligently
    search_agent = initialize_agent(
        tools,
        llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=3,
        handle_parsing_errors=True,
        verbose=False,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            raw_result = search_agent.run(prompt, callbacks=[st_cb])

            # Clean and summarize final result
            summary_prompt = f"""
            Summarize the following search results clearly and concisely.
            Combine information from Wikipedia, Arxiv, and DuckDuckGo if available.
            Avoid repeating the same text.

            Results:
            {raw_result}
            """

            summary = llm.invoke(summary_prompt).content

            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.write(summary)

        except Exception as e:
            st.error(f"Error: {e}")
