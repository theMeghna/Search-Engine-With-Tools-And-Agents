import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Load environment variables
load_dotenv()

# --- Streamlit page setup ---
st.set_page_config(page_title="AI Research & Web Search", page_icon="🔍", layout="wide")
st.title("🔎 AI Research & Web Search Agent (Wikipedia + Arxiv + DuckDuckGo)")

# --- API keys ---
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.sidebar.error("❌ GROQ_API_KEY not found in .env or Streamlit Secrets!")

# --- Initialize Groq model ---
llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=groq_api_key)

# --- Tool 1: Wikipedia ---
wikipedia = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia Search",
    func=wikipedia.run,
    description="Get general information and factual summaries from Wikipedia."
)

# --- Tool 2: Arxiv ---
arxiv = ArxivAPIWrapper()
arxiv_tool = Tool(
    name="Arxiv Research",
    func=arxiv.run,
    description="Search for research papers and scientific studies from Arxiv."
)

# --- Tool 3: DuckDuckGo ---
def safe_duckduckgo_search(query: str) -> str:
    """Run a DuckDuckGo search with rate-limit handling."""
    for attempt in range(3):  # up to 3 retries
        try:
            ddg = DuckDuckGoSearchRun()
            results = ddg.run(query)
            return results
        except Exception as e:
            if "rate limit" in str(e).lower():
                st.warning(f"⚠️ DuckDuckGo rate limit hit. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
            else:
                raise e
    return "⚠️ DuckDuckGo search temporarily unavailable due to rate limits."

duckduckgo_tool = Tool(
    name="DuckDuckGo Web Search",
    func=safe_duckduckgo_search,
    description="Search current web results using DuckDuckGo safely with retries."
)

# --- Combine tools ---
tools = [wiki_tool, arxiv_tool, duckduckgo_tool]

# --- Initialize Agent ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Streamlit input ---
query = st.text_input("💬 Ask your question (e.g., 'Recent advancements in Generative AI'):")

if query:
    with st.spinner("🔍 Searching Wikipedia, Arxiv, and DuckDuckGo..."):
        try:
            response = agent.run(query)
            st.success("✅ Search complete!")
            st.write(response)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, LangChain, and Groq LLMs.")
