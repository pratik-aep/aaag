import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import calculator, word_counter

# --- CONFIGURATION ---
st.set_page_config(page_title="Gemini AI Agent", page_icon="🤖")

def get_api_key():
    """Fetch API key from Secrets (Cloud) or .env (Local)"""
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def setup_agent():
    """Initializes the Agent once and caches it for performance"""
    api_key = get_api_key()
    
    if not api_key:
        st.error("Missing GOOGLE_API_KEY. Please set it in Streamlit Secrets or .env file.")
        st.stop()

    # Model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # Note: 2.5 is not a standard version yet, 1.5 is the current stable flash
        google_api_key=api_key,
        temperature=0
    )

    # Tools
    tools = [calculator, word_counter]

    # Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful AI assistant. Answer using your own knowledge "
            "and use tools only when necessary. For math, use the 'calculator' tool. "
            "For counting words, use the 'word_counter' tool."
        )),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # Build Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- UI LOGIC ---
st.title("🤖 Gemini Agent Bot")
st.markdown("Ask me to calculate something or count words in a text!")

# Initialize the agent
executor = setup_agent()

query = st.text_input("How can I help you today?", placeholder="e.g., What is 55 * 12?")

if st.button("Run Agent", type="primary"):
    if query:
        with st.spinner("Thinking..."):
            try:
                response = executor.invoke({"input": query})
                st.subheader("Response")
                st.write(response["output"])
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query first.")
