from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

os.environ['GOOGLE_API_KEY']=st.secrets['GOOGLE_API_KEY']

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.7
)

SYSTEM_PROMPT = """
You are the most intelligent being in the world.
Your responsibilities:
- Give accurate and clear answers.
- Explain difficult concepts in simple language.
- If you don't know something, clearly say that you don't know.
- Do not make up information.
- Be concise but provide enough explanation to be useful.
"""


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    messages_with_system_prompt = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + messages
    response = llm.invoke(messages_with_system_prompt)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
