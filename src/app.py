from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage , AIMessage , SystemMessage
from langgraph.graph import StateGraph, START , END
from langgraph.graph.message import add_messages
from tools import extract_text
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import tools_condition , ToolNode

load_dotenv()

llm = ChatGroq(
    model= "openai/gpt-oss-20b", 
    temperature= 0
)

tools = [extract_text]

llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages : Annotated[list[AnyMessage] , add_messages]

def assistant(state: AgentState):
    system_message = SystemMessage(content="""
    You are a helpful gala planning assistant.

    You have access to a gotham guest database through the extract_text tool.

    When a user asks a question about guests or information
    stored in the guest database, use extract_text before answering.

    For general questions unrelated to the guest database,
    answer directly.
    """)

    response = llm_with_tools.invoke(
        [system_message] + state["messages"]
    )

    print("TOOL CALLS:", response.tool_calls)

    return {"messages": [response]}

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools" , ToolNode(tools))

builder.add_edge(START , "assistant")
builder.add_conditional_edges("assistant" , tools_condition)
builder.add_edge("tools" , "assistant")

alfred = builder.compile()

messages = [HumanMessage(content="tell me sometihng funney about the joker from his description")]
response = alfred.invoke({"messages" : messages})

print(response["messages"])

