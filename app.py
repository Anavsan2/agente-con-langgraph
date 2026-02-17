import os
import functools
import streamlit as st
from typing import Annotated, Literal, TypedDict

# Importaciones de LangChain y LangGraph
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ==========================================
# 1. Configuración de la Interfaz de Streamlit
# ==========================================
st.set_page_config(page_title="News Writer AI", page_icon="📰", layout="centered")
st.title("📰 AI News Writer Agent")
st.write("Escribe un tema y el agente buscará noticias, creará un esquema y redactará un artículo.")

# Barra lateral para credenciales
with st.sidebar:
    st.header("🔑 Configuración de API")
    google_api_key = st.text_input("Google API Key (Gemini)", type="password")
    tavily_api_key = st.text_input("Tavily API Key (Search)", type="password")
    st.markdown("---")
    st.info("El agente utiliza Gemini 2.5 Flash y Tavily Search.")

# ==========================================
# 2. Definición del Grafo (LangGraph)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

SEARCH_TEMPLATE = """Your job is to search the web for related news that would be relevant to generate the article described by the user.
NOTE: Do not write the article. Just search the web for related news if needed and then forward that news to the outliner node."""

OUTLINER_TEMPLATE = """Your job is to take as input a list of articles from the web along with users instruction on what article they want to write and generate an outline for the article."""

WRITER_TEMPLATE = """Your job is to write an article based on the information and outline provided by the previous nodes.

You must STRICTLY adhere to this format:
TITLE: <title>
BODY: <body>

CRITICAL RULES:
1. DO NOT include any introductory text, internal reasoning, or comments.
2. DO NOT mention irrelevant news or other people with similar names found in the search.
3. Output ONLY the exact 'TITLE:' and 'BODY:' sections. Nothing else.
4. Do not copy the outline, use it to write the final text."""

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm

def agent_node(state: AgentState, agent, name: str):
    result = agent.invoke(state)
    return {'messages': [result]}

def should_search(state: AgentState) -> Literal["tools", "outliner"]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "outliner"

# Cacheamos la creación del grafo para no recompilarlo en cada recarga de Streamlit
@st.cache_resource
def build_graph(google_key: str, tavily_key: str):
    os.environ["GOOGLE_API_KEY"] = google_key
    os.environ["TAVILY_API_KEY"] = tavily_key
    
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.2)
    tools = [TavilySearch(max_results=5)]

    search_agent = create_agent(llm, tools, SEARCH_TEMPLATE)
    outliner_agent = create_agent(llm, [], OUTLINER_TEMPLATE)
    writer_agent = create_agent(llm, [], WRITER_TEMPLATE)

    search_node = functools.partial(agent_node, agent=search_agent, name="Search")
    outliner_node = functools.partial(agent_node, agent=outliner_agent, name="Outliner")
    writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer")
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("outliner", outliner_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("search")
    workflow.add_conditional_edges("search", should_search)
    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()

# ==========================================
# 3. Lógica de la Aplicación Streamlit
# ==========================================

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial de mensajes en la UI
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# Capturar input del usuario
if user_query := st.chat_input("Ejemplo: Novedades sobre la exploración de Marte..."):
    if not google_api_key or not tavily_api_key:
        st.warning("⚠️ Por favor, introduce tus claves de API en la barra lateral primero.")
        st.stop()

    # Añadir mensaje del usuario al estado y a la UI
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.write(user_query)

    # Preparar el grafo
    app_graph = build_graph(google_api_key, tavily_api_key)

    with st.chat_message("assistant"):
        final_response = ""
        last_msg = None
        
        # Usamos un status_container para mostrar el progreso interno
        with st.status("Procesando tu solicitud...", expanded=True) as status:
            try:
                for event in app_graph.stream({"messages": [HumanMessage(content=user_query)]}):
                    for node_name, node_state in event.items():
                        st.write(f"✅ Nodo ejecutado: **{node_name}**")
                        
                        # Extraemos el último mensaje del estado
                        if "messages" in node_state and len(node_state["messages"]) > 0:
                            last_msg = node_state["messages"][-1]
                            raw_content = getattr(last_msg, 'content', "")
                            
                            # Si Gemini devuelve una lista, extraemos el texto
                            if isinstance(raw_content, list):
                                extracted_text = [
                                    str(item.get("text", item)) if isinstance(item, dict) else str(item) 
                                    for item in raw_content
                                ]
                                final_response = " ".join(extracted_text)
                            else:
                                final_response = str(raw_content)
                                
                status.update(label="¡Proceso finalizado!", state="complete", expanded=False)
            
            except Exception as e:
                status.update(label="Ocurrió un error", state="error")
                st.error(f"Error de ejecución: {str(e)}")
        
        # Limpieza de formato Markdown (quita los ``` si el modelo los añade)
        clean_response = final_response.replace("```markdown", "").replace("```", "").strip()
        
        # Validación final para mostrar en pantalla
        if clean_response:
            st.markdown(clean_response)
            st.session_state.messages.append(AIMessage(content=clean_response))
        else:
            st.error("❌ La respuesta final está vacía.")
            with st.expander("Ver detalles del error interno (Debug)"):
                st.write("Esto es lo que devolvió Gemini internamente. Si ves información sobre 'SAFETY', el filtro bloqueó el contenido por tratar temas delicados:")
                if last_msg:
                    try:
                        st.json(last_msg.dict())
                    except:
                        st.write(last_msg)
                else:
                    st.write("No se generó ningún mensaje en los nodos.")
