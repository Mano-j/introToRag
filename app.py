from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

from CustomTools.pdfEngine import create_pdf_embeddings
from CustomTools.tools import (
    save_to_notes,
    get_response_from_learned_data,
    print_to_screen,
)
from ui import init_page, print_text

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
tools = [save_to_notes, get_response_from_learned_data]
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)


def on_input_change(user_question):
    query = "Try to get response data by using the tool 'get_response_from_learned_data tool': my query is="
    agent_respone = agent_executor.invoke({"input": query+user_question})
    print_text(agent_respone["output"])


def main():
    event_handlers = {
        "on_input": on_input_change,
        "on_pdf_upload": create_pdf_embeddings,
    }
    init_page(event_handlers)


if __name__ == "__main__":
    main()
