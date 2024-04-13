import os
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from ui import print_text

FIASS_DIR_NAME = "embeddings"


def get_prompt():
    prompt_template = """
    Provide a detailed response to the query making use of the data provided in the context field. When there is NO relevant information in the context simply return "Not able to find any relevant information in the provided context." DO NOT try to provide answer if you could NOT find it inside the provided context\n
    Context: {context}?\n
    Question: {question}\n

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt

def get_conversational_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = get_prompt()
    chain = load_qa_chain(llm=llm, prompt=prompt)
    return chain



class LearnedDataInput(BaseModel):
    user_question: str = Field(
        description="query of the user for which the response has to be found from the learned data"
    )

@tool("get_response_from_learned_data", args_schema=LearnedDataInput, return_direct=False)
def get_response_from_learned_data(user_question):
    """Use this tool when you have to find any relevant information for the user query"""
    
    if os.path.exists(FIASS_DIR_NAME):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local(
            FIASS_DIR_NAME, embeddings, allow_dangerous_deserialization=True
        )
        docs = new_db.similarity_search(user_question)
        # print(len(docs))
        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True
        )
        return response["output_text"]
    else:
        return "Please provide relevant PDF before querying"


class NotesInput(BaseModel):
    text_to_save: str = Field(description="string value that should be saved")

@tool("save_to_notes", args_schema=NotesInput, return_direct=True)
def save_to_notes(txt_to_save: str):
    "Use this tool only when you want to save the response. Use this only when prompted by the user"
    note_file = os.path.join("data", "notes.txt")

    if not os.path.exists(note_file):
        open(note_file, "w")

    with open(note_file, "a") as f:
        f.writelines([txt_to_save + "\n"])

    return "note saved\n"


class PrintInput(BaseModel):
    text_to_print: str = Field(description="Provide the text that has to be displayed to the user")

@tool("print_to_screen", args_schema=PrintInput)
def print_to_screen(text_to_print):
    """Use this tool to print any message to the user"""
    print_text(text_to_print)