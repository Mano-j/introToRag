import streamlit as st

def print_text(text_to_print):
    st.write("Message from agent: ", text_to_print)
    
def init_page(event_handlers):
    st.set_page_config("RAG demo")
    st.header("RAG with PDF files")

    user_question = st.text_input("Your query goes here")

    if user_question:
        event_handlers["on_input"](user_question)

    with st.sidebar:
        st.title("PDF uploader:")
        pdf_docs = st.file_uploader(
            "Select your PDF(s) and Click on the Submit & Process Button",
            type='pdf',
            accept_multiple_files=True,
        )
        if st.button("Upload & Process"):
            with st.spinner("Processing..."):
                event_handlers["on_pdf_upload"](pdf_docs)
                st.success("Done!")
