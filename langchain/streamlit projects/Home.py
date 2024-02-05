import streamlit as st
from pages import email, rag, weaviate

def main():
    pages = {
        "email": email,
        "rag": rag,
        "weaviate": weaviate
    }

    # st.sidebar.title("Navigation")
    # selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))

    # Display the selected page
    # pages[selected_page].app()

if __name__ == "__main__":
    main()
