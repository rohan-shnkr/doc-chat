import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

def pdf_loader(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def MistralPDF(pdf_file):
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    file_path = pdf_file
    loader = PyPDFLoader(file_path)
    
    docs = loader.load()
    
    # print(len(docs))
    # print(docs[0].page_content[0:100])
    # print(docs[0].metadata)

    os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_AI_API_KEY")
    llm = ChatMistralAI(model="mistral-large-latest")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    persist_directory = "chroma_db"

    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
                                        persist_directory=persist_directory)

    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # results = rag_chain.invoke({"input": "Who is John D Rockefeller?"})

    # print(results)

    return rag_chain

if __name__ == "__main__":
    MistralPDF("C:/Users/sriva/OneDrive/My Tech Projects/doc-chat/doc-chat/test-file/sample-pdf.pdf")
