import langchain_pdf_retriever

def rag_agent(input_text, filename):
    rag_chain = langchain_pdf_retriever.MistralPDF(filename)
    results = rag_chain.invoke(input_text)

    print(results)

if __name__ == "__main__":
    rag_agent({"input": "Who was Rockefeller?"}, filename = "C:/Users/sriva/OneDrive/My Tech Projects/doc-chat/doc-chat/test-file/sample-pdf.pdf")
