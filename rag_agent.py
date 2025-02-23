import langchain_pdf_retriever

def setup_pdf_rag(filename):
    rag_chain = langchain_pdf_retriever.MistralPDF(filename)
    return rag_chain

def rag_agent_response(input_text, rag_chain, chat_history):
    formatted_history = "\n".join([
            f"Human: {h[0]}\nAssistant: {h[1]}" 
            for h in chat_history
        ])
    input_text["chat_history"] = formatted_history
    results = rag_chain.invoke(input_text)
    print(results)
    return results

if __name__ == "__main__":
    rag_chain = setup_pdf_rag(filename = "./test-file/sample-pdf.pdf")
    rag_agent_response({"input": "Who was Rockefeller?"}, rag_chain)
