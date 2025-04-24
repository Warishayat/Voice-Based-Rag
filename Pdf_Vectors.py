#in this phase i will upload pdf
#parse the pdf 
#split that pdf text into chunks
#store into faiss-vectorstore after creting embeddings.
#will check the final response everthing is working fine or not.

import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def Pdf_parsing(pdf_path:str):
    md_text_images = pymupdf4llm.to_markdown(
        pdf_path,
        write_images=True,
        image_path="images",
        image_format="png",
        page_chunks=True
    )

    # print(f"The content of the first page is {md_text_images[0]}")

    docs = []
    for page in md_text_images:
        text = page["text"]
        page_num = md_text_images.index(page) + 1
        docs.append(Document(page_content=text,metadata={"page_number":page_num}))
    
    return docs

def textSplitter(documents,c_size=1200,c_overlap=200):
    "Chunks docs into smaller points"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = c_size,
        chunk_overlap = c_overlap
    )
    chunked_docs=text_splitter.split_documents(documents)

    #now make the embeddings of the chunked text
    Embed = HuggingFaceEmbeddings()

    #store to the vectorstore
    vectorstore = FAISS.from_documents(
        documents= chunked_docs,
        embedding = Embed
    )
    retriver = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":2}
    )

    return retriver

    print("All set till here")
    


if __name__ == "__main__":
    pdf_path = r"C:\Users\HP\Desktop\Voice-Base-Rag\Voice-Based-Rag\Deepseek.pdf"
    docs = Pdf_parsing(pdf_path=pdf_path)
    retriver=textSplitter(documents=docs)
    print(retriver.invoke("What is deepseek r1?"))