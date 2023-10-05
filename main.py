import os
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CORPUS_PATH = os.environ.get("CORPUS_PATH")
question = "Describe the rope walk method."

def create_docsearch(path=CORPUS_PATH):
    f = open(path, "r")
    text = f.read()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separator="\n"
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(chunks, embeddings)
    return docsearch

def main():
    if not question:
        print("No question was provided. Exiting.")
        return

    print(f"Considering: {question}")
    print("creating docsearch")
    docsearch = create_docsearch()
    docs = docsearch.similarity_search(question)

    print("loading llm")
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    print("creating chain")
    chain = load_qa_chain(llm, chain_type="stuff")
    print("waiting for callback...")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
        print(f"> {response}")
    print("done.")

if __name__ == "__main__":
    main()
