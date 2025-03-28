from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document  # Import the correct Document class
import openai
import re
import config_path

help_file_path = f'{config_path.Database_PATH}/openfoam_command_helps.txt'

loader = TextLoader(help_file_path)
pages = loader.load()


pattern = re.compile(r"```input_file_begin:(.*?)input_file_end.```", re.DOTALL)
matches = pattern.findall(pages[0].page_content)
pages = [Document(page_content=match.strip(), metadata={'source': help_file_path}) for match in matches]

persist_directory = f'{config_path.Database_PATH}/openfoam_command_helps'

batch_size = config_path.batchsize

for i in range(0, len(pages), batch_size):
    print("i:",i)
    if(i+batch_size<=len(pages)-1):
        batch = pages[i:i + batch_size]
    elif(i<=len(pages)-2):
        batch = pages[i:]

    try:
        if(i==0):
            vectordb = FAISS.from_documents(
                documents=batch, 
                embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"))
        else:
            vectordb.add_documents(documents=batch)

    except openai.error.APIError as e:
        print(f"Error processing batch {i//batch_size + 1}: {e}")
        break
vectordb.save_local(persist_directory)
