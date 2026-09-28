from langchain_community.document_loaders import TextLoader, DirectoryLoader


file_path = "/Users/apple/Desktop/learning/genai/document-ask/data/java_multithreading.txt"

file_loader = TextLoader(
    file_path=file_path,
    encoding="utf-8"
)



dir_loader = DirectoryLoader(
    "data/",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={
        "encoding": "utf-8"
    }
)
