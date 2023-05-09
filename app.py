import os
import time
import hashlib

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from langchain.text_splitter import MarkdownTextSplitter

load_dotenv()

app = Flask(__name__)

client = chromadb.Client(
    # Settings(
    #     chroma_api_impl="rest",
    #     chroma_server_host="localhost",
    #     chroma_server_http_port="8000",
    # )
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY"),
    )
)

ef = embedding_functions.InstructorEmbeddingFunction(
    model_name="hkunlp/instructor-xl", device="cpu"
)

markdown_splitter = MarkdownTextSplitter(chunk_size=1024, chunk_overlap=100)


def create_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_collection(name):
    return client.get_or_create_collection(name=name, embedding_function=ef)


def get_files_content(directory):
    print(directory)
    file_contents = []
    for folder, _subfolders, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(folder, file)
            with open(file_path, "r", encoding="utf-8") as file_descriptor:
                file_content = file_descriptor.read()
                file_contents.append(
                    {"content": file_content, "path": file_path}
                )
    return file_contents


@app.route("/api/collection/<path:name>/index", methods=["GET"])
def index(name):
    start_time = time.time()
    collection = get_collection(name)
    files = get_files_content(os.getenv("DATA_DIRECTORY"))

    print("Collection name:", name)
    print(
        len(files),
        "file(s) found in directory:",
        '"' + os.getenv("DATA_DIRECTORY") + '"',
    )

    for file_idx, file in enumerate(files):
        print(
            "--->",
            str(file_idx + 1) + "/" + str(len(files)),
            "Processing file:",
            file["path"],
        )

        file["chunks"] = markdown_splitter.create_documents(
            texts=[file["content"]],
            metadatas=[
                {
                    "file_path": file["path"],
                    "file_name": os.path.basename(file["path"]),
                }
            ],
        )

        print(
            "    File",
            '"' + os.path.basename(file["path"]) + '"',
            "has been splitted into",
            len(file["chunks"]),
            "chunks",
        )

        ids = []
        texts = []
        metadatas = []
        for idx, document in enumerate(file["chunks"]):
            ids.append(file["path"] + "-" + str(idx))
            texts.append(document.page_content)
            metadatas.append(document.metadata)

        print(
            "    Adding",
            len(file["chunks"]),
            "chunks to",
            '"' + name + '"',
            "collection",
        )

        add_start_time = time.time()
        collection.add(ids=ids, documents=texts, metadatas=metadatas)

        print(
            "    ",
            len(file["chunks"]),
            "chunks added in:",
            round(time.time() - add_start_time, 2),
            "secondes",
        )

    print(
        "All filed processed in:",
        round(time.time() - start_time, 2),
        "secondes",
    )

    return jsonify({"success": True})


@app.route("/api/collection/<path:name>", methods=["GET"])
def query_collection(name):
    query = request.args.get("query")
    collection = get_collection(name)
    result = collection.query(query_texts=[query], n_results=3)
    print(result)
    return jsonify({"result": result})


@app.route("/api/collection/<path:name>/info", methods=["GET"])
def info(name):
    collection = get_collection(name)
    return jsonify({"count": collection.count(), "peek": collection.peek()})


@app.route("/api/collections", methods=["GET"])
def list_collections():
    return jsonify({"collections": client.list_collections()})


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"success": True})


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=8080)
    # app.run(debug=True, host='0.0.0.0')
