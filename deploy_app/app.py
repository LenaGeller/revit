import streamlit as st
import json
import chromadb
import os
import numpy as np
from openai import OpenAI
import fitz
import requests
import zipfile
import shutil

# ==================================================
# BASE
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================================================
# INIT
# ==================================================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

TOP_K = 3
EMBED_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4.1-mini"

# ==================================================
# DOWNLOAD ASSETS
# ==================================================
@st.cache_resource
def download_assets():

    files = {
        "chroma_db.zip":
        "https://huggingface.co/datasets/LenaGeller/revit/resolve/main/chroma_db.zip",

        "cropped_images.zip":
        "https://huggingface.co/datasets/LenaGeller/revit/resolve/main/cropped_images.zip",

        "revit_mep_2011_user_guide_deu.pdf":
        "https://huggingface.co/datasets/LenaGeller/revit/resolve/main/revit_mep_2011_user_guide_deu.pdf"
    }

    for filename, url in files.items():

        target_exists = (
            os.path.exists(
                os.path.join(BASE_DIR, filename.replace(".zip", ""))
            )
            if filename.endswith(".zip")
            else os.path.exists(
                os.path.join(BASE_DIR, filename)
            )
        )

        if target_exists:
            continue

        st.write(f"⬇️ Lade {filename} ...")

        target_file = os.path.join(BASE_DIR, filename)

        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()

        with open(target_file, "wb") as f:
            shutil.copyfileobj(r.raw, f)

        if filename.endswith(".zip"):
            with zipfile.ZipFile(target_file, "r") as zip_ref:
                zip_ref.extractall(BASE_DIR)

            os.remove(target_file)

# ==================================================
# LOAD SYSTEM
# ==================================================
@st.cache_resource
def load_system():

    with open(
        os.path.join(BASE_DIR, "final_chunks_stage2.json"),
        "r",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    with open(
        os.path.join(BASE_DIR, "image_map.json"),
        "r",
        encoding="utf-8"
    ) as f:
        image_map = json.load(f)

    chroma_client = chromadb.PersistentClient(
        path=os.path.join(BASE_DIR, "chroma_db")
    )

    collection = chroma_client.get_or_create_collection(
        "tutorial_chunks"
    )

    if collection.count() == 0:
        st.error("Chroma Datenbank leer.")
        st.stop()

    return chunks, image_map, collection


download_assets()
chunks, image_map, collection = load_system()

# ==================================================
# RETRIEVE
# ==================================================
def retrieve(query, top_k=TOP_K):

    query_embedding = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding

    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    best_idx = [
        meta["idx"]
        for meta in search_results["metadatas"][0]
    ]

    results = []

    for idx in best_idx:

        chunk = chunks[idx]
        context = chunk["context"]
        metadata = chunk["metadata"]

        has_table = metadata.get("has_table", False)

        if has_table and "|" in context:
            paragraphs = [
                line.strip()
                for line in context.splitlines()
                if line.strip().startswith("|")
            ]
        else:
            paragraphs = [
                p.strip()
                for p in context.split("\n\n")
                if p.strip()
            ]

        if paragraphs:

            para_embeddings = []

            for para in paragraphs:
                emb = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=para
                ).data[0].embedding

                para_embeddings.append(emb)

            para_embeddings = np.array(para_embeddings)

            query_vec = np.array(query_embedding)

            para_embeddings = para_embeddings / np.linalg.norm(
                para_embeddings,
                axis=1,
                keepdims=True
            )

            query_vec = query_vec / np.linalg.norm(query_vec)

            scores = np.dot(para_embeddings, query_vec)

            best_passage = paragraphs[np.argmax(scores)]

        else:
            best_passage = context

        matching_images = []

        for image_id in metadata.get("image_ids", [])[:2]:

            image_path = os.path.join(
                BASE_DIR,
                "cropped_images",
                f"{image_id}.png"
            )

            if os.path.exists(image_path):
                matching_images.append({
                    "image_id": image_id,
                    "path": image_path
                })

        results.append({
            "chunk": chunk,
            "best_passage": best_passage,
            "images": matching_images
        })

    return results

# ==================================================
# GPT ANSWER
# ==================================================
def generate_answer(query, retrieved_chunks):

    context_parts = []

    for item in retrieved_chunks:

        path = item["chunk"]["metadata"]["section_path"]

        context_parts.append(
            f"SECTION: {' > '.join(path)}\n{item['best_passage']}"
        )

    full_context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content":
                "Du bist ein technischer Revit Assistent. "
                "Antworte präzise anhand des Kontexts."
                "Nutze nur Text, der wirklich im Kontext steht."
            },
            {
                "role": "user",
                "content":
                f"Frage: {query}\n\nKontext:\n{full_context}"
            }
        ]
    )

    return response.choices[0].message.content

# ==================================================
# PDF PAGE
# ==================================================
def render_pdf_page(pdf_path, page_number):

    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]

    pix = page.get_pixmap(dpi=150)

    return pix.tobytes("png")

# ==================================================
# UI
# ==================================================
st.title("🔧 Revit MEP Knowledge Assistant")

query = st.text_input("Frage stellen")

if query:

    results = retrieve(query)

    answer = generate_answer(query, results)

    st.subheader("Antwort")
    st.write(answer)

    if results:

        best = results[0]
        meta = best["chunk"]["metadata"]

        for image in best["images"]:
            st.image(image["path"])

        st.subheader("Quelle")
        st.write("Kapitel:", " > ".join(meta["section_path"]))

        page_number = None

        if best["images"]:
            first_id = best["images"][0]["image_id"]

            match = next(
                (
                    img for img in image_map
                    if img["image_id"] == first_id
                ),
                None
            )

            if match:
                page_number = match.get("page")

        if page_number:

            with st.expander(
                f"📄 PDF Seite {page_number}"
            ):

                page_img = render_pdf_page(
                    os.path.join(
                        BASE_DIR,
                        meta["source_doc"]
                    ),
                    page_number
                )

                st.image(
                    page_img,
                    use_container_width=True
                )
