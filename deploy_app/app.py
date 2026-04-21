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


# =========================
# INIT
# =========================

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

TOP_K = 3
EMBED_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4.1-mini"

@st.cache_resource
def download_assets():

    files = {
        "chroma_db.zip": "https://huggingface.co/datasets/LenaGeller/revit/resolve/main/chroma_db.zip",
        "cropped_images.zip": "https://huggingface.co/datasets/LenaGeller/revit/resolve/main/cropped_images.zip",
        "revit_mep_2011_user_guide_deu.pdf": "https://huggingface.co/datasets/LenaGeller/revit/resolve/main/revit_mep_2011_user_guide_deu.pdf"
    }

    for filename, url in files.items():

        target_exists = (
            os.path.exists(filename.replace(".zip", ""))
            if filename.endswith(".zip")
            else os.path.exists(filename)
        )

        if target_exists:
            continue

        st.write(f"⬇️ Lade {filename} ...")

        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)

        if filename.endswith(".zip"):
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(".")
            os.remove(filename)
                
# =========================
# DATA LOAD
# =========================
@st.cache_resource
def load_system():
    with open("final_chunks_stage2.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open("image_map.json", "r", encoding="utf-8") as f:
        image_map = json.load(f)

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("tutorial_chunks")

    if collection.count() == 0:
        st.error("Chroma Datenbank leer oder nicht korrekt entpackt.")
        st.stop()

    return chunks, image_map, collection

download_assets()
chunks, image_map, collection = load_system()


# =========================
# RETRIEVAL
# =========================
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

        # =========================
        # PASSAGE-LEVEL FILTER
        # =========================
        metadata = chunk.get("metadata", {})
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

            # Cosine ähnlich wie vorher
            query_vec = np.array(query_embedding)
            para_norms = np.linalg.norm(para_embeddings, axis=1, keepdims=True)
            query_norm = np.linalg.norm(query_vec)

            para_embeddings = para_embeddings / para_norms
            query_vec = query_vec / query_norm

            para_scores = np.dot(para_embeddings, query_vec)
            best_para_idx = np.argmax(para_scores)

            best_passage = paragraphs[best_para_idx]
        else:
            best_passage = context

        # =========================
        # IMAGE FILTER
        # =========================
        matching_images = []

        # =========================
        # TABLE SPECIAL CASE
        # =========================
        if has_table:
            for image_id in metadata.get("image_ids", []):
                image_path = os.path.join(
                    "cropped_images",
                    f"{image_id}.png"
                )

                matching_images.append({
                    "image_id": image_id,
                    "path": image_path
                })

        # =========================
        # NORMAL PASSAGE IMAGE MATCH
        # =========================
        else:
            for image in image_map:
                anchor = image.get("anchor_text")
                if not anchor:
                    continue

                anchor_short = anchor[:120]

                if (
                    anchor_short in best_passage
                    or best_passage[:80] in anchor
                ):
                    image_path = os.path.join(
                        "cropped_images",
                        f"{image['image_id']}.png"
                    )

                    enriched_image = image.copy()
                    enriched_image["path"] = image_path
                    matching_images.append(enriched_image)

            # normaler fallback
            if not matching_images:
                for image_id in metadata.get("image_ids", [])[:1]:
                    image_path = os.path.join(
                        "cropped_images",
                        f"{image_id}.png"
                    )

                    matching_images.append({
                        "image_id": image_id,
                        "path": image_path
                    })

        results.append({
            "score": 1.0,
            "chunk": chunk,
            "best_passage": best_passage,
            "images": matching_images
        })

    return results


# =========================
# GPT
# =========================
def generate_answer(query, retrieved_chunks):
    context_parts = []

    for item in retrieved_chunks:
        chunk = item["chunk"]
        path = chunk["metadata"]["section_path"]

        context_parts.append(
            f"SECTION: {' > '.join(path)}\n{item['best_passage']}"
        )

    full_context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Du bist ein technischer Tutorial-Assistent. "
                    "Beantworte die Frage präzise anhand des Kontexts."
                )
            },
            {
                "role": "user",
                "content": f"Frage: {query}\n\nKontext:\n{full_context}"
            }
        ]
    )

    return response.choices[0].message.content


def render_pdf_page(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]

    pix = page.get_pixmap(dpi=150)
    return pix.tobytes("png")


# =========================
# UI
# =========================
st.title("🔧 Revit MEP Knowledge Assistant")

query = st.text_input("Stelle eine Frage zum Handbuch")

if query:
    results = retrieve(query)
    answer = generate_answer(query, results)

    st.subheader("Antwort")
    st.write(answer)


    shown = set()

    if results:
        best_result = results[0]

        for image in best_result["images"]:
            path = image["path"]

            if path not in shown:
                if os.path.exists(path):
                    st.image(path)
                shown.add(path)


        st.subheader("Quelle")
        best = results[0]["chunk"]
        meta = best["metadata"]

        st.write("Kapitel:", " > ".join(meta["section_path"]))
        st.write("Dokument:", meta["source_doc"])

        page_number = None

        # 1) zuerst aus Bild-Match
        if best_result["images"]:
            page_number = best_result["images"][0].get("page")

        # 2) fallback: aus metadata, falls später vorhanden
        if not page_number:
            image_ids = meta.get("image_ids", [])
            if image_ids:
                first_id = image_ids[0]

                match = next(
                    (img for img in image_map if img["image_id"] == first_id),
                    None
                )

                if match:
                    page_number = match.get("page")

        if page_number:
            with st.expander(f"📄 PDF Seite {page_number} anzeigen"):
                page_img = render_pdf_page(
                    meta["source_doc"],
                    page_number
                )
                st.image(page_img, use_container_width=True)
