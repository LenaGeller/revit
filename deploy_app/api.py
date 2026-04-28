import os
import json
import chromadb
import numpy as np
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
import fitz
from fastapi.responses import Response
import dotenv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv.load_dotenv((os.path.join(BASE_DIR, ".env")))
print("KEY FOUND:", os.getenv("OPENAI_API_KEY") is not None)



TOP_K = 3
EMBED_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4.1-mini"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Revit MEP Knowledge Assistant API")


class AskRequest(BaseModel):
    question: str


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

    return chunks, image_map, collection


chunks, image_map, collection = load_system()


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

        matching_images = []

        for image_id in metadata.get("image_ids", [])[:2]:
            image_path = os.path.join(
                BASE_DIR,
                "cropped_images",
                f"{image_id}.png"
            )

            page_number = None

            match = next(
                (img for img in image_map if img["image_id"] == image_id),
                None
            )
            if match:
                page_number = match.get("page")
         
                matching_images.append({
                    "image_id": image_id,
                    "path": image_path,
                    "page": page_number
                })
    

        results.append({
            "chunk": chunk,
            "images": matching_images
        })

    return results


def generate_answer(query, retrieved_chunks):
    context_parts = []

    for item in retrieved_chunks:
        chunk = item["chunk"]
        path = chunk["metadata"]["section_path"]

        context_parts.append(
            f"SECTION: {' > '.join(path)}\n{chunk['context']}"
        )

    full_context = "\n\n".join(context_parts)

    system_prompt = """
Du bist ein technischer Dokumentations-Assistent.

Nutze ausschließlich Informationen aus dem bereitgestellten Kontext.

Regeln:
- Erfinde nichts.
- Nutze keine Informationen aus allgemeinem Wissen.
- Antworte nur aus dem Kontext.
- Wenn im Kontext konkrete Schritte stehen, gib diese Schritte vollständig wieder.
- Übernimm Tastenkombinationen, Buttonnamen, Fensternamen und Begriffe exakt.
- Bevorzuge Originalsätze aus dem Kontext.
- Du darfst Sätze nur leicht kürzen oder verbinden, aber keine neuen Inhalte hinzufügen.
- Wenn eine Überschrift zur Frage passt, nutze den Text unter dieser Überschrift.
- Wenn keine passende Information im Kontext steht, antworte exakt: "Nicht im Kontext enthalten."
- Antworte kompakt in maximal 5 Punkten.
- Keine allgemeinen Autodesk-, Revit- oder Software-Erklärungen.
"""

    user_prompt = f"""
FRAGE:
{query}

KONTEXT:
{full_context}

AUFGABE:
Beantworte die Frage ausschließlich mit Informationen aus dem KONTEXT.
Wenn der Kontext einen passenden Abschnitt enthält, gib die relevanten Schritte und Hinweise daraus wieder.
"""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content

def render_pdf_page(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]

    pix = page.get_pixmap(dpi=150)
    return pix.tobytes("png")


@app.post("/ask")
def ask(request: AskRequest):
    results = retrieve(request.question)
    answer = generate_answer(request.question, results)

    sources = []

    for item in results:
        meta = item["chunk"]["metadata"]

        sources.append({
            "section_path": meta.get("section_path"),
            "source_doc": meta.get("source_doc"),
            "image_ids": meta.get("image_ids", []),
            "physical_page_start": meta.get("physical_page_start"),
            "physical_page_end": meta.get("physical_page_end")
        })

    images = []

    if results:
        for image in results[0]["images"]:
            images.append({
                "image_id": image["image_id"],
                "path": image["path"]
            })

    return {
        "question": request.question,
        "answer": answer,
        "sources": sources,
        "images": images
    }

@app.get("/pdf-page/{page_number}")
def get_pdf_page(page_number: int, source_doc: str):
    pdf_path = os.path.join(BASE_DIR, source_doc)

    image_bytes = render_pdf_page(pdf_path, page_number)

    return Response(
        content=image_bytes,
        media_type="image/png"
    )