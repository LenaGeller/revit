import streamlit as st
import requests
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
API_URL = f"{API_BASE_URL}/ask"

st.title("🔧 Revit MEP Knowledge Assistant")
st.caption("Frontend → FastAPI Backend → RAG/LLM Pipeline")

query = st.text_input("Frage stellen")

if query:
    with st.spinner("Antwort wird generiert..."):
        response = requests.post(
            API_URL,
            json={"question": query},
            timeout=120
        )

    if response.status_code != 200:
        st.error(f"API-Fehler: {response.status_code}")
        st.write(response.text)
    else:
        data = response.json()

        st.subheader("Antwort")
        st.write(data["answer"])

        if data.get("images"):
            st.subheader("Bilder")
            for image in data["images"]:
                st.image(image["path"])

        if data.get("sources"):
            st.subheader("Quellen")

            shown_pages = set()

            for i, source in enumerate(data["sources"], start=1):
                section_path = source.get("section_path", [])
                source_doc = source.get("source_doc", "")
                page_start = source.get("physical_page_start")
                page_end = source.get("physical_page_end")

                # Schlüssel für Duplikat-Prüfung
                page_key = (source_doc, page_start, page_end)

                # wenn diese Seite schon gezeigt wurde -> überspringen
                if page_key in shown_pages:
                    continue

                shown_pages.add(page_key)

                st.write(f"**Quelle {i}:** " + " > ".join(section_path))
                st.caption(source_doc)

                if page_start:
                    label = (
                        f"📄 PDF Seiten {page_start}–{page_end}"
                        if page_end and page_start != page_end
                        else f"📄 PDF Seite {page_start}"
                    )

                    with st.expander(label):
                        pdf_response = requests.get(
                            f"{API_BASE_URL}/pdf-page/{page_start}",
                            params={"source_doc": source_doc},
                            timeout=120
                        )

                        if pdf_response.status_code == 200:
                            st.image(pdf_response.content, use_container_width=True)
                        else:
                            st.error("PDF-Seite konnte nicht geladen werden.")
                            st.write(pdf_response.text)
