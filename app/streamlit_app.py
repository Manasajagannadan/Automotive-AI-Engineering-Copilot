import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

from src.vlm.vlm_client import (
    VLMNotConfigured,
    analyze_image,
    vlm_available,
)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="Automotive AI Engineering Copilot",
    page_icon="🚗",
    layout="wide",
)


st.title("🚗 Automotive AI Engineering Copilot")
st.caption("RAG + LLM + VLM assistant for automotive engineering")


rag_tab, vlm_tab = st.tabs(["📚 RAG Copilot", "🖼️ VLM Image Analysis"])


with rag_tab:
    st.subheader("Ask the Automotive Engineering Copilot")

    question = st.text_area(
        "Ask an automotive engineering question",
        placeholder="Example: What is the accessory weight threshold for the Load Carrying Capacity Reduced label?",
        height=120,
    )

    if st.button("Ask Copilot", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching technical documents..."):
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"question": question},
                        timeout=60,
                    )

                response.raise_for_status()
                result = response.json()

                st.subheader("Answer")
                st.markdown(result.get("answer", "No answer returned."))

                sources = result.get("sources", [])
                if sources:
                    st.subheader("Sources")
                    for source in sources:
                        st.write(f"- {source}")

                confidence = result.get("retrieval_confidence")
                latency = result.get("latency_ms")

                col1, col2 = st.columns(2)

                with col1:
                    if confidence is not None:
                        st.metric(
                            "Retrieval confidence",
                            f"{confidence:.3f}",
                        )

                with col2:
                    if latency is not None:
                        st.metric(
                            "Latency",
                            f"{latency:.0f} ms",
                        )

            except requests.exceptions.RequestException as exc:
                st.error(f"API connection error: {exc}")


with vlm_tab:
    st.subheader("Automotive Image Analysis")

    google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()

    if not google_api_key:
        st.info(
            "VLM not configured. "
            "GOOGLE_API_KEY is optional. "
            "The core RAG system works without it."
        )
    else:
        st.success("VLM is configured.")

        uploaded_image = st.file_uploader(
            "Upload an automotive image",
            type=["png", "jpg", "jpeg"],
        )

        image_question = st.text_input(
            "What would you like to know about the image?",
            placeholder="Example: What warning indicators are visible?",
        )

        if uploaded_image:
            st.image(
                uploaded_image,
                caption="Uploaded image",
                use_container_width=True,
            )

            if st.button("Analyze Image", type="primary"):
                if not image_question.strip():
                    st.warning("Please enter a question about the image.")
                else:
                    try:
                        with st.spinner("Analyzing image..."):
                            answer = analyze_image(
                                uploaded_image.getvalue(),
                                image_question,
                            )

                        st.subheader("VLM Analysis")
                        st.markdown(answer)

                    except VLMNotConfigured as exc:
                        st.info(str(exc))

                    except Exception as exc:
                        st.error(f"VLM analysis error: {exc}")