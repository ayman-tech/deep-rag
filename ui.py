# run docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

from nicegui import ui, app as nicegui_app, events, run
from src.retrieval import hybrid_retrieve
from src.generation import generate_reasoned_answer
from src.ingestion import ingest_pdf
from fastapi import UploadFile, File
from pydantic import BaseModel
import shutil
import os

# ── FastAPI endpoints (mounted directly on NiceGUI's internal FastAPI app) ──

class QueryRequest(BaseModel):
    query: str

@nicegui_app.post("/upload-pdf")
async def upload_document(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        ingest_pdf(temp_path)
        return {"message": f"Successfully processed {file.filename}"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@nicegui_app.post("/ask")
async def ask_endpoint(request: QueryRequest):
    context_chunks = hybrid_retrieve(request.query)
    result = generate_reasoned_answer(request.query, context_chunks)
    return {
        "query": request.query,
        "reasoning": result["thought"],
        "answer": result["answer"],
        "sources_used": len(context_chunks),
    }

# ── NiceGUI Frontend ──

@ui.page("/")
def index():
    # run docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    ui.query("body").style(
        "margin:0; font-family: 'Segoe UI', sans-serif;"
    )

    # Shared reactive state
    result_data = {"value": None}

    # Header
    with ui.header().classes("bg-purple-700 text-white shadow-lg items-center"):
        ui.icon("psychology", size="sm").classes("text-3xl")
        ui.label("Gold Standard RAG").classes("text-2xl font-bold ml-2")
        ui.space()
        ui.label("Upload a PDF, then ask anything").classes("text-sm opacity-80")

    # ── Upload handler ──
    async def handle_upload(e: events.UploadEventArguments):
        try:
            name = e.file.name
            temp_path = f"temp_{name}"
            content = await e.file.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            try:
                await run.io_bound(ingest_pdf, temp_path)
                ui.notify(f"Successfully processed {name}", type="positive")
                upload_status.set_text(f"Loaded: {name}")
                upload_status.classes(replace="text-green-600 font-semibold")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as ex:
            ui.notify(f"Error: {ex}", type="negative")
            upload_status.set_text(f"Error: {ex}")
            upload_status.classes(replace="text-red-600 text-sm")

    # ── Ask handler ──
    async def handle_ask():
        query = query_input.value
        if not query or not query.strip():
            ui.notify("Please enter a question", type="warning")
            return
        ask_button.disable()
        spinner.set_visibility(True)
        answer_card.clear()
        try:
            context_chunks = await run.io_bound(hybrid_retrieve, query)
            result = await run.io_bound(generate_reasoned_answer, query, context_chunks)
            result_data["value"] = result
            with answer_card:
                with ui.card().classes("w-full bg-blue-50 p-4"):
                    ui.label("Answer").classes("text-lg font-bold text-blue-800")
                    ui.markdown(result["answer"]).classes("text-gray-800")
                reasoning = result.get("thought", "")
                if reasoning and reasoning != "Reasoning not available":
                    with ui.expansion("Reasoning", icon="lightbulb").classes("w-full"):
                        ui.markdown(reasoning).classes("text-gray-600 italic")
                ui.label(f"Sources used: {len(context_chunks)}").classes("text-sm text-gray-500 mt-2")
        except Exception as ex:
            with answer_card:
                ui.label(f"Error: {ex}").classes("text-red-600")
        finally:
            ask_button.enable()
            spinner.set_visibility(False)

    # Main two-column layout
    with ui.column().classes("w-full max-w-6xl mx-auto p-6 gap-6"):
        with ui.row().classes("w-full gap-6 items-start flex-wrap"):

            # LEFT PANEL
            with ui.column().classes("flex-1 min-w-[360px] gap-4"):

                # Upload card
                with ui.card().classes("w-full p-4 shadow-md"):
                    ui.label("Upload PDF").classes("text-lg font-bold text-purple-700")
                    ui.upload(
                        label="Drag & drop or click to upload",
                        on_upload=handle_upload,
                        auto_upload=True,
                    ).props('accept=".pdf"').classes("w-full")
                    upload_status = ui.label("No file uploaded yet").classes("text-sm text-gray-400")

                # Query card
                with ui.card().classes("w-full p-4 shadow-md"):
                    ui.label("Ask a Question").classes("text-lg font-bold text-purple-700")
                    query_input = ui.textarea(
                        placeholder="e.g. What skills does Ayman have?",
                    ).classes("w-full")
                    with ui.row().classes("w-full items-center gap-4 mt-2"):
                        ask_button = ui.button("Ask", icon="search", on_click=handle_ask).classes(
                            "bg-purple-700 text-white"
                        )
                        spinner = ui.spinner("dots", size="lg", color="purple")
                        spinner.set_visibility(False)

            # RIGHT PANEL
            with ui.column().classes("flex-1 min-w-[360px] gap-4"):
                with ui.card().classes("w-full p-4 min-h-[300px] shadow-md"):
                    ui.label("Results").classes("text-lg font-bold text-purple-700")
                    answer_card = ui.column().classes("w-full gap-3")
                    with answer_card:
                        ui.label("Results will appear here...").classes("text-gray-400 italic")

    # Footer
    with ui.footer().classes("bg-gray-100 text-center text-sm text-gray-500 py-3"):
        ui.label("Powered by FastAPI  |  Qdrant  |  DeepSeek  |  NiceGUI")


# Launch
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(port=8000, title="Gold Standard RAG", reload=False)
