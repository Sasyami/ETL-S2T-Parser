import os
import logging
import json
import datetime
from threading import Lock
from typing import List, Any, Dict, Optional
from flask import Flask, request, jsonify, render_template, send_from_directory
from agents.agent import get_model_name, agent_chat
from agents.sheet_group_classifier import classify_file_sheet_groups
from services.analysis import (
    finish_analysis,
    try_generate_description,
    try_generate_summary,
    try_refresh_s2t_transformations,
    try_sync_file_graph,
)
from services.graph_sync import clear_graph_projection
from storage.database import clear_all_data, get_file, init_db, store_excel_data
from processing.excel import (
    allowed_file,
    convert_to_serializable,
    get_preview_headers,
    parse_excel_with_decisions,
)
from storage.s2t import (
    clear_s2t_transformations,
    list_s2t_transformations,
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHAT_HISTORY_MAX_MESSAGES = 12
CHAT_HISTORY_MAX_MESSAGE_CHARS = 8000
CHAT_HISTORY_MAX_TOTAL_CHARS = 16000
CHAT_SESSION_ID_MAX_CHARS = 200
PROGRESS_EVENT_FIELDS = (
    "status",
    "phase",
    "percent",
    "message",
    "detail",
    "sheet_name",
    "sheet_index",
    "sheet_count",
)
HEADER_PREVIEW_OPTIONS = {
    "1": (0, 1),
    "2": (1, 1),
    "3": (0, 2),
}
file_bytes_cache: Dict[int, bytes] = {}
analysis_progress = {}
analysis_progress_lock = Lock()

init_db()


def _normalize_chat_history(value: Any) -> List[Dict[str, str]]:
    """Validate and bound browser-session chat context before sending it to the LLM."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("history must be an array")

    normalized: List[Dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"history[{index}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"}:
            raise ValueError(f"history[{index}].role must be 'user' or 'assistant'")
        if not isinstance(content, str):
            raise ValueError(f"history[{index}].content must be a string")
        text = content.strip()
        if not text:
            continue
        normalized.append({
            "role": role,
            "content": text[:CHAT_HISTORY_MAX_MESSAGE_CHARS],
        })

    normalized = normalized[-CHAT_HISTORY_MAX_MESSAGES:]
    while (
        len(normalized) > 1
        and sum(len(item["content"]) for item in normalized)
        > CHAT_HISTORY_MAX_TOTAL_CHARS
    ):
        normalized.pop(0)
    if normalized:
        normalized[-1]["content"] = normalized[-1]["content"][:CHAT_HISTORY_MAX_TOTAL_CHARS]
    return normalized


def _normalize_chat_session_id(value: Any) -> Optional[str]:
    """Validate an optional browser-session identifier used for tracing."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("session_id must be a string")
    text = value.strip()
    if not text:
        return None
    return text[:CHAT_SESSION_ID_MAX_CHARS]

def _progress_now() -> str:
    return datetime.datetime.now().isoformat()


def _set_analysis_progress(upload_id: Optional[str], **updates) -> None:
    if not upload_id:
        return
    payload = convert_to_serializable(updates)
    with analysis_progress_lock:
        current = analysis_progress.setdefault(
            upload_id,
            {
                "upload_id": upload_id,
                "status": "running",
                "phase": "queued",
                "percent": 0,
                "message": "Ожидаю начала анализа...",
                "history": [],
            },
        )
        current.update(payload)
        current["updated_at"] = _progress_now()
        event = {
            "timestamp": current["updated_at"],
            **{field: current.get(field) for field in PROGRESS_EVENT_FIELDS},
        }
        current.setdefault("history", []).append(event)
        current["history"] = current["history"][-40:]


def _get_analysis_progress(upload_id: str) -> Optional[Dict[str, Any]]:
    with analysis_progress_lock:
        progress = analysis_progress.get(upload_id)
        if progress is None:
            return None
        return json.loads(json.dumps(convert_to_serializable(progress), ensure_ascii=False))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat_app')
def chat_app():
    return render_template('chat_app.html')


@app.route('/analysis_progress/<upload_id>', methods=['GET'])
def get_analysis_progress(upload_id):
    progress = _get_analysis_progress(upload_id)
    if progress is None:
        return jsonify({"error": "Progress not found"}), 404
    return jsonify(progress), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    upload_id = request.form.get("upload_id")
    include_hidden_rows = (
        str(request.form.get("include_hidden_rows", "")).strip().casefold()
        in {"1", "true", "yes", "on"}
    )
    _set_analysis_progress(
        upload_id,
        status="running",
        phase="received",
        percent=2,
        message="Файл получен сервером...",
        detail=file.filename,
    )

    try:
        file_bytes = file.read()
        if len(file_bytes) == 0:
            _set_analysis_progress(
                upload_id,
                status="error",
                phase="validate",
                percent=100,
                message="Файл пустой",
                detail=file.filename,
            )
            return jsonify({'error': 'Empty file'}), 400

        _set_analysis_progress(
            upload_id,
            status="running",
            phase="parse",
            percent=5,
            message="Начинаю анализ Excel...",
            detail=file.filename,
        )
        sheets = parse_excel_with_decisions(
            file_bytes,
            corrections=None,
            progress_callback=lambda update: _set_analysis_progress(upload_id, **update),
            include_hidden_rows=include_hidden_rows,
        )

        _set_analysis_progress(
            upload_id,
            status="running",
            phase="store",
            percent=60,
            message="Сохраняю структуру и данные в SQLite...",
            detail=file.filename,
        )
        file_id = store_excel_data(file_bytes, file.filename, get_model_name(), sheets)
        file_bytes_cache[file_id] = file_bytes
        response = finish_analysis(
            file_id,
            file.filename,
            sheets,
            progress_callback=lambda update: _set_analysis_progress(
                upload_id,
                **update,
            ),
        )
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Error parsing Excel file")
        _set_analysis_progress(
            upload_id,
            status="error",
            phase="error",
            percent=100,
            message="Ошибка анализа файла",
            detail=str(e),
        )
        return jsonify({'error': f'Failed to parse Excel file: {str(e)}'}), 400

@app.route('/apply_corrections', methods=['POST'])
def apply_corrections():
    data = request.get_json()
    try:
        file_id = int(data.get("file_id"))
    except (TypeError, ValueError):
        file_id = None
    corrections = data.get("corrections", [])
    include_hidden_rows = data.get("include_hidden_rows") is True

    if not file_id:
        return jsonify({"error": "Missing file_id"}), 400

    if file_id not in file_bytes_cache:
        return jsonify({"error": "File bytes not found. Please re-upload."}), 404

    file_bytes = file_bytes_cache[file_id]
    file_record = get_file(file_id)
    if file_record is None:
        return jsonify({"error": "File not found"}), 404
    original_filename = file_record["filename"] or "unknown.xlsx"

    skipped_sheets = []
    header_corrections = {}
    for corr in corrections:
        sheet_name = corr["sheet_name"]
        if corr.get("skipped", False):
            skipped_sheets.append(sheet_name)
        else:
            header_corrections[sheet_name] = {
                "header_start_row": corr["header_start_row"],
                "header_rows_count": corr["header_rows_count"]
            }

    try:
        sheets = parse_excel_with_decisions(
            file_bytes,
            corrections=header_corrections,
            skip_sheets=skipped_sheets,
            include_hidden_rows=include_hidden_rows,
        )
        store_excel_data(
            file_bytes,
            original_filename,
            get_model_name(),
            sheets,
            file_id=file_id,
        )
        response = finish_analysis(
            file_id,
            original_filename,
            sheets,
        )
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Error applying corrections")
        return jsonify({"error": str(e)}), 500

@app.route('/preview_headers', methods=['POST'])
def preview_headers():
    data = request.get_json()
    try:
        file_id = int(data.get("file_id"))
    except (TypeError, ValueError):
        file_id = None
    sheet_name = data.get("sheet_name")
    option = data.get("option")

    if not file_id or not sheet_name or not option:
        return jsonify({"error": "Missing parameters"}), 400

    if file_id not in file_bytes_cache:
        return jsonify({"error": "File not found"}), 404

    file_bytes = file_bytes_cache[file_id]

    if option not in HEADER_PREVIEW_OPTIONS:
        return jsonify({"error": "Invalid option"}), 400
    start_row, header_rows = HEADER_PREVIEW_OPTIONS[option]

    headers = get_preview_headers(file_bytes, sheet_name, start_row, header_rows)
    if headers and isinstance(headers[0], list):
        flat = [" > ".join(str(p) for p in col if p) for col in headers]
    else:
        flat = [str(h) if h is not None else "" for h in headers]

    return jsonify({"headers": flat}), 200

@app.route('/summary/<int:file_id>', methods=['GET'])
def get_summary(file_id: int):
    file_record = get_file(file_id)
    if file_record and file_record["summary"]:
        return jsonify({"file_id": file_id, "summary": file_record["summary"]}), 200
    summary, summary_error = try_generate_summary(file_id)
    return jsonify(
        {"file_id": file_id, "summary": summary, "summary_error": summary_error}
    ), 200


@app.route('/description/<int:file_id>', methods=['GET'])
def get_description(file_id: int):
    refresh = request.args.get("refresh", "false").lower() == "true"
    file_record = get_file(file_id)
    if file_record and file_record["description"] and not refresh:
        return jsonify(
            {"file_id": file_id, "description": file_record["description"]}
        ), 200
    description, description_error = try_generate_description(file_id, refresh=refresh)
    return jsonify(
        {
            "file_id": file_id,
            "description": description,
            "description_error": description_error,
        }
    ), 200

@app.route('/transformations/<int:file_id>', methods=['GET'])
def get_transformations(file_id: int):
    try:
        limit = request.args.get("limit", 200, type=int)
        q = request.args.get("q", "", type=str).strip()
        return jsonify(list_s2t_transformations(file_id, limit=limit, q=q or None)), 200
    except Exception as e:
        logger.exception("Failed to load S2T transformations")
        return jsonify({"error": str(e)}), 500


@app.route('/transformations/<int:file_id>', methods=['DELETE'])
def delete_transformations(file_id: int):
    try:
        deleted = clear_s2t_transformations(file_id)
        return jsonify({"file_id": file_id, "deleted": deleted}), 200
    except Exception as e:
        logger.exception("Failed to clear S2T transformations")
        return jsonify({"error": str(e)}), 500


@app.route('/storage', methods=['DELETE'])
def delete_all_storage():
    try:
        graph_deleted = clear_graph_projection()
        sqlite_deleted = clear_all_data()
        cached_files = len(file_bytes_cache)
        file_bytes_cache.clear()
        with analysis_progress_lock:
            progress_entries = len(analysis_progress)
            analysis_progress.clear()
        return jsonify(
            {
                "sqlite_deleted": sqlite_deleted,
                "neo4j_deleted": graph_deleted,
                "memory_deleted": {
                    "file_bytes": cached_files,
                    "progress_entries": progress_entries,
                },
            }
        ), 200
    except Exception as e:
        logger.exception("Failed to clear all application storage")
        return jsonify({"error": str(e)}), 500


@app.route('/transformations/<int:file_id>/refresh', methods=['POST'])
def refresh_transformations(file_id: int):
    try:
        count, extraction_error, extraction_report = try_refresh_s2t_transformations(file_id)
        graph_sync_report, graph_sync_error = (
            try_sync_file_graph(file_id)
            if extraction_error is None
            else (None, None)
        )
        return jsonify({
            "file_id": file_id,
            "count": count,
            "s2t_transformations_count": count,
            "s2t_transformations_error": extraction_error,
            "s2t_extraction_report": extraction_report,
            "graph_sync_report": graph_sync_report,
            "graph_sync_error": graph_sync_error,
        }), 200
    except Exception as e:
        logger.exception("Failed to rebuild S2T transformations")
        return jsonify({"error": str(e)}), 500


@app.route('/sheet_groups/<int:file_id>/classify', methods=['GET'])
def classify_sheet_groups_route(file_id: int):
    try:
        use_llm = request.args.get("llm", "0").lower() in {"1", "true", "yes", "y"}
        return jsonify(
            classify_file_sheet_groups(
                file_id,
                use_llm=use_llm,
                persist_aliases=False,
            )
        ), 200
    except Exception as e:
        logger.exception("Failed to classify sheet groups")
        return jsonify({"error": str(e)}), 500


@app.route('/exports/sql/<path:filename>', methods=['GET'])
def download_sql_export(filename):
    from agents.tools import SQL_EXPORT_DIR

    return send_from_directory(
        str(SQL_EXPORT_DIR),
        filename,
        as_attachment=True,
    )


@app.route('/chat', methods=['POST'])
def chat():
    """Natural language query endpoint."""
    data = request.get_json(silent=True) or {}
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Missing query"}), 400
    try:
        raw_file_id = data.get("file_id")
        if raw_file_id is None:
            file_id = None
        else:
            try:
                file_id = int(raw_file_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("file_id must be an integer") from exc
            if file_id <= 0:
                raise ValueError("file_id must be a positive integer")
        session_id = _normalize_chat_session_id(data.get("session_id"))
        history = _normalize_chat_history(data.get("history"))
        agent_kwargs: Dict[str, Any] = {}
        if file_id:
            agent_kwargs["file_id"] = file_id
        if history:
            agent_kwargs["history"] = history
        if session_id:
            agent_kwargs["session_id"] = session_id
        answer = agent_chat(query.strip(), **agent_kwargs)
        return jsonify({"answer": answer}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Chat agent failed")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, use_reloader=debug)
