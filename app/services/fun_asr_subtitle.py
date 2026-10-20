"""Fun-ASR subtitle transcription helpers.

The Bailian path intentionally uses the REST API because the official Fun-ASR
recorded-file API supports temporary `oss://` resources only through REST.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from loguru import logger

from app.utils import utils

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com"
UPLOAD_POLICY_URL = f"{DASHSCOPE_BASE_URL}/api/v1/uploads"
TRANSCRIPTION_URL = f"{DASHSCOPE_BASE_URL}/api/v1/services/audio/asr/transcription"
TASK_URL_TEMPLATE = f"{DASHSCOPE_BASE_URL}/api/v1/tasks/{{task_id}}"
MODEL_NAME = "fun-asr"
LOCAL_FUN_ASR_OPENAI_MODEL = "sensevoice"
LOCAL_FUN_ASR_API_URL = "http://127.0.0.1:7860"
LOCAL_FIRERED_ASR_API_URL = "http://127.0.0.1:7867"
TERMINAL_FAILED_STATUSES = {"FAILED", "CANCELED", "UNKNOWN"}
PUNCTUATION_BREAKS = set("，。！？；,.!?;")


class FunAsrError(RuntimeError):
    """Raised for user-actionable Fun-ASR transcription failures."""


@dataclass
class UploadPolicy:
    upload_host: str
    upload_dir: str
    policy: str
    signature: str
    oss_access_key_id: str
    x_oss_object_acl: str = "private"
    x_oss_forbid_overwrite: str = "true"
    max_file_size_mb: Optional[float] = None


def _auth_headers(api_key: str, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def _raise_for_http(response: requests.Response, action: str) -> None:
    try:
        response.raise_for_status()
    except Exception as exc:  # requests may be mocked with generic exceptions
        raise FunAsrError(f"{action}失败，请检查阿里百炼 API Key、网络或服务状态") from exc


def _json(response: requests.Response, action: str) -> dict[str, Any]:
    _raise_for_http(response, action)
    try:
        data = response.json()
    except Exception as exc:
        raise FunAsrError(f"{action}返回了无效 JSON") from exc
    if not isinstance(data, dict):
        raise FunAsrError(f"{action}返回格式无效")
    return data


def _require_api_key(api_key: str) -> str:
    api_key = (api_key or "").strip()
    if not api_key:
        raise FunAsrError("请先输入阿里百炼 API Key")
    return api_key


def _safe_upload_name(local_file: str) -> str:
    name = os.path.basename(local_file).strip() or f"audio_{int(time.time())}.wav"
    return name.replace("/", "_").replace("\\", "_")


def _session_get(session, url: str, **kwargs):
    return session.get(url, **kwargs)


def _session_post(session, url: str, **kwargs):
    return session.post(url, **kwargs)


def _require_local_file(local_file: str) -> None:
    if not os.path.isfile(local_file):
        raise FunAsrError(f"待转写文件不存在: {local_file}")


def _normalize_local_api_url(api_url: str = "") -> str:
    api_url = (api_url or LOCAL_FUN_ASR_API_URL).strip().rstrip("/")
    if not api_url:
        raise FunAsrError("请先填写本地 FunASR-Pack API 地址")
    if "://" not in api_url:
        api_url = f"http://{api_url}"
    return api_url


def _local_base_url(api_url: str = "") -> str:
    api_url = _normalize_local_api_url(api_url)
    parsed = urlparse(api_url)
    path = parsed.path.rstrip("/")
    for suffix in ("/v1/audio/transcriptions", "/v1", "/asr"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    return urlunparse(parsed._replace(path=path, params="", query="", fragment="")).rstrip("/")


def _local_asr_url(api_url: str = "") -> str:
    api_url = _normalize_local_api_url(api_url)
    path = urlparse(api_url).path.rstrip("/")
    if path.endswith("/asr"):
        return api_url
    if path.endswith("/v1") or path.endswith("/v1/audio/transcriptions"):
        return f"{_local_base_url(api_url)}/asr"
    return f"{api_url}/asr"


def _local_openai_transcriptions_url(api_url: str = "") -> str:
    api_url = _normalize_local_api_url(api_url)
    path = urlparse(api_url).path.rstrip("/")
    if path.endswith("/v1/audio/transcriptions"):
        return api_url
    if path.endswith("/v1"):
        return f"{api_url}/audio/transcriptions"
    return f"{_local_base_url(api_url)}/v1/audio/transcriptions"


def _local_fun_asr_prefers_openai(api_url: str = "") -> bool:
    path = urlparse(_normalize_local_api_url(api_url)).path.rstrip("/")
    return path.endswith("/v1") or path.endswith("/v1/audio/transcriptions")


def _is_not_found_response(response: requests.Response) -> bool:
    return getattr(response, "status_code", 200) == 404


def _absolute_local_download_url(api_url: str, download_url: str) -> str:
    download_url = (download_url or "").strip()
    if not download_url:
        return ""
    if urlparse(download_url).scheme:
        return download_url
    return urljoin(f"{_local_base_url(api_url)}/", download_url)


def _raise_for_local_http(
    response: requests.Response,
    action: str,
    service_name: str = "本地 FunASR-Pack 服务",
) -> None:
    status_code = getattr(response, "status_code", 200)
    if status_code and status_code >= 400:
        detail = ""
        try:
            data = response.json()
            if isinstance(data, dict):
                detail = str(data.get("detail") or "")
        except Exception:
            detail = ""
        suffix = f": {detail}" if detail else ""
        raise FunAsrError(f"{action}失败{suffix}，请确认{service_name}可用")

    try:
        response.raise_for_status()
    except Exception as exc:
        raise FunAsrError(f"{action}失败，请确认{service_name}可用") from exc


def _local_json(
    response: requests.Response,
    action: str,
    service_name: str = "本地 FunASR-Pack 服务",
) -> dict[str, Any]:
    _raise_for_local_http(response, action, service_name=service_name)
    try:
        data = response.json()
    except Exception as exc:
        raise FunAsrError(f"{action}返回了无效 JSON") from exc
    if not isinstance(data, dict):
        raise FunAsrError(f"{action}返回格式无效")
    return data


def _response_text(response: requests.Response) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(response, "content", b"")
    if isinstance(content, bytes):
        return content.decode("utf-8")
    return str(content)


def request_upload_policy(api_key: str, model: str = MODEL_NAME, session=requests) -> UploadPolicy:
    """Request Bailian temporary-storage upload policy for the target model."""
    api_key = _require_api_key(api_key)
    response = _session_get(
        session,
        UPLOAD_POLICY_URL,
        params={"action": "getPolicy", "model": model},
        headers=_auth_headers(api_key),
        timeout=30,
    )
    data = _json(response, "获取临时存储上传凭证")
    policy_data = data.get("data") or {}
    required = ["upload_host", "upload_dir", "policy", "signature", "oss_access_key_id"]
    missing = [field for field in required if not policy_data.get(field)]
    if missing:
        raise FunAsrError(f"临时存储上传凭证缺少字段: {', '.join(missing)}")

    return UploadPolicy(
        upload_host=str(policy_data["upload_host"]),
        upload_dir=str(policy_data["upload_dir"]).rstrip("/"),
        policy=str(policy_data["policy"]),
        signature=str(policy_data["signature"]),
        oss_access_key_id=str(policy_data["oss_access_key_id"]),
        x_oss_object_acl=str(policy_data.get("x_oss_object_acl") or "private"),
        x_oss_forbid_overwrite=str(policy_data.get("x_oss_forbid_overwrite") or "true"),
        max_file_size_mb=policy_data.get("max_file_size_mb"),
    )


def _validate_file_size(local_file: str, policy: UploadPolicy) -> None:
    if policy.max_file_size_mb is None:
        return
    max_bytes = float(policy.max_file_size_mb) * 1024 * 1024
    size = os.path.getsize(local_file)
    if size > max_bytes:
        raise FunAsrError(
            f"文件大小超过阿里百炼临时存储限制: {size / 1024 / 1024:.2f}MB > {float(policy.max_file_size_mb):.2f}MB"
        )


def upload_to_temporary_oss(local_file: str, policy: UploadPolicy, session=requests) -> str:
    """Upload local file to temporary OSS and return `oss://...` URL."""
    if not os.path.isfile(local_file):
        raise FunAsrError(f"待转写文件不存在: {local_file}")
    _validate_file_size(local_file, policy)

    key = f"{policy.upload_dir}/{_safe_upload_name(local_file)}"
    data = {
        "OSSAccessKeyId": policy.oss_access_key_id,
        "policy": policy.policy,
        "Signature": policy.signature,
        "key": key,
        "x-oss-object-acl": policy.x_oss_object_acl,
        "x-oss-forbid-overwrite": policy.x_oss_forbid_overwrite,
        "success_action_status": "200",
    }
    with open(local_file, "rb") as file_obj:
        files = {"file": (_safe_upload_name(local_file), file_obj)}
        response = _session_post(session, policy.upload_host, data=data, files=files, timeout=120)
    _raise_for_http(response, "上传文件到阿里百炼临时存储")
    return f"oss://{key}"


def submit_transcription_task(
    api_key: str,
    oss_url: str,
    speaker_count: Optional[int] = None,
    model: str = MODEL_NAME,
    session=requests,
) -> str:
    """Submit async Fun-ASR task and return task_id."""
    api_key = _require_api_key(api_key)
    parameters: dict[str, Any] = {"diarization_enabled": True}
    if speaker_count:
        parameters["speaker_count"] = int(speaker_count)

    payload = {
        "model": model,
        "input": {"file_urls": [oss_url]},
        "parameters": parameters,
    }
    response = _session_post(
        session,
        TRANSCRIPTION_URL,
        headers=_auth_headers(
            api_key,
            {
                "X-DashScope-Async": "enable",
                "X-DashScope-OssResourceResolve": "enable",
            },
        ),
        json=payload,
        timeout=30,
    )
    data = _json(response, "提交 Fun-ASR 转写任务")
    task_id = ((data.get("output") or {}).get("task_id") or "").strip()
    if not task_id:
        raise FunAsrError("提交 Fun-ASR 转写任务失败：未返回 task_id")
    return task_id


def poll_transcription_task(
    api_key: str,
    task_id: str,
    poll_interval: float = 2.0,
    timeout: float = 600.0,
    session=requests,
) -> dict[str, Any]:
    """Poll task until terminal status and return successful result item."""
    api_key = _require_api_key(api_key)
    deadline = time.time() + timeout
    last_status = "PENDING"
    while time.time() < deadline:
        response = _session_post(
            session,
            TASK_URL_TEMPLATE.format(task_id=task_id),
            headers=_auth_headers(api_key),
            timeout=30,
        )
        data = _json(response, "查询 Fun-ASR 转写任务")
        output = data.get("output") or {}
        last_status = str(output.get("task_status") or "").upper()

        if last_status == "SUCCEEDED":
            results = output.get("results") or []
            for result in results:
                subtask_status = str(result.get("subtask_status") or "").upper()
                if subtask_status and subtask_status != "SUCCEEDED":
                    raise FunAsrError(f"Fun-ASR 子任务失败: {subtask_status}")
            if not results:
                raise FunAsrError("Fun-ASR 转写成功但未返回结果")
            return results[0]

        if last_status in TERMINAL_FAILED_STATUSES:
            raise FunAsrError(f"Fun-ASR 转写任务失败: {last_status}")

        time.sleep(poll_interval)

    raise FunAsrError(f"Fun-ASR 转写任务超时，最后状态: {last_status}")


def download_transcription_result(transcription_url: str, session=requests) -> dict[str, Any]:
    if not transcription_url:
        raise FunAsrError("Fun-ASR 结果缺少 transcription_url")
    response = _session_get(session, transcription_url, timeout=60)
    return _json(response, "下载 Fun-ASR 转写结果")


def _ms_to_srt_time(ms: float) -> str:
    total_ms = max(0, int(round(float(ms))))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    seconds = total_ms // 1_000
    milliseconds = total_ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _srt_block(index: int, start_ms: float, end_ms: float, text: str) -> str:
    if end_ms <= start_ms:
        end_ms = start_ms + 500
    return f"{index}\n{_ms_to_srt_time(start_ms)} --> {_ms_to_srt_time(end_ms)}\n{text.strip()}\n"


def _timestamp_ms(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise FunAsrError(f"Fun-ASR 转写结果时间戳无效: {field_name}={value!r}") from exc


def _speaker_prefix(speaker_id: Any) -> str:
    if speaker_id is None or speaker_id == "":
        return ""
    try:
        label = int(speaker_id) + 1
    except (TypeError, ValueError):
        label = str(speaker_id)
    return f"说话人{label}: "


def _iter_sentences(result_json: dict[str, Any]):
    transcripts = result_json.get("transcripts")
    if transcripts is None and "sentences" in result_json:
        transcripts = [{"sentences": result_json.get("sentences") or []}]
    if not transcripts:
        raise FunAsrError("Fun-ASR 转写结果为空：未找到 transcripts")
    for transcript in transcripts:
        for sentence in transcript.get("sentences") or []:
            yield sentence


def _word_text(word: dict[str, Any]) -> str:
    text = str(word.get("text") or word.get("word") or "")
    punctuation = str(word.get("punctuation") or "")
    if punctuation and not text.endswith(punctuation):
        text += punctuation
    return text


def _flush_block(blocks: list[dict[str, Any]], current: dict[str, Any]) -> None:
    text = current.get("text", "").strip()
    if text:
        blocks.append(current.copy())


def _blocks_from_words(sentence: dict[str, Any], max_chars: int, max_duration: float) -> list[dict[str, Any]]:
    words = sentence.get("words") or []
    blocks: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    max_duration_ms = max_duration * 1000
    sentence_speaker = sentence.get("speaker_id")

    for word in words:
        text = _word_text(word)
        if not text:
            continue
        start = word.get("begin_time", word.get("start_time"))
        end = word.get("end_time")
        if start is None or end is None:
            continue
        speaker_id = word.get("speaker_id", sentence_speaker)
        start_ms = _timestamp_ms(start, "word.begin_time")
        end_ms = _timestamp_ms(end, "word.end_time")

        if current is None:
            current = {"start": start_ms, "end": end_ms, "text": text, "speaker_id": speaker_id}
        else:
            should_split_before = (
                speaker_id != current.get("speaker_id")
                or len(current["text"] + text) > max_chars
                or (end_ms - current["start"]) > max_duration_ms
            )
            if should_split_before:
                _flush_block(blocks, current)
                current = {"start": start_ms, "end": end_ms, "text": text, "speaker_id": speaker_id}
            else:
                current["text"] += text
                current["end"] = end_ms

        if current and text[-1:] in PUNCTUATION_BREAKS:
            _flush_block(blocks, current)
            current = None

    if current:
        _flush_block(blocks, current)
    return blocks


def _split_text(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in PUNCTUATION_BREAKS or len(current) >= max_chars:
            chunks.append(current.strip())
            current = ""
    if current.strip():
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def _blocks_from_sentence(sentence: dict[str, Any], max_chars: int) -> list[dict[str, Any]]:
    text = str(sentence.get("text") or "").strip()
    if not text:
        return []
    start = sentence.get("begin_time", 0)
    end = sentence.get("end_time")
    start_ms = _timestamp_ms(start, "sentence.begin_time")
    end_ms = _timestamp_ms(end, "sentence.end_time") if end is not None else start_ms + 500
    chunks = _split_text(text, max_chars)
    if not chunks:
        return []
    duration = max(500.0, end_ms - start_ms)
    total_chars = max(1, sum(len(chunk) for chunk in chunks))
    cursor = start_ms
    blocks: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        if i == len(chunks) - 1:
            chunk_end = end_ms
        else:
            chunk_end = cursor + duration * (len(chunk) / total_chars)
        blocks.append(
            {
                "start": cursor,
                "end": max(cursor + 200, chunk_end),
                "text": chunk,
                "speaker_id": sentence.get("speaker_id"),
            }
        )
        cursor = chunk_end
    return blocks


def fun_asr_result_to_srt(result_json: dict[str, Any], max_chars: int = 20, max_duration: float = 3.5) -> str:
    """Convert downloaded Fun-ASR JSON into fine-grained SRT.

    Official downloaded schema is `transcripts[*].sentences[*].words[*]`.
    Fun-ASR timestamps are milliseconds.
    """
    blocks: list[dict[str, Any]] = []
    for sentence in _iter_sentences(result_json):
        sentence_blocks = _blocks_from_words(sentence, max_chars, max_duration)
        if not sentence_blocks:
            sentence_blocks = _blocks_from_sentence(sentence, max_chars)
        blocks.extend(sentence_blocks)

    if not blocks:
        raise FunAsrError("Fun-ASR 转写结果为空：未找到可用字幕内容")

    lines = []
    for index, block in enumerate(blocks, start=1):
        text = f"{_speaker_prefix(block.get('speaker_id'))}{block['text']}"
        lines.append(_srt_block(index, block["start"], block["end"], text))
    return "\n".join(lines).rstrip() + "\n"


def write_srt_file(srt_content: str, subtitle_file: str = "") -> str:
    if not subtitle_file:
        subtitle_file = os.path.join(utils.subtitle_dir(), f"fun_asr_{int(time.time())}.srt")
    parent = os.path.dirname(subtitle_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return subtitle_file


def copy_srt_file(source_file: str, subtitle_file: str = "") -> str:
    """Copy an existing SRT file into NarratoAI's subtitle directory."""
    if not os.path.isfile(source_file):
        raise FunAsrError(f"本地 FunASR-Pack 返回的字幕文件不存在: {source_file}")
    if not subtitle_file:
        subtitle_file = os.path.join(utils.subtitle_dir(), f"fun_asr_local_{int(time.time())}.srt")
    parent = os.path.dirname(subtitle_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.abspath(source_file) != os.path.abspath(subtitle_file):
        shutil.copyfile(source_file, subtitle_file)
    return subtitle_file


def request_local_fun_asr_health(api_url: str = LOCAL_FUN_ASR_API_URL, session=requests) -> dict[str, Any]:
    """Fetch FunASR-Pack health metadata from the local service."""
    response = _session_get(session, f"{_local_base_url(api_url)}/health", timeout=10)
    return _local_json(response, "检查本地 FunASR-Pack 服务")


def request_local_firered_asr_health(
    api_url: str = LOCAL_FIRERED_ASR_API_URL,
    session=requests,
) -> dict[str, Any]:
    """Fetch FireRedASR2-AED-Pack health metadata from the local service."""
    response = _session_get(session, f"{_local_base_url(api_url)}/health", timeout=10)
    return _local_json(
        response,
        "检查本地 FireRedASR2-AED-Pack 服务",
        service_name="本地 FireRedASR2-AED-Pack 服务",
    )


def request_local_fun_asr(
    local_file: str,
    api_url: str = LOCAL_FUN_ASR_API_URL,
    hotword: str = "",
    enable_spk: Optional[bool] = None,
    model: str = LOCAL_FUN_ASR_OPENAI_MODEL,
    timeout: float = 600.0,
    session=requests,
) -> dict[str, Any]:
    """Call the local FunASR-Pack API and return its JSON result."""
    _require_local_file(local_file)
    rest_data: dict[str, str] = {}
    if hotword.strip():
        rest_data["hotword"] = hotword.strip()
    if enable_spk is not None:
        rest_data["enable_spk"] = "true" if enable_spk else "false"

    openai_data: dict[str, str] = {
        "model": (model or LOCAL_FUN_ASR_OPENAI_MODEL).strip() or LOCAL_FUN_ASR_OPENAI_MODEL,
        "response_format": "verbose_json",
    }
    if enable_spk is not None:
        openai_data["spk"] = "true" if enable_spk else "false"

    rest_url = _local_asr_url(api_url)
    openai_url = _local_openai_transcriptions_url(api_url)
    attempts = [
        (openai_url, openai_data),
        (rest_url, rest_data),
    ] if _local_fun_asr_prefers_openai(api_url) else [
        (rest_url, rest_data),
        (openai_url, openai_data),
    ]

    last_response = None
    for index, (url, data) in enumerate(attempts):
        with open(local_file, "rb") as file_obj:
            files = {"file": (_safe_upload_name(local_file), file_obj)}
            response = _session_post(
                session,
                url,
                data=data,
                files=files,
                timeout=timeout,
            )
        if index == 0 and _is_not_found_response(response):
            last_response = response
            continue
        return _local_json(response, "调用本地 FunASR-Pack ASR API")

    return _local_json(last_response, "调用本地 FunASR-Pack ASR API")


def request_local_firered_asr(
    local_file: str,
    api_url: str = LOCAL_FIRERED_ASR_API_URL,
    enable_vad: Optional[bool] = True,
    enable_lid: Optional[bool] = True,
    enable_punc: Optional[bool] = True,
    return_timestamp: Optional[bool] = True,
    timeout: float = 600.0,
    session=requests,
) -> dict[str, Any]:
    """Call the local FireRedASR2-AED-Pack `/asr` API and return its JSON result."""
    _require_local_file(local_file)
    data: dict[str, str] = {}
    options = {
        "enable_vad": enable_vad,
        "enable_lid": enable_lid,
        "enable_punc": enable_punc,
        "return_timestamp": return_timestamp,
    }
    for key, value in options.items():
        if value is not None:
            data[key] = "true" if value else "false"

    with open(local_file, "rb") as file_obj:
        files = {"file": (_safe_upload_name(local_file), file_obj)}
        response = _session_post(
            session,
            _local_asr_url(api_url),
            data=data,
            files=files,
            timeout=timeout,
        )
    return _local_json(
        response,
        "调用本地 FireRedASR2-AED-Pack ASR API",
        service_name="本地 FireRedASR2-AED-Pack 服务",
    )


def download_local_srt(
    download_url: str,
    api_url: str = LOCAL_FUN_ASR_API_URL,
    subtitle_file: str = "",
    session=requests,
    service_name: str = "本地 FunASR-Pack 服务",
) -> str:
    """Download an SRT exposed by FunASR-Pack and save it as a NarratoAI subtitle."""
    absolute_url = _absolute_local_download_url(api_url, download_url)
    if not absolute_url:
        raise FunAsrError("本地 FunASR-Pack 结果缺少 SRT 下载地址")
    response = _session_get(session, absolute_url, timeout=60)
    _raise_for_local_http(response, "下载本地 SRT", service_name=service_name)
    srt_content = _response_text(response)
    if not srt_content.strip():
        raise FunAsrError(f"{service_name}返回了空 SRT")
    return write_srt_file(srt_content, subtitle_file)


def _local_result_items(result_json: dict[str, Any]):
    raw = result_json.get("raw")
    if isinstance(raw, dict):
        yield raw
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item
    elif result_json.get("text"):
        yield result_json


def _openai_segment_ms(value: Any, field_name: str) -> float:
    return _timestamp_ms(value, field_name) * 1000


def _blocks_from_openai_segments(result_json: dict[str, Any], max_chars: int) -> list[dict[str, Any]]:
    segments = result_json.get("segments") or []
    if not isinstance(segments, list):
        return []

    blocks: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = segment.get("start", segment.get("start_time", 0))
        end = segment.get("end", segment.get("end_time"))
        start_ms = _openai_segment_ms(start, "openai.segment.start")
        end_ms = _openai_segment_ms(end, "openai.segment.end") if end is not None else start_ms + 500
        blocks.extend(
            _blocks_from_sentence(
                {
                    "begin_time": start_ms,
                    "end_time": end_ms,
                    "text": text,
                    "speaker_id": segment.get("speaker"),
                },
                max_chars=max_chars,
            )
        )
    return blocks


def _blocks_from_local_timestamp(item: dict[str, Any], max_chars: int, max_duration: float) -> list[dict[str, Any]]:
    text = str(item.get("text") or "").strip()
    timestamps = item.get("timestamp") or []
    if not text or not isinstance(timestamps, list):
        return []

    non_space_chars = [char for char in text if char.strip()]
    consume_punctuation = len(timestamps) >= len(non_space_chars)
    blocks: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    timestamp_index = 0
    last_end = 0.0
    max_duration_ms = max_duration * 1000

    for char in text:
        if not char.strip():
            continue

        is_punctuation = char in PUNCTUATION_BREAKS
        consume_timestamp = consume_punctuation or not is_punctuation
        if consume_timestamp and timestamp_index < len(timestamps):
            pair = timestamps[timestamp_index]
            timestamp_index += 1
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            start_ms = _timestamp_ms(pair[0], "local.timestamp.start")
            end_ms = _timestamp_ms(pair[1], "local.timestamp.end")
            last_end = end_ms
        else:
            start_ms = last_end
            end_ms = last_end if is_punctuation else last_end + 200
            last_end = end_ms

        if current is None:
            current = {"start": start_ms, "end": end_ms, "text": char}
        else:
            should_split_before = (
                len(current["text"] + char) > max_chars
                or (end_ms - current["start"]) > max_duration_ms
            )
            if should_split_before:
                _flush_block(blocks, current)
                current = {"start": start_ms, "end": end_ms, "text": char}
            else:
                current["text"] += char
                current["end"] = end_ms

        if current and is_punctuation:
            _flush_block(blocks, current)
            current = None

    if current:
        _flush_block(blocks, current)
    return blocks


def local_fun_asr_result_to_srt(
    result_json: dict[str, Any],
    max_chars: int = 20,
    max_duration: float = 3.5,
) -> str:
    """Convert a FunASR-Pack JSON response into SRT when the API SRT is unavailable."""
    blocks = _blocks_from_openai_segments(result_json, max_chars=max_chars)
    for item in _local_result_items(result_json):
        item_blocks = _blocks_from_local_timestamp(item, max_chars, max_duration)
        if not item_blocks:
            text = str(item.get("text") or "").strip()
            if text:
                item_blocks = _blocks_from_sentence(
                    {
                        "begin_time": 0,
                        "end_time": max(1500, len(text) * 180),
                        "text": text,
                    },
                    max_chars=max_chars,
                )
        blocks.extend(item_blocks)

    if not blocks:
        raise FunAsrError("本地 FunASR-Pack 转写结果为空：未找到可用字幕内容")

    lines = []
    for index, block in enumerate(blocks, start=1):
        lines.append(_srt_block(index, block["start"], block["end"], block["text"]))
    return "\n".join(lines).rstrip() + "\n"


def firered_asr_result_to_srt(result_json: dict[str, Any]) -> str:
    """Convert a FireRedASR2-AED-Pack JSON response into SRT when no SRT URL is returned."""
    blocks: list[dict[str, Any]] = []
    sentences = result_json.get("sentences")
    if isinstance(sentences, list):
        for sentence in sentences:
            if not isinstance(sentence, dict):
                continue
            text = str(sentence.get("text") or "").strip()
            if not text:
                continue
            start = sentence.get("start_ms", sentence.get("begin_time", sentence.get("start_time", 0)))
            end = sentence.get("end_ms", sentence.get("end_time"))
            start_ms = _timestamp_ms(start, "firered.sentence.start_ms")
            end_ms = _timestamp_ms(end, "firered.sentence.end_ms") if end is not None else start_ms + 500
            blocks.append({"start": start_ms, "end": end_ms, "text": text})

    if not blocks:
        return local_fun_asr_result_to_srt(result_json)

    lines = []
    for index, block in enumerate(blocks, start=1):
        lines.append(_srt_block(index, block["start"], block["end"], block["text"]))
    return "\n".join(lines).rstrip() + "\n"


def _get_local_srt_download_url(result_json: dict[str, Any]) -> str:
    downloads = result_json.get("downloads") or {}
    if isinstance(downloads, dict):
        download_url = downloads.get("srt")
        if download_url:
            return str(download_url)
    for key in ("srt_url", "srt_download_url", "download_url"):
        download_url = result_json.get(key)
        if download_url:
            return str(download_url)
    return ""


def create_with_local_fun_asr(
    local_file: str,
    subtitle_file: str = "",
    api_url: str = LOCAL_FUN_ASR_API_URL,
    hotword: str = "",
    enable_spk: Optional[bool] = None,
    timeout: float = 600.0,
    session=requests,
) -> Optional[str]:
    """Create an SRT file through a locally running FunASR-Pack API."""
    try:
        result_json = request_local_fun_asr(
            local_file=local_file,
            api_url=api_url,
            hotword=hotword,
            enable_spk=enable_spk,
            timeout=timeout,
            session=session,
        )

        srt_file = result_json.get("srt_file")
        if isinstance(srt_file, str) and srt_file and os.path.isfile(srt_file):
            output_file = copy_srt_file(srt_file, subtitle_file)
        else:
            download_url = _get_local_srt_download_url(result_json)
            if download_url:
                output_file = download_local_srt(
                    download_url,
                    api_url=api_url,
                    subtitle_file=subtitle_file,
                    session=session,
                )
            else:
                srt_content = local_fun_asr_result_to_srt(result_json)
                output_file = write_srt_file(srt_content, subtitle_file)

        logger.info(f"本地 FunASR-Pack 字幕文件已生成: {output_file}")
        return output_file
    except FunAsrError:
        raise
    except Exception as exc:
        raise FunAsrError("本地 FunASR-Pack 字幕转写失败，请检查服务地址、文件或模型状态") from exc

def create_with_local_firered_asr(
    local_file: str,
    subtitle_file: str = "",
    api_url: str = LOCAL_FIRERED_ASR_API_URL,
    enable_vad: Optional[bool] = True,
    enable_lid: Optional[bool] = True,
    enable_punc: Optional[bool] = True,
    return_timestamp: Optional[bool] = True,
    timeout: float = 600.0,
    session=requests,
) -> Optional[str]:
    """
    【已被劫持】原本是调用本地 FireRedASR，现在直接转为调用本地 Whisper
    """
    logger.info("🚀 拦截到 FireRedASR 请求，已切换为本地 Whisper-large-v3 直接处理")
    return _run_local_whisper_direct(local_file, subtitle_file)


def create_with_fun_asr(
    local_file: str,
    subtitle_file: str = "",
    api_key: str = "",
    speaker_count: Optional[int] = None,
    poll_interval: float = 2.0,
    timeout: float = 600.0,
    session=requests,
) -> Optional[str]:
    """Upload local media to Bailian temporary storage and create a Fun-ASR SRT file."""
    api_key = _require_api_key(api_key)
    try:
        policy = request_upload_policy(api_key, session=session)
        oss_url = upload_to_temporary_oss(local_file, policy, session=session)
        task_id = submit_transcription_task(api_key, oss_url, speaker_count=speaker_count, session=session)
        task_result = poll_transcription_task(
            api_key,
            task_id,
            poll_interval=poll_interval,
            timeout=timeout,
            session=session,
        )
        transcription_url = task_result.get("transcription_url")
        result_json = download_transcription_result(transcription_url, session=session)
        srt_content = fun_asr_result_to_srt(result_json)
        output_file = write_srt_file(srt_content, subtitle_file)
        logger.info(f"Fun-ASR 字幕文件已生成: {output_file}")
        return output_file
    except FunAsrError:
        raise
    except Exception as exc:
        raise FunAsrError("Fun-ASR 字幕转写失败，请检查文件、网络或阿里百炼服务状态") from exc


def _run_local_whisper_direct(
    local_file: str,
    subtitle_file: str = "",
    model_path: str = "/home/ken/data/youdub/submodule/models/whisper-large-v3",
) -> Optional[str]:
    """
    【满分神作】基于 wtpsplit 高斯长度约束的语义切分 + 毫秒级时间戳回填
    """
    from faster_whisper import WhisperModel
    import gc
    import os
    import time
    import json
    import hashlib
    import re
    from loguru import logger
    import opencc

    sat_model = None
    def _get_sat_model():
        nonlocal sat_model
        if sat_model is None:
            from wtpsplit import SaT
            # 使用纯 ONNX 版本
            sat_model = SaT("sat-3l-sm", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        return sat_model

    def _smart_join(words_list):
        text = ""
        for w in words_list:
            word_str = w["word"] if isinstance(w, dict) else w
            if not text:
                text = word_str
                continue
            
            prev_word_clean = text.strip().split(" ")[-1].strip(",.?!").lower()
            curr_word_clean = word_str.strip().strip(",.?!").lower()
            if prev_word_clean == curr_word_clean and prev_word_clean not in {"no", "yeah", "yes", "oh", "well"}:
                continue
                
            prev_char = text[-1]
            curr_char = word_str[0]
            
            if curr_char in {'.', ',', '?', '!', '。', '，', '？', '！', ';', '；', ':', '：', "'", '"'}:
                text += word_str
            elif re.match(r'[\u4e00-\u9fa5]', prev_char) or re.match(r'[\u4e00-\u9fa5]', curr_char):
                if text.endswith(" "): text = text[:-1]
                text += word_str
            else:
                if word_str.startswith(" ") or text.endswith(" "):
                    text += word_str
                else:
                    text += " " + word_str
        return re.sub(r'\s+', ' ', text).strip()

    # 计算文本中中文字符比例
    def _get_chinese_ratio(text):
        if not text: return 0.0
        total = len(text.replace(" ", ""))
        if total == 0: return 0.0
        ch_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        return ch_chars / total

    if not subtitle_file:
        subtitle_file = os.path.join(utils.subtitle_dir(), f"whisper_{int(time.time())}.srt")
    
    output_file = subtitle_file
    audio_basename = os.path.basename(local_file)
    try:
        audio_size = os.path.getsize(local_file)
    except OSError:
        audio_size = 0
    
    file_fingerprint = hashlib.md5(f"{audio_basename}_{audio_size}".encode('utf-8')).hexdigest()[:8]
    cache_dir = os.path.join(utils.subtitle_dir(), "whisper_cache")
    os.makedirs(cache_dir, exist_ok=True)
    json_file_path = os.path.join(cache_dir, f"{file_fingerprint}_asr_word_level.json")

    all_words = []
    whisper_model = None

    try:
        # 1. 读取缓存或执行转录
        if os.path.exists(json_file_path):
            logger.info(f"♻️ 读取缓存: {json_file_path}")
            with open(json_file_path, "r", encoding="utf-8") as f:
                all_words = json.load(f)
        else:
            logger.info(f"⏳ 加载 Whisper 模型: {model_path}")
            whisper_model = WhisperModel(model_path, device="cuda", compute_type="float16")
            segments, info = whisper_model.transcribe(
                local_file, beam_size=5, word_timestamps=True, language=None,
                initial_prompt="This is an interview. Please transcribe accurately.",
                vad_filter=True, condition_on_previous_text=False
            )
            for segment in segments:
                if not segment.words: continue
                for word in segment.words:
                    if not word.word.strip(): continue
                    all_words.append({
                        "word": word.word, "start": word.start, "end": word.end,
                        "probability": getattr(word, "probability", 1.0)
                    })
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(all_words, f, ensure_ascii=False, indent=2)

        all_words.sort(key=lambda x: x["start"])
        
        # ==========================================
        # 2. 物理预切分：把全篇切成“段落级”大块（防模型输入过长）
        # ==========================================
        logger.info("🔪 第一层：物理预切分为大段落...")
        paragraphs = []
        current_para = []
        for w in all_words:
            current_para.append(w)
            # 遇到句号，或者超过 15 秒，强制切成一个大段落
            if w["word"].strip()[-1:] in {'.', '?', '!', '。', '？', '！'} or (current_para[-1]["end"] - current_para[0]["start"]) > 15.0:
                paragraphs.append(current_para)
                current_para = []
        if current_para:
            paragraphs.append(current_para)

        # ==========================================
        # 3. 第二层：利用 wtpsplit 的高斯长度约束进行语义细分
        # ==========================================
        logger.info("🧠 第二层：wtpsplit 语义细分配置...")
        semantic_blocks = []
        model = _get_sat_model()

        for para in paragraphs:
            para_text = _smart_join(para)
            if not para_text.strip(): continue

            cn_ratio = _get_chinese_ratio(para_text)
            # 动态配置高斯先验：中文目标 15 字，英文目标 25 字符
            if cn_ratio > 0.3:
                target_len, max_len = 15, 22
            else:
                target_len, max_len = 25, 42

            # 调用 wtpsplit 长度约束（源码明确说明：传了约束后，内部会做 Viterbi 寻优）
            try:
                sentences = model.split(
                    para_text, 
                    max_length=max_len, 
                    prior_type="gaussian", 
                    prior_kwargs={"target_length": target_len}
                )
            except Exception as e:
                logger.warning(f"wtpsplit 失败，降级为物理切分: {e}")
                sentences = [para_text] # 降级处理

            # ==========================================
            # 4. 毫秒级时间戳回填（按字符顺序严格匹配）
            # ==========================================
            word_idx = 0
            for sent in sentences:
                sent = sent.strip()
                if not sent: continue
                
                # 源码注释：使用约束时，句子内部可能包含换行，拼接使用 "".join()
                # 这里为了匹配，我们直接把 sent 里的所有空白去掉，用纯字符匹配
                sent_clean = re.sub(r'\s+', '', sent)
                if not sent_clean: continue
                
                sub_words = []
                char_consumed = 0
                
                # 遍历当前大段落的词，直到字符数匹配上
                while word_idx < len(para) and char_consumed < len(sent_clean):
                    w = para[word_idx]
                    sub_words.append(w)
                    # 累加去掉空白的字符数
                    char_consumed += len(re.sub(r'\s+', '', w["word"]))
                    word_idx += 1
                
                if sub_words:
                    semantic_blocks.append({
                        "start": sub_words[0]["start"] * 1000,
                        "end": sub_words[-1]["end"] * 1000,
                        "text": _smart_join(sub_words), 
                        "words": sub_words
                    })

        # ==========================================
        # 5. 安全兜底与极致抛光
        # ==========================================
        logger.info("🪄 第三层：悬空词迁移与碎片合并...")
        HANGING_WORDS = {"the", "a", "an", "of", "to", "in", "for", "on", "at", "by", "with", "and", "or", "but", "that"}
        
        # 5.1 悬空词迁移（同样需要严格防重叠）
        for i in range(len(semantic_blocks) - 1):
            words_in_block = semantic_blocks[i]["words"]
            if not words_in_block: continue
            
            # 如果本身就带着强标点，不要迁移
            if words_in_block[-1]["word"].strip()[-1:] in {'.', '?', '!', '。', '？', '！'}:
                continue
                
            while len(words_in_block) > 1:
                last_word = words_in_block[-1]["word"].strip().lower()
                last_word_clean = re.sub(r'[^\w\s]', '', last_word).strip()
                if last_word_clean in HANGING_WORDS:
                    hanging_word = words_in_block.pop()
                    
                    semantic_blocks[i]["text"] = _smart_join(words_in_block)
                    semantic_blocks[i]["end"] = words_in_block[-1]["end"] * 1000
                    
                    semantic_blocks[i+1]["words"].insert(0, hanging_word)
                    semantic_blocks[i+1]["start"] = max(
                        semantic_blocks[i+1]["start"],
                        semantic_blocks[i]["end"] + 10,
                        hanging_word["start"] * 1000
                    )
                    semantic_blocks[i+1]["text"] = _smart_join(semantic_blocks[i+1]["words"])
                else:
                    break

        # 5.2 极短碎片安全合并（严禁超 3.5 秒）
        final_blocks = []
        for block in semantic_blocks:
            duration = block["end"] - block["start"]
            if duration < 800 and final_blocks:
                merged_duration = block["end"] - final_blocks[-1]["start"]
                if merged_duration <= 3500: 
                    prev_text = final_blocks[-1]["text"]
                    curr_text = block["text"]
                    
                    if curr_text[0] in {'.', ',', '?', '!', '。', '，', '？', '！'}:
                        final_blocks[-1]["text"] = prev_text + curr_text
                    elif re.match(r'[\u4e00-\u9fa5]', prev_text[-1]) and re.match(r'[\u4e00-\u9fa5]', curr_text[0]):
                        final_blocks[-1]["text"] = prev_text + curr_text
                    else:
                        final_blocks[-1]["text"] = prev_text + " " + curr_text
                        
                    final_blocks[-1]["end"] = block["end"]
                    continue
            final_blocks.append(block)

        # ==========================================
        # 6. 生成 SRT
        # ==========================================
        lines = []
        for index, block in enumerate(final_blocks, start=1):
            lines.append(_srt_block(index, block["start"], block["end"], block["text"]))
        srt_content = "\n".join(lines).rstrip() + "\n"

        try:
            cc = opencc.OpenCC('t2s')
            srt_content = cc.convert(srt_content)
        except Exception:
            pass

        output_file = write_srt_file(srt_content, output_file)
        logger.info(f"✅ 生成成功，共 {len(final_blocks)} 条: {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"❌ 失败: {e}")
        raise FunAsrError(f"失败: {e}")

    finally:
        if whisper_model is not None: del whisper_model
        if sat_model is not None: del sat_model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError: pass



