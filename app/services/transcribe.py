"""Fun-ASR subtitle transcription helpers.

The Bailian path intentionally uses the REST API because the official Fun-ASR
recorded-file API supports temporary `oss://` resources only through REST.

v3.9 变更：
  · feat 顶部 soundfile patch：torchaudio.load 绕过 torchcodec（torchaudio 2.8+ 强制 API）
  · feat _pyannote_available()：未装 pyannote 时快速降级，不浪费加载时间
  · feat pyannote 3.4 + 4.0 双兼容：getattr(output, "speaker_diarization", output)
  · feat YOUDUB_ASR_ENABLE_DIARIZATION 环境变量：可完全跳过 speaker
  · 保留 v3.8 全部：faster-whisper + pyannote，无 whisperx

v3.8 变更：
  · 彻底去除 whisperx 依赖，改用 faster-whisper + pyannote.audio
  · ASR 用 faster-whisper（自带 word_timestamps），Diarize 用 pyannote.Pipeline
  · 自写 _assign_speakers_with_pyannote（word 中点落段）

v3.6 变更：
  · _semantic_split_paragraph 补 start/end；_enforce_max_duration 队列循环；
    _migrate_hanging_words 返回 total_moved

依赖：
  · requests / loguru / app.utils.utils
  · faster-whisper（Whisper 路径）
  · pyannote.audio >=3.1,<4.0（speaker，可选）
  · soundfile（torchaudio patch，随 pyannote 装）
  · wtpsplit（语义切分）
  · opencc（繁转简，可选）
  · tomllib / tomli（config.toml，可选）
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from loguru import logger

from app.utils import utils


# ============================================================================
# torchaudio.load 兼容补丁（必须最早执行，pyannote / speechbrain import 之前）
# ============================================================================
def _install_soundfile_backend_for_torchaudio() -> None:
    """把 torchaudio.load 替换为 soundfile 实现。

    torchaudio 2.8+ 的 load() 强制走 torchcodec，torchcodec 又依赖系统
    FFmpeg 共享库（libavutil.so.*），容器/裸机常缺。soundfile 用 libsndfile
    （wheel 自带），零系统依赖，对 wav/flac/mp3/ogg 全支持。

    保留原函数引用在 torchaudio._torchaudio_load_original，便于回滚。
    幂等：重复调用不会叠加 patch。
    """
    try:
        import torchaudio
        import soundfile as sf
        import torch
        import numpy as np
    except ImportError:
        return

    if getattr(torchaudio, "_soundfile_patched", False):
        return

    _orig_load = torchaudio.load

    def _soundfile_load(
        filepath,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: str = None,
        **kwargs,
    ):
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(np.ascontiguousarray(data))  # (frames, channels)
        if frame_offset or num_frames > 0:
            end = None if num_frames < 0 else frame_offset + num_frames
            waveform = waveform[frame_offset:end]
        if channels_first:
            waveform = waveform.transpose(0, 1).contiguous()  # (channels, frames)
        if not normalize:
            waveform = (waveform * 32768.0).clamp(-32768, 32767).to(torch.int16)
        return waveform, int(sr)

    torchaudio._torchaudio_load_original = _orig_load
    torchaudio.load = _soundfile_load
    torchaudio._soundfile_patched = True


_install_soundfile_backend_for_torchaudio()


# ============================================================================
# 常量
# ============================================================================
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

_BYTES_PER_MB = 1024 ** 2


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


# ============================================================================
# 通用 HTTP 工具
# ============================================================================
def _auth_headers(api_key: str, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra:
        headers.update(extra)
    return headers


def _raise_for_http(response: requests.Response, action: str) -> None:
    try:
        response.raise_for_status()
    except Exception as exc:
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
    response: requests.Response, action: str,
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
    response: requests.Response, action: str,
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


# ============================================================================
# Bailian REST 路径
# ============================================================================
def request_upload_policy(api_key: str, model: str = MODEL_NAME, session=requests) -> UploadPolicy:
    api_key = _require_api_key(api_key)
    response = _session_get(
        session, UPLOAD_POLICY_URL,
        params={"action": "getPolicy", "model": model},
        headers=_auth_headers(api_key), timeout=30,
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
        raise FunAsrError(f"文件大小超过阿里百炼临时存储限制: {size/1024/1024:.2f}MB > {float(policy.max_file_size_mb):.2f}MB")


def upload_to_temporary_oss(local_file: str, policy: UploadPolicy, session=requests) -> str:
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
    api_key: str, oss_url: str, speaker_count: Optional[int] = None,
    model: str = MODEL_NAME, session=requests,
) -> str:
    api_key = _require_api_key(api_key)
    parameters: dict[str, Any] = {"diarization_enabled": True}
    if speaker_count:
        parameters["speaker_count"] = int(speaker_count)
    payload = {"model": model, "input": {"file_urls": [oss_url]}, "parameters": parameters}
    response = _session_post(
        session, TRANSCRIPTION_URL,
        headers=_auth_headers(api_key, {
            "X-DashScope-Async": "enable",
            "X-DashScope-OssResourceResolve": "enable",
        }),
        json=payload, timeout=30,
    )
    data = _json(response, "提交 Fun-ASR 转写任务")
    task_id = ((data.get("output") or {}).get("task_id") or "").strip()
    if not task_id:
        raise FunAsrError("提交 Fun-ASR 转写任务失败：未返回 task_id")
    return task_id


def poll_transcription_task(
    api_key: str, task_id: str, poll_interval: float = 2.0,
    timeout: float = 600.0, session=requests,
) -> dict[str, Any]:
    api_key = _require_api_key(api_key)
    deadline = time.time() + timeout
    last_status = "PENDING"
    while time.time() < deadline:
        response = _session_post(
            session, TASK_URL_TEMPLATE.format(task_id=task_id),
            headers=_auth_headers(api_key), timeout=30,
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


# ============================================================================
# SRT 工具
# ============================================================================
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
        blocks.append({
            "start": cursor, "end": max(cursor + 200, chunk_end),
            "text": chunk, "speaker_id": sentence.get("speaker_id"),
        })
        cursor = chunk_end
    return blocks


def fun_asr_result_to_srt(result_json: dict[str, Any], max_chars: int = 20, max_duration: float = 3.5) -> str:
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


# ============================================================================
# 本地 FunASR-Pack / FireRed 路径
# ============================================================================
def request_local_fun_asr_health(api_url: str = LOCAL_FUN_ASR_API_URL, session=requests) -> dict[str, Any]:
    response = _session_get(session, f"{_local_base_url(api_url)}/health", timeout=10)
    return _local_json(response, "检查本地 FunASR-Pack 服务")


def request_local_firered_asr_health(api_url: str = LOCAL_FIRERED_ASR_API_URL, session=requests) -> dict[str, Any]:
    response = _session_get(session, f"{_local_base_url(api_url)}/health", timeout=10)
    return _local_json(response, "检查本地 FireRedASR2-AED-Pack 服务", service_name="本地 FireRedASR2-AED-Pack 服务")


def request_local_fun_asr(
    local_file: str, api_url: str = LOCAL_FUN_ASR_API_URL,
    hotword: str = "", enable_spk: Optional[bool] = None,
    model: str = LOCAL_FUN_ASR_OPENAI_MODEL, timeout: float = 600.0, session=requests,
) -> dict[str, Any]:
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
    attempts = [(openai_url, openai_data), (rest_url, rest_data)] if _local_fun_asr_prefers_openai(api_url) else [(rest_url, rest_data), (openai_url, openai_data)]
    last_response = None
    for index, (url, data) in enumerate(attempts):
        with open(local_file, "rb") as file_obj:
            files = {"file": (_safe_upload_name(local_file), file_obj)}
            response = _session_post(session, url, data=data, files=files, timeout=timeout)
        if index == 0 and _is_not_found_response(response):
            last_response = response
            continue
        return _local_json(response, "调用本地 FunASR-Pack ASR API")
    return _local_json(last_response, "调用本地 FunASR-Pack ASR API")


def request_local_firered_asr(
    local_file: str, api_url: str = LOCAL_FIRERED_ASR_API_URL,
    enable_vad: Optional[bool] = True, enable_lid: Optional[bool] = True,
    enable_punc: Optional[bool] = True, return_timestamp: Optional[bool] = True,
    timeout: float = 600.0, session=requests,
) -> dict[str, Any]:
    _require_local_file(local_file)
    data: dict[str, str] = {}
    options = {"enable_vad": enable_vad, "enable_lid": enable_lid, "enable_punc": enable_punc, "return_timestamp": return_timestamp}
    for key, value in options.items():
        if value is not None:
            data[key] = "true" if value else "false"
    with open(local_file, "rb") as file_obj:
        files = {"file": (_safe_upload_name(local_file), file_obj)}
        response = _session_post(session, _local_asr_url(api_url), data=data, files=files, timeout=timeout)
    return _local_json(response, "调用本地 FireRedASR2-AED-Pack ASR API", service_name="本地 FireRedASR2-AED-Pack 服务")


def download_local_srt(
    download_url: str, api_url: str = LOCAL_FUN_ASR_API_URL,
    subtitle_file: str = "", session=requests, service_name: str = "本地 FunASR-Pack 服务",
) -> str:
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
        blocks.extend(_blocks_from_sentence(
            {"begin_time": start_ms, "end_time": end_ms, "text": text, "speaker_id": segment.get("speaker")},
            max_chars=max_chars,
        ))
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


def local_fun_asr_result_to_srt(result_json: dict[str, Any], max_chars: int = 20, max_duration: float = 3.5) -> str:
    blocks = _blocks_from_openai_segments(result_json, max_chars=max_chars)
    for item in _local_result_items(result_json):
        item_blocks = _blocks_from_local_timestamp(item, max_chars, max_duration)
        if not item_blocks:
            text = str(item.get("text") or "").strip()
            if text:
                item_blocks = _blocks_from_sentence(
                    {"begin_time": 0, "end_time": max(1500, len(text) * 180), "text": text},
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
    local_file: str, subtitle_file: str = "", api_url: str = LOCAL_FUN_ASR_API_URL,
    hotword: str = "", enable_spk: Optional[bool] = None,
    timeout: float = 600.0, session=requests,
) -> Optional[str]:
    try:
        result_json = request_local_fun_asr(
            local_file=local_file, api_url=api_url, hotword=hotword,
            enable_spk=enable_spk, timeout=timeout, session=session,
        )
        srt_file = result_json.get("srt_file")
        if isinstance(srt_file, str) and srt_file and os.path.isfile(srt_file):
            output_file = copy_srt_file(srt_file, subtitle_file)
        else:
            download_url = _get_local_srt_download_url(result_json)
            if download_url:
                output_file = download_local_srt(download_url, api_url=api_url, subtitle_file=subtitle_file, session=session)
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
    local_file: str, subtitle_file: str = "", api_url: str = LOCAL_FIRERED_ASR_API_URL,
    enable_vad: Optional[bool] = True, enable_lid: Optional[bool] = True,
    enable_punc: Optional[bool] = True, return_timestamp: Optional[bool] = True,
    timeout: float = 600.0, session=requests,
) -> Optional[str]:
    """【已被劫持】调用本地 Whisper + pyannote diarization（speaker 可选）。"""
    logger.info("🚀 拦截到 FireRedASR 请求，已切换为本地 Whisper + pyannote diarization")
    return _run_local_whisper_direct(
        local_file, subtitle_file,
        enable_diarization=ENABLE_WHISPER_DIARIZATION,
    )


def create_with_fun_asr(
    local_file: str, subtitle_file: str = "", api_key: str = "",
    speaker_count: Optional[int] = None, poll_interval: float = 2.0,
    timeout: float = 600.0, session=requests,
) -> Optional[str]:
    api_key = _require_api_key(api_key)
    try:
        policy = request_upload_policy(api_key, session=session)
        oss_url = upload_to_temporary_oss(local_file, policy, session=session)
        task_id = submit_transcription_task(api_key, oss_url, speaker_count=speaker_count, session=session)
        task_result = poll_transcription_task(api_key, task_id, poll_interval=poll_interval, timeout=timeout, session=session)
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


# ============================================================================
# Whisper 显存 / batch 工具
# ============================================================================
def _parse_device_index(device: str) -> int:
    if ":" in device:
        try:
            return int(device.split(":")[1])
        except ValueError:
            return 0
    return 0


def _vram(tag: str, device_index: int = 0, aggressive: bool = False) -> None:
    """清缓存（可选）+ 记录显存。logger.debug 不污染 stdout。"""
    import torch
    if not torch.cuda.is_available():
        return
    if aggressive:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    torch.cuda.empty_cache()
    if aggressive:
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()
    try:
        free, _total = torch.cuda.mem_get_info(device_index)
        allocated = torch.cuda.memory_allocated(device_index) / _BYTES_PER_MB
        reserved = torch.cuda.memory_reserved(device_index) / _BYTES_PER_MB
        logger.debug(
            f"[{tag}] free={free / _BYTES_PER_MB:.0f}MB "
            f"alloc={allocated:.0f}MB resv={reserved:.0f}MB"
        )
    except Exception:
        pass


# ============================================================================
# Diarization 配置
# ============================================================================
DIARIZE_MODEL_PATH = Path(
    os.getenv(
        "YOUDUB_DIARIZE_MODEL",
        "/home/ken/data/youdub/submodule/models/speaker-diarization-community-1",
    )
)
DIARIZE_MODEL_FALLBACK = "pyannote/speaker-diarization-3.1"
CONFIG_TOML_PATH = Path(
    os.getenv("YOUDUB_CONFIG_TOML", str(Path(".").resolve() / "config.toml"))
)

ENABLE_WHISPER_DIARIZATION = (
    os.getenv("YOUDUB_ASR_ENABLE_DIARIZATION", "1") == "1"
)

try:
    import tomllib as _toml_lib  # Python 3.11+
except ImportError:
    try:
        import tomli as _toml_lib  # type: ignore
    except ImportError:
        _toml_lib = None  # type: ignore


def _load_hf_token() -> Optional[str]:
    """按优先级读取 HF_TOKEN：环境变量 > config.toml。"""
    env_token = (os.getenv("HF_TOKEN") or "").strip()
    if env_token:
        return env_token
    if _toml_lib is None or not CONFIG_TOML_PATH.exists():
        return None
    try:
        with open(CONFIG_TOML_PATH, "rb") as f:
            cfg = _toml_lib.load(f)
    except Exception as e:
        logger.warning(f"读取 config.toml 失败（HF_TOKEN 不可用）: {e}")
        return None
    for path in (
        ("HF_TOKEN",), ("hf", "token"), ("huggingface", "token"),
        ("huggingface", "hf_token"), ("api", "hf_token"),
        ("asr", "hf_token"), ("asr", "HF_TOKEN"),
    ):
        node = cfg
        ok = True
        for key in path:
            if not isinstance(node, dict) or key not in node:
                ok = False
                break
            node = node[key]
        if ok and node:
            token = str(node).strip()
            if token:
                return token
    return None


def _pyannote_available() -> bool:
    """pyannote.audio 是否可 import（find_spec 零开销，不实际 import）。"""
    import importlib.util
    return importlib.util.find_spec("pyannote.audio") is not None


def _diarization_available(hf_token: Optional[str]) -> bool:
    """pyannote 已装 且（本地模型有效 或 有 HF token）时才可用。"""
    if not _pyannote_available():
        return False
    if DIARIZE_MODEL_PATH.exists():
        if (
            (DIARIZE_MODEL_PATH / "config.yaml").exists()
            or any(DIARIZE_MODEL_PATH.glob("*.bin"))
            or any(DIARIZE_MODEL_PATH.glob("*.pt"))
            or any(DIARIZE_MODEL_PATH.glob("*.safetensors"))
        ):
            return True
    return bool(hf_token)


# ============================================================================
# faster-whisper 抗幻觉参数
# ============================================================================
_FASTER_WHISPER_KWARGS = dict(
    language=None,
    beam_size=1,
    best_of=1,
    temperature=0.0,
    word_timestamps=True,
    vad_filter=True,
    condition_on_previous_text=False,
    no_speech_threshold=0.6,
    log_prob_threshold=-1.0,
    compression_ratio_threshold=2.4,
    hallucination_silence_threshold=2.0,
    initial_prompt="This is an interview. Please transcribe accurately.",
)


# ============================================================================
# 断行常量（与 _4_asr_fix.py v18.5 对齐）
# ============================================================================
_WHISPER_END_PUNCT = set("，。！？；,.!?;")
_WHISPER_STRONG_END_PUNCT = set("。！？；.!?;")
_CJK_RE = re.compile(r"[\u4e00-\u9fa5]")

# 30 词功能词表：冠词、介词、并列连词、wh- 关系词、从属连词、比较连词
# 明确排除 be 动词（is/are/was/were）：疑问句中前置到句中，行尾合理，迁移误伤
_WHISPER_HANGING_WORDS = {
    "the", "a", "an",
    "of", "to", "in", "for", "on", "at", "by", "with",
    "and", "or", "but",
    "that", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "what",
    "if", "whether", "because", "although", "though",
    "than",
}

WHISPER_PRESPLIT_MAX_SECONDS = 15.0
WHISPER_MIN_FRAGMENT_MS = 800.0
WHISPER_MAX_MERGE_MS = 3500.0
WHISPER_MAX_SEGMENT_DURATION_MS = 5000
WHISPER_TARGET_LEN_CN = 15
WHISPER_TARGET_LEN_EN = 25
WHISPER_MAX_LEN_CN = 22
WHISPER_MAX_LEN_EN = 42


def _ends_with_strong_punct(text: str) -> bool:
    s = str(text or "").strip()
    return bool(s) and s[-1] in _WHISPER_STRONG_END_PUNCT


def _get_chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(re.sub(r"\s+", "", text))
    if total == 0:
        return 0.0
    return len(_CJK_RE.findall(text)) / total


def _dedupe_whisper_words(words: list[dict]) -> list[dict]:
    """仅删"文本相同且时间戳完全重叠"的重复词，避免误删 very very。"""
    out: list[dict] = []
    for w in words:
        if out:
            p = out[-1]
            if (
                str(p.get("word", "")).strip() == str(w.get("word", "")).strip()
                and p.get("start") == w.get("start")
                and p.get("end") == w.get("end")
            ):
                continue
        out.append(w)
    return out


def _smart_join_with_index(words: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """一次遍历同时产出 text 和每词的 (char_start, char_end)，无字符漂移。"""
    text_parts: list[str] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    last_char = ""
    for idx, w in enumerate(words):
        word_str = str(w.get("word") or "")
        if not word_str:
            ranges.append((cursor, cursor))
            continue
        if idx == 0:
            word_str = word_str.lstrip()
        else:
            curr_char = word_str[0]
            need_space = True
            if curr_char in _WHISPER_END_PUNCT:
                need_space = False
            elif _CJK_RE.match(last_char) or _CJK_RE.match(curr_char):
                need_space = False
            elif last_char == " " or word_str.startswith(" "):
                need_space = False
            if need_space:
                text_parts.append(" ")
                cursor += 1
            if last_char == " " and word_str.startswith(" "):
                word_str = word_str.lstrip()
        start = cursor
        text_parts.append(word_str)
        cursor += len(word_str)
        if word_str:
            last_char = word_str[-1]
        ranges.append((start, cursor))
    return "".join(text_parts), ranges


def _smart_join(words: list[dict]) -> str:
    return _smart_join_with_index(words)[0].strip()


# ============================================================================
# pyannote diarize + word 中点分配
# ============================================================================
def _assign_speakers_with_pyannote(
    audio_file: Path,
    all_words: list[dict],
    hf_token: Optional[str],
    device: str,
) -> None:
    """pyannote 3.4 / 4.0 兼容 diarize，按 word 中点分配 speaker_id。

    · 3.4 pipeline(audio) 返回 Annotation（有 itertracks）
    · 4.0 返回 DiarizeOutput（有 speaker_diarization / exclusive_speaker_diarization）
    · 用 in-memory 波形避免 torchcodec 依赖（配合顶部 soundfile patch 双保险）
    """
    import torch
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as exc:
        raise FunAsrError("未安装 pyannote.audio，无法做说话人识别") from exc

    # 1. 模型源（本地优先）
    if DIARIZE_MODEL_PATH.exists():
        model_src = str(DIARIZE_MODEL_PATH.resolve())
        logger.info(f"🎙️ 说话人识别（本地模型）: {model_src}")
    else:
        model_src = DIARIZE_MODEL_FALLBACK
        logger.info(f"🎙️ 说话人识别（远端模型）: {model_src}")

    # 2. 加载 pipeline（兼容 use_auth_token / token 两种参数名）
    try:
        pipeline = Pipeline.from_pretrained(model_src, use_auth_token=hf_token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_src, token=hf_token)
    if pipeline is None:
        raise FunAsrError("pyannote pipeline 加载失败（可能需 HF_TOKEN 或接受用户协议）")

    # 3. 移到 GPU（失败则留 CPU）
    try:
        pipeline.to(torch.device(device))
    except Exception as e:
        logger.warning(f"pyannote 移到 {device} 失败，使用 CPU: {e}")

    # 4. 预加载音频（in-memory 波形，绕开 torchcodec / torchaudio.load 路径）
    from faster_whisper import decode_audio
    audio_array = decode_audio(str(audio_file), sampling_rate=16000)
    waveform = torch.from_numpy(audio_array).unsqueeze(0)  # (1, samples)
    audio_in_memory = {"waveform": waveform, "sample_rate": 16000}

    # 5. 跑 diarize
    try:
        with torch.inference_mode():
            output = pipeline(audio_in_memory)
    finally:
        del pipeline
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # 6. pyannote 3.4 / 4.0 双兼容取 annotation
    #    3.4: output 就是 Annotation
    #    4.0: output 是 DiarizeOutput，优先取 exclusive_speaker_diarization（不重叠，字幕场景更合适）
    annotation = output
    if hasattr(output, "exclusive_speaker_diarization"):
        annotation = output.exclusive_speaker_diarization
    elif hasattr(output, "speaker_diarization"):
        annotation = output.speaker_diarization

    # 7. 收集 turns
    turns: list[tuple[float, float, str]] = []
    try:
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append((float(turn.start), float(turn.end), str(speaker)))
    except AttributeError:
        # 极端情况：返回结构完全变了，用 serialize 解析 RTTM
        logger.warning("pyannote output 结构异常，尝试 serialize 解析")
        try:
            for line in annotation.serialize():
                parts = line.strip().split()
                if len(parts) >= 8:
                    turns.append((float(parts[3]), float(parts[4]), parts[7]))
        except Exception as e:
            raise FunAsrError(f"pyannote 输出解析失败: {e}") from e

    logger.info(f"   检测到 {len(turns)} 个 speaker turn")
    if not turns:
        logger.warning("pyannote 未检测到任何 speaker turn")
        return

    # 8. 按 word 中点分配（turns 时间有序，线性推进）
    assigned = 0
    turn_idx = 0
    for w in all_words:
        midpoint = (w["start"] + w["end"]) / 2
        while turn_idx < len(turns) and turns[turn_idx][1] < midpoint:
            turn_idx += 1
        if turn_idx < len(turns):
            start, end, label = turns[turn_idx]
            if start <= midpoint <= end:
                w["speaker_id"] = label
                assigned += 1
            else:
                w["speaker_id"] = None
        else:
            w["speaker_id"] = None
    logger.info(f"   speaker 分配: {assigned}/{len(all_words)} 词")


# ============================================================================
# faster-whisper 转录 + pyannote diarize
# ============================================================================
def _load_or_transcribe_whisper_words(
    local_file: str,
    cache_dir: str,
    model_path: str,
    enable_diarization: bool = True,
) -> list[dict]:
    """读缓存或 faster-whisper 转录 + pyannote diarize。

    · ASR: faster-whisper（自带 word_timestamps，无需 align 模型）
    · Diarize: pyannote.audio（本地模型优先，HF_TOKEN 兜底）
    · 输出 word 带 speaker_id（SPEAKER_00 / None）
    · 缓存 key 含 diarization 标记 + version（_v4，区分旧 whisperx 版）
    """
    try:
        audio_size = os.path.getsize(local_file)
        audio_mtime = int(os.path.getmtime(local_file))
    except OSError:
        audio_size, audio_mtime = 0, 0
    fingerprint = hashlib.md5(
        f"{os.path.basename(local_file)}_{audio_size}_{audio_mtime}"
        f"_diar={int(enable_diarization)}_v4".encode("utf-8")
    ).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f"{fingerprint}_asr_word_level.json")

    if os.path.exists(cache_path):
        logger.info(f"♻️ 读取缓存: {cache_path}")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, list) and cached:
                return cached
        except Exception as e:
            logger.warning(f"缓存损坏，重新转录: {e}")

    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    device_index = _parse_device_index(device)

    # ---- 阶段 1：faster-whisper 转录 ----
    logger.info(f"⏳ [1/2] 加载 faster-whisper 模型: {model_path}")
    whisper_model = WhisperModel(model_path, device=device, compute_type=compute_type)
    all_words: list[dict] = []
    try:
        with torch.inference_mode():
            segments, info = whisper_model.transcribe(
                str(local_file), **_FASTER_WHISPER_KWARGS
            )
            detected_lang = getattr(info, "language", None) or "en"
            logger.info(f"   检测语言: {detected_lang}")

            for seg in segments:  # 生成器，逐个消费
                if not seg.words:
                    continue
                for w in seg.words:
                    word_text = str(w.word or "")
                    if not word_text.strip():
                        continue
                    all_words.append({
                        "word": word_text,
                        "start": float(w.start),
                        "end": float(w.end),
                        "probability": float(getattr(w, "probability", 1.0)),
                        "speaker_id": None,
                    })
    finally:
        del whisper_model
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        _vram("faster-whisper 释放", device_index, aggressive=True)

    all_words.sort(key=lambda x: x["start"])
    if not all_words:
        raise FunAsrError("Whisper 转录结果为空，未找到任何词")

    # ---- 阶段 2：pyannote diarize（可选，失败降级）----
    hf_token = _load_hf_token()
    if enable_diarization:
        if not _pyannote_available():
            logger.info(
                "pyannote.audio 未安装，跳过说话人识别"
                "（如需 speaker: uv add 'pyannote.audio>=3.1,<4.0'）"
            )
        elif not _diarization_available(hf_token):
            logger.warning(
                "说话人模型不可用：本地模型不存在且 config.toml / 环境变量未配置 HF_TOKEN，"
                "降级为无 speaker（speaker_id=None）"
            )
        else:
            logger.info("🎙️ [2/2] 说话人识别...")
            try:
                _assign_speakers_with_pyannote(
                    Path(local_file), all_words, hf_token, device
                )
            except Exception as e:
                logger.warning(f"pyannote diarize 失败，降级为无 speaker: {e}")
    else:
        logger.info("⏭️ [2/2] 跳过说话人识别")

    # ---- 缓存写回 ----
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_words, f, ensure_ascii=False, indent=2)
    return all_words


# ============================================================================
# 断行后处理
# ============================================================================
def _presplit_whisper_paragraphs(words: list[dict]) -> list[list[dict]]:
    """15s / 强标点 / speaker 变化 硬切。speaker_id=None 视为同一 speaker。"""
    paragraphs: list[list[dict]] = []
    current: list[dict] = []
    block_start: float | None = None
    last_speaker = None
    for w in words:
        spk = w.get("speaker_id")
        if current and spk != last_speaker and spk is not None and last_speaker is not None:
            paragraphs.append(current)
            current = []
            block_start = None
        current.append(w)
        if block_start is None:
            block_start = w["start"]
        last_speaker = spk
        is_strong = _ends_with_strong_punct(w["word"])
        is_too_long = (w["end"] - block_start) > WHISPER_PRESPLIT_MAX_SECONDS
        if is_strong:
            paragraphs.append(current)
            current = []
            block_start = None
        elif is_too_long and len(current) > 1:
            paragraphs.append(current[:-1])
            current = [current[-1]]
            block_start = current[0]["start"]
    if current:
        paragraphs.append(current)
    return paragraphs


def _semantic_split_paragraph(
    para: list[dict], model, target_len: int, max_len: int
) -> list[dict]:
    """wtpsplit 语义细分，用 char↔word 索引精确回填。返回含 start/end（毫秒）。"""
    text, ranges = _smart_join_with_index(para)
    if not text.strip():
        return []
    lang_code = "zh" if _get_chinese_ratio(text) > 0.3 else "en"
    prior_kwargs = {"lang_code": lang_code, "target_length": target_len}
    try:
        segs = list(model.split(
            text, max_length=max_len, prior_type="gaussian",
            prior_kwargs=prior_kwargs,
        ))
    except (TypeError, ValueError):
        prior_kwargs.pop("target_length", None)
        segs = list(model.split(
            text, max_length=max_len, prior_type="gaussian",
            prior_kwargs=prior_kwargs,
        ))
    result: list[dict] = []
    pos = 0
    word_idx = 0
    n = len(ranges)
    for seg in segs:
        seg_end = pos + len(seg)
        s = pos
        while s < seg_end and text[s].isspace():
            s += 1
        e = seg_end
        while e > s and text[e - 1].isspace():
            e -= 1
        pos = seg_end
        if s >= e:
            continue
        while word_idx < n and ranges[word_idx][1] <= s:
            word_idx += 1
        w_start = word_idx
        w_end = w_start
        while w_end < n and ranges[w_end][0] < e:
            w_end += 1
        if w_start >= w_end:
            continue
        sub_words = para[w_start:w_end]
        result.append({
            "words": sub_words,
            "text": _smart_join(sub_words),
            "start": sub_words[0]["start"] * 1000,   # 秒 → 毫秒
            "end": sub_words[-1]["end"] * 1000,
        })
        word_idx = w_end
    return result


def _enforce_max_duration_whisper(blocks: list[dict]) -> list[dict]:
    """单条 ≤ 5s 时间兜底（队列循环，递归切到 ≤5s）。"""
    out: list[dict] = []
    queue: list[dict] = list(blocks)
    while queue:
        block = queue.pop(0)
        words = block.get("words") or []
        duration = block.get("end", 0) - block.get("start", 0)
        if duration <= WHISPER_MAX_SEGMENT_DURATION_MS or len(words) <= 1:
            out.append(block)
            continue
        mid = len(words) // 2
        split_at = None
        for i in range(mid, 0, -1):
            if _ends_with_strong_punct(words[i - 1]["word"]):
                split_at = i
                break
        if split_at is None:
            for i in range(mid + 1, len(words)):
                if _ends_with_strong_punct(words[i - 1]["word"]):
                    split_at = i
                    break
        if split_at is None or split_at <= 0 or split_at >= len(words):
            split_at = mid
        left_words = words[:split_at]
        right_words = words[split_at:]
        queue.insert(0, {
            "words": right_words,
            "text": _smart_join(right_words),
            "start": right_words[0]["start"] * 1000,
            "end": right_words[-1]["end"] * 1000,
        })
        queue.insert(0, {
            "words": left_words,
            "text": _smart_join(left_words),
            "start": left_words[0]["start"] * 1000,
            "end": left_words[-1]["end"] * 1000,
        })
    return out


def _migrate_hanging_words_whisper(blocks: list[dict]) -> int:
    """悬空词迁移（支持单 hanging word utt 被相邻句吸收）。返回实际迁移词数。"""
    total_moved = 0
    i = 0
    while i < len(blocks) - 1:
        cur_words = blocks[i]["words"]
        nxt_words = blocks[i + 1]["words"]
        if not cur_words or not nxt_words:
            i += 1
            continue
        if _ends_with_strong_punct(cur_words[-1]["word"]):
            i += 1
            continue
        cur_spk = cur_words[-1].get("speaker_id")
        nxt_spk = nxt_words[0].get("speaker_id")
        if cur_spk != nxt_spk and cur_spk is not None and nxt_spk is not None:
            i += 1
            continue
        moved_words: list[dict] = []
        while cur_words:
            last_word = str(cur_words[-1]["word"]).strip().lower()
            last_clean = re.sub(r"[^\w\s]", "", last_word).strip()
            if last_clean not in _WHISPER_HANGING_WORDS:
                break
            moved_words.insert(0, cur_words.pop())
        if not moved_words:
            i += 1
            continue
        total_moved += len(moved_words)
        candidate_nxt_words = moved_words + nxt_words
        if not cur_words:
            blocks[i + 1] = {
                "words": candidate_nxt_words,
                "text": _smart_join(candidate_nxt_words),
                "start": candidate_nxt_words[0]["start"] * 1000,
                "end": candidate_nxt_words[-1]["end"] * 1000,
            }
            del blocks[i]
            continue
        else:
            blocks[i] = {
                "words": cur_words,
                "text": _smart_join(cur_words),
                "start": cur_words[0]["start"] * 1000,
                "end": cur_words[-1]["end"] * 1000,
            }
            blocks[i + 1] = {
                "words": candidate_nxt_words,
                "text": _smart_join(candidate_nxt_words),
                "start": candidate_nxt_words[0]["start"] * 1000,
                "end": candidate_nxt_words[-1]["end"] * 1000,
            }
            i += 1
    return total_moved


def _merge_fragments_whisper(blocks: list[dict]) -> tuple[list[dict], int]:
    """<800ms 碎片与前一组合并（≤3.5s）。跨 speaker 不合并。"""
    merged = 0
    out: list[dict] = []
    for block in blocks:
        if not out:
            out.append(block)
            continue
        prev = out[-1]
        prev_words = prev.get("words") or []
        cur_words = block.get("words") or []
        prev_spk = prev_words[0].get("speaker_id") if prev_words else None
        cur_spk = cur_words[0].get("speaker_id") if cur_words else None
        if prev_spk != cur_spk and prev_spk is not None and cur_spk is not None:
            out.append(block)
            continue
        dur = block["end"] - block["start"]
        merged_dur = block["end"] - prev["start"]
        if dur < WHISPER_MIN_FRAGMENT_MS and merged_dur <= WHISPER_MAX_MERGE_MS:
            merged_words = prev_words + cur_words
            out[-1] = {
                "words": merged_words,
                "text": _smart_join(merged_words),
                "start": prev["start"],
                "end": block["end"],
            }
            merged += 1
            continue
        out.append(block)
    return out, merged


# ============================================================================
# 主入口
# ============================================================================
def _run_local_whisper_direct(
    local_file: str,
    subtitle_file: str = "",
    model_path: str = "/home/ken/data/youdub/submodule/models/whisper-large-v3",
    enable_diarization: bool = True,
) -> Optional[str]:
    """本地 Whisper + pyannote diarization（v3.9）。

    流水线：
      _load_or_transcribe_whisper_words（faster-whisper + pyannote）
      → _dedupe_whisper_words
      → _presplit_whisper_paragraphs（15s / 强标点 / speaker 变化硬切）
      → _semantic_split_paragraph（wtpsplit gaussian prior + 索引回填）
      → _enforce_max_duration_whisper（单条 ≤5s 时间兜底）
      → _migrate_hanging_words_whisper（30 词功能词表 + 单 hanging word utt 吸收）
      → _merge_fragments_whisper（<800ms 合并 ≤3.5s，speaker 保护）
      → SRT
    """
    from wtpsplit import SaT

    if not subtitle_file:
        subtitle_file = os.path.join(utils.subtitle_dir(), f"whisper_{int(time.time())}.srt")
    cache_dir = os.path.join(utils.subtitle_dir(), "whisper_cache")
    os.makedirs(cache_dir, exist_ok=True)

    sat_model = None
    try:
        all_words = _load_or_transcribe_whisper_words(
            local_file, cache_dir, model_path,
            enable_diarization=enable_diarization,
        )
        all_words = _dedupe_whisper_words(all_words)
        logger.info(
            f"词数: {len(all_words)}, 是否含 speaker: "
            f"{all_words[0].get('speaker_id') is not None if all_words else False}"
        )

        logger.info("🔪 第一层：物理预切分为大段落...")
        paragraphs = _presplit_whisper_paragraphs(all_words)
        logger.info(f"   段落数: {len(paragraphs)}")

        logger.info("🧠 第二层：wtpsplit 语义细分...")
        sat_model = SaT(
            "sat-3l-sm",
            ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        semantic_blocks: list[dict] = []
        for para in paragraphs:
            para_text = _smart_join(para)
            if not para_text.strip():
                continue
            if _get_chinese_ratio(para_text) > 0.3:
                target_len, max_len = WHISPER_TARGET_LEN_CN, WHISPER_MAX_LEN_CN
            else:
                target_len, max_len = WHISPER_TARGET_LEN_EN, WHISPER_MAX_LEN_EN
            semantic_blocks.extend(
                _semantic_split_paragraph(para, sat_model, target_len, max_len)
            )

        logger.info("⏱️ 时间兜底：单条 ≤5s...")
        before = len(semantic_blocks)
        semantic_blocks = _enforce_max_duration_whisper(semantic_blocks)
        if len(semantic_blocks) > before:
            logger.info(f"   超时二次切: {len(semantic_blocks) - before} 次")

        logger.info("🪄 第三层：悬空词迁移 + 碎片合并...")
        moved = _migrate_hanging_words_whisper(semantic_blocks)
        final_blocks, merged = _merge_fragments_whisper(semantic_blocks)
        logger.info(f"   悬空词迁移: {moved} 个，碎片合并: {merged} 次")

        if not final_blocks:
            raise FunAsrError("Whisper 切分结果为空")

        lines = []
        for index, block in enumerate(final_blocks, start=1):
            lines.append(_srt_block(index, block["start"], block["end"], block["text"]))
        srt_content = "\n".join(lines).rstrip() + "\n"

        try:
            import opencc
            srt_content = opencc.OpenCC("t2s").convert(srt_content)
        except Exception:
            pass

        output_file = write_srt_file(srt_content, subtitle_file)
        logger.info(f"✅ 生成成功，共 {len(final_blocks)} 条: {output_file}")
        return output_file

    except FunAsrError:
        raise
    except Exception as e:
        logger.error(f"❌ 失败: {e}")
        raise FunAsrError(f"本地 Whisper 字幕生成失败: {e}") from e
    finally:
        sat_model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass