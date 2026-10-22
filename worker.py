# worker.py
import os
import sys
import time
import socket
import traceback
import threading
import json
import re
import argparse
import subprocess
from datetime import datetime
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from loguru import logger
from app.config import config
from app.services.worker_paths import make_work_dir, paths, update_meta
from app.services.youtube_downloader import download_youtube_video
from app.models.schema import VideoClipParams
from database import db

# ============================================================
# 统一 LLM 体系
# ============================================================
from app.services.prompts import PromptManager
from app.services.llm.manager import LLMServiceManager
from app.services.llm.unified_service import UnifiedLLMService
from app.services.llm.validators import OutputValidator, ValidationError
from app.services.llm.migration_adapter import _run_async_safely


HOSTNAME = socket.gethostname()
COOKIES_FILE = os.path.join(ROOT_DIR, "cookies.txt")

HEARTBEAT_INTERVAL = 15
STALE_RESET_INTERVAL = 60
STALE_TIMEOUT = 120
FAILED_REQUEUE_HOURS = 6

# 默认 prompt_category（项目当前主做播客访谈）
DEFAULT_PROMPT_CATEGORY = "podcast_interview"

# pipeline phase 拆为 6 个 stage：
#   script        → LLM 生成 video_script.json
#   tts           → voice.tts_multiple → tts_results.json + audio_*.wav
#   clip          → clip_video.clip_video_unified → clip_result.json + clips/*.mp4
#   merge_audio   → update_script + audio_merger + script_subtitle
#   merge_video   → merger_video.combine_clip_videos → merger.mp4
#   final_compose → generate_video.merge_materials → combined.mp4
PHASE_STAGES = {
    "preprocess":  ["download", "transcribe"],
    "pipeline":    ["script", "tts", "clip", "merge_audio", "merge_video", "final_compose"],
    "postprocess": ["publish"],
}
PHASE_NEXT = {
    "preprocess":  "pipeline",
    "pipeline":    "done",
    "postprocess": "done",
}
ACTIVE_PHASES = ["preprocess", "pipeline"]


# ============================================================
# 一、路径重定向
# ============================================================

from app.utils import utils as narrato_utils

_thread_local = threading.local()


def _get_task_dir_cache() -> dict:
    cache = getattr(_thread_local, "task_dir_cache", None)
    if cache is None:
        cache = {}
        _thread_local.task_dir_cache = cache
    return cache


_original_task_dir = narrato_utils.task_dir


def _patched_task_dir(sub_dir: str = ""):
    if sub_dir and sub_dir.startswith("proj_"):
        cache = _get_task_dir_cache()
        if sub_dir in cache:
            return cache[sub_dir]
        try:
            pid = int(sub_dir.split("_", 1)[1])
            proj = db.get_project(pid, with_stages=False)
            if proj and proj.get("work_dir"):
                d = os.path.join(proj["work_dir"], "tasks")
                os.makedirs(d, exist_ok=True)
                cache[sub_dir] = d
                return d
        except Exception as e:
            print(f"[Worker] task_dir 重定向失败，回退全局: {e}", flush=True)
    return _original_task_dir(sub_dir)


narrato_utils.task_dir = _patched_task_dir


from app.services import clip_video as _clip_module
_original_clip_video_unified = _clip_module.clip_video_unified


def _patched_clip_video_unified(video_origin_path, script_list, tts_results,
                                output_dir=None, task_id=None, video_origin_paths=None):
    if output_dir is None:
        wd = getattr(_thread_local, "current_work_dir", None)
        if wd:
            output_dir = os.path.join(wd, "clips")
            os.makedirs(output_dir, exist_ok=True)
    return _original_clip_video_unified(
        video_origin_path=video_origin_path,
        script_list=script_list,
        tts_results=tts_results,
        output_dir=output_dir,
        task_id=task_id,
        video_origin_paths=video_origin_paths,
    )


_clip_module.clip_video_unified = _patched_clip_video_unified


# ============================================================
# 二、工具函数
# ============================================================

def get_storage_root() -> str:
    root = (config.app.get("worker_storage_dir", "") or "").strip()
    return root or os.path.join(ROOT_DIR, "storage", "projects")


def heartbeat_thread(project_id: int, stop_event: threading.Event):
    while not stop_event.wait(HEARTBEAT_INTERVAL):
        db.heartbeat(project_id)


def _probe_media(media_path: str) -> dict:
    try:
        from moviepy import VideoFileClip
        clip = VideoFileClip(media_path)
        info = {
            "duration_sec": round(float(clip.duration), 3),
            "resolution": f"{clip.size[0]}x{clip.size[1]}",
            "source_size": os.path.getsize(media_path),
        }
        clip.close()
        return info
    except Exception:
        return {}


def _infer_aspect(resolution: str) -> str:
    try:
        w, h = map(int, str(resolution).lower().split("x"))
        ratio = w / h
        if ratio >= 1.5: return "16:9"
        if ratio >= 1.2: return "4:3"
        if ratio <= 0.65: return "9:16"
        if ratio <= 0.85: return "3:4"
        return "1:1"
    except Exception:
        return "16:9"


def _clean_srt_duplicates(srt_content: str) -> str:
    pattern = re.compile(r'\b(\w+)([.,!?;]?)\s+\1([.,!?;]?)', re.IGNORECASE)
    def _dedup(m):
        return f"{m.group(1)}{m.group(3) or m.group(2)}"
    for _ in range(3):
        new = pattern.sub(_dedup, srt_content)
        if new == srt_content:
            break
        srt_content = new
    return srt_content


def _detect_is_chinese(text: str, language_hint: str = "") -> bool:
    lang = (language_hint or "").strip()
    if any(k in lang for k in ("中文", "简体", "繁体", "Chinese", "zh")):
        return True
    if any(k in lang for k in ("英语", "English", "en-US", "en-GB")):
        return False
    text = text or ""
    if not text:
        return True
    zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return zh_chars >= 10


def _estimate_audio_sec(text: str, language_hint: str = "") -> float:
    text = text or ""
    if not text.strip():
        return 0.0
    if _detect_is_chinese(text, language_hint):
        n = sum(1 for c in text if not c.isspace())
        return n / 5.0
    words = len(re.findall(r"[A-Za-z0-9']+", text))
    if words > 0:
        return words / 2.5      # ← 2.8 → 2.5
    return len(text) / 14.0


# ============================================================
# 三、统一 LLM 调用原语
# ============================================================

def _render_prompt(category: str, name: str, parameters: dict) -> tuple[str, str]:
    prompt = PromptManager.get_prompt(category=category, name=name, parameters=parameters)
    prompt_object = PromptManager.get_prompt_object(category=category, name=name)
    system_prompt = prompt_object.get_system_prompt() or ""
    return prompt, system_prompt


def _llm_text(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    response_format: Optional[str] = None,
    on_chunk=None,
) -> str:
    if on_chunk:
        return _run_async_safely(
            UnifiedLLMService.generate_text_stream,
            prompt=prompt,
            system_prompt=system_prompt or None,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            on_chunk=on_chunk,
        )
    return _run_async_safely(
        UnifiedLLMService.generate_text,
        prompt=prompt,
        system_prompt=system_prompt or None,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )


def _validate_json(raw_text: str, fallback: dict = None) -> dict:
    try:
        return OutputValidator.validate_json_output(raw_text)
    except ValidationError:
        pass

    text = (raw_text or "").strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        candidate = m.group(0)
        if candidate.count('{') > candidate.count('}'):
            candidate += '}' * (candidate.count('{') - candidate.count('}'))
        if candidate.count('[') > candidate.count(']'):
            candidate += ']' * (candidate.count('[') - candidate.count(']'))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    if fallback is not None:
        return fallback
    raise ValueError(f"无法解析为合法的 JSON 格式: {text[:200]}")


# ============================================================
# 四、视频信息获取 + 标题翻译
# ============================================================

def _fetch_video_info(url: str) -> dict:
    import yt_dlp
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": False,
    }
    if os.path.isfile(COOKIES_FILE):
        ydl_opts["cookiefile"] = COOKIES_FILE
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False) or {}
    except Exception as e:
        print(f"[Worker] 获取视频信息失败: {e}", flush=True)
        return {}


def _translate_title_to_chinese(original_title: str) -> str:
    if not original_title or not original_title.strip():
        return ""
    prompt = f"""请将以下视频标题翻译为简体中文，用于创建文件夹名。
要求：
1. 保留人名、品牌名、专业术语的英文原文
2. 去掉繁复的标点、引号、括号内容
3. 简短有力，控制在 30 字以内
4. 只输出翻译后的标题文本，不要引号、不要解释

原标题：{original_title}"""

    try:
        result = _llm_text(
            prompt,
            "你是一位专业视频标题翻译。只输出翻译结果。",
            temperature=0.3,
        )
        cn = (result or "").strip().strip('"').strip("'").strip("《》「」").strip()
        cn = re.sub(r"\s+", " ", cn).strip()
        return cn
    except Exception as e:
        print(f"[Worker] 标题翻译失败: {e}", flush=True)
        return original_title


# ============================================================
# 五、播客访谈配置读取器
# ============================================================

def _get_film_cfg(project: dict) -> dict:
    options = project.get("options") or {}
    a = config.app

    def _pick(opt_key: str, cfg_key: str, default, cast):
        v = options.get(opt_key)
        if v is not None and str(v).strip() != "":
            try:
                return cast(v)
            except (TypeError, ValueError):
                pass
        cv = a.get(cfg_key, default)
        try:
            return cast(cv) if cv is not None else default
        except (TypeError, ValueError):
            return default

    return {
        "prompt_category":      _pick("prompt_category",     "narration_prompt_category",      DEFAULT_PROMPT_CATEGORY, str),
        "drama_genre":          _pick("drama_genre",         "narration_drama_genre",          "观点/思想",           str),
        "narration_language":   _pick("narration_language",  "narration_language",             "简体中文",             str),
        "word_count_override":  _pick("narration_word_count","narration_word_count",           0,                    int),
        "words_per_second":     _pick("words_per_second",    "narration_words_per_second",     1.75,                 float),
        "original_sound_ratio": _pick("original_sound_ratio","narration_original_sound_ratio", 60,                   int),
        "temperature":          _pick("temperature",         "narration_temperature",          0.7,                  float),
        "max_retries":          _pick("script_max_retries",  "narration_script_max_retries",   3,                    int),
        "min_words_ratio":      _pick("min_words_ratio",     "narration_min_words_ratio",      0.7,                  float),
    }



def _calc_target_words(duration_sec: float, cfg: dict) -> int:
    if cfg["word_count_override"] > 0:
        return cfg["word_count_override"]
    return max(200, int(duration_sec * cfg["words_per_second"]))


# ============================================================
# 六、VideoClipParams 构建 + JSON I/O
# ============================================================

def _build_video_clip_params(project: dict, work_dir: str, p: dict) -> VideoClipParams:
    """所有 stage 共用的 VideoClipParams 构建。"""
    ui_config = config.ui.copy()
    valid_fields = set(VideoClipParams.model_fields.keys())
    ui_kwargs = {k: v for k, v in ui_config.items() if k in valid_fields}

    params = VideoClipParams(
        video_clip_json_path=p["video_script"],
        video_origin_path=p["source"],
        video_origin_paths=[p["source"]],
        video_aspect=_infer_aspect(project.get("resolution")),
        subtitle_enabled=True,
        subtitle_auto_transcribe_enabled=True,
        subtitle_auto_transcribe_backend="local",
        **ui_kwargs,
    )
    if not params.voice_name:
        raise RuntimeError(
            "config.toml [ui] 里缺少 voice_name 配置。请先在 WebUI 里选择一次音色"
        )
    return params


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 七、stage 函数
# ============================================================

def stage_download(project, work_dir, p):
    pid = project["id"]
    db.start_stage(pid, "download")

    def on_progress(pct):
        db.update_stage(pid, "download", progress=pct)

    video_path = download_youtube_video(
        url=project["youtube_url"],
        save_dir=work_dir,
        progress_callback=on_progress,
        max_height=1080,
        cookies_file=COOKIES_FILE,
    )
    target = p["source"]
    if os.path.abspath(video_path) != os.path.abspath(target):
        if os.path.exists(target):
            os.remove(target)
        os.replace(video_path, target)

    info = _probe_media(target)
    if info:
        db.update_project(pid, **info)

    db.complete_stage(pid, "download", output_path=target)
    print(f"[preprocess] P{pid} 下载完成: {target}", flush=True)
    return target


def stage_transcribe(project, work_dir, p):
    pid = project["id"]
    db.start_stage(pid, "transcribe")
    db.update_stage(pid, "transcribe", progress=10)

    from app.services import transcribe
    model_path = config.app.get(
        "whisper_model_path",
        "/home/ken/data/youdub/submodule/models/whisper-large-v3",
    )
    generated = transcribe._run_local_whisper_direct(
        local_file=p["source"],
        subtitle_file=p["subtitle"],
        model_path=model_path,
    )
    if not generated or not os.path.isfile(generated):
        raise RuntimeError(f"Whisper 未生成字幕: {generated}")
    if os.path.abspath(generated) != os.path.abspath(p["subtitle"]):
        import shutil
        shutil.move(generated, p["subtitle"])

    db.complete_stage(pid, "transcribe", output_path=p["subtitle"])
    print(f"[preprocess] P{pid} 转写完成: {p['subtitle']}", flush=True)
    return p["subtitle"]


def stage_script(project, work_dir, p):
    """Stage: LLM 生成播客访谈解说脚本。输出 script/video_script.json

    走 copy 驱动路径：
      plot_analysis → narration_copy（写串场解说） → script_matching（匹配到原声时间轴）
    """
    from app.services.narration_validation import (
        build_subtitle_index, validate_narration_script_items,
        parse_script_timestamp_range, ms_to_timestamp,
        NARRATION_PROFILE_PODCAST, NARRATION_PROFILE_SHORT_DRAMA,
    )

    pid = project["id"]
    db.start_stage(pid, "script")

    with open(p["subtitle"], "r", encoding="utf-8") as f:
        srt_content = _clean_srt_duplicates(f.read())

    film_cfg = _get_film_cfg(project)
    strict_mode = bool(config.app.get("film_tv_strict_mode", True))
    title = project.get("title", "") or "Untitled"
    duration_sec = float(project.get("duration_sec") or 600)

    prompt_category      = film_cfg["prompt_category"]
    drama_genre          = film_cfg["drama_genre"]
    narration_language   = film_cfg["narration_language"]
    original_sound_ratio = film_cfg["original_sound_ratio"]
    temperature          = film_cfg["temperature"]
    max_retries          = max(1, film_cfg["max_retries"])
    min_words_ratio      = film_cfg["min_words_ratio"]
    target_words         = _calc_target_words(duration_sec, film_cfg)
    min_words_threshold  = int(target_words * min_words_ratio)

    # 按 prompt_category 选叙事规则 profile
    if prompt_category == "short_drama_narration":
        narration_profile = NARRATION_PROFILE_SHORT_DRAMA
    else:
        narration_profile = NARRATION_PROFILE_PODCAST

    print(
        f"[pipeline] P{pid} script 配置: category={prompt_category}, genre={drama_genre}, "
        f"lang={narration_language}, target_words={target_words}, "
        f"ratio={original_sound_ratio}%, temp={temperature}, "
        f"max_retries={max_retries}, strict_mode={strict_mode}, "
        f"profile={narration_profile.name}"
        f"(opening_hook={narration_profile.require_opening_hook},"
        f"max_original={narration_profile.max_consecutive_original})",
        flush=True,
    )
    db.update_stage(pid, "script", progress=5)

    # 1. plot_analysis：分析访谈，提炼主线 + 候选片段
    plot_analysis = _run_async_safely(
        UnifiedLLMService.analyze_subtitle,
        subtitle_content=srt_content,
        temperature=temperature,
        prompt_category=prompt_category,
        validate_output=False,
    )
    db.update_stage(pid, "script", progress=20)

    # 2. narration_copy：写串场解说正文
    prompt, system_prompt = _render_prompt(
        prompt_category, "narration_copy",
        parameters={
            "drama_name": title, "drama_genre": drama_genre,
            "plot_analysis": plot_analysis, "subtitle_content": srt_content,
            "narration_language": narration_language,
            "narration_word_count": int(target_words),
        },
    )
    narration_copy = _llm_text(prompt, system_prompt, temperature=temperature)

    os.makedirs(p["script_dir"], exist_ok=True)
    with open(p["narration"], "w", encoding="utf-8") as f:
        f.write(narration_copy)
    db.update_stage(pid, "script", progress=40)

    # 3. script_matching：把解说匹配到原声时间轴（访谈第一段必须是 OST=1）
    prompt, system_prompt = _render_prompt(
        prompt_category, "script_matching",
        parameters={
            "drama_name": title, "drama_genre": drama_genre,
            "plot_analysis": plot_analysis, "subtitle_content": srt_content,
            "narration_copy": narration_copy, "narration_language": narration_language,
            "original_sound_ratio": int(original_sound_ratio),
        },
    )
    raw_script = _llm_text(
        prompt, system_prompt,
        temperature=temperature,
        response_format="json",
    )
    data = _validate_json(raw_script)
    items = data.get("items", [])
    db.update_stage(pid, "script", progress=70)

    # 4. 校验 + repair
    subtitle_index = build_subtitle_index(srt_content, [p["source"]])

    def _normalize_items(raw_items):
        if not subtitle_index:
            return raw_items
        min_ms = min(cue.start_ms for cue in subtitle_index)
        max_ms = max(cue.end_ms for cue in subtitle_index)

        fixed = []
        for idx, it in enumerate(raw_items):
            try:
                s_ms, e_ms, _ = parse_script_timestamp_range(it.get("timestamp", ""))
            except Exception:
                continue
            s_ms = max(min_ms, min(s_ms, max_ms - 500))
            e_ms = max(s_ms + 500, min(e_ms, max_ms))
            if fixed:
                prev_end_ms = parse_script_timestamp_range(fixed[-1]["timestamp"])[1]
                if s_ms < prev_end_ms:
                    s_ms = prev_end_ms
                    if e_ms <= s_ms:
                        e_ms = s_ms + 800
            it["timestamp"] = f"{ms_to_timestamp(s_ms)}-{ms_to_timestamp(e_ms)}"
            if it.get("OST") == 1:
                it["_id"] = it.get("_id", idx + 1)
                it["narration"] = f"播放原片{it['_id']}"
            it.setdefault("_id", idx + 1)
            it.setdefault("video_id", 1)
            it.setdefault("video_name", "original.mp4")
            it.setdefault("picture", "画面")
            it.setdefault("narration", "")
            fixed.append(it)
        return fixed

    if not strict_mode:
        items = _normalize_items(items)
        print(f"[pipeline] P{pid} 宽松模式预修复完成: {len(items)} 段", flush=True)

    best_items = items

    for attempt in range(1, max_retries + 1):
        result = validate_narration_script_items(
            items, subtitle_index, [p["source"]], profile=narration_profile
        )

        if _detect_is_chinese(narration_copy, narration_language):
            generated_words = sum(
                sum(1 for c in (i.get("narration") or "") if not c.isspace())
                for i in items if i.get("OST") == 0
            )
        else:
            generated_words = sum(
                len(re.findall(r"[A-Za-z0-9']+", i.get("narration") or ""))
                for i in items if i.get("OST") == 0
            )

        err_reasons = []
        if not result.valid:
            err_reasons.extend(result.errors)
        if generated_words < min_words_threshold:
            err_reasons.append(
                f"文案字数严重不足：当前 {generated_words}，至少需 {min_words_threshold}"
                f"（目标 {target_words}）"
            )

        for item in items:
            if item.get("OST") != 0:
                continue
            try:
                start_ms, end_ms, _ = parse_script_timestamp_range(item["timestamp"])
                video_sec = (end_ms - start_ms) / 1000.0
                text = item.get("narration", "")
                audio_sec = _estimate_audio_sec(text, narration_language)
                if video_sec > (audio_sec * 2.0 + 2.0):
                    err_reasons.append(
                        f"片段 _id={item['_id']} 时间跨度 {video_sec:.1f}s "
                        f"超过解说时长 {audio_sec:.1f}s 的 2 倍"
                    )
            except Exception:
                pass

        if not err_reasons:
            print(
                f"[pipeline] P{pid} script 校验通过"
                f"（第 {attempt}/{max_retries} 次，{len(items)} 段，"
                f"{generated_words}/{target_words}）",
                flush=True,
            )
            best_items = items
            break

        err_msg = "\n".join(err_reasons[:8])
        print(f"[pipeline] P{pid} script 校验失败（第 {attempt}/{max_retries} 次）：\n{err_msg}", flush=True)

        if not strict_mode:
            print(f"[pipeline] P{pid} 宽松模式：忽略以上 {len(err_reasons)} 条错误，继续出片", flush=True)
            best_items = items
            break

        if generated_words > 0 and len(items) >= len(best_items):
            best_items = items

        if attempt >= max_retries:
            print(f"[pipeline] P{pid} script 达到最大重试次数，落盘 best_items ({len(best_items)} 段)", flush=True)
            break

        repair_prompt, repair_system = _render_prompt(
            prompt_category, "script_repair",
            parameters={
                "drama_name": title, "drama_genre": drama_genre,
                "plot_analysis": plot_analysis, "subtitle_content": srt_content,
                "invalid_script": json.dumps(data, ensure_ascii=False),
                "validation_errors": err_msg, "narration_language": narration_language,
            },
        )
        try:
            raw_fixed = _llm_text(
                repair_prompt, repair_system,
                temperature=0.6,
                response_format="json",
            )
            new_data = _validate_json(raw_fixed, fallback=data)
            if new_data and isinstance(new_data.get("items"), list) and new_data["items"]:
                data = new_data
                items = data["items"]
                if not strict_mode:
                    items = _normalize_items(items)
        except ValueError as e:
            print(f"[pipeline] P{pid} script_repair JSON 无法解析，沿用上一版: {e}", flush=True)
            continue

    final_items = items if items else best_items
    script_path = p["video_script"]
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(final_items, f, ensure_ascii=False, indent=2)

    db.complete_stage(pid, "script", output_path=script_path)
    print(f"[pipeline] P{pid} script 完成: {script_path} ({len(final_items)} 段)", flush=True)
    return script_path


def stage_tts(project, work_dir, p):
    """Stage: TTS 生成音频。输出 tts_results.json + tasks/audio_*.wav"""
    from app.services import voice

    pid = project["id"]
    db.start_stage(pid, "tts")

    list_script = _load_json(p["video_script"])
    tts_segments = [seg for seg in list_script if seg["OST"] in [0, 2]]
    print(f"[pipeline] P{pid} 需要生成TTS的片段数: {len(tts_segments)}", flush=True)

    if not tts_segments:
        _save_json(p["tts_results"], [])
        db.complete_stage(pid, "tts", output_path=p["tts_results"])
        return p["tts_results"]

    params = _build_video_clip_params(project, work_dir, p)

    tts_results = voice.tts_multiple(
        task_id=f"proj_{pid}",
        list_script=tts_segments,
        tts_engine=params.tts_engine,
        voice_name=params.voice_name,
        voice_rate=params.voice_rate,
        voice_pitch=params.voice_pitch,
    )

    _save_json(p["tts_results"], tts_results)
    db.complete_stage(pid, "tts", output_path=p["tts_results"])
    print(f"[pipeline] P{pid} tts 完成: {len(tts_results)} 段", flush=True)
    return p["tts_results"]


def stage_clip(project, work_dir, p):
    """Stage: 按脚本裁剪视频。输出 clips/*.mp4 + clip_result.json"""
    from app.services import clip_video

    pid = project["id"]
    db.start_stage(pid, "clip")

    list_script = _load_json(p["video_script"])
    tts_results = _load_json(p["tts_results"])

    _thread_local.current_work_dir = work_dir
    _thread_local.current_task_id = f"proj_{pid}"
    try:
        video_clip_result = clip_video.clip_video_unified(
            video_origin_path=p["source"],
            video_origin_paths=[p["source"]],
            script_list=list_script,
            tts_results=tts_results,
        )
    finally:
        _thread_local.current_work_dir = None
        _thread_local.current_task_id = None

    _save_json(p["clip_result"], video_clip_result)
    db.complete_stage(pid, "clip", output_path=p["clip_result"])
    print(f"[pipeline] P{pid} clip 完成: {len(video_clip_result)} 段", flush=True)
    return p["clip_result"]


def stage_merge_audio(project, work_dir, p):
    """Stage: 更新脚本时间戳 + 合并音频 + 生成字幕。"""
    from app.services import audio_merger, script_subtitle, update_script

    pid = project["id"]
    db.start_stage(pid, "merge_audio")

    list_script = _load_json(p["video_script"])
    tts_results = _load_json(p["tts_results"])

    # ⚠️ clip_result.json 是 {str(_id): path}（JSON 序列化后 int key 变 str），
    # 而 list_script 里的 _id 是 int。update_script_timestamps 用 int 匹配，
    # 不转换的话 video 字段会全部为空，merge_video 会失败。
    video_clip_result_raw = _load_json(p["clip_result"])
    video_clip_result = {
        (int(k) if isinstance(k, str) and str(k).isdigit() else k): v
        for k, v in video_clip_result_raw.items()
    }

    params = _build_video_clip_params(project, work_dir, p)

    # 1. 更新脚本里的 audio / video / subtitle / duration / editedTimeRange
    tts_map = {tr["_id"]: tr["audio_file"] for tr in tts_results}
    sub_map = {tr["_id"]: tr["subtitle_file"] for tr in tts_results}
    new_script_list = update_script.update_script_timestamps(
        list_script, video_clip_result, tts_map, sub_map
    )
    _save_json(p["new_script_list"], new_script_list)
    db.update_stage(pid, "merge_audio", progress=40)

    # 2. 合并音频
    total_duration = sum(seg["duration"] for seg in new_script_list)
    merged_audio_path = ""
    if tts_results:
        merged_audio_path = audio_merger.merge_audio_files(
            task_id=f"proj_{pid}",
            total_duration=total_duration,
            list_script=new_script_list,
        )
    db.update_stage(pid, "merge_audio", progress=75)

    # 3. 生成程序化字幕
    merged_subtitle_path = ""
    if getattr(params, "subtitle_enabled", True):
        try:
            merged_subtitle_path = script_subtitle.create_script_subtitle_file(
                task_id=f"proj_{pid}",
                list_script=new_script_list,
                original_subtitle_paths=[],
                video_origin_paths=[p["source"]],
            )
        except Exception as e:
            print(f"[pipeline] P{pid} 程序化字幕生成失败: {e}", flush=True)

    db.complete_stage(
        pid, "merge_audio",
        output_path=merged_audio_path or merged_subtitle_path,
    )
    print(
        f"[pipeline] P{pid} merge_audio 完成: audio={bool(merged_audio_path)}, "
        f"subtitle={bool(merged_subtitle_path)}",
        flush=True,
    )
    return merged_audio_path


def stage_merge_video(project, work_dir, p):
    """Stage: 合并视频流 + 音频。输出 merger.mp4"""
    from app.services import merger_video

    pid = project["id"]
    db.start_stage(pid, "merge_video")

    new_script_list = _load_json(p["new_script_list"])
    list_script = _load_json(p["video_script"])
    params = _build_video_clip_params(project, work_dir, p)

    video_clips = []
    for seg in new_script_list:
        video_path = seg.get("video")
        if video_path and os.path.exists(video_path):
            video_clips.append(video_path)
        else:
            print(
                f"[pipeline] P{pid} 片段 {seg.get('_id')} 视频文件缺失: {video_path}",
                flush=True,
            )

    if not video_clips:
        raise RuntimeError("merge_video: 没有任何可合并的视频片段")

    video_ost_list = [seg["OST"] for seg in list_script]

    merger_video.combine_clip_videos(
        output_video_path=p["merger"],
        video_paths=video_clips,
        video_ost_list=video_ost_list,
        video_aspect=params.video_aspect,
        threads=params.n_threads,
    )

    db.complete_stage(pid, "merge_video", output_path=p["merger"])
    print(f"[pipeline] P{pid} merge_video 完成: {p['merger']}", flush=True)
    return p["merger"]


def stage_final_compose(project, work_dir, p):
    """Stage: 最终合成字幕 / BGM / 配音 / 视频。输出 combined.mp4"""
    from app.services import generate_video
    from app.config.audio_config import get_recommended_volumes_for_content

    pid = project["id"]
    db.start_stage(pid, "final_compose")

    if not os.path.isfile(p["merger"]):
        raise RuntimeError(f"final_compose: merger.mp4 缺失: {p['merger']}")

    params = _build_video_clip_params(project, work_dir, p)
    list_script = _load_json(p["video_script"])

    # BGM（Sonilo 可选，失败回退随机）
    bgm_path = _resolve_bgm_path_for_worker(pid, params, p["merger"], work_dir)

    # 音量配置
    optimized_volumes = get_recommended_volumes_for_content("mixed")
    has_original_audio_segments = any(seg["OST"] == 1 for seg in list_script)

    final_tts_volume = (
        params.tts_volume
        if hasattr(params, "tts_volume") and params.tts_volume != 1.0
        else optimized_volumes["tts_volume"]
    )
    if has_original_audio_segments:
        final_original_volume = 1.0
        print(f"[pipeline] P{pid} 检测到原声片段，原声音量设为 1.0", flush=True)
    else:
        final_original_volume = (
            params.original_volume
            if hasattr(params, "original_volume") and params.original_volume != 0.7
            else optimized_volumes["original_volume"]
        )
    final_bgm_volume = (
        params.bgm_volume
        if hasattr(params, "bgm_volume") and params.bgm_volume != 0.3
        else optimized_volumes["bgm_volume"]
    )

    print(
        f"[pipeline] P{pid} 音量配置: TTS={final_tts_volume}, "
        f"原声={final_original_volume}, BGM={final_bgm_volume}",
        flush=True,
    )

    subtitle_path = p["script_subtitles"] if os.path.isfile(p["script_subtitles"]) else ""
    audio_path = p["merger_audio"] if os.path.isfile(p["merger_audio"]) else ""

    options = {
        "voice_volume": final_tts_volume,
        "bgm_volume": final_bgm_volume,
        "original_audio_volume": final_original_volume,
        "keep_original_audio": True,
        "subtitle_enabled": params.subtitle_enabled,
        "subtitle_font": params.font_name,
        "subtitle_font_size": params.font_size,
        "subtitle_color": params.text_fore_color,
        "subtitle_bg_color": None,
        "subtitle_position": params.subtitle_position,
        "custom_position": params.custom_position,
        "threads": params.n_threads,
        **_build_subtitle_mask_options_for_worker(params),
    }

    def _on_ffmpeg_progress(percent: float):
        pct = int(10 + (max(0.0, min(100.0, percent)) / 100) * 85)
        db.update_stage(pid, "final_compose", progress=pct)

    generate_video.merge_materials(
        video_path=p["merger"],
        audio_path=audio_path,
        subtitle_path=subtitle_path,
        bgm_path=bgm_path,
        output_path=p["final"],
        options=options,
        progress_callback=_on_ffmpeg_progress,
    )

    db.complete_stage(pid, "final_compose", output_path=p["final"])
    print(f"[pipeline] P{pid} final_compose 完成: {p['final']}", flush=True)
    return p["final"]


def _resolve_bgm_path_for_worker(
    pid: int, params, combined_video_path: str, work_dir: str
) -> str:
    """BGM 解析：Sonilo 可选，失败回退随机。"""
    try:
        from app.services import sonilo
    except ImportError:
        sonilo = None

    if getattr(params, "bgm_type", "") == "sonilo" and sonilo:
        save_path = os.path.join(work_dir, "tasks", "sonilo_bgm.m4a")
        bgm = sonilo.generate_bgm(combined_video_path, save_path)
        if bgm:
            return bgm
        print(f"[pipeline] P{pid} Sonilo 配乐失败，回退随机 BGM", flush=True)

    from app.utils import utils as _utils
    return _utils.get_bgm_file(
        bgm_type=getattr(params, "bgm_type", "random"),
        bgm_file=getattr(params, "bgm_file", ""),
    )


def _build_subtitle_mask_options_for_worker(params) -> dict:
    mask_configured = bool(
        getattr(params, "subtitle_enabled", True)
        and getattr(params, "subtitle_mask_enabled", False)
    )
    return {
        "subtitle_mask_enabled": mask_configured,
        "subtitle_mask_landscape_x_percent": getattr(params, "subtitle_mask_landscape_x_percent", 10.0),
        "subtitle_mask_landscape_y_percent": getattr(params, "subtitle_mask_landscape_y_percent", 78.0),
        "subtitle_mask_landscape_width_percent": getattr(params, "subtitle_mask_landscape_width_percent", 80.0),
        "subtitle_mask_landscape_height_percent": getattr(params, "subtitle_mask_landscape_height_percent", 14.0),
        "subtitle_mask_landscape_blur_radius": getattr(params, "subtitle_mask_landscape_blur_radius", 18),
        "subtitle_mask_landscape_opacity_percent": getattr(params, "subtitle_mask_landscape_opacity_percent", 82),
        "subtitle_mask_portrait_x_percent": getattr(params, "subtitle_mask_portrait_x_percent", 8.0),
        "subtitle_mask_portrait_y_percent": getattr(params, "subtitle_mask_portrait_y_percent", 79.0),
        "subtitle_mask_portrait_width_percent": getattr(params, "subtitle_mask_portrait_width_percent", 84.0),
        "subtitle_mask_portrait_height_percent": getattr(params, "subtitle_mask_portrait_height_percent", 16.0),
        "subtitle_mask_portrait_blur_radius": getattr(params, "subtitle_mask_portrait_blur_radius", 26),
        "subtitle_mask_portrait_opacity_percent": getattr(params, "subtitle_mask_portrait_opacity_percent", 84),
        "subtitle_position_landscape_y_percent": getattr(params, "subtitle_position_landscape_y_percent", 85.0),
        "subtitle_position_portrait_y_percent": getattr(params, "subtitle_position_portrait_y_percent", 82.0),
    }


STAGE_FUNCS = {
    "download":       stage_download,
    "transcribe":     stage_transcribe,
    "script":         stage_script,
    "tts":            stage_tts,
    "clip":           stage_clip,
    "merge_audio":    stage_merge_audio,
    "merge_video":    stage_merge_video,
    "final_compose":  stage_final_compose,
}


# ============================================================
# 八、流水线
# ============================================================

def _initialize_project_once(project: dict, storage_root: str, phase: str) -> dict:
    pid = project["id"]

    print(f"[{phase}] P{pid} 获取视频信息...", flush=True)
    info = _fetch_video_info(project["youtube_url"])
    original_title = (info.get("title") or "").strip()

    if not original_title:
        original_title = project.get("youtube_id") or f"video_{pid}"

    print(f"[{phase}] P{pid} 原标题: {original_title}", flush=True)
    cn_title = _translate_title_to_chinese(original_title) or original_title
    if not cn_title:
        cn_title = f"video_{pid}"
    print(f"[{phase}] P{pid} 中文标题: {cn_title}", flush=True)

    work_dir = make_work_dir(storage_root, pid, cn_title)
    p = paths(work_dir)

    duration_sec = info.get("duration")
    width = info.get("width")
    height = info.get("height")
    resolution = f"{width}x{height}" if width and height else None

    db.update_project(
        pid,
        title=cn_title,
        work_dir=work_dir,
        slug=os.path.basename(work_dir),
        source_path=p["source"],
        subtitle_path=p["subtitle"],
        final_path=p["final"],
        **({"duration_sec": float(duration_sec)} if duration_sec else {}),
        **({"resolution": resolution} if resolution else {}),
    )

    update_meta(
        work_dir,
        project_id=pid,
        title=cn_title,
        original_title=original_title,
        youtube_url=project["youtube_url"],
        youtube_id=project.get("youtube_id") or "",
        status="running",
        created_at=datetime.now().isoformat(),
    )

    project.update({
        "title": cn_title,
        "work_dir": work_dir,
        "source_path": p["source"],
        "subtitle_path": p["subtitle"],
        "final_path": p["final"],
    })
    if duration_sec:
        project["duration_sec"] = float(duration_sec)
    if resolution:
        project["resolution"] = resolution

    return project


def run_phase(project: dict, phase: str, storage_root: str):
    pid = project["id"]

    if not project.get("work_dir"):
        try:
            project = _initialize_project_once(project, storage_root, phase)
        except Exception as e:
            print(f"[{phase}] P{pid} 初始化失败: {e}", flush=True)
            traceback.print_exc()
            raise

    work_dir = project["work_dir"]
    p = paths(work_dir)

    stages = db.get_stages(pid)
    completed = {s["stage_name"] for s in stages if s["status"] == "completed"}

    stop_hb = threading.Event()
    threading.Thread(target=heartbeat_thread, args=(pid, stop_hb), daemon=True).start()

    current_stage = None
    try:
        for stage_name in PHASE_STAGES[phase]:
            if stage_name in completed:
                print(f"[{phase}] P{pid} 跳过已完成的 {stage_name}", flush=True)
                continue
            current_stage = stage_name
            STAGE_FUNCS[stage_name](project, work_dir, p)

        next_phase = PHASE_NEXT.get(phase, "done")
        if next_phase == "done":
            db.update_project(pid, status="completed", phase="done",
                              progress=100, finished_at=datetime.now())
            update_meta(work_dir, status="completed")
            print(f"[{phase}] P{pid} 全部完成", flush=True)
        else:
            db.update_project(pid, phase=next_phase, status="queued", progress=0)
            print(f"[{phase}] P{pid} phase 完成 → {next_phase}", flush=True)

    except Exception as e:
        stage_label = current_stage or "unknown"
        print(f"[{phase}] P{pid} 失败于 {stage_label}: {e}", flush=True)
        traceback.print_exc()
        try: db.fail_stage(pid, stage_label, str(e))
        except Exception: pass
        try: db.mark_failed(pid, stage_label, str(e))
        except Exception: pass
        try: update_meta(work_dir, status="failed", error=str(e))
        except Exception: pass
        raise
    finally:
        stop_hb.set()


def run_phase_loop(phase: str, storage_root: str):
    worker_id = f"{HOSTNAME}-{os.getpid()}-{phase}"
    print(f"[{phase}] 队列进程启动 worker_id={worker_id}", flush=True)

    stop_event = threading.Event()

    def recovery_loop():
        def _recover_once():
            try:
                db.reset_stale_tasks(timeout_seconds=STALE_TIMEOUT)
                db.requeue_failed_tasks(max_age_hours=FAILED_REQUEUE_HOURS)
            except Exception as e:
                print(f"[{phase}] 恢复扫描失败: {e}", flush=True)

        _recover_once()
        while not stop_event.wait(STALE_RESET_INTERVAL):
            _recover_once()

    threading.Thread(target=recovery_loop, daemon=True).start()

    try:
        while not stop_event.is_set():
            try:
                project = db.claim_task(worker_id, phase=phase)
                if project:
                    print(f"[{phase}] 抢到 P{project['id']} {project['title']}", flush=True)
                    try:
                        run_phase(project, phase, storage_root)
                    except Exception:
                        pass
                else:
                    time.sleep(3)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[{phase}] 主循环异常: {e}", flush=True)
                traceback.print_exc()
                time.sleep(3)
    finally:
        stop_event.set()


# ============================================================
# 九、主入口
# ============================================================

def _ensure_llm_providers_registered():
    if LLMServiceManager.is_registered():
        return
    from app.services.llm.providers import register_all_providers
    register_all_providers()
    logger.info(
        f"[Worker] LLM providers 已注册: "
        f"{LLMServiceManager.get_registered_providers_info()}"
    )


def main():
    parser = argparse.ArgumentParser(description="NarratoAI Worker")
    parser.add_argument("--phase", default="all",
                        help="preprocess | pipeline | postprocess | all")
    parser.add_argument("--no-retry-failed", action="store_true",
                        help="启动时不自动重试 failed 任务（自动化部署时建议开启）")

    # 项目管理命令（执行后退出，不进队列循环）
    admin = parser.add_argument_group("项目管理命令（执行后退出）")
    admin.add_argument("--retry-project", type=int, metavar="PID",
                       help="重试指定项目（保留已完成 stage，从失败 stage 继续）")
    admin.add_argument("--reset-project", type=int, metavar="PID",
                       help="重置指定项目（清空全部或 --from-stage 起的 stage）")
    admin.add_argument("--from-stage", metavar="STAGE",
                       help="配合 --reset-project：从该 stage 起重置（含）")
    admin.add_argument("--clean-artifacts", action="store_true",
                       help="配合 --reset-project：同时删除产物文件")
    args = parser.parse_args()

    phase_arg = args.phase.strip().lower()

    # ---- 管理命令优先：执行后退出 ----
    if args.retry_project is not None or args.reset_project is not None:
        db.init()
        try:
            if args.retry_project is not None:
                result = db.retry_project(args.retry_project)
            else:
                result = db.reset_project(
                    args.reset_project,
                    from_stage=args.from_stage,
                    clean_artifacts=args.clean_artifacts,
                )
        finally:
            db.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0 if result.get("ok") else 1)

    # ---- 队列循环 ----
    if phase_arg == "all":
        phases = list(ACTIVE_PHASES)
        print(f"[Worker] 启动 {len(phases)} 个子进程: {phases}", flush=True)

        procs = []
        for p in phases:
            cmd = [sys.executable, os.path.abspath(__file__), "--phase", p]
            if args.no_retry_failed:
                cmd.append("--no-retry-failed")
            print(f"[Worker]   → {' '.join(cmd)}", flush=True)
            procs.append(subprocess.Popen(cmd))

        try:
            for proc in procs:
                proc.wait()
        except KeyboardInterrupt:
            print("[Worker] 收到 Ctrl+C，终止所有子进程", flush=True)
            for proc in procs:
                proc.terminate()
            for proc in procs:
                proc.wait()
        return

    if phase_arg not in ACTIVE_PHASES:
        print(f"[Worker] 无效的 phase: {phase_arg}，可用: {ACTIVE_PHASES}", flush=True)
        sys.exit(1)

    db.init()
    _ensure_llm_providers_registered()

    # 启动时无条件重试 failed 任务（一次性，不影响周期性 recovery 扫描）
    if not args.no_retry_failed:
        n = db.requeue_failed_for_phase(phase_arg)
        if n > 0:
            print(f"[Worker] 启动时已重置 {n} 个 {phase_arg} 的 failed 任务", flush=True)
    else:
        print("[Worker] --no-retry-failed 已启用，跳过启动时的 failed 任务重置", flush=True)

    storage_root = get_storage_root()

    print(f"[Worker] phase={phase_arg}", flush=True)
    print(f"[Worker] 存储根目录: {storage_root}", flush=True)

    try:
        run_phase_loop(phase_arg, storage_root)
    except KeyboardInterrupt:
        print(f"[{phase_arg}] 收到 Ctrl+C，退出", flush=True)
    finally:
        db.close()

if __name__ == "__main__":
    main()