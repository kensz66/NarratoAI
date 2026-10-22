# app/services/worker_paths.py
"""
项目工作目录路径管理。

目录结构：
    {WORKER_STORAGE_DIR}/{pid}_{中文标题}/
    ├── original.mp4              ← 下载的源视频
    ├── auto.srt                  ← 原片 Whisper 转写字幕
    ├── meta.json                 ← 项目元数据
    ├── cover/                    ← 封面
    ├── script/
    │   ├── narration.txt         ← LLM 解说词
    │   └── video_script.json     ← 编排脚本
    ├── clips/                    ← clip_video 裁剪的片段
    └── tasks/
        ├── tts_results.json      ← TTS 结果（stage: tts 输出）
        ├── audio_*.wav
        ├── clip_result.json      ← 裁剪结果（stage: clip 输出）
        ├── new_script_list.json  ← 更新后的脚本（stage: merge_audio 输出）
        ├── merger_audio.mp3      ← 合并音频（stage: merge_audio 输出）
        ├── script_subtitles.srt  ← 程序化字幕（stage: merge_audio 输出）
        ├── merger.mp4            ← 合并后的视频（stage: merge_video 输出）
        └── combined.mp4          ← 最终成品（stage: final_compose 输出）
"""
import os
import re
import json
from typing import Dict


def slugify(text: str, max_len: int = 50) -> str:
    text = (text or "untitled").strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._")[:max_len] or "untitled"
    return text


def make_work_dir(storage_root: str, project_id: int, title: str) -> str:
    session = f"{project_id}_{slugify(title)}"
    work_dir = os.path.abspath(os.path.join(storage_root, session))
    for sub in ("script", "tasks", "clips", "cover"):
        os.makedirs(os.path.join(work_dir, sub), exist_ok=True)
    return work_dir


def paths(work_dir: str) -> Dict[str, str]:
    """返回 work_dir 内所有标准路径。

    拆 stage 后新增 6 个字段，stage 之间通过文件系统传递状态：
    - tts_results / clip_result / new_script_list：中间 JSON
    - merger / merger_audio / script_subtitles：中间媒体
    """
    tasks_dir = os.path.join(work_dir, "tasks")
    return {
        "root":              work_dir,
        "source":            os.path.join(work_dir, "original.mp4"),
        "subtitle":          os.path.join(work_dir, "auto.srt"),
        "script_dir":        os.path.join(work_dir, "script"),
        "narration":         os.path.join(work_dir, "script", "narration.txt"),
        "video_script":      os.path.join(work_dir, "script", "video_script.json"),
        "tasks_dir":         tasks_dir,
        "clips_dir":         os.path.join(work_dir, "clips"),
        "cover_dir":         os.path.join(work_dir, "cover"),
        "meta":              os.path.join(work_dir, "meta.json"),

        # stage 产物
        "tts_results":       os.path.join(tasks_dir, "tts_results.json"),
        "clip_result":       os.path.join(tasks_dir, "clip_result.json"),
        "new_script_list":   os.path.join(tasks_dir, "new_script_list.json"),
        "merger":            os.path.join(tasks_dir, "merger.mp4"),
        "merger_audio":      os.path.join(tasks_dir, "merger_audio.mp3"),
        "script_subtitles":  os.path.join(tasks_dir, "script_subtitles.srt"),
        "final":             os.path.join(tasks_dir, "combined.mp4"),
    }


def load_meta(work_dir: str) -> Dict:
    meta_path = os.path.join(work_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_meta(work_dir: str, meta: Dict) -> None:
    meta_path = os.path.join(work_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def update_meta(work_dir: str, **kwargs) -> Dict:
    meta = load_meta(work_dir)
    meta.update(kwargs)
    save_meta(work_dir, meta)
    return meta