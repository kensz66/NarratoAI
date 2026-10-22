# app/services/youtube_downloader.py
import os
import glob
from typing import Callable, Optional

from loguru import logger


def download_youtube_video(
    url: str,
    save_dir: str,
    progress_callback: Optional[Callable[[int], None]] = None,
    max_height: Optional[int] = None,
    cookies_file: str = "",
) -> str:
    """
    用 yt-dlp 下载 YouTube 视频（同步，带进度回调）。

    Args:
        url: YouTube 链接
        save_dir: 保存目录（通常是 work_dir/source）
        progress_callback: 进度回调 0-100
        max_height: 最大高度限制，例如 1080 表示不超过 1080p；None 为不限制
        cookies_file: 可选 cookies.txt 路径（处理需要登录的视频）

    Returns:
        下载得到的 mp4 绝对路径
    """
    import yt_dlp

    os.makedirs(save_dir, exist_ok=True)

    # 输出模板：{save_dir}/<title>.mp4
    out_tmpl = os.path.join(save_dir, "%(title).100s.%(ext)s")

    # 格式选择
    if max_height:
        fmt = (
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
            f"/bestvideo[height<={max_height}]+bestaudio"
            f"/best[height<={max_height}]"
        )
    else:
        fmt = (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]"
            "/bestvideo+bestaudio"
            "/best"
        )

    def _hook(d):
        if not progress_callback:
            return
        try:
            status = d.get("status")
            if status == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded = d.get("downloaded_bytes") or 0
                if total > 0:
                    pct = int(downloaded / total * 100)
                    progress_callback(min(pct, 99))
            elif status == "finished":
                # 下载完成，等待 ffmpeg 合并
                progress_callback(99)
        except Exception as e:
            logger.debug(f"yt-dlp progress hook error: {e}")

    ydl_opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": out_tmpl,
        "progress_hooks": [_hook],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "concurrent_fragment_downloads": 4,
        "windowsfilenames": True,
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
    }

    if cookies_file and os.path.isfile(cookies_file):
        ydl_opts["cookiefile"] = cookies_file

    logger.info(f"开始下载: {url} -> {save_dir}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        mp4_path = os.path.splitext(filename)[0] + ".mp4"

        # 兜底：找最新 mp4
        if not os.path.isfile(mp4_path):
            candidates = sorted(
                glob.glob(os.path.join(save_dir, "*.mp4")),
                key=os.path.getmtime,
                reverse=True,
            )
            if not candidates:
                raise RuntimeError(f"下载完成但未找到 mp4 文件: {save_dir}")
            mp4_path = candidates[0]

    logger.success(f"下载完成: {mp4_path}")
    if progress_callback:
        progress_callback(100)
    return mp4_path