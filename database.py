# database.py
import glob
import os
import shutil
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor
from loguru import logger

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ken:886482++@localhost:5432/narrato",
)


class Database:
    """统一数据库操作类（连接池 + 事务 + CRUD）"""

    DEFAULT_STAGES = (
        "download",
        "transcribe",
        "script",
        "tts",
        "clip",
        "merge_audio",
        "merge_video",
        "final_compose",
        "publish",
    )

    STAGE_TO_PHASE = {
        "download":       "preprocess",
        "transcribe":     "preprocess",
        "script":         "pipeline",
        "tts":            "pipeline",
        "clip":           "pipeline",
        "merge_audio":    "pipeline",
        "merge_video":    "pipeline",
        "final_compose":  "pipeline",
        "publish":        "postprocess",
    }
    ALL_PHASES = ("preprocess", "pipeline", "postprocess")

    # 每个 stage 的产物路径（相对 work_dir），支持 glob 通配符
    STAGE_ARTIFACTS = {
        "download":       ["original.mp4"],
        "transcribe":     ["auto.srt"],
        "script":         ["script/video_script.json", "script/narration.txt"],
        "tts":            ["tasks/tts_results.json", "tasks/audio_*.wav"],
        "clip":           ["tasks/clip_result.json", "clips/*.mp4"],
        "merge_audio":    ["tasks/new_script_list.json", "tasks/merger_audio.mp3",
                           "tasks/script_subtitles.srt"],
        "merge_video":    ["tasks/merger.mp4"],
        "final_compose":  ["tasks/combined.mp4"],
        "publish":        ["cover/*"],
    }

    PROJECT_FIELDS = {
        "title",
        "status", "progress", "slug", "work_dir", "phase",
        "priority", "retry_count", "max_retries",
        "worker_id", "heartbeat",
        "error_msg", "error_stage",
        "duration_sec", "resolution", "source_size",
        "source_path", "subtitle_path", "final_path", "cover_path",
        "started_at", "finished_at", "scheduled_at",
        "options", "meta",
    }
    STAGE_FIELDS = {"status", "progress", "error_msg", "output_path",
                    "started_at", "finished_at"}

    def __init__(self, dsn: str = DATABASE_URL, minconn: int = 1, maxconn: int = 10):
        self.dsn = dsn
        self.minconn = minconn
        self.maxconn = maxconn
        self._pool: Optional[pg_pool.ThreadedConnectionPool] = None
        self._initialized = False

    # ---------- 生命周期 ----------

    def init(self) -> "Database":
        if self._initialized:
            return self
        self._pool = pg_pool.ThreadedConnectionPool(
            self.minconn, self.maxconn, dsn=self.dsn
        )
        self._create_tables()
        self._initialized = True
        logger.info(f"[DB] 初始化完成 (dsn={self._masked_dsn()})")
        return self

    def close(self):
        if self._pool:
            self._pool.closeall()
            self._pool = None
            self._initialized = False

    def _ensure(self):
        if not self._initialized:
            self.init()

    def _masked_dsn(self):
        try:
            from urllib.parse import urlparse
            u = urlparse(self.dsn)
            if u.password:
                return self.dsn.replace(u.password, "***")
        except Exception:
            pass
        return self.dsn

    @contextmanager
    def _conn(self):
        self._ensure()
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def _transaction(self):
        self._ensure()
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    # ---------- 建表（不变） ----------

    def _create_tables(self):
        sql = """
        CREATE TABLE IF NOT EXISTS projects (
            id              SERIAL PRIMARY KEY,
            title           VARCHAR(255) NOT NULL,
            youtube_url     VARCHAR(512) NOT NULL,
            youtube_id      VARCHAR(32),
            slug            VARCHAR(120),
            work_dir        TEXT,

            status          VARCHAR(50) NOT NULL DEFAULT 'queued',
            phase           VARCHAR(20) NOT NULL DEFAULT 'preprocess',
            progress        INT NOT NULL DEFAULT 0,

            priority        INT NOT NULL DEFAULT 0,
            retry_count     INT NOT NULL DEFAULT 0,
            max_retries     INT NOT NULL DEFAULT 3,
            worker_id       VARCHAR(64),
            heartbeat       TIMESTAMP,

            error_msg       TEXT,
            error_stage     VARCHAR(50),

            duration_sec    FLOAT,
            resolution      VARCHAR(20),
            source_size     BIGINT,

            source_path     TEXT,
            subtitle_path   TEXT,
            final_path      TEXT,
            cover_path      TEXT,

            options         JSONB NOT NULL DEFAULT '{}'::jsonb,
            meta            JSONB NOT NULL DEFAULT '{}'::jsonb,

            created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at      TIMESTAMP,
            finished_at     TIMESTAMP,
            scheduled_at    TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS project_stages (
            id              SERIAL PRIMARY KEY,
            project_id      INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            stage_name      VARCHAR(50) NOT NULL,
            status          VARCHAR(20) NOT NULL DEFAULT 'pending',
            progress        INT NOT NULL DEFAULT 0,
            error_msg       TEXT,
            output_path     TEXT,
            started_at      TIMESTAMP,
            finished_at     TIMESTAMP,
            updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(project_id, stage_name)
        );

        ALTER TABLE projects ADD COLUMN IF NOT EXISTS youtube_id VARCHAR(32);
        ALTER TABLE projects ADD COLUMN IF NOT EXISTS phase VARCHAR(20) NOT NULL DEFAULT 'preprocess';

        CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_youtube_id
            ON projects (youtube_id) WHERE youtube_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_projects_dispatch
            ON projects (status, phase, priority DESC, id ASC);
        CREATE INDEX IF NOT EXISTS idx_projects_heartbeat
            ON projects (heartbeat) WHERE status = 'running';
        CREATE INDEX IF NOT EXISTS idx_stages_lookup
            ON project_stages (project_id, stage_name);
        """
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    # ---------- CRUD（不变） ----------

    def create_project(self, title: str, youtube_url: str,
                       youtube_id: Optional[str] = None,
                       options: Optional[dict] = None) -> Optional[int]:
        try:
            with self._transaction() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        INSERT INTO projects (title, youtube_url, youtube_id, options)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (title, youtube_url, youtube_id, options or {}),
                    )
                    project_id = cur.fetchone()["id"]
                    for stage in self.DEFAULT_STAGES:
                        cur.execute(
                            "INSERT INTO project_stages (project_id, stage_name) VALUES (%s, %s)",
                            (project_id, stage),
                        )
            logger.info(f"[DB] create_project P{project_id}")
            return project_id
        except psycopg2.IntegrityError as e:
            if "idx_projects_youtube_id" in str(e) or "youtube_id" in str(e):
                logger.warning(f"[DB] youtube_id={youtube_id} 已存在，拒绝重复建任务")
                return None
            raise

    def find_by_youtube_id(self, youtube_id: str) -> Optional[dict]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM projects WHERE youtube_id=%s", (youtube_id,))
                return cur.fetchone()

    def get_project(self, project_id: int, with_stages: bool = True) -> Optional[dict]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM projects WHERE id=%s", (project_id,))
                project = cur.fetchone()
                if not project:
                    return None
                if with_stages:
                    cur.execute(
                        "SELECT * FROM project_stages WHERE project_id=%s ORDER BY id",
                        (project_id,),
                    )
                    project["stages"] = cur.fetchall()
                return project

    def list_projects(self, limit: int = 200) -> list:
        sql = """
        SELECT p.*,
          COALESCE(
            json_agg(json_build_object(
              'stage_name',  s.stage_name,
              'status',      s.status,
              'progress',    s.progress,
              'output_path', s.output_path,
              'error_msg',   s.error_msg
            ) ORDER BY s.id) FILTER (WHERE s.id IS NOT NULL),
            '[]'::json
          ) AS stages
        FROM projects p
        LEFT JOIN project_stages s ON p.id = s.project_id
        GROUP BY p.id
        ORDER BY p.id DESC
        LIMIT %s
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (limit,))
                return cur.fetchall()

    def update_project(self, project_id: int, **kwargs) -> None:
        fields, values = [], []
        for key, val in kwargs.items():
            if key not in self.PROJECT_FIELDS:
                logger.warning(f"[DB] update_project 忽略未知字段: {key}")
                continue
            fields.append(f"{key}=%s")
            values.append(val)
        if not fields:
            return
        fields.append("updated_at=NOW()")
        values.append(project_id)
        sql = f"UPDATE projects SET {', '.join(fields)} WHERE id=%s"
        try:
            with self._transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, values)
        except Exception as e:
            logger.error(f"[DB] update_project P{project_id} 失败: {e}")

    def mark_failed(self, project_id: int, stage_name: str, error_msg: str) -> None:
        try:
            with self._transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE projects
                        SET status='failed',
                            error_stage=%s,
                            error_msg=%s,
                            retry_count=retry_count+1,
                            finished_at=NOW(),
                            updated_at=NOW()
                        WHERE id=%s
                        """,
                        (stage_name, error_msg, project_id),
                    )
        except Exception as e:
            logger.error(f"[DB] mark_failed P{project_id} 失败: {e}")

    def delete_project(self, project_id: int) -> bool:
        try:
            with self._transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM projects WHERE id=%s", (project_id,))
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"[DB] delete_project P{project_id} 失败: {e}")
            return False

    # ---------- 调度（不变） ----------

    def claim_task(self, worker_id: str, phase: str) -> Optional[dict]:
        sql = """
        UPDATE projects
        SET status='running',
            worker_id=%s,
            started_at=COALESCE(started_at, NOW()),
            heartbeat=NOW(),
            updated_at=NOW()
        WHERE id = (
            SELECT id FROM projects
            WHERE status='queued'
              AND phase=%s
              AND retry_count < max_retries
            ORDER BY priority DESC, id ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, title, youtube_url, youtube_id, work_dir, phase, options,
                  duration_sec, resolution, source_size
        """
        try:
            with self._transaction() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, (worker_id, phase))
                    return cur.fetchone()
        except Exception as e:
            logger.error(f"[DB] claim_task({phase}) 失败: {e}")
            return None

    def heartbeat(self, project_id: int) -> None:
        try:
            with self._transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE projects SET heartbeat=NOW() WHERE id=%s", (project_id,))
        except Exception as e:
            logger.warning(f"[DB] heartbeat P{project_id} 失败: {e}")

    # ---------- 自动恢复（不变） ----------

    def reset_stale_tasks(self, timeout_seconds: int = 120) -> int:
        sql = f"""
        UPDATE projects
        SET status='queued',
            worker_id=NULL,
            retry_count=retry_count+1,
            error_msg=COALESCE(error_msg,'') || ' [heartbeat timeout, requeued]',
            updated_at=NOW()
        WHERE status='running'
          AND (heartbeat IS NULL OR heartbeat < NOW() - INTERVAL '{int(timeout_seconds)} seconds')
        """
        try:
            with self._transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    n = cur.rowcount
            if n:
                logger.warning(f"[DB] 重置 {n} 个心跳超时任务")
            return n
        except Exception as e:
            logger.error(f"[DB] reset_stale_tasks 失败: {e}")
            return 0

    def requeue_failed_tasks(self, max_age_hours: int = 6) -> int:
        sql = f"""
        UPDATE projects
        SET status='queued',
            worker_id=NULL,
            error_msg=COALESCE(error_msg,'') || ' [auto requeue]',
            updated_at=NOW()
        WHERE status='failed'
          AND retry_count < max_retries
          AND (finished_at IS NULL OR finished_at > NOW() - INTERVAL '{int(max_age_hours)} hours')
        """
        try:
            with self._transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    n = cur.rowcount
            if n:
                logger.warning(f"[DB] 自动重新入队 {n} 个 failed 任务")
            return n
        except Exception as e:
            logger.error(f"[DB] requeue_failed_tasks 失败: {e}")
            return 0

    # ---------- 阶段（不变） ----------

    def get_stages(self, project_id: int) -> list:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM project_stages WHERE project_id=%s ORDER BY id",
                    (project_id,),
                )
                return cur.fetchall()

    def start_stage(self, project_id: int, stage_name: str) -> None:
        try:
            with self._transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO project_stages
                            (project_id, stage_name, status, progress, started_at)
                        VALUES (%s, %s, 'running', 0, NOW())
                        ON CONFLICT (project_id, stage_name) DO UPDATE
                        SET status='running',
                            progress=0,
                            error_msg=NULL,
                            started_at=COALESCE(project_stages.started_at, NOW()),
                            updated_at=NOW()
                        """,
                        (project_id, stage_name),
                    )
        except Exception as e:
            logger.error(f"[DB] start_stage P{project_id}/{stage_name} 失败: {e}")

    def update_stage(self, project_id: int, stage_name: str, **kwargs) -> None:
        fields, values = [], []
        for key, val in kwargs.items():
            if key not in self.STAGE_FIELDS:
                logger.warning(f"[DB] update_stage 忽略未知字段: {key}")
                continue
            fields.append(f"{key}=%s")
            values.append(val)
        if not fields:
            return
        fields.append("updated_at=NOW()")
        if kwargs.get("status") in ("completed", "failed", "skipped"):
            fields.append("finished_at=NOW()")
        values.extend([project_id, stage_name])
        sql = (
            f"UPDATE project_stages SET {', '.join(fields)} "
            f"WHERE project_id=%s AND stage_name=%s"
        )
        try:
            with self._transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, values)
        except Exception as e:
            logger.error(f"[DB] update_stage P{project_id}/{stage_name} 失败: {e}")

    def complete_stage(self, project_id: int, stage_name: str,
                       output_path: Optional[str] = None) -> None:
        self.update_stage(project_id, stage_name,
                          status="completed", progress=100, output_path=output_path)
        self._recompute_project_progress(project_id)

    def fail_stage(self, project_id: int, stage_name: str, error_msg: str) -> None:
        self.update_stage(project_id, stage_name,
                          status="failed", error_msg=error_msg)

    def _recompute_project_progress(self, project_id: int) -> None:
        try:
            with self._transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE projects
                        SET progress = COALESCE((
                            SELECT ROUND(
                                100.0 * COUNT(*) FILTER (WHERE status IN ('completed', 'skipped'))
                                / NULLIF(COUNT(*), 0)
                            )::int
                            FROM project_stages
                            WHERE project_id = %s
                        ), 0),
                        updated_at = NOW()
                        WHERE id = %s
                        """,
                        (project_id, project_id),
                    )
        except Exception as e:
            logger.error(f"[DB] _recompute_project_progress P{project_id} 失败: {e}")

    # ============================================================
    # 项目生命周期管理：重试 / 重置
    # ============================================================

    @classmethod
    def _stages_from(cls, from_stage: str) -> List[str]:
        """返回 from_stage 及之后的所有 stage 名称（按 DEFAULT_STAGES 顺序）。"""
        stages = list(cls.DEFAULT_STAGES)
        try:
            idx = stages.index(from_stage)
        except ValueError:
            raise ValueError(f"未知 stage: {from_stage}，可用: {stages}")
        return stages[idx:]

    def retry_project(self, project_id: int, force: bool = False) -> Dict[str, Any]:
        """重试任务：保留已完成 stage，从失败 stage 继续。

        适用：
          - status='failed'（业务异常）
          - status='running' 且 force=True（强制抢占已卡住的任务）

        拒绝：
          - status='completed'（用 reset_project）
          - status='queued'（已在队列中）
          - status='running' 且 force=False（可能真在跑）
        """
        with self._transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, status, phase FROM projects WHERE id=%s FOR UPDATE",
                    (project_id,),
                )
                row = cur.fetchone()
                if not row:
                    return {"ok": False, "reason": "not_found"}

                status = row["status"]
                if status == "completed":
                    return {"ok": False, "reason": "completed",
                            "hint": "项目已完成，用 reset_project 重置"}
                if status == "queued":
                    return {"ok": False, "reason": "already_queued"}
                if status == "running" and not force:
                    return {"ok": False, "reason": "running",
                            "hint": "任务可能正在运行，用 force=True 强制重试"}

                cur.execute(
                    """
                    UPDATE projects
                    SET status='queued',
                        worker_id=NULL,
                        retry_count=0,
                        error_msg=NULL,
                        error_stage=NULL,
                        finished_at=NULL,
                        heartbeat=NULL,
                        updated_at=NOW()
                    WHERE id=%s
                    """,
                    (project_id,),
                )
        logger.info(f"[DB] 重试 P{project_id}（保留已完成 stage，从失败 stage 继续）")
        return {"ok": True, "project_id": project_id, "phase": row["phase"]}

    def reset_project(
        self,
        project_id: int,
        from_stage: Optional[str] = None,
        clean_artifacts: bool = False,
    ) -> Dict[str, Any]:
        """重置任务：清空 from_stage 及之后所有 stage 的状态（默认全部）。

        Args:
            project_id: 项目 ID
            from_stage: 从哪个 stage 起重置（含）；None 表示从 download 起全部重置
            clean_artifacts: 是否删除对应产物文件（默认 False，只改 DB 状态）
        """
        with self._transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, status, phase, work_dir FROM projects WHERE id=%s FOR UPDATE",
                    (project_id,),
                )
                row = cur.fetchone()
                if not row:
                    return {"ok": False, "reason": "not_found"}

                reset_stages = (
                    self._stages_from(from_stage) if from_stage
                    else list(self.DEFAULT_STAGES)
                )
                new_phase = self.STAGE_TO_PHASE.get(reset_stages[0], "preprocess")

                cur.execute(
                    """
                    UPDATE project_stages
                    SET status='pending',
                        progress=0,
                        error_msg=NULL,
                        output_path=NULL,
                        started_at=NULL,
                        finished_at=NULL,
                        updated_at=NOW()
                    WHERE project_id=%s AND stage_name = ANY(%s)
                    """,
                    (project_id, reset_stages),
                )

                cur.execute(
                    """
                    UPDATE projects
                    SET status='queued',
                        phase=%s,
                        progress=0,
                        worker_id=NULL,
                        retry_count=0,
                        error_msg=NULL,
                        error_stage=NULL,
                        started_at=NULL,
                        finished_at=NULL,
                        heartbeat=NULL,
                        updated_at=NOW()
                    WHERE id=%s
                    """,
                    (new_phase, project_id),
                )

        artifacts_removed: List[str] = []
        artifacts_failed: List[str] = []
        if clean_artifacts and row["work_dir"]:
            artifacts_removed, artifacts_failed = self._clean_stage_artifacts(
                row["work_dir"], reset_stages
            )

        logger.info(
            f"[DB] 重置 P{project_id} from_stage={from_stage or 'ALL'} "
            f"stages={reset_stages} clean={clean_artifacts}"
        )
        return {
            "ok": True,
            "project_id": project_id,
            "stages_reset": reset_stages,
            "artifacts_removed": artifacts_removed,
            "artifacts_failed": artifacts_failed,
        }

    def _clean_stage_artifacts(
        self, work_dir: str, stage_names: Sequence[str]
    ) -> Tuple[List[str], List[str]]:
        """删除指定 stage 的产物文件，返回 (removed, failed)。"""
        removed: List[str] = []
        failed: List[str] = []
        for stage in stage_names:
            for pattern in self.STAGE_ARTIFACTS.get(stage, []):
                full_pattern = os.path.join(work_dir, pattern)
                for path in glob.glob(full_pattern):
                    try:
                        if os.path.isfile(path):
                            os.remove(path)
                            removed.append(path)
                        elif os.path.isdir(path):
                            shutil.rmtree(path)
                            removed.append(path)
                    except OSError as e:
                        failed.append(f"{path}: {e}")
        return removed, failed

    def requeue_failed_for_phase(self, phase: str) -> int:
        """启动时无条件重试指定 phase 的所有 failed 任务（不检查 retry_count / finished_at）。"""
        sql = """
        UPDATE projects
        SET status='queued',
            worker_id=NULL,
            retry_count=0,
            finished_at=NULL,
            error_msg=COALESCE(error_msg, '') || ' [requeued on worker startup]',
            updated_at=NOW()
        WHERE status='failed' AND phase=%s
        """
        try:
            with self._transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (phase,))
                    n = cur.rowcount
            if n:
                logger.warning(f"[DB] 启动时无条件重试 {n} 个 {phase} 的 failed 任务")
            return n
        except Exception as e:
            logger.error(f"[DB] requeue_failed_for_phase 失败: {e}")
            return 0


db = Database(DATABASE_URL)


def init_db() -> Database:
    return db.init()


if __name__ == "__main__":
    init_db()
    print("[DB] tables ready")
    db.close()