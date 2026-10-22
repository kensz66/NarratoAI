// front/app/projects/page.tsx
'use client';

import { useMemo, useState } from 'react';
import useSWR from 'swr';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  IconButton,
  Chip,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tooltip,
  Stack,
  CircularProgress,
} from '@mui/material';
import {
  DragIndicator,
  Edit,
  Delete,
  Add,
  Refresh,
} from '@mui/icons-material';
import { fetcher } from '@/lib/api';

// ==================== 类型 ====================

type Stage = {
  stage_name: string;
  status: string;
  progress: number;
  output_path?: string | null;
  error_msg?: string | null;
};

type Project = {
  id: number;
  title: string;
  youtube_url: string;
  status: string;
  progress: number;
  work_dir?: string | null;
  error_msg?: string | null;
  error_stage?: string | null;
  duration_sec?: number | null;
  resolution?: string | null;
  source_size?: number | null;
  created_at?: string;
  updated_at?: string;
  started_at?: string | null;
  finished_at?: string | null;
  stages?: Stage[];
};

// ==================== 常量 ====================

const STATUS_MAP: Record<string, { label: string; color: any }> = {
  queued:    { label: '排队中', color: 'info' },
  running:   { label: '处理中', color: 'warning' },
  paused:    { label: '已暂停', color: 'default' },
  completed: { label: '已完成', color: 'success' },
  failed:    { label: '失败',   color: 'error' },
  canceled:  { label: '已取消', color: 'default' },
};

const STAGE_LABEL: Record<string, string> = {
  download:   '下载',
  transcribe: '转写',
  script:     '文案',
  render:     '渲染',
};

const STAGE_ORDER = ['download', 'transcribe', 'script', 'render'];

// ==================== 小组件 ====================

function StatusChip({ status }: { status: string }) {
  const cfg = STATUS_MAP[status] || { label: status, color: 'default' };
  return <Chip label={cfg.label} color={cfg.color} size="small" variant="outlined" />;
}

/** 显示迷你阶段进度条：下载 → 转写 → 文案 → 渲染 */
function StageStrip({ stages }: { stages?: Stage[] }) {
  if (!stages || stages.length === 0) return null;

  const map = new Map(stages.map((s) => [s.stage_name, s]));

  return (
    <Stack direction="row" spacing={0.5} sx={{ alignItems: 'center' }}>
      {STAGE_ORDER.map((name) => {
        const stage = map.get(name);
        const status = stage?.status || 'pending';
        const color =
          status === 'completed'
            ? 'success.main'
            : status === 'running'
              ? 'warning.main'
              : status === 'failed'
                ? 'error.main'
                : 'text.disabled';
        return (
          <Tooltip
            key={name}
            title={`${STAGE_LABEL[name] || name}：${
              status === 'completed'
                ? '已完成'
                : status === 'running'
                  ? `进行中 ${stage?.progress ?? 0}%`
                  : status === 'failed'
                    ? `失败：${stage?.error_msg || ''}`
                    : '待处理'
            }`}
          >
            <Box
              sx={{
                width: 18,
                height: 6,
                borderRadius: 3,
                bgcolor: color,
                opacity: status === 'pending' ? 0.35 : 1,
                transition: 'all .2s',
              }}
            />
          </Tooltip>
        );
      })}
    </Stack>
  );
}

/** 把秒数格式化成 mm:ss */
function fmtDuration(sec?: number | null) {
  if (!sec || sec <= 0) return '';
  const s = Math.round(sec);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, '0')}`;
}

// ==================== 页面 ====================

export default function ProjectsPage() {
  const { data: projects = [], isLoading, mutate } = useSWR<Project[]>(
    '/api/projects',
    fetcher,
    { refreshInterval: 3000 },
  );

  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState('');
  const [url, setUrl] = useState('');
  const [submitting, setSubmitting] = useState(false);

  // 汇总统计
  const stats = useMemo(() => {
    const running = projects.filter((p) => p.status === 'running').length;
    const queued = projects.filter((p) => p.status === 'queued').length;
    const failed = projects.filter((p) => p.status === 'failed').length;
    const completed = projects.filter((p) => p.status === 'completed').length;
    return { running, queued, failed, completed };
  }, [projects]);

  const handleCreate = async () => {
    if (!url.trim()) return;
    setSubmitting(true);
    try {
      const res = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: title.trim(), youtube_url: url.trim() }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert(`创建失败：${err.error || res.status}`);
        return;
      }
      setOpen(false);
      setTitle('');
      setUrl('');
      mutate();
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm(`确定删除项目 P${id}？(只会删数据库记录，不会删磁盘文件)`)) return;
    await fetch(`/api/projects/${id}`, { method: 'DELETE' });
    mutate();
  };

  return (
    <Box sx={{ p: 4, maxWidth: 1300, mx: 'auto' }}>
      {/* 顶部标题栏 */}
      <Stack
        direction="row"
        sx={{ justifyContent: 'space-between', alignItems: 'center', mb: 3 }}
      >
        <Box>
          <Typography variant="h6" sx={{ mb: 0.5 }}>
            视频列表 ({projects.length})
          </Typography>
          <Stack direction="row" spacing={2} sx={{ color: 'text.secondary' }}>
            <Typography variant="caption">排队中 {stats.queued}</Typography>
            <Typography variant="caption">处理中 {stats.running}</Typography>
            <Typography variant="caption">已完成 {stats.completed}</Typography>
            <Typography variant="caption" sx={{ color: 'error.main' }}>
              失败 {stats.failed}
            </Typography>
          </Stack>
        </Box>

        <Stack direction="row" spacing={1}>
          <Tooltip title="刷新">
            <IconButton onClick={() => mutate()}>
              <Refresh />
            </IconButton>
          </Tooltip>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setOpen(true)}
          >
            新建项目
          </Button>
        </Stack>
      </Stack>

      {/* 列表 */}
      <Paper elevation={0} sx={{ bgcolor: 'transparent' }}>
        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 6 }}>
            <CircularProgress />
          </Box>
        ) : (
          <List disablePadding>
            {projects.map((p) => {
              const activeStage = (p.stages || []).find(
                (s) => s.status === 'running',
              );
              const isRunning = p.status === 'running';
              const isFailed = p.status === 'failed';

              return (
                <ListItem
                  key={p.id}
                  sx={{
                    mb: 1.5,
                    bgcolor: '#26262e',
                    borderRadius: 2,
                    border: '1px solid rgba(255,255,255,0.06)',
                    '&:hover': { borderColor: 'rgba(74,144,226,0.4)' },
                  }}
                  secondaryAction={
                    <Stack direction="row" spacing={0.5}>
                      <Tooltip title="编辑">
                        <IconButton
                          size="small"
                          href={`/projects/${p.id}`}
                        >
                          <Edit fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="删除">
                        <IconButton
                          size="small"
                          onClick={() => handleDelete(p.id)}
                        >
                          <Delete fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Stack>
                  }
                >
                  {/* 拖拽手柄 */}
                  <DragIndicator
                    sx={{
                      color: 'text.disabled',
                      mr: 1.5,
                      cursor: 'grab',
                    }}
                  />

                  {/* 编号 */}
                  <Typography
                    sx={{
                      color: 'text.secondary',
                      fontFamily: 'monospace',
                      mr: 2,
                      fontSize: 13,
                    }}
                  >
                    P{p.id}
                  </Typography>

                  {/* 主内容 */}
                  <Box sx={{ flex: 1, mr: 2, minWidth: 0 }}>
                    <Typography
                      noWrap
                      sx={{ color: 'primary.main', fontWeight: 500, fontSize: 14 }}
                    >
                      {p.title}
                    </Typography>

                    <Stack
                      direction="row"
                      spacing={2}
                      sx={{
                        mt: 0.5,
                        alignItems: 'center',
                        color: 'text.secondary',
                        fontSize: 12,
                      }}
                    >
                      {p.duration_sec ? (
                        <Typography variant="caption">
                          {fmtDuration(p.duration_sec)}
                        </Typography>
                      ) : null}
                      {p.resolution ? (
                        <Typography variant="caption">{p.resolution}</Typography>
                      ) : null}
                      {p.created_at ? (
                        <Typography variant="caption">
                          {new Date(p.created_at).toLocaleString()}
                        </Typography>
                      ) : null}
                    </Stack>
                  </Box>

                  {/* 阶段小进度条 */}
                  <Box sx={{ mr: 2 }}>
                    <StageStrip stages={p.stages} />
                  </Box>

                  {/* 状态徽章 */}
                  <StatusChip status={p.status} />

                  {/* 运行中：显示当前阶段 + 进度环 */}
                  {isRunning && (
                    <Stack
                      direction="row"
                      spacing={1}
                      sx={{ ml: 2, alignItems: 'center' }}
                    >
                      <CircularProgress
                        variant="determinate"
                        value={p.progress}
                        size={18}
                        thickness={5}
                      />
                      <Typography
                        variant="caption"
                        sx={{ color: 'text.secondary', minWidth: 60 }}
                      >
                        {activeStage
                          ? `${STAGE_LABEL[activeStage.stage_name] || activeStage.stage_name} `
                          : ''}
                        {p.progress}%
                      </Typography>
                    </Stack>
                  )}

                  {/* 失败：显示错误提示 */}
                  {isFailed && (
                    <Tooltip
                      title={p.error_msg || '未知错误'}
                      placement="top"
                      slotProps={{ tooltip: { sx: { maxWidth: 480 } } }}
                    >
                      <Typography
                        variant="caption"
                        sx={{
                          ml: 2,
                          color: 'error.main',
                          maxWidth: 200,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {p.error_stage
                          ? `${STAGE_LABEL[p.error_stage] || p.error_stage}失败`
                          : '失败'}
                      </Typography>
                    </Tooltip>
                  )}
                </ListItem>
              );
            })}

            {projects.length === 0 && (
              <Typography
                sx={{ color: 'text.secondary', textAlign: 'center', py: 6 }}
              >
                暂无项目，点击右上角"新建项目"
              </Typography>
            )}
          </List>
        )}
      </Paper>

      {/* 新建项目弹窗 */}
      <Dialog open={open} onClose={() => setOpen(false)} fullWidth maxWidth="sm">
        <DialogTitle>新建项目</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField
              label="标题（可选）"
              fullWidth
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="留空则下载完成后回填原视频标题"
            />
            <TextField
              label="YouTube 链接"
              fullWidth
              required
              placeholder="https://www.youtube.com/watch?v=..."
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              helperText="粘贴 YouTube 视频或 Shorts 链接，Worker 会自动下载并按流水线处理"
            />
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>取消</Button>
          <Button
            variant="contained"
            onClick={handleCreate}
            disabled={submitting || !url.trim()}
          >
            {submitting ? '提交中…' : '创建'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}