// front/app/api/projects/route.ts
import { NextResponse } from 'next/server';
import { pool } from '@/lib/db';

const DEFAULT_STAGES = ['download', 'transcribe', 'script', 'render'] as const;

/**
 * GET /api/projects
 * 一次查询返回主表 + stages 聚合，前端无需二次请求。
 */
export async function GET() {
  const { rows } = await pool.query(`
    SELECT
      p.*,
      COALESCE(
        json_agg(
          json_build_object(
            'stage_name',  s.stage_name,
            'status',      s.status,
            'progress',    s.progress,
            'output_path', s.output_path,
            'error_msg',   s.error_msg
          ) ORDER BY s.id
        ) FILTER (WHERE s.id IS NOT NULL),
        '[]'::json
      ) AS stages
    FROM projects p
    LEFT JOIN project_stages s ON p.id = s.project_id
    GROUP BY p.id
    ORDER BY p.id DESC
    LIMIT 200
  `);
  return NextResponse.json(rows);
}

/**
 * POST /api/projects
 * 事务里插 project + 4 个 stage 占位。
 */
export async function POST(req: Request) {
  let body: any;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'invalid json' }, { status: 400 });
  }
  const { title, youtube_url } = body || {};
  if (!youtube_url || typeof youtube_url !== 'string') {
    return NextResponse.json({ error: 'youtube_url is required' }, { status: 400 });
  }

  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    const { rows } = await client.query(
      `INSERT INTO projects (title, youtube_url)
       VALUES ($1, $2)
       RETURNING *`,
      [title || 'Untitled', youtube_url.trim()],
    );
    const project = rows[0];

    for (const stage of DEFAULT_STAGES) {
      await client.query(
        `INSERT INTO project_stages (project_id, stage_name)
         VALUES ($1, $2)`,
        [project.id, stage],
      );
    }

    await client.query('COMMIT');
    return NextResponse.json({ ...project, stages: [] });
  } catch (e: any) {
    await client.query('ROLLBACK');
    console.error('[api/projects POST] failed:', e);
    return NextResponse.json({ error: e?.message || 'insert failed' }, { status: 500 });
  } finally {
    client.release();
  }
}