// front/app/api/projects/[id]/route.ts
import { NextResponse } from 'next/server';
import { pool } from '@/lib/db';

export async function GET(
  _: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const pid = Number(id);
  if (!Number.isFinite(pid)) {
    return NextResponse.json({ error: 'invalid id' }, { status: 400 });
  }

  const { rows: projects } = await pool.query(
    'SELECT * FROM projects WHERE id=$1',
    [pid],
  );
  if (!projects[0]) {
    return NextResponse.json({ error: 'not found' }, { status: 404 });
  }

  const { rows: stages } = await pool.query(
    'SELECT * FROM project_stages WHERE project_id=$1 ORDER BY id',
    [pid],
  );
  return NextResponse.json({ ...projects[0], stages });
}

export async function DELETE(
  _: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const pid = Number(id);
  if (!Number.isFinite(pid)) {
    return NextResponse.json({ error: 'invalid id' }, { status: 400 });
  }
  await pool.query('DELETE FROM projects WHERE id=$1', [pid]);
  return NextResponse.json({ ok: true });
}