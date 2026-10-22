// front/lib/db.ts
import { Pool } from 'pg';

const connectionString =
  process.env.DATABASE_URL ||
  'postgresql://ken:886482++@localhost:5432/narrato';

const globalForPg = global as unknown as { pgPool?: Pool };

export const pool =
  globalForPg.pgPool ||
  new Pool({ connectionString, max: 10 });

if (process.env.NODE_ENV !== 'production') globalForPg.pgPool = pool;