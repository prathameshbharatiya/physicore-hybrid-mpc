import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';

// ── Types ────────────────────────────────────────────────────────────────────

interface SessionRecord {
  session_id:  string;
  robot_id:    string;
  platform:    string;
  started_at:  number;
  ended_at:    number;
  step_count:  number;
  meta:        Record<string, unknown>;
}

interface SeriesPoint {
  step:      number;
  timestamp: number;
  value:     number;
}

interface SessionData {
  [key: string]: SeriesPoint[];
}

interface ComparisonReport {
  winner:                   string;
  session_a:                string;
  session_b:                string;
  residual_improvement_pct: number;
  convergence_speedup:      number;
  param_deltas:             Record<string, number>;
  summary_text:             string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const METRIC_KEYS = ['residual', 'uncertainty', 'sysid_loss'];
const PARAM_KEYS  = ['param_mass', 'param_friction', 'param_inertia', 'param_damping'];
const KEY_COLORS: Record<string, string> = {
  residual:      '#ef4444',
  uncertainty:   '#f59e0b',
  sysid_loss:    '#8b5cf6',
  param_mass:    '#06b6d4',
  param_friction:'#22c55e',
  param_inertia: '#ec4899',
  param_damping: '#a78bfa',
};

function fmt_date(ts: number): string {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function fmt_duration(started: number, ended: number): string {
  if (!ended || ended <= started) return '—';
  const s = Math.round(ended - started);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

function AnomalyBadge({ score }: { score: number | null }) {
  if (score === null) return null;
  const label = score < 2 ? 'Normal' : score < 4 ? 'Unusual' : 'Anomalous';
  const cls   = score < 2
    ? 'bg-emerald-900/40 text-emerald-400 border-emerald-800/40'
    : score < 4
      ? 'bg-amber-900/40 text-amber-400 border-amber-800/40'
      : 'bg-red-900/40 text-red-400 border-red-800/40';
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded-full border font-medium ${cls}`}>
      {label} ({score.toFixed(2)})
    </span>
  );
}

// ── MiniSparkline ─────────────────────────────────────────────────────────────

function MiniSparkline({ points, color }: { points: SeriesPoint[]; color: string }) {
  const data = points.map((p, i) => ({ i, value: p.value }));
  return (
    <ResponsiveContainer width="100%" height={40}>
      <AreaChart data={data} margin={{ top: 2, right: 0, left: 0, bottom: 0 }}>
        <Area
          type="monotone" dataKey="value"
          stroke={color} fill={color} fillOpacity={0.15}
          dot={false} isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ── MetricChart ───────────────────────────────────────────────────────────────

function MetricChart({ label, points, color }: { label: string; points: SeriesPoint[]; color: string }) {
  const data = points.slice(-300).map((p, i) => ({ i, step: p.step, value: p.value }));
  return (
    <div className="bg-[#0f172a] border border-slate-700/40 rounded-xl p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">{label}</span>
        {data.length > 0 && (
          <span className="text-[10px] font-mono text-slate-500">
            latest: {data[data.length - 1].value.toFixed(5)}
          </span>
        )}
      </div>
      {data.length === 0 ? (
        <div className="h-20 flex items-center justify-center text-slate-600 text-xs">No data</div>
      ) : (
        <ResponsiveContainer width="100%" height={80}>
          <AreaChart data={data} margin={{ top: 2, right: 4, left: -28, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="step" tick={{ fontSize: 9, fill: '#475569' }} />
            <YAxis tick={{ fontSize: 9, fill: '#475569' }} />
            <Tooltip
              contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6, fontSize: 11 }}
              labelFormatter={(l: number) => `step ${l}`}
            />
            <Area
              type="monotone" dataKey="value"
              stroke={color} fill={color} fillOpacity={0.15}
              dot={false} isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// ── ComparePanel ──────────────────────────────────────────────────────────────

function ComparePanel({
  sessionA, sessions, apiBase, onClose,
}: {
  sessionA:  SessionRecord;
  sessions:  SessionRecord[];
  apiBase:   string;
  onClose:   () => void;
}) {
  const [selectedB, setSelectedB]   = useState<string>('');
  const [report,    setReport]      = useState<ComparisonReport | null>(null);
  const [loading,   setLoading]     = useState(false);
  const [error,     setError]       = useState<string | null>(null);

  const runCompare = useCallback(async () => {
    if (!selectedB) return;
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${apiBase}/api/analytics/compare?a=${sessionA.session_id}&b=${selectedB}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setReport(await r.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [apiBase, sessionA.session_id, selectedB]);

  const othersB = sessions.filter(s => s.session_id !== sessionA.session_id);

  return (
    <div className="bg-[#0a0e1a] border border-slate-700/50 rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Compare Sessions</span>
        <button onClick={onClose} className="text-slate-600 hover:text-slate-400 text-xs">Close</button>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">Session A</label>
          <div className="text-xs font-mono text-cyan-400 bg-slate-800/50 px-2 py-1.5 rounded">
            {sessionA.session_id.slice(0, 12)}… ({sessionA.platform})
          </div>
        </div>
        <div>
          <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">Session B</label>
          <select
            value={selectedB}
            onChange={e => setSelectedB(e.target.value)}
            className="w-full text-xs font-mono bg-slate-800/80 border border-slate-700/50 text-slate-300 rounded px-2 py-1.5 outline-none"
          >
            <option value="">— select session —</option>
            {othersB.map(s => (
              <option key={s.session_id} value={s.session_id}>
                {s.session_id.slice(0, 12)}… ({s.platform}, {fmt_date(s.started_at)})
              </option>
            ))}
          </select>
        </div>
      </div>

      <button
        onClick={runCompare}
        disabled={!selectedB || loading}
        className="w-full py-1.5 text-xs font-bold uppercase tracking-wider rounded bg-violet-700 text-white hover:bg-violet-600 disabled:opacity-40 transition-colors"
      >
        {loading ? 'Comparing…' : 'Run Comparison'}
      </button>

      {error && <p className="text-red-400 text-xs">{error}</p>}

      {report && (
        <div className="space-y-3">
          <div className="rounded-lg bg-slate-900/60 border border-slate-700/30 p-3">
            <p className="text-xs text-slate-300 mb-2">{report.summary_text}</p>
            <div className="grid grid-cols-2 gap-2 text-[10px]">
              <div>
                <span className="text-slate-500">Winner</span>
                <div className="font-mono text-emerald-400">{report.winner.slice(0, 12)}…</div>
              </div>
              <div>
                <span className="text-slate-500">Residual Δ</span>
                <div className={`font-mono ${report.residual_improvement_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {report.residual_improvement_pct >= 0 ? '+' : ''}{report.residual_improvement_pct.toFixed(2)}%
                </div>
              </div>
            </div>
          </div>
          {Object.keys(report.param_deltas).length > 0 && (
            <div className="rounded-lg bg-slate-900/60 border border-slate-700/30 p-3">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Parameter Deltas (B − A)</div>
              <div className="space-y-1">
                {Object.entries(report.param_deltas).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-[10px]">
                    <span className="text-slate-400 font-mono">{k}</span>
                    <span className={`font-mono ${(v as number) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {(v as number) >= 0 ? '+' : ''}{(v as number).toFixed(5)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── SessionDetail ─────────────────────────────────────────────────────────────

function SessionDetail({
  session, sessions, apiBase,
}: {
  session:  SessionRecord;
  sessions: SessionRecord[];
  apiBase:  string;
}) {
  const [data,         setData]         = useState<SessionData>({});
  const [anomaly,      setAnomaly]      = useState<number | null>(null);
  const [showCompare,  setShowCompare]  = useState(false);
  const [loadingData,  setLoadingData]  = useState(true);
  const [exportLoading, setExportLoading] = useState(false);

  useEffect(() => {
    setLoadingData(true);
    const keys = [...METRIC_KEYS, ...PARAM_KEYS];
    Promise.all(
      keys.map(key =>
        fetch(`${apiBase}/api/telemetry/sessions/${session.session_id}/data?key=${key}&limit=2000`)
          .then(r => r.ok ? r.json() : { points: [] })
          .then(d => [key, d.points ?? []] as [string, SeriesPoint[]])
          .catch(() => [key, []] as [string, SeriesPoint[]])
      )
    ).then(results => {
      const d: SessionData = {};
      results.forEach(([k, pts]) => { d[k] = pts; });
      setData(d);
      setLoadingData(false);
    });

    fetch(`${apiBase}/api/analytics/session/${session.session_id}/anomaly`)
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setAnomaly(d.anomaly_score ?? null))
      .catch(() => {});
  }, [session.session_id, apiBase]);

  const handleExport = useCallback(async () => {
    setExportLoading(true);
    try {
      const r = await fetch(
        `${apiBase}/api/telemetry/sessions/${session.session_id}/export.csv`
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const blob = await r.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = `physicore_session_${session.session_id}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // ignore — user sees no action
    } finally {
      setExportLoading(false);
    }
  }, [apiBase, session.session_id]);

  return (
    <div className="space-y-4">
      {/* Metadata card */}
      <div className="bg-[#0f172a] border border-slate-700/40 rounded-xl p-4">
        <div className="flex items-start justify-between gap-3 mb-3">
          <div>
            <div className="font-mono text-xs text-cyan-400 mb-1">{session.session_id}</div>
            <div className="text-[10px] text-slate-500 space-y-0.5">
              <div>Robot: <span className="text-slate-300">{session.robot_id}</span></div>
              <div>Platform: <span className="text-slate-300">{session.platform}</span></div>
              <div>Started: <span className="text-slate-300">{fmt_date(session.started_at)}</span></div>
              <div>Duration: <span className="text-slate-300">{fmt_duration(session.started_at, session.ended_at)}</span></div>
              <div>Steps: <span className="text-slate-300">{session.step_count.toLocaleString()}</span></div>
            </div>
          </div>
          <div className="flex flex-col items-end gap-2">
            <AnomalyBadge score={anomaly} />
            <button
              onClick={handleExport}
              disabled={exportLoading}
              className="text-[10px] px-3 py-1 rounded bg-slate-800 text-slate-300 hover:text-cyan-400 hover:bg-slate-700 transition-colors disabled:opacity-40"
            >
              {exportLoading ? 'Exporting…' : 'Export CSV'}
            </button>
            <button
              onClick={() => setShowCompare(v => !v)}
              className="text-[10px] px-3 py-1 rounded bg-violet-900/30 text-violet-400 hover:bg-violet-800/30 transition-colors"
            >
              Compare
            </button>
          </div>
        </div>

        {Object.keys(session.meta).length > 0 && (
          <div className="mt-2 pt-2 border-t border-slate-800">
            <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1">Meta</div>
            <pre className="text-[10px] text-slate-500 whitespace-pre-wrap">
              {JSON.stringify(session.meta, null, 2)}
            </pre>
          </div>
        )}
      </div>

      {showCompare && (
        <ComparePanel
          sessionA={session}
          sessions={sessions}
          apiBase={apiBase}
          onClose={() => setShowCompare(false)}
        />
      )}

      {loadingData ? (
        <div className="text-center text-slate-500 text-xs py-8 animate-pulse">Loading telemetry…</div>
      ) : (
        <div className="space-y-3">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider">Metrics</div>
          <div className="grid grid-cols-1 gap-3">
            {METRIC_KEYS.map(key => (
              <MetricChart
                key={key}
                label={key}
                points={data[key] ?? []}
                color={KEY_COLORS[key] ?? '#06b6d4'}
              />
            ))}
          </div>

          {PARAM_KEYS.some(k => (data[k] ?? []).length > 0) && (
            <>
              <div className="text-[10px] text-slate-500 uppercase tracking-wider mt-4">Learned Parameters</div>
              <div className="grid grid-cols-2 gap-3">
                {PARAM_KEYS.map(key => {
                  const pts = data[key] ?? [];
                  if (pts.length === 0) return null;
                  return (
                    <MetricChart
                      key={key}
                      label={key.replace('param_', '')}
                      points={pts}
                      color={KEY_COLORS[key] ?? '#06b6d4'}
                    />
                  );
                })}
              </div>
            </>
          )}

          {METRIC_KEYS.every(k => (data[k] ?? []).length === 0) &&
           PARAM_KEYS.every(k => (data[k] ?? []).length === 0) && (
            <div className="text-center text-slate-600 text-xs py-8">
              No telemetry data for this session.
              <br />
              Telemetry is written automatically when the engine runs with
              an attached TelemetryStore.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── SessionList ───────────────────────────────────────────────────────────────

function SessionList({
  sessions,
  selectedId,
  onSelect,
  filterPlatform,
  filterRobot,
  onFilterPlatform,
  onFilterRobot,
}: {
  sessions:        SessionRecord[];
  selectedId:      string | null;
  onSelect:        (s: SessionRecord) => void;
  filterPlatform:  string;
  filterRobot:     string;
  onFilterPlatform:(v: string) => void;
  onFilterRobot:   (v: string) => void;
}) {
  const platforms = [...new Set(sessions.map(s => s.platform))];
  const robots    = [...new Set(sessions.map(s => s.robot_id))];
  const filtered  = sessions.filter(s =>
    (filterPlatform === '' || s.platform === filterPlatform) &&
    (filterRobot    === '' || s.robot_id === filterRobot)
  );

  return (
    <div className="flex flex-col h-full">
      {/* Filters */}
      <div className="space-y-2 mb-3">
        <select
          value={filterPlatform}
          onChange={e => onFilterPlatform(e.target.value)}
          className="w-full text-[10px] bg-slate-800/80 border border-slate-700/50 text-slate-400 rounded px-2 py-1 outline-none"
        >
          <option value="">All platforms</option>
          {platforms.map(p => <option key={p} value={p}>{p}</option>)}
        </select>
        <select
          value={filterRobot}
          onChange={e => onFilterRobot(e.target.value)}
          className="w-full text-[10px] bg-slate-800/80 border border-slate-700/50 text-slate-400 rounded px-2 py-1 outline-none"
        >
          <option value="">All robots</option>
          {robots.map(r => <option key={r} value={r}>{r}</option>)}
        </select>
      </div>

      <div className="text-[10px] text-slate-600 mb-2">{filtered.length} session{filtered.length !== 1 ? 's' : ''}</div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto space-y-1.5 pr-1" style={{ maxHeight: '70vh' }}>
        {filtered.map(s => {
          const active = s.session_id === selectedId;
          return (
            <button
              key={s.session_id}
              onClick={() => onSelect(s)}
              className={`w-full text-left p-2.5 rounded-lg border transition-all ${
                active
                  ? 'bg-cyan-900/20 border-cyan-700/50'
                  : 'bg-slate-900/40 border-slate-700/30 hover:bg-slate-800/40 hover:border-slate-600/40'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-[10px] text-cyan-400 truncate">
                  {s.session_id.slice(0, 12)}…
                </span>
                <span className="text-[9px] text-slate-600 shrink-0 ml-2">
                  {s.step_count.toLocaleString()} steps
                </span>
              </div>
              <div className="text-[9px] text-slate-500">
                {s.platform} · {s.robot_id}
              </div>
              <div className="text-[9px] text-slate-600 mt-0.5">
                {fmt_date(s.started_at)}
              </div>
            </button>
          );
        })}
        {filtered.length === 0 && (
          <div className="text-slate-600 text-xs text-center py-6">No sessions match the filters</div>
        )}
      </div>
    </div>
  );
}

// ── Main TelemetryExplorer ────────────────────────────────────────────────────

interface TelemetryExplorerProps {
  apiBase?: string;
}

const TelemetryExplorer: React.FC<TelemetryExplorerProps> = ({
  apiBase = 'http://localhost:8000',
}) => {
  const [sessions,       setSessions]       = useState<SessionRecord[]>([]);
  const [selected,       setSelected]       = useState<SessionRecord | null>(null);
  const [loading,        setLoading]        = useState(true);
  const [fetchError,     setFetchError]     = useState<string | null>(null);
  const [filterPlatform, setFilterPlatform] = useState('');
  const [filterRobot,    setFilterRobot]    = useState('');
  const [stats,          setStats]          = useState<Record<string, unknown> | null>(null);

  const fetchSessions = useCallback(async () => {
    try {
      const r = await fetch(`${apiBase}/api/telemetry/sessions?limit=200`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setSessions(d.sessions ?? []);
      setFetchError(null);
    } catch (e: unknown) {
      setFetchError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  const fetchStats = useCallback(async () => {
    try {
      const r = await fetch(`${apiBase}/api/telemetry/stats`);
      if (r.ok) setStats(await r.json());
    } catch {
      // stats are optional
    }
  }, [apiBase]);

  useEffect(() => {
    fetchSessions();
    fetchStats();
  }, [fetchSessions, fetchStats]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-40 text-slate-500 text-sm animate-pulse">
        Loading sessions…
      </div>
    );
  }

  if (fetchError) {
    return (
      <div className="rounded-xl border border-red-800/50 bg-red-950/20 p-6 text-center">
        <p className="text-red-400 text-sm font-medium mb-1">Telemetry store unavailable</p>
        <p className="text-red-600 text-xs mb-3">{fetchError}</p>
        <button
          onClick={() => { setLoading(true); fetchSessions(); }}
          className="text-xs px-3 py-1.5 rounded bg-red-900/30 text-red-400 hover:bg-red-800/30 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col gap-4">
      {/* Stats bar */}
      {stats && (
        <div className="flex gap-4 flex-wrap">
          {[
            ['Sessions', (stats.total_sessions as number)?.toLocaleString() ?? '—'],
            ['Rows', (stats.total_rows as number)?.toLocaleString() ?? '—'],
            ['DB Size', stats.db_size_mb ? `${stats.db_size_mb} MB` : '—'],
          ].map(([label, value]) => (
            <div key={label} className="bg-[#0f172a] border border-slate-700/40 rounded-lg px-4 py-2 flex flex-col">
              <span className="text-[9px] text-slate-500 uppercase tracking-wider">{label}</span>
              <span className="text-sm font-bold font-mono text-cyan-400">{value}</span>
            </div>
          ))}
          <button
            onClick={() => { setLoading(true); fetchSessions(); fetchStats(); }}
            className="ml-auto text-[10px] px-3 py-1 rounded bg-slate-800 text-slate-400 hover:text-cyan-400 self-center transition-colors"
          >
            Refresh
          </button>
        </div>
      )}

      {/* Main two-panel layout */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Left: session list */}
        <div className="w-72 shrink-0">
          <SessionList
            sessions={sessions}
            selectedId={selected?.session_id ?? null}
            onSelect={setSelected}
            filterPlatform={filterPlatform}
            filterRobot={filterRobot}
            onFilterPlatform={setFilterPlatform}
            onFilterRobot={setFilterRobot}
          />
        </div>

        {/* Right: session detail */}
        <div className="flex-1 overflow-y-auto">
          {selected ? (
            <SessionDetail
              session={selected}
              sessions={sessions}
              apiBase={apiBase}
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center py-20">
              <div className="text-slate-600 text-sm mb-2">No session selected</div>
              <div className="text-slate-700 text-xs">
                {sessions.length === 0
                  ? 'No sessions in the telemetry store yet. Run the engine to start recording.'
                  : 'Select a session from the left panel to explore its telemetry.'}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TelemetryExplorer;
