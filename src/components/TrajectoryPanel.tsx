import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';

// ─── Types ───────────────────────────────────────────────────────────────────

interface TrajectoryPoint {
  t: number;
  q: number[];
  qd?: number[] | null;
  ee_pos?: number[] | null;
}

interface Trajectory {
  trajectory_id: string;
  duration: number;
  dof: number;
  n_points: number;
  metadata: Record<string, unknown>;
  points: TrajectoryPoint[];
}

interface ExecutionResult {
  trajectory_id: string;
  status: 'idle' | 'running' | 'completed' | 'aborted' | 'error';
  elapsed_s: number;
  mean_tracking_error: number;
  max_tracking_error: number;
  final_q: number[] | null;
  message: string;
}

interface TrajectoryPanelProps {
  baseUrl?: string;
  dof?: number;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const JOINT_COLORS = [
  '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4',
];

// ─── Helpers ──────────────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function JointSlider({
  index, value, onChange,
}: {
  index: number; value: number; onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-3 py-1">
      <span className="text-xs font-mono text-gray-400 w-8">q{index}</span>
      <input
        type="range"
        min={-3.14}
        max={3.14}
        step={0.01}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="flex-1 accent-indigo-500"
      />
      <span className="text-xs font-mono text-gray-300 w-14 text-right">
        {value.toFixed(3)}
      </span>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    idle:      'bg-gray-700 text-gray-300',
    running:   'bg-blue-900 text-blue-300 animate-pulse',
    completed: 'bg-emerald-900 text-emerald-300',
    aborted:   'bg-amber-900 text-amber-300',
    error:     'bg-red-900 text-red-300',
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-semibold uppercase ${colors[status] ?? colors.idle}`}>
      {status}
    </span>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

const TrajectoryPanel: React.FC<TrajectoryPanelProps> = ({
  baseUrl = 'http://localhost:8000',
  dof = 6,
}) => {
  // ── Planning mode
  const [planMode, setPlanMode] = useState<'joint' | 'task' | 'waypoints' | 'circular'>('joint');

  // ── Joint sliders for start + goal
  const [qStart, setQStart] = useState<number[]>(() => Array(dof).fill(0));
  const [qGoal,  setQGoal]  = useState<number[]>(() => Array(dof).fill(0));

  // ── Task-space target
  const [taskTarget, setTaskTarget] = useState({ x: '0.5', y: '0', z: '0.4' });

  // ── Waypoints (joint configs separated by ;)
  const [waypointsText, setWaypointsText] = useState('0,0,0,0,0,0;0.5,0.5,0,0,0,0;1,0,-0.5,0,0,0');

  // ── Circular arc params
  const [circCenter, setCircCenter] = useState({ x: '0.5', y: '0', z: '0.3' });
  const [circNormal, setCircNormal] = useState({ x: '0', y: '0', z: '1' });
  const [circAngle, setCircAngle]   = useState('3.14');

  // ── Trajectory state
  const [trajectory, setTrajectory]       = useState<Trajectory | null>(null);
  const [planLoading, setPlanLoading]     = useState(false);
  const [planError, setPlanError]         = useState<string | null>(null);

  // ── Execution state
  const [execResult, setExecResult]       = useState<ExecutionResult | null>(null);
  const [execStatus, setExecStatus]       = useState<string>('idle');
  const [execLoading, setExecLoading]     = useState(false);
  const [trackingData, setTrackingData]   = useState<{ t: number; error: number }[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Chart data derived from trajectory points
  const chartData = trajectory
    ? trajectory.points.map(pt => {
        const row: Record<string, number> = { t: parseFloat(pt.t.toFixed(3)) };
        pt.q.slice(0, 6).forEach((v, i) => { row[`q${i}`] = parseFloat(v.toFixed(4)); });
        return row;
      })
    : [];

  // ── Plan ─────────────────────────────────────────────────────────────────

  const handlePlan = useCallback(async () => {
    setPlanLoading(true);
    setPlanError(null);
    setTrajectory(null);
    setExecResult(null);
    setTrackingData([]);
    setExecStatus('idle');

    try {
      let url = '';
      let body: Record<string, unknown> = {};

      if (planMode === 'joint') {
        url  = `${baseUrl}/api/plan/joint_space`;
        body = { q_start: qStart, q_goal: qGoal };
      } else if (planMode === 'task') {
        url  = `${baseUrl}/api/plan/task_space`;
        body = {
          q_start: qStart,
          target_pos: [
            parseFloat(taskTarget.x),
            parseFloat(taskTarget.y),
            parseFloat(taskTarget.z),
          ],
        };
      } else if (planMode === 'waypoints') {
        const wps = waypointsText.split(';').map(seg =>
          seg.split(',').map(Number)
        );
        url  = `${baseUrl}/api/plan/waypoints`;
        body = { waypoints: wps };
      } else {
        url  = `${baseUrl}/api/plan/circular`;
        body = {
          q_start: qStart,
          center: [parseFloat(circCenter.x), parseFloat(circCenter.y), parseFloat(circCenter.z)],
          normal: [parseFloat(circNormal.x), parseFloat(circNormal.y), parseFloat(circNormal.z)],
          angle_rad: parseFloat(circAngle),
        };
      }

      const res  = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status}: ${text}`);
      }
      const data: Trajectory = await res.json();
      setTrajectory(data);
    } catch (err) {
      setPlanError(String(err));
    } finally {
      setPlanLoading(false);
    }
  }, [planMode, qStart, qGoal, taskTarget, waypointsText, circCenter, circNormal, circAngle, baseUrl]);

  // ── Execute ───────────────────────────────────────────────────────────────

  const handleExecute = useCallback(async () => {
    if (!trajectory) return;
    setExecLoading(true);
    setExecResult(null);
    setTrackingData([]);
    setExecStatus('running');

    try {
      const res = await fetch(`${baseUrl}/api/execute/${trajectory.trajectory_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ async_mode: true }),
      });
      if (!res.ok) throw new Error(await res.text());

      // Poll for status
      let ticks = 0;
      pollRef.current = setInterval(async () => {
        try {
          const sr = await fetch(`${baseUrl}/api/execute/status?trajectory_id=${trajectory.trajectory_id}`);
          if (!sr.ok) return;
          const sd: ExecutionResult = await sr.json();
          setExecStatus(sd.status);
          setTrackingData(prev => [
            ...prev,
            { t: parseFloat((ticks * 0.1).toFixed(1)), error: sd.mean_tracking_error },
          ]);
          ticks++;
          if (sd.status !== 'running') {
            clearInterval(pollRef.current!);
            setExecResult(sd);
            setExecLoading(false);
          }
        } catch {
          // ignore poll errors
        }
      }, 100);
    } catch (err) {
      setPlanError(String(err));
      setExecLoading(false);
      setExecStatus('error');
    }
  }, [trajectory, baseUrl]);

  const handleAbort = useCallback(async () => {
    if (!trajectory) return;
    clearInterval(pollRef.current!);
    try {
      await fetch(`${baseUrl}/api/execute/abort`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trajectory_id: trajectory.trajectory_id }),
      });
      setExecStatus('aborted');
    } catch {
      // ignore
    } finally {
      setExecLoading(false);
    }
  }, [trajectory, baseUrl]);

  // Cleanup poll on unmount
  useEffect(() => () => { clearInterval(pollRef.current!); }, []);

  // ── Helpers ───────────────────────────────────────────────────────────────

  const setQ = (which: 'start' | 'goal', idx: number, v: number) => {
    if (which === 'start') setQStart(prev => prev.map((x, i) => i === idx ? clamp(v, -3.14, 3.14) : x));
    else                   setQGoal( prev => prev.map((x, i) => i === idx ? clamp(v, -3.14, 3.14) : x));
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-6 p-4 text-sm text-gray-200">

      {/* ── Plan Mode Selector ── */}
      <div className="flex gap-2">
        {(['joint', 'task', 'waypoints', 'circular'] as const).map(m => (
          <button
            key={m}
            onClick={() => setPlanMode(m)}
            className={`px-3 py-1.5 rounded text-xs font-semibold capitalize transition-colors
              ${planMode === m
                ? 'bg-emerald-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
          >
            {m === 'joint' ? 'Joint Space' : m === 'task' ? 'Task Space' : m === 'waypoints' ? 'Waypoints' : 'Circular Arc'}
          </button>
        ))}
      </div>

      {/* ── Configuration panel ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* Start joints (always shown) */}
        {(planMode === 'joint' || planMode === 'task' || planMode === 'circular') && (
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wider">
              Start Configuration
            </h3>
            {qStart.map((v, i) => (
              <JointSlider key={i} index={i} value={v} onChange={nv => setQ('start', i, nv)} />
            ))}
          </div>
        )}

        {/* Goal joints (joint space only) */}
        {planMode === 'joint' && (
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wider">
              Goal Configuration
            </h3>
            {qGoal.map((v, i) => (
              <JointSlider key={i} index={i} value={v} onChange={nv => setQ('goal', i, nv)} />
            ))}
          </div>
        )}

        {/* Task space target */}
        {planMode === 'task' && (
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-xs font-semibold text-gray-400 mb-3 uppercase tracking-wider">
              Target EE Position
            </h3>
            {(['x', 'y', 'z'] as const).map(ax => (
              <div key={ax} className="flex items-center gap-2 mb-2">
                <label className="text-gray-400 w-4">{ax}</label>
                <input
                  type="number" step="0.05"
                  value={taskTarget[ax]}
                  onChange={e => setTaskTarget(t => ({ ...t, [ax]: e.target.value }))}
                  className="flex-1 bg-gray-700 rounded px-2 py-1 text-gray-200 text-xs"
                />
              </div>
            ))}
          </div>
        )}

        {/* Waypoints text */}
        {planMode === 'waypoints' && (
          <div className="bg-gray-800 rounded-lg p-4 md:col-span-2">
            <h3 className="text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wider">
              Waypoints (semicolon-separated, comma-separated joint angles)
            </h3>
            <textarea
              value={waypointsText}
              onChange={e => setWaypointsText(e.target.value)}
              rows={4}
              className="w-full bg-gray-700 rounded px-2 py-1 text-xs font-mono text-gray-200 resize-y"
            />
          </div>
        )}

        {/* Circular arc params */}
        {planMode === 'circular' && (
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-xs font-semibold text-gray-400 mb-3 uppercase tracking-wider">
              Arc Parameters
            </h3>
            <div className="mb-2">
              <span className="text-gray-400 text-xs mb-1 block">Center (x, y, z)</span>
              <div className="flex gap-2">
                {(['x', 'y', 'z'] as const).map(ax => (
                  <input key={ax} type="number" step="0.05" value={circCenter[ax]}
                    onChange={e => setCircCenter(c => ({ ...c, [ax]: e.target.value }))}
                    className="flex-1 bg-gray-700 rounded px-2 py-1 text-gray-200 text-xs"
                    placeholder={ax}
                  />
                ))}
              </div>
            </div>
            <div className="mb-2">
              <span className="text-gray-400 text-xs mb-1 block">Normal (x, y, z)</span>
              <div className="flex gap-2">
                {(['x', 'y', 'z'] as const).map(ax => (
                  <input key={ax} type="number" step="0.1" value={circNormal[ax]}
                    onChange={e => setCircNormal(n => ({ ...n, [ax]: e.target.value }))}
                    className="flex-1 bg-gray-700 rounded px-2 py-1 text-gray-200 text-xs"
                    placeholder={ax}
                  />
                ))}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-gray-400 text-xs">Angle (rad)</label>
              <input type="number" step="0.1" value={circAngle}
                onChange={e => setCircAngle(e.target.value)}
                className="w-24 bg-gray-700 rounded px-2 py-1 text-gray-200 text-xs"
              />
            </div>
          </div>
        )}
      </div>

      {/* ── Plan button ── */}
      <div className="flex items-center gap-3">
        <button
          onClick={handlePlan}
          disabled={planLoading}
          className="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50
                     text-white rounded font-semibold text-sm transition-colors"
        >
          {planLoading ? 'Planning…' : '▶ Plan Trajectory'}
        </button>
        {planError && (
          <span className="text-red-400 text-xs">{planError}</span>
        )}
      </div>

      {/* ── Trajectory preview chart ── */}
      {trajectory && chartData.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Trajectory Preview — {trajectory.dof} joints, {trajectory.duration.toFixed(2)}s
            </h3>
            <span className="text-xs text-gray-500 font-mono">{trajectory.trajectory_id.slice(0, 8)}</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={chartData} margin={{ top: 0, right: 10, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="t" tick={{ fill: '#9ca3af', fontSize: 10 }} label={{ value: 't (s)', position: 'insideBottomRight', offset: 0, fill: '#6b7280', fontSize: 10 }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', fontSize: 11 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {Array.from({ length: Math.min(trajectory.dof, 6) }, (_, i) => (
                <Area key={i} type="monotone" dataKey={`q${i}`}
                  stroke={JOINT_COLORS[i]} fill={JOINT_COLORS[i]}
                  fillOpacity={0.08} strokeWidth={1.5}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>

          {/* metadata */}
          {trajectory.metadata.collision && (
            <div className={`mt-2 text-xs px-3 py-1.5 rounded ${
              (trajectory.metadata.collision as { in_collision: boolean }).in_collision
                ? 'bg-red-900 text-red-300'
                : 'bg-emerald-900 text-emerald-300'
            }`}>
              {(trajectory.metadata.collision as { in_collision: boolean }).in_collision
                ? `⚠ Collision detected: ${(trajectory.metadata.collision as { obstacle_name: string }).obstacle_name}`
                : '✓ Path is clear'}
            </div>
          )}
        </div>
      )}

      {/* ── Execute controls ── */}
      {trajectory && (
        <div className="flex items-center gap-3">
          <button
            onClick={handleExecute}
            disabled={execLoading || execStatus === 'running'}
            className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50
                       text-white rounded font-semibold text-sm transition-colors"
          >
            {execLoading ? 'Executing…' : '⚡ Execute'}
          </button>
          {execStatus === 'running' && (
            <button
              onClick={handleAbort}
              className="px-4 py-2 bg-red-700 hover:bg-red-600 text-white rounded font-semibold text-sm"
            >
              ■ Abort
            </button>
          )}
          <StatusBadge status={execStatus} />
        </div>
      )}

      {/* ── Live tracking error chart ── */}
      {trackingData.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Live Tracking Error
          </h3>
          <ResponsiveContainer width="100%" height={140}>
            <LineChart data={trackingData} margin={{ top: 0, right: 10, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="t" tick={{ fill: '#9ca3af', fontSize: 10 }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', fontSize: 11 }} />
              <Line type="monotone" dataKey="error" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Execution result summary ── */}
      {execResult && execResult.status !== 'running' && (
        <div className={`rounded-lg p-4 text-sm ${
          execResult.status === 'completed'
            ? 'bg-emerald-900/30 border border-emerald-700'
            : 'bg-amber-900/30 border border-amber-700'
        }`}>
          <div className="flex justify-between items-start">
            <div>
              <div className="font-semibold mb-1">
                {execResult.status === 'completed' ? '✓ Execution Complete' : '⚠ Execution Stopped'}
              </div>
              <div className="text-xs text-gray-400 space-y-0.5">
                <div>Duration: <span className="text-gray-200">{execResult.elapsed_s.toFixed(2)}s</span></div>
                <div>Mean tracking error: <span className="text-gray-200">{execResult.mean_tracking_error.toFixed(5)}</span></div>
                <div>Max tracking error: <span className="text-gray-200">{execResult.max_tracking_error.toFixed(5)}</span></div>
                {execResult.message && (
                  <div>Message: <span className="text-gray-200">{execResult.message}</span></div>
                )}
              </div>
            </div>
            <StatusBadge status={execResult.status} />
          </div>
          {execResult.final_q && (
            <div className="mt-2 text-xs text-gray-400">
              Final q: <span className="font-mono text-gray-300">
                [{execResult.final_q.map(v => v.toFixed(3)).join(', ')}]
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TrajectoryPanel;
