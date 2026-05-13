// src/components/SafetyPanel.tsx

import React, { useState, useEffect, useCallback, useRef } from 'react';

// Safety state from backend /api/safety/status
interface SafetyStatus {
  armed: boolean;
  is_estopped: boolean;
  escalation_level: 'NOMINAL' | 'WARNING' | 'SOFT_STOP' | 'HARD_STOP' | 'E_STOP';
  violations: ViolationEntry[];
  action_dim: number;
  last_action: number[];
  torque_limits: number[] | null;
  workspace_inside: boolean;
}

interface ViolationEntry {
  kind: string;
  joint: number;
  value: number;
  limit: number;
  timestamp: number;
}

interface SafetyPanelProps {
  apiBase?: string;  // defaults to 'https://physicore-hybrid-mpc-production.up.railway.app'
}

export const SafetyPanel: React.FC<SafetyPanelProps> = ({ apiBase = 'https://physicore-hybrid-mpc-production.up.railway.app' }) => {
  // State
  const [status, setStatus] = useState<SafetyStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isEstopping, setIsEstopping] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll /api/safety/status every 500ms
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/safety/status`);
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
        setError(null);
      } else if (res.status === 404 || res.status === 503) {
        // Safety not configured yet — show placeholder
        setStatus(null);
      }
    } catch {
      setError('Backend unreachable');
    }
  }, [apiBase]);

  useEffect(() => {
    fetchStatus();
    pollRef.current = setInterval(fetchStatus, 500);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [fetchStatus]);

  // API actions
  const arm = async () => {
    setLoading(true);
    try { await fetch(`${apiBase}/api/safety/arm`, { method: 'POST' }); await fetchStatus(); }
    finally { setLoading(false); }
  };

  const disarm = async () => {
    setLoading(true);
    try { await fetch(`${apiBase}/api/safety/disarm`, { method: 'POST' }); await fetchStatus(); }
    finally { setLoading(false); }
  };

  const estop = async () => {
    setIsEstopping(true);
    try { await fetch(`${apiBase}/api/safety/estop`, { method: 'POST' }); await fetchStatus(); }
    finally { setIsEstopping(false); }
  };

  const resetEstop = async () => {
    setLoading(true);
    try { await fetch(`${apiBase}/api/safety/reset_estop`, { method: 'POST' }); await fetchStatus(); }
    finally { setLoading(false); }
  };

  // Helper: escalation color
  const escalationColor = (level: string) => {
    switch (level) {
      case 'NOMINAL':   return 'text-emerald-400';
      case 'WARNING':   return 'text-amber-400';
      case 'SOFT_STOP': return 'text-orange-400';
      case 'HARD_STOP': return 'text-red-400';
      case 'E_STOP':    return 'text-red-600 animate-pulse';
      default:          return 'text-slate-400';
    }
  };

  const badgeBg = (level: string) => {
    switch (level) {
      case 'NOMINAL':   return 'bg-emerald-900/40 border-emerald-600/50 text-emerald-300';
      case 'WARNING':   return 'bg-amber-900/40 border-amber-600/50 text-amber-300';
      case 'SOFT_STOP': return 'bg-orange-900/40 border-orange-600/50 text-orange-300';
      case 'HARD_STOP': return 'bg-red-900/40 border-red-600/50 text-red-300';
      case 'E_STOP':    return 'bg-red-950/80 border-red-500 text-red-200 animate-pulse';
      default:          return 'bg-slate-800/40 border-slate-600/50 text-slate-400';
    }
  };

  // Render placeholder when safety not configured
  if (!status && !error) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-3 text-slate-500">
        <div className="text-4xl opacity-30">🛡</div>
        <p className="font-mono text-xs uppercase tracking-widest">Safety interlock not configured</p>
        <p className="font-mono text-[10px] text-slate-600">Configure engine first, then POST /api/safety/arm</p>
      </div>
    );
  }

  const estopped = status?.is_estopped ?? false;
  const armed    = status?.armed ?? false;
  const level    = status?.escalation_level ?? 'NOMINAL';

  return (
    <div className="flex flex-col gap-3 h-full overflow-y-auto p-3">
      {/* ── Header: state badge ──────────────────────────────── */}
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] text-slate-500 uppercase tracking-widest">Safety Interlock</span>
        {status && (
          <span className={`font-mono text-[9px] px-2 py-0.5 rounded border font-bold uppercase ${badgeBg(level)}`}>
            {estopped ? 'E-STOP' : level}
          </span>
        )}
      </div>

      {error && (
        <div className="bg-red-950/50 border border-red-700/50 rounded px-3 py-2 font-mono text-[10px] text-red-300">
          {error}
        </div>
      )}

      {status && (
        <>
          {/* ── E-STOP Button ───────────────────────────────── */}
          <div className="flex flex-col gap-2">
            {!estopped ? (
              <button
                onClick={estop}
                disabled={isEstopping}
                className="w-full py-4 rounded-lg font-bold text-white text-lg tracking-widest uppercase transition-all
                           bg-red-700 hover:bg-red-600 active:scale-95 border-2 border-red-500
                           disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-red-900/40"
              >
                {isEstopping ? 'STOPPING…' : '⏹ EMERGENCY STOP'}
              </button>
            ) : (
              <button
                onClick={resetEstop}
                disabled={loading}
                className="w-full py-4 rounded-lg font-bold text-white text-base tracking-widest uppercase transition-all
                           bg-amber-700 hover:bg-amber-600 active:scale-95 border-2 border-amber-500
                           disabled:opacity-50"
              >
                RESET E-STOP
              </button>
            )}
          </div>

          {/* ── ARM / DISARM toggle ──────────────────────────── */}
          <div className="flex gap-2">
            <button
              onClick={arm}
              disabled={loading || armed || estopped}
              className="flex-1 py-2 rounded font-mono text-[10px] uppercase tracking-widest transition-all
                         bg-emerald-900/40 border border-emerald-700/60 text-emerald-300
                         hover:bg-emerald-800/50 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              ARM
            </button>
            <button
              onClick={disarm}
              disabled={loading || !armed || estopped}
              className="flex-1 py-2 rounded font-mono text-[10px] uppercase tracking-widest transition-all
                         bg-slate-800/60 border border-slate-600/50 text-slate-300
                         hover:bg-slate-700/50 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              DISARM
            </button>
          </div>

          {/* ── Status grid ─────────────────────────────────── */}
          <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
            <div className="bg-slate-900/50 border border-slate-700/40 rounded px-2 py-1.5">
              <div className="text-slate-500 uppercase tracking-widest text-[9px] mb-0.5">ARM STATE</div>
              <div className={armed ? 'text-emerald-400 font-bold' : 'text-slate-400'}>
                {armed ? 'ARMED' : 'DISARMED'}
              </div>
            </div>
            <div className="bg-slate-900/50 border border-slate-700/40 rounded px-2 py-1.5">
              <div className="text-slate-500 uppercase tracking-widest text-[9px] mb-0.5">ESCALATION</div>
              <div className={`font-bold ${escalationColor(level)}`}>{level.replace('_', ' ')}</div>
            </div>
            <div className="bg-slate-900/50 border border-slate-700/40 rounded px-2 py-1.5">
              <div className="text-slate-500 uppercase tracking-widest text-[9px] mb-0.5">WORKSPACE</div>
              <div className={status.workspace_inside ? 'text-emerald-400' : 'text-red-400 font-bold'}>
                {status.workspace_inside ? 'INSIDE' : 'OUTSIDE'}
              </div>
            </div>
            <div className="bg-slate-900/50 border border-slate-700/40 rounded px-2 py-1.5">
              <div className="text-slate-500 uppercase tracking-widest text-[9px] mb-0.5">VIOLATIONS</div>
              <div className={status.violations.length > 0 ? 'text-amber-400 font-bold' : 'text-slate-400'}>
                {status.violations.length} recent
              </div>
            </div>
          </div>

          {/* ── Per-joint torque bars ────────────────────────── */}
          {status.torque_limits && status.last_action && status.last_action.length > 0 && (
            <div className="flex flex-col gap-1">
              <div className="font-mono text-[9px] text-slate-500 uppercase tracking-widest">Torque vs Limit</div>
              {status.last_action.slice(0, 8).map((torque, i) => {
                const limit = status.torque_limits![i] ?? 200;
                const pct = Math.min(100, (Math.abs(torque) / limit) * 100);
                const barColor = pct > 90 ? 'bg-red-500' : pct > 70 ? 'bg-amber-500' : 'bg-cyan-500';
                return (
                  <div key={i} className="flex items-center gap-2">
                    <span className="font-mono text-[9px] text-slate-500 w-8 text-right">J{i}</span>
                    <div className="flex-1 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${pct}%` }} />
                    </div>
                    <span className="font-mono text-[9px] text-slate-400 w-16 text-right">
                      {torque.toFixed(1)} / {limit.toFixed(0)}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {/* ── Violation log ────────────────────────────────── */}
          <div className="flex flex-col gap-1">
            <div className="font-mono text-[9px] text-slate-500 uppercase tracking-widest">Recent Violations</div>
            <div className="flex flex-col gap-0.5 max-h-40 overflow-y-auto">
              {status.violations.length === 0 ? (
                <div className="font-mono text-[9px] text-slate-600 py-1">No violations</div>
              ) : (
                [...status.violations].reverse().slice(0, 20).map((v, i) => (
                  <div key={i} className="flex items-center gap-2 px-1.5 py-0.5 rounded bg-slate-900/40 border border-slate-800/50">
                    <span className={`font-mono text-[9px] uppercase w-16 font-bold ${
                      v.kind === 'workspace' ? 'text-red-400' :
                      v.kind === 'thermal'   ? 'text-orange-400' : 'text-amber-400'
                    }`}>{v.kind}</span>
                    <span className="font-mono text-[9px] text-slate-500">
                      {v.joint >= 0 ? `J${v.joint} ` : ''}
                      {v.value.toFixed(2)} / {v.limit.toFixed(2)}
                    </span>
                    <span className="ml-auto font-mono text-[9px] text-slate-600">
                      {new Date(v.timestamp * 1000).toLocaleTimeString()}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default SafetyPanel;

