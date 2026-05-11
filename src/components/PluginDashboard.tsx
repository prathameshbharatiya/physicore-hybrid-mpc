import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';

// ── Types ────────────────────────────────────────────────────────────────────

interface PanelPosition {
  row: number;
  col: number;
  w: number;
  h: number;
}

interface DashboardPanelSpec {
  panel_id:      string;
  title:         string;
  chart_type:    'line' | 'bar' | 'gauge' | 'value' | 'heatmap' | 'custom';
  data_endpoint: string;
  refresh_hz:    number;
  position:      PanelPosition;
  extra?:        Record<string, unknown>;
}

interface PluginStatus {
  plugin_id:   string;
  name:        string;
  version:     string;
  disabled:    boolean;
  error_count: number;
  last_error:  string | null;
}

interface SeriesPoint {
  time:  number;
  value: number;
}

interface Series {
  name:   string;
  color?: string;
  points: SeriesPoint[];
}

interface LineBarData {
  series: Series[];
}

interface GaugeData {
  value:   number;
  max:     number;
  percent: number;
  unit:    string;
  color?:  string;
}

interface ValueData {
  value:  number | string;
  unit?:  string;
  label?: string;
  color?: string;
}

interface BarChartData {
  labels: string[];
  values: number[];
  colors?: string[];
}

// ── Colour helpers ────────────────────────────────────────────────────────────

const CHART_COLORS = ['#06b6d4', '#f59e0b', '#22c55e', '#8b5cf6', '#ec4899'];

// ── Gauge SVG ────────────────────────────────────────────────────────────────

const GaugeChart: React.FC<{ data: GaugeData }> = ({ data }) => {
  const pct     = Math.min(Math.max(data.percent ?? 0, 0), 100);
  const angle   = -135 + pct * 2.7;   // -135° to +135°
  const color   = data.color ?? '#06b6d4';
  const r       = 48;
  const cx      = 60;
  const cy      = 65;
  const startA  = -135 * (Math.PI / 180);
  const endA    = angle * (Math.PI / 180);
  const x1 = cx + r * Math.cos(startA);
  const y1 = cy + r * Math.sin(startA);
  const x2 = cx + r * Math.cos(endA);
  const y2 = cy + r * Math.sin(endA);
  const large = pct > 50 ? 1 : 0;

  return (
    <div className="flex flex-col items-center justify-center h-full">
      <svg width="120" height="80" viewBox="0 0 120 80">
        <path
          d={`M ${cx + r * Math.cos(-135 * Math.PI / 180)} ${cy + r * Math.sin(-135 * Math.PI / 180)} A ${r} ${r} 0 1 1 ${cx + r * Math.cos(135 * Math.PI / 180)} ${cy + r * Math.sin(135 * Math.PI / 180)}`}
          fill="none" stroke="#1e293b" strokeWidth="10" strokeLinecap="round"
        />
        {pct > 0 && (
          <path
            d={`M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`}
            fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
          />
        )}
        <text x={cx} y={cy - 8} textAnchor="middle" fill="#f1f5f9" fontSize="16" fontWeight="bold">
          {typeof data.value === 'number' ? data.value.toFixed(1) : data.value}
        </text>
        <text x={cx} y={cy + 6} textAnchor="middle" fill="#94a3b8" fontSize="8">
          {data.unit}
        </text>
      </svg>
      <span className="text-xs text-slate-400">{pct.toFixed(1)}%</span>
    </div>
  );
};

// ── Value display ─────────────────────────────────────────────────────────────

const ValueDisplay: React.FC<{ data: ValueData }> = ({ data }) => (
  <div className="flex flex-col items-center justify-center h-full gap-1">
    {data.label && <span className="text-xs text-slate-400 uppercase tracking-widest">{data.label}</span>}
    <span
      className="text-3xl font-bold font-mono"
      style={{ color: data.color ?? '#06b6d4' }}
    >
      {typeof data.value === 'number' ? data.value.toFixed(3) : data.value}
    </span>
    {data.unit && <span className="text-xs text-slate-500">{data.unit}</span>}
  </div>
);

// ── Line / Area chart ─────────────────────────────────────────────────────────

const LinePanel: React.FC<{ data: LineBarData }> = ({ data }) => {
  const series = data.series ?? [];
  const merged: Record<string, number>[] = [];

  if (series.length > 0) {
    const pts = series[0].points ?? [];
    pts.forEach((p, i) => {
      const row: Record<string, number> = { i };
      series.forEach(s => { row[s.name] = s.points[i]?.value ?? 0; });
      merged.push(row);
    });
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={merged} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
        <XAxis dataKey="i" hide />
        <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
        <Tooltip
          contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6 }}
          labelStyle={{ color: '#94a3b8' }}
        />
        {series.map((s, idx) => (
          <Area
            key={s.name}
            type="monotone"
            dataKey={s.name}
            stroke={s.color ?? CHART_COLORS[idx % CHART_COLORS.length]}
            fill={s.color ?? CHART_COLORS[idx % CHART_COLORS.length]}
            fillOpacity={0.15}
            dot={false}
            isAnimationActive={false}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
};

// ── Bar chart ─────────────────────────────────────────────────────────────────

const BarPanel: React.FC<{ data: BarChartData }> = ({ data }) => {
  const chartData = (data.labels ?? []).map((label, i) => ({
    label,
    value: data.values?.[i] ?? 0,
    fill:  data.colors?.[i] ?? CHART_COLORS[i % CHART_COLORS.length],
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 20 }}>
        <XAxis
          dataKey="label"
          tick={{ fontSize: 9, fill: '#94a3b8' }}
          angle={-20}
          textAnchor="end"
        />
        <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
        <Tooltip
          contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6 }}
          labelStyle={{ color: '#94a3b8' }}
        />
        <Bar dataKey="value" isAnimationActive={false}>
          {chartData.map((entry, i) => (
            <rect key={i} fill={entry.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

// ── Plugin Panel ──────────────────────────────────────────────────────────────

interface PluginPanelProps {
  pluginId: string;
  panel:    DashboardPanelSpec;
  apiBase:  string;
}

const PluginPanel: React.FC<PluginPanelProps> = ({ pluginId, panel, apiBase }) => {
  const [data, setData]   = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef          = useRef<number | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const url = `${apiBase}/plugins/${pluginId}/${panel.panel_id}/data`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [apiBase, pluginId, panel.panel_id]);

  useEffect(() => {
    fetchData();
    const ms = Math.round(1000 / Math.max(panel.refresh_hz, 0.1));
    timerRef.current = window.setInterval(fetchData, ms);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [fetchData, panel.refresh_hz]);

  const renderContent = () => {
    if (error) return <div className="text-red-400 text-xs p-2">{error}</div>;
    if (data == null) return <div className="text-slate-500 text-xs p-2 animate-pulse">Loading…</div>;

    switch (panel.chart_type) {
      case 'line':
        return <LinePanel data={data as LineBarData} />;
      case 'bar':
        return <BarPanel data={data as BarChartData} />;
      case 'gauge':
        return <GaugeChart data={data as GaugeData} />;
      case 'value':
        return <ValueDisplay data={data as ValueData} />;
      case 'heatmap':
        return (
          <pre className="text-xs text-cyan-300 p-2 overflow-auto h-full">
            {JSON.stringify(data, null, 2)}
          </pre>
        );
      case 'custom':
        return (
          <iframe
            src={`${apiBase}/plugins/${pluginId}/${panel.panel_id}/frame`}
            className="w-full h-full border-0 rounded"
            title={panel.title}
            sandbox="allow-scripts allow-same-origin"
          />
        );
      default:
        return <div className="text-slate-400 text-xs p-2">Unsupported chart type</div>;
    }
  };

  return (
    <div className="bg-[#0f172a] border border-slate-700/50 rounded-xl p-3 flex flex-col gap-2 h-full">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">{panel.title}</span>
        <span className="text-[10px] text-slate-600 font-mono">{panel.refresh_hz}Hz</span>
      </div>
      <div className="flex-1 min-h-0">
        {renderContent()}
      </div>
    </div>
  );
};

// ── Plugin Manager Panel ──────────────────────────────────────────────────────

interface PluginManagerPanelProps {
  apiBase:  string;
  plugins:  PluginStatus[];
  onReload: (id: string) => Promise<void>;
  onRefresh: () => void;
}

const PluginManagerPanel: React.FC<PluginManagerPanelProps> = ({
  apiBase, plugins, onReload, onRefresh,
}) => (
  <div className="bg-[#0f172a] border border-slate-700/50 rounded-xl p-4">
    <div className="flex items-center justify-between mb-3">
      <h3 className="text-sm font-semibold text-slate-200">Plugin Manager</h3>
      <button
        onClick={onRefresh}
        className="text-xs px-2 py-1 rounded bg-slate-800 text-slate-400 hover:text-cyan-400 transition-colors"
      >
        Refresh
      </button>
    </div>

    {plugins.length === 0 ? (
      <p className="text-slate-500 text-xs">No plugins loaded. Drop plugins into ./plugins/ and reload.</p>
    ) : (
      <div className="space-y-2">
        {plugins.map(p => (
          <div
            key={p.plugin_id}
            className="flex items-center justify-between px-3 py-2 rounded-lg bg-slate-800/50 border border-slate-700/30"
          >
            <div className="flex flex-col gap-0.5">
              <span className="text-xs font-medium text-slate-200">{p.name}</span>
              <span className="text-[10px] text-slate-500 font-mono">{p.plugin_id} v{p.version}</span>
              {p.last_error && (
                <span className="text-[10px] text-red-400 truncate max-w-xs">{p.last_error}</span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                p.disabled
                  ? 'bg-red-900/40 text-red-400'
                  : p.error_count > 0
                    ? 'bg-amber-900/40 text-amber-400'
                    : 'bg-emerald-900/40 text-emerald-400'
              }`}>
                {p.disabled ? 'DISABLED' : p.error_count > 0 ? `${p.error_count} err` : 'OK'}
              </span>
              <button
                onClick={() => onReload(p.plugin_id)}
                className="text-[10px] px-2 py-1 rounded bg-cyan-900/30 text-cyan-400 hover:bg-cyan-800/30 transition-colors"
              >
                Reload
              </button>
            </div>
          </div>
        ))}
      </div>
    )}
  </div>
);

// ── Main PluginDashboard ──────────────────────────────────────────────────────

interface PluginDashboardProps {
  apiBase?: string;
}

const PluginDashboard: React.FC<PluginDashboardProps> = ({
  apiBase = 'http://localhost:8000',
}) => {
  const [plugins,   setPlugins]   = useState<PluginStatus[]>([]);
  const [panelMap,  setPanelMap]  = useState<Record<string, DashboardPanelSpec[]>>({});
  const [loading,   setLoading]   = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const fetchPlugins = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/plugins/`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const statuses: PluginStatus[] = await res.json();
      setPlugins(statuses);
      setFetchError(null);

      // Fetch panel specs for each plugin
      const panelEntries = await Promise.all(
        statuses
          .filter(p => !p.disabled)
          .map(async p => {
            try {
              const r = await fetch(`${apiBase}/plugins/${p.plugin_id}/panels`);
              if (!r.ok) return [p.plugin_id, []] as const;
              const panels: DashboardPanelSpec[] = await r.json();
              return [p.plugin_id, panels] as const;
            } catch {
              return [p.plugin_id, []] as const;
            }
          })
      );

      setPanelMap(Object.fromEntries(panelEntries));
    } catch (e: unknown) {
      setFetchError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    fetchPlugins();
    const t = window.setInterval(fetchPlugins, 5000);
    return () => clearInterval(t);
  }, [fetchPlugins]);

  const handleReload = useCallback(async (pluginId: string) => {
    try {
      await fetch(`${apiBase}/plugins/${pluginId}/reload`, { method: 'POST' });
      await fetchPlugins();
    } catch {
      // ignore — fetchPlugins will surface errors
    }
  }, [apiBase, fetchPlugins]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-40 text-slate-500 text-sm animate-pulse">
        Loading plugins…
      </div>
    );
  }

  if (fetchError) {
    return (
      <div className="rounded-xl border border-red-800/50 bg-red-950/20 p-6 text-center">
        <p className="text-red-400 text-sm font-medium mb-1">Plugin system unavailable</p>
        <p className="text-red-600 text-xs">{fetchError}</p>
        <button
          onClick={fetchPlugins}
          className="mt-3 text-xs px-3 py-1.5 rounded bg-red-900/30 text-red-400 hover:bg-red-800/30 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Manager */}
      <PluginManagerPanel
        apiBase={apiBase}
        plugins={plugins}
        onReload={handleReload}
        onRefresh={fetchPlugins}
      />

      {/* Panels per plugin */}
      {plugins.filter(p => !p.disabled).map(plugin => {
        const panels = panelMap[plugin.plugin_id] ?? [];
        if (panels.length === 0) return null;

        return (
          <div key={plugin.plugin_id}>
            <div className="flex items-center gap-2 mb-3">
              <span className="h-px flex-1 bg-slate-800" />
              <span className="text-xs text-slate-500 uppercase tracking-widest font-medium">
                {plugin.name}
              </span>
              <span className="h-px flex-1 bg-slate-800" />
            </div>

            <div
              className="grid gap-3"
              style={{ gridTemplateColumns: 'repeat(12, 1fr)' }}
            >
              {panels.map(panel => (
                <div
                  key={panel.panel_id}
                  style={{
                    gridColumn: `span ${Math.min(panel.position.w, 12)}`,
                    minHeight:  `${panel.position.h * 60}px`,
                  }}
                >
                  <PluginPanel
                    pluginId={plugin.plugin_id}
                    panel={panel}
                    apiBase={apiBase}
                  />
                </div>
              ))}
            </div>
          </div>
        );
      })}

      {plugins.every(p => p.disabled) && plugins.length > 0 && (
        <div className="text-center text-slate-600 text-xs py-6">
          All plugins are disabled due to errors. Check logs and reload.
        </div>
      )}

      {plugins.length === 0 && (
        <div className="text-center text-slate-600 text-xs py-12">
          <p className="text-slate-400 mb-1">No plugins installed</p>
          <p>Drop a plugin directory or .py file into <code className="text-cyan-700">./plugins/</code> and restart the backend.</p>
        </div>
      )}
    </div>
  );
};

export default PluginDashboard;
