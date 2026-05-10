import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { AlertTriangle, Plus, Trash2, Wifi, WifiOff } from "lucide-react";

// ── Types ────────────────────────────────────────────────────────────────────

type RobotStatus = "healthy" | "degraded" | "critical";

interface RobotParams {
  mass: number;
  friction: number;
  inertia: number;
}

interface RobotHealth {
  robot_id: string;
  platform: string;
  status: RobotStatus;
  step_count: number;
  loop_time_ms: number;
  residual_norm: number;
  uncertainty: number;
  params: RobotParams;
}

interface FleetHealthResponse {
  total: number;
  healthy: number;
  degraded: number;
  critical: number;
  robots: Record<string, RobotHealth>;
}

// ── Constants ────────────────────────────────────────────────────────────────

const API_BASE = "http://localhost:8000";
const POLL_INTERVAL_MS = 500;

const PLATFORM_DYNAMICS = [
  "quadrotor",
  "balancing_bot",
  "manipulator_arm",
  "ground_rover",
  "bipedal_walker",
  "fixed_wing",
  "underwater_vehicle",
  "legged_hexapod",
] as const;

type Platform = (typeof PLATFORM_DYNAMICS)[number];

// ── Helpers ──────────────────────────────────────────────────────────────────

const statusColor = (s: RobotStatus): string =>
  s === "healthy" ? "text-green" : s === "degraded" ? "text-amber" : "text-red";

const statusBgBorder = (s: RobotStatus): string =>
  s === "healthy"
    ? "bg-green/10 border-green text-green"
    : s === "degraded"
      ? "bg-amber/10 border-amber text-amber"
      : "bg-red/10 border-red text-red";

const residualColor = (v: number): string =>
  v < 0.1 ? "bg-green" : v < 0.5 ? "bg-amber" : "bg-red";

const clamp01 = (v: number): number => Math.min(1, Math.max(0, v));

// ── Sub-components ────────────────────────────────────────────────────────────

interface BarProps {
  value: number; // 0-1
  colorClass: string;
  label: string;
}

const MetricBar: React.FC<BarProps> = ({ value, colorClass, label }) => (
  <div className="space-y-0.5">
    <div className="flex justify-between items-center">
      <span className="font-mono text-[9px] text-textDim uppercase tracking-widest">
        {label}
      </span>
      <span className="font-mono text-[9px] text-textSecondary">
        {value.toFixed(3)}
      </span>
    </div>
    <div className="h-1 bg-bg rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-300 ${colorClass}`}
        style={{ width: `${clamp01(value) * 100}%` }}
      />
    </div>
  </div>
);

interface ParamRowProps {
  label: string;
  value: number;
}

const ParamRow: React.FC<ParamRowProps> = ({ label, value }) => (
  <div className="flex justify-between">
    <span className="font-mono text-[9px] text-textDim">{label}</span>
    <span className="font-mono text-[9px] text-textSecondary">{value.toFixed(4)}</span>
  </div>
);

interface RobotCardProps {
  robot: RobotHealth;
  onRemove: (id: string) => void;
  removing: boolean;
}

const RobotCard: React.FC<RobotCardProps> = ({ robot, onRemove, removing }) => (
  <div className="bg-bgRaised border border-border rounded p-4 flex flex-col gap-3 relative">
    {/* Header */}
    <div className="flex items-start justify-between gap-2">
      <div className="flex flex-col gap-0.5 min-w-0">
        <span className="font-mono font-bold text-white text-sm truncate">
          {robot.robot_id}
        </span>
        <span className="font-mono text-[10px] text-textDim truncate">
          {robot.platform}
        </span>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <span
          className={`inline-block px-2 py-0.5 rounded-full border font-display text-[8px] font-bold tracking-widest uppercase ${statusBgBorder(robot.status)}`}
        >
          {robot.status}
        </span>
        <button
          onClick={() => onRemove(robot.robot_id)}
          disabled={removing}
          className="p-1 text-textDim hover:text-red transition-colors disabled:opacity-40"
          aria-label={`Remove ${robot.robot_id}`}
          title="Remove robot"
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>

    {/* Metrics */}
    <div className="space-y-2">
      <MetricBar
        value={robot.residual_norm}
        colorClass={residualColor(robot.residual_norm)}
        label="Residual"
      />
      <MetricBar
        value={robot.uncertainty}
        colorClass="bg-cyan"
        label="Uncertainty"
      />
    </div>

    {/* Loop time + steps */}
    <div className="flex gap-4">
      <div className="flex flex-col gap-0.5">
        <span className="font-mono text-[8px] text-textDim uppercase tracking-widest">
          Loop
        </span>
        <span className="font-mono text-[11px] text-white">
          {robot.loop_time_ms.toFixed(1)}
          <span className="text-textDim text-[9px]"> ms</span>
        </span>
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="font-mono text-[8px] text-textDim uppercase tracking-widest">
          Steps
        </span>
        <span className="font-mono text-[11px] text-white">
          {robot.step_count.toLocaleString()}
        </span>
      </div>
    </div>

    {/* Learned params */}
    <div className="border-t border-borderDim pt-2 space-y-1">
      <span className="font-display text-[8px] text-textDim uppercase tracking-widest">
        Learned Params
      </span>
      <ParamRow label="mass" value={robot.params.mass} />
      <ParamRow label="friction" value={robot.params.friction} />
      <ParamRow label="inertia" value={robot.params.inertia} />
    </div>
  </div>
);

// ── Main Component ────────────────────────────────────────────────────────────

export const FleetDashboard: React.FC = () => {
  const [data, setData] = useState<FleetHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [stale, setStale] = useState(false);
  const [removingIds, setRemovingIds] = useState<Set<string>>(new Set());

  // Add robot form state
  const [newRobotId, setNewRobotId] = useState("");
  const [newPlatform, setNewPlatform] = useState<Platform | "">("");
  const [urdfPath, setUrdfPath] = useState("");
  const [adding, setAdding] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Polling ──────────────────────────────────────────────────────────────

  const fetchHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/fleet/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = (await res.json()) as FleetHealthResponse;
      setData(json);
      setStale(false);
      if (loading) setLoading(false);
    } catch {
      setStale(true);
      if (loading) setLoading(false);
    }
  }, [loading]);

  useEffect(() => {
    void fetchHealth();
    intervalRef.current = setInterval(() => {
      void fetchHealth();
    }, POLL_INTERVAL_MS);
    return () => {
      if (intervalRef.current !== null) clearInterval(intervalRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Remove robot ─────────────────────────────────────────────────────────

  const handleRemove = useCallback(async (id: string) => {
    setRemovingIds((prev) => new Set([...prev, id]));
    try {
      await fetch(`${API_BASE}/fleet/robot/${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
      // Next poll will update data; force immediate refresh
      await fetchHealth();
    } catch {
      // Silently fail; stale banner will appear if polling is also broken
    } finally {
      setRemovingIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  }, [fetchHealth]);

  // ── Add robot ────────────────────────────────────────────────────────────

  const handleAdd = useCallback(async () => {
    if (!newRobotId.trim()) {
      setAddError("Robot ID is required.");
      return;
    }
    if (!newPlatform && !urdfPath.trim()) {
      setAddError("Select a platform or enter a URDF path.");
      return;
    }
    setAddError(null);
    setAdding(true);
    try {
      const body: Record<string, string> = { robot_id: newRobotId.trim() };
      if (urdfPath.trim()) {
        body.urdf_path = urdfPath.trim();
      } else if (newPlatform) {
        body.platform = newPlatform;
      }
      const res = await fetch(`${API_BASE}/fleet/add`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const text = await res.text();
        setAddError(`Failed: ${text || res.statusText}`);
        return;
      }
      setNewRobotId("");
      setNewPlatform("");
      setUrdfPath("");
      await fetchHealth();
    } catch (err) {
      setAddError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setAdding(false);
    }
  }, [newRobotId, newPlatform, urdfPath, fetchHealth]);

  // ── Render ───────────────────────────────────────────────────────────────

  const robots = data ? Object.values(data.robots) : [];

  return (
    <div className="min-h-screen bg-void text-white p-4 space-y-4">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <h1 className="font-display text-[13px] uppercase tracking-widest text-white">
          Fleet Dashboard
        </h1>
        <div className="flex items-center gap-1.5">
          {stale ? (
            <WifiOff size={12} className="text-amber" />
          ) : (
            <Wifi size={12} className="text-green" />
          )}
          <span
            className={`font-mono text-[9px] ${stale ? "text-amber" : "text-green"}`}
          >
            {stale ? "STALE" : "LIVE"}
          </span>
        </div>
      </div>

      {/* Stale banner */}
      {stale && data && (
        <div className="flex items-center gap-2 px-3 py-2 border border-amber bg-amber/10 rounded">
          <AlertTriangle size={12} className="text-amber shrink-0" />
          <span className="font-mono text-[10px] text-amber">
            Connection lost — showing last known data
          </span>
        </div>
      )}

      {/* Fleet summary bar */}
      {data && (
        <div className="flex flex-wrap gap-4 border border-border bg-bgRaised rounded px-4 py-3">
          <SummaryPill label="TOTAL" value={data.total} colorClass="text-white" />
          <SummaryPill label="HEALTHY" value={data.healthy} colorClass="text-green" />
          <SummaryPill label="DEGRADED" value={data.degraded} colorClass="text-amber" />
          <SummaryPill label="CRITICAL" value={data.critical} colorClass="text-red" />
        </div>
      )}

      {/* Add robot section */}
      <div className="border border-border bg-bgRaised rounded p-4 space-y-3">
        <div className="flex items-center gap-2">
          <Plus size={12} className="text-cyan" />
          <span className="font-display text-[10px] uppercase tracking-widest text-cyan">
            Add Robot
          </span>
        </div>

        <div className="flex flex-wrap gap-2 items-end">
          {/* Robot ID */}
          <div className="flex flex-col gap-1">
            <label className="font-mono text-[9px] text-textDim uppercase tracking-widest">
              Robot ID
            </label>
            <input
              type="text"
              value={newRobotId}
              onChange={(e) => setNewRobotId(e.target.value)}
              placeholder="e.g. arm2"
              className="bg-bg border border-borderDim rounded px-2 py-1.5 font-mono text-[11px] text-white placeholder:text-textDim focus:outline-none focus:border-cyan w-32"
            />
          </div>

          {/* Platform dropdown */}
          <div className="flex flex-col gap-1">
            <label className="font-mono text-[9px] text-textDim uppercase tracking-widest">
              Platform
            </label>
            <select
              value={newPlatform}
              onChange={(e) => {
                setNewPlatform(e.target.value as Platform | "");
                if (e.target.value) setUrdfPath("");
              }}
              className="bg-bg border border-borderDim rounded px-2 py-1.5 font-mono text-[11px] text-white focus:outline-none focus:border-cyan w-44 appearance-none"
            >
              <option value="">— select platform —</option>
              {PLATFORM_DYNAMICS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          <span className="font-mono text-[10px] text-textDim self-end mb-2">or</span>

          {/* URDF path */}
          <div className="flex flex-col gap-1">
            <label className="font-mono text-[9px] text-textDim uppercase tracking-widest">
              URDF Path
            </label>
            <input
              type="text"
              value={urdfPath}
              onChange={(e) => {
                setUrdfPath(e.target.value);
                if (e.target.value) setNewPlatform("");
              }}
              placeholder="/path/to/robot.urdf"
              className="bg-bg border border-borderDim rounded px-2 py-1.5 font-mono text-[11px] text-white placeholder:text-textDim focus:outline-none focus:border-cyan w-52"
            />
          </div>

          <button
            onClick={() => void handleAdd()}
            disabled={adding}
            className="px-4 py-2 bg-green text-black font-display text-[11px] font-bold uppercase tracking-widest hover:bg-white transition-all disabled:opacity-50 disabled:cursor-not-allowed rounded self-end"
          >
            {adding ? "Adding..." : "Add"}
          </button>
        </div>

        {addError && (
          <p className="font-mono text-[10px] text-red">{addError}</p>
        )}
      </div>

      {/* Robot grid */}
      {loading && !data ? (
        <div className="flex items-center justify-center py-16 gap-2">
          <div className="w-2 h-2 bg-cyan rounded-full animate-pulse" />
          <span className="font-mono text-[11px] text-textDim">Connecting...</span>
        </div>
      ) : robots.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 gap-2 text-textDim">
          <span className="font-display text-[11px] uppercase tracking-widest">
            No robots in fleet
          </span>
          <span className="font-mono text-[10px]">
            Add a robot above to get started
          </span>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {robots.map((robot) => (
            <RobotCard
              key={robot.robot_id}
              robot={robot}
              onRemove={(id) => void handleRemove(id)}
              removing={removingIds.has(robot.robot_id)}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// ── Summary pill ─────────────────────────────────────────────────────────────

interface SummaryPillProps {
  label: string;
  value: number;
  colorClass: string;
}

const SummaryPill: React.FC<SummaryPillProps> = ({ label, value, colorClass }) => (
  <div className="flex items-baseline gap-1.5">
    <span className={`font-mono text-lg font-bold ${colorClass}`}>{value}</span>
    <span className="font-display text-[9px] uppercase tracking-widest text-textDim">
      {label}
    </span>
  </div>
);
