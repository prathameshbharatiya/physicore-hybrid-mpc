import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

// ─── Types ────────────────────────────────────────────────────────────────────

interface OrgMember {
  user_id: string;
  org_id: string;
  role: string;
  email: string;
  joined_at: number;
  permissions: string[];
}

interface QuotaBar {
  resource: string;
  used: number;
  limit: number;
  pct: number;
  status: 'ok' | 'warning' | 'exceeded';
}

interface OrgUsage {
  org_id: string;
  plan: string;
  robots: QuotaBar;
  plugins: QuotaBar;
  storage_mb: number;
  retention_days: number;
}

interface Organization {
  org_id: string;
  name: string;
  plan: string;
  created_at: number;
  member_ids: string[];
  robot_quota: number;
  plugin_quota: number;
  data_retention_days: number;
  owner_id: string;
}

interface UsageSummary {
  org_id: string;
  period: string;
  steps_this_period: number;
  robots_active: number;
  storage_mb: number;
  plugins_loaded: number;
  plan_limits: Record<string, number>;
}

interface StepsPerDay {
  day: string;
  steps: number;
}

interface AuditEvent {
  event_id: string;
  user_id: string;
  action: string;
  resource: string;
  timestamp: number;
  status: string;
  ip: string;
}

interface TeamDashboardProps {
  baseUrl?: string;
  userId?: string;
  userEmail?: string;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function planBadgeColor(plan: string) {
  if (plan === 'enterprise') return 'bg-purple-700 text-white';
  if (plan === 'pro') return 'bg-blue-700 text-white';
  return 'bg-gray-600 text-gray-200';
}

function roleBadgeColor(role: string) {
  if (role === 'owner') return 'bg-amber-700 text-amber-100';
  if (role === 'admin') return 'bg-indigo-700 text-indigo-100';
  if (role === 'viewer') return 'bg-gray-700 text-gray-300';
  return 'bg-emerald-800 text-emerald-200';
}

function statusColor(status: 'ok' | 'warning' | 'exceeded') {
  if (status === 'exceeded') return 'bg-red-500';
  if (status === 'warning')  return 'bg-amber-400';
  return 'bg-emerald-500';
}

function fmtTs(ts: number) {
  return new Date(ts * 1000).toLocaleString();
}

function QuotaProgress({ label, used, limit, pct, status }: QuotaBar & { label: string }) {
  return (
    <div className="mb-3">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-400">{label}</span>
        <span className={status === 'exceeded' ? 'text-red-400' : status === 'warning' ? 'text-amber-400' : 'text-gray-300'}>
          {used} / {limit} ({pct.toFixed(0)}%)
        </span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${statusColor(status)}`}
          style={{ width: `${Math.min(100, pct)}%` }}
        />
      </div>
      {status !== 'ok' && (
        <p className={`text-xs mt-1 ${status === 'exceeded' ? 'text-red-400' : 'text-amber-400'}`}>
          {status === 'exceeded' ? '⚠ Quota exceeded — upgrade plan' : '⚠ Approaching quota limit'}
        </p>
      )}
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

const TABS = ['overview', 'members', 'usage', 'audit'] as const;
type DashTab = typeof TABS[number];

const TeamDashboard: React.FC<TeamDashboardProps> = ({
  baseUrl = 'https://physicore-hybrid-mpc-production.up.railway.app',
  userId = '',
  userEmail = '',
}) => {
  const [orgId, setOrgId]         = useState('');
  const [orgIdInput, setOrgIdInput] = useState('');
  const [org, setOrg]             = useState<Organization | null>(null);
  const [usage, setUsage]         = useState<OrgUsage | null>(null);
  const [summary, setSummary]     = useState<UsageSummary | null>(null);
  const [members, setMembers]     = useState<OrgMember[]>([]);
  const [stepsData, setStepsData] = useState<StepsPerDay[]>([]);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [tab, setTab]             = useState<DashTab>('overview');
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);

  // Create org form
  const [createName, setCreateName] = useState('');
  const [showCreate, setShowCreate] = useState(false);

  // Invite form
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole]   = useState('member');

  const headers = { 'Content-Type': 'application/json', 'X-User-Id': userId };

  const load = useCallback(async (id: string) => {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const [orgRes, membersRes, usageRes, summaryRes, stepsRes, auditRes] = await Promise.all([
        fetch(`${baseUrl}/api/orgs/${id}`, { headers }),
        fetch(`${baseUrl}/api/orgs/${id}/members`, { headers }),
        fetch(`${baseUrl}/api/orgs/${id}/usage`, { headers }),
        fetch(`${baseUrl}/api/billing/usage?org_id=${id}`, { headers }),
        fetch(`${baseUrl}/api/billing/steps_per_day?org_id=${id}&days=30`, { headers }),
        fetch(`${baseUrl}/api/audit/events?org_id=${id}&limit=50`, { headers }),
      ]);
      if (orgRes.ok) setOrg(await orgRes.json());
      if (membersRes.ok) setMembers(await membersRes.json());
      if (usageRes.ok) setUsage(await usageRes.json());
      if (summaryRes.ok) setSummary(await summaryRes.json());
      if (stepsRes.ok) setStepsData(await stepsRes.json());
      if (auditRes.ok) setAuditEvents(await auditRes.json());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [baseUrl, userId]);

  useEffect(() => {
    if (orgId) load(orgId);
  }, [orgId, load]);

  const handleCreateOrg = async () => {
    if (!createName.trim()) return;
    try {
      const res = await fetch(`${baseUrl}/api/orgs`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ name: createName, owner_id: userId || 'local' }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data: Organization = await res.json();
      setOrgId(data.org_id);
      setShowCreate(false);
      setCreateName('');
    } catch (e) {
      setError(String(e));
    }
  };

  const handleInvite = async () => {
    if (!inviteEmail || !orgId) return;
    try {
      await fetch(`${baseUrl}/api/orgs/${orgId}/invite`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ email: inviteEmail, role: inviteRole }),
      });
      setInviteEmail('');
      load(orgId);
    } catch (e) {
      setError(String(e));
    }
  };

  const handleRemoveMember = async (uid: string) => {
    if (!orgId) return;
    await fetch(`${baseUrl}/api/orgs/${orgId}/members/${uid}`, {
      method: 'DELETE', headers,
    });
    load(orgId);
  };

  const handleChangeRole = async (uid: string, newRole: string) => {
    if (!orgId) return;
    await fetch(`${baseUrl}/api/orgs/${orgId}/members/${uid}`, {
      method: 'PUT',
      headers,
      body: JSON.stringify({ role: newRole }),
    });
    load(orgId);
  };

  const myRole = members.find(m => m.user_id === userId)?.role || 'viewer';
  const canAdmin = myRole === 'owner' || myRole === 'admin';

  // ── Org picker ────────────────────────────────────────────────────────────

  if (!orgId) {
    return (
      <div className="flex flex-col gap-6 p-6 text-gray-200">
        <div className="flex items-center gap-4">
          <input
            placeholder="Enter org ID..."
            value={orgIdInput}
            onChange={e => setOrgIdInput(e.target.value)}
            className="flex-1 bg-gray-800 rounded px-3 py-2 text-sm text-gray-200"
            onKeyDown={e => e.key === 'Enter' && setOrgId(orgIdInput.trim())}
          />
          <button
            onClick={() => setOrgId(orgIdInput.trim())}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-sm font-semibold"
          >Load Org</button>
          <button
            onClick={() => setShowCreate(!showCreate)}
            className="px-4 py-2 bg-emerald-700 hover:bg-emerald-600 rounded text-sm font-semibold"
          >+ New Org</button>
        </div>
        {showCreate && (
          <div className="bg-gray-800 rounded-lg p-4 flex gap-3 items-center">
            <input
              placeholder="Organization name..."
              value={createName}
              onChange={e => setCreateName(e.target.value)}
              className="flex-1 bg-gray-700 rounded px-3 py-2 text-sm text-gray-200"
            />
            <button
              onClick={handleCreateOrg}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded text-sm font-semibold"
            >Create</button>
          </div>
        )}
        {error && <p className="text-red-400 text-sm">{error}</p>}
      </div>
    );
  }

  if (loading) {
    return <div className="p-8 text-gray-400 text-sm">Loading organization…</div>;
  }

  // ── Main dashboard ────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-4 text-sm text-gray-200">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="font-bold text-white text-base">{org?.name ?? orgId}</span>
          {org && (
            <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${planBadgeColor(org.plan)}`}>
              {org.plan}
            </span>
          )}
        </div>
        <button
          onClick={() => { setOrgId(''); setOrg(null); }}
          className="text-xs text-gray-500 hover:text-gray-300"
        >← Switch org</button>
      </div>

      {/* Quota warnings */}
      {usage && (
        <>
          {usage.robots.status !== 'ok' && (
            <div className="bg-amber-900/40 border border-amber-700 rounded px-3 py-2 text-xs text-amber-300">
              ⚠ Robot quota at {usage.robots.pct.toFixed(0)}% — {usage.robots.used}/{usage.robots.limit} used
            </div>
          )}
          {usage.plugins.status !== 'ok' && (
            <div className="bg-amber-900/40 border border-amber-700 rounded px-3 py-2 text-xs text-amber-300">
              ⚠ Plugin quota at {usage.plugins.pct.toFixed(0)}% — {usage.plugins.used}/{usage.plugins.limit} used
            </div>
          )}
        </>
      )}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-700 pb-1">
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-3 py-1.5 text-xs font-semibold uppercase tracking-wider rounded-t transition-colors
              ${tab === t ? 'bg-indigo-700 text-white' : 'text-gray-400 hover:text-gray-200'}`}
          >{t}</button>
        ))}
      </div>

      {/* ── Overview tab ── */}
      {tab === 'overview' && org && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-xs text-gray-400 font-semibold uppercase tracking-wider mb-3">Quotas</h3>
            {usage && (
              <>
                <QuotaProgress label="Robots" {...usage.robots} />
                <QuotaProgress label="Plugins" {...usage.plugins} />
                <div className="text-xs text-gray-400 mt-2">
                  Storage: <span className="text-gray-200">{usage.storage_mb.toFixed(1)} MB</span>
                  <span className="mx-2">·</span>
                  Retention: <span className="text-gray-200">{usage.retention_days} days</span>
                </div>
              </>
            )}
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-xs text-gray-400 font-semibold uppercase tracking-wider mb-3">Org Info</h3>
            <div className="space-y-1.5 text-xs">
              <div><span className="text-gray-500">ID:</span> <span className="font-mono text-gray-300">{org.org_id}</span></div>
              <div><span className="text-gray-500">Created:</span> <span>{fmtTs(org.created_at)}</span></div>
              <div><span className="text-gray-500">Members:</span> <span>{org.member_ids.length}</span></div>
              <div><span className="text-gray-500">Your role:</span> <span className={`ml-1 px-1.5 py-0.5 rounded text-xs ${roleBadgeColor(myRole)}`}>{myRole}</span></div>
            </div>
          </div>
        </div>
      )}

      {/* ── Members tab ── */}
      {tab === 'members' && (
        <div className="flex flex-col gap-4">
          {canAdmin && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-xs text-gray-400 font-semibold uppercase tracking-wider mb-3">Invite Member</h3>
              <div className="flex gap-2">
                <input
                  placeholder="Email address..."
                  value={inviteEmail}
                  onChange={e => setInviteEmail(e.target.value)}
                  className="flex-1 bg-gray-700 rounded px-3 py-1.5 text-sm text-gray-200"
                />
                <select
                  value={inviteRole}
                  onChange={e => setInviteRole(e.target.value)}
                  className="bg-gray-700 rounded px-2 py-1.5 text-sm text-gray-200"
                >
                  {['member', 'admin', 'viewer'].map(r => (
                    <option key={r} value={r}>{r}</option>
                  ))}
                </select>
                <button onClick={handleInvite}
                  className="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 rounded text-sm font-semibold">
                  Invite
                </button>
              </div>
            </div>
          )}
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-400 border-b border-gray-700">
                  <th className="text-left px-4 py-2">User ID</th>
                  <th className="text-left px-4 py-2">Email</th>
                  <th className="text-left px-4 py-2">Role</th>
                  <th className="text-left px-4 py-2">Joined</th>
                  {canAdmin && <th className="px-4 py-2" />}
                </tr>
              </thead>
              <tbody>
                {members.map(m => (
                  <tr key={m.user_id} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                    <td className="px-4 py-2 font-mono text-gray-300">{m.user_id.slice(0, 12)}</td>
                    <td className="px-4 py-2 text-gray-300">{m.email || '—'}</td>
                    <td className="px-4 py-2">
                      {canAdmin && m.role !== 'owner' ? (
                        <select
                          value={m.role}
                          onChange={e => handleChangeRole(m.user_id, e.target.value)}
                          className={`rounded px-1.5 py-0.5 text-xs ${roleBadgeColor(m.role)}`}
                        >
                          {['member', 'admin', 'viewer'].map(r => (
                            <option key={r} value={r}>{r}</option>
                          ))}
                        </select>
                      ) : (
                        <span className={`px-1.5 py-0.5 rounded text-xs ${roleBadgeColor(m.role)}`}>{m.role}</span>
                      )}
                    </td>
                    <td className="px-4 py-2 text-gray-500">{fmtTs(m.joined_at)}</td>
                    {canAdmin && (
                      <td className="px-4 py-2">
                        {m.role !== 'owner' && m.user_id !== userId && (
                          <button
                            onClick={() => handleRemoveMember(m.user_id)}
                            className="text-red-500 hover:text-red-400 text-xs"
                          >Remove</button>
                        )}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Usage tab ── */}
      {tab === 'usage' && summary && (
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: 'Steps (30d)', value: summary.steps_this_period.toLocaleString() },
              { label: 'Robots Active', value: summary.robots_active },
              { label: 'Storage', value: `${summary.storage_mb.toFixed(1)} MB` },
              { label: 'Plugins', value: summary.plugins_loaded },
            ].map(({ label, value }) => (
              <div key={label} className="bg-gray-800 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">{label}</div>
                <div className="text-lg font-bold text-white">{value}</div>
              </div>
            ))}
          </div>
          {stepsData.length > 0 && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-xs text-gray-400 font-semibold uppercase tracking-wider mb-3">
                Control Steps per Day (last 30 days)
              </h3>
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={stepsData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="day" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', fontSize: 11 }} />
                  <Line type="monotone" dataKey="steps" stroke="#6366f1" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-xs text-gray-400 font-semibold uppercase tracking-wider mb-3">Plan Limits</h3>
            {Object.entries(summary.plan_limits).map(([k, v]) => (
              <div key={k} className="flex justify-between text-xs py-1 border-b border-gray-700/50">
                <span className="text-gray-400">{k}</span>
                <span className="text-gray-200">{v.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Audit tab ── */}
      {tab === 'audit' && (
        <div className="bg-gray-800 rounded-lg overflow-hidden">
          <div className="flex justify-between items-center px-4 py-3 border-b border-gray-700">
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Audit Log</span>
            <button
              onClick={() => window.open(`${baseUrl}/api/audit/export.csv?org_id=${orgId}`)}
              className="text-xs text-indigo-400 hover:text-indigo-300"
            >Export CSV</button>
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-400 border-b border-gray-700">
                <th className="text-left px-4 py-2">Time</th>
                <th className="text-left px-4 py-2">User</th>
                <th className="text-left px-4 py-2">Action</th>
                <th className="text-left px-4 py-2">Resource</th>
                <th className="text-left px-4 py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {auditEvents.map(ev => (
                <tr key={ev.event_id} className="border-b border-gray-700/40 hover:bg-gray-700/20">
                  <td className="px-4 py-2 text-gray-500 font-mono">{fmtTs(ev.timestamp)}</td>
                  <td className="px-4 py-2 text-gray-400 font-mono">{ev.user_id.slice(0, 10)}</td>
                  <td className="px-4 py-2 text-gray-200 font-mono">{ev.action}</td>
                  <td className="px-4 py-2 text-gray-400">{ev.resource}</td>
                  <td className="px-4 py-2">
                    <span className={`px-1.5 py-0.5 rounded text-xs ${ev.status === 'ok' ? 'bg-emerald-900 text-emerald-300' : 'bg-red-900 text-red-300'}`}>
                      {ev.status}
                    </span>
                  </td>
                </tr>
              ))}
              {auditEvents.length === 0 && (
                <tr><td colSpan={5} className="px-4 py-6 text-center text-gray-500">No audit events found</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {error && <p className="text-red-400 text-xs mt-2">{error}</p>}
    </div>
  );
};

export default TeamDashboard;

