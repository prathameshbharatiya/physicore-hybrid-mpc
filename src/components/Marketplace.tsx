import React, { useState, useEffect, useCallback } from 'react';

// ─── Types ────────────────────────────────────────────────────────────────────

interface Review {
  review_id: string;
  author_id: string;
  rating: number;
  text: string;
  created_at: number;
}

interface PluginEntry {
  plugin_id: string;
  name: string;
  version: string;
  description: string;
  author: string;
  author_id: string;
  category: string;
  tags: string[];
  download_count: number;
  rating: number;
  reviews: Review[];
  verified: boolean;
  price: number;
  screenshots: string[];
  submitted_at: number;
  versions: string[];
}

interface MarketplaceProps {
  baseUrl?: string;
  userId?: string;
  installDir?: string;
}

const CATEGORIES = ['all', 'perception', 'safety', 'analytics', 'actuation', 'communication', 'demo'];

// ─── Sub-components ───────────────────────────────────────────────────────────

function StarRating({ value, size = 14 }: { value: number; size?: number }) {
  return (
    <span className="inline-flex gap-0.5" aria-label={`${value.toFixed(1)} stars`}>
      {[1, 2, 3, 4, 5].map(i => (
        <svg key={i} width={size} height={size} viewBox="0 0 24 24" fill={i <= Math.round(value) ? '#f59e0b' : 'none'}
          stroke="#f59e0b" strokeWidth="2">
          <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26" />
        </svg>
      ))}
    </span>
  );
}

function VerifiedBadge() {
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 bg-blue-900 text-blue-300 text-xs rounded font-semibold">
      <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
      Verified
    </span>
  );
}

function CategoryBadge({ cat }: { cat: string }) {
  const colors: Record<string, string> = {
    perception: 'bg-violet-900 text-violet-300',
    safety: 'bg-red-900 text-red-300',
    analytics: 'bg-sky-900 text-sky-300',
    actuation: 'bg-emerald-900 text-emerald-300',
    communication: 'bg-amber-900 text-amber-300',
    demo: 'bg-gray-700 text-gray-300',
  };
  return (
    <span className={`px-1.5 py-0.5 rounded text-xs font-semibold capitalize ${colors[cat] ?? colors.demo}`}>
      {cat}
    </span>
  );
}

function PluginCard({
  entry,
  onSelect,
  onInstall,
}: {
  entry: PluginEntry;
  onSelect: () => void;
  onInstall: () => void;
}) {
  return (
    <div
      className="bg-gray-800 rounded-lg p-4 flex flex-col gap-2 cursor-pointer hover:bg-gray-750 hover:ring-1 hover:ring-indigo-500 transition-all"
      onClick={onSelect}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-semibold text-white text-sm">{entry.name}</span>
            {entry.verified && <VerifiedBadge />}
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            <CategoryBadge cat={entry.category} />
            <span className="text-xs text-gray-500">v{entry.version}</span>
          </div>
        </div>
        <div className="text-right shrink-0">
          <div className="text-xs font-bold text-emerald-400">
            {entry.price === 0 ? 'Free' : `$${entry.price.toFixed(2)}`}
          </div>
        </div>
      </div>

      <p className="text-xs text-gray-400 line-clamp-2">{entry.description}</p>

      <div className="flex items-center justify-between mt-auto">
        <div className="flex items-center gap-2">
          <StarRating value={entry.rating} size={12} />
          <span className="text-xs text-gray-500">({entry.reviews.length})</span>
          <span className="text-xs text-gray-600">·</span>
          <span className="text-xs text-gray-500">{entry.download_count.toLocaleString()} installs</span>
        </div>
        <button
          onClick={e => { e.stopPropagation(); onInstall(); }}
          className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white text-xs rounded font-semibold"
        >
          Install
        </button>
      </div>
    </div>
  );
}

function PluginModal({
  entry,
  onClose,
  onInstall,
  onRate,
  userId,
}: {
  entry: PluginEntry;
  onClose: () => void;
  onInstall: () => void;
  onRate: (rating: number, text: string) => void;
  userId: string;
}) {
  const [ratingVal, setRatingVal] = useState(5);
  const [reviewText, setReviewText] = useState('');
  const [showReview, setShowReview] = useState(false);

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl max-w-2xl w-full max-h-[85vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        <div className="p-6 flex flex-col gap-4">
          {/* Header */}
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-white">{entry.name}</h2>
                {entry.verified && <VerifiedBadge />}
              </div>
              <div className="flex items-center gap-2 mt-1 flex-wrap">
                <CategoryBadge cat={entry.category} />
                <span className="text-xs text-gray-400">by {entry.author || entry.author_id}</span>
                <span className="text-xs text-gray-500">v{entry.version}</span>
              </div>
            </div>
            <button onClick={onClose} className="text-gray-400 hover:text-white text-xl">✕</button>
          </div>

          {/* Rating summary */}
          <div className="flex items-center gap-3">
            <StarRating value={entry.rating} size={16} />
            <span className="text-sm text-gray-300">{entry.rating.toFixed(1)}</span>
            <span className="text-xs text-gray-500">{entry.reviews.length} reviews</span>
            <span className="text-gray-600">·</span>
            <span className="text-xs text-gray-500">{entry.download_count.toLocaleString()} installs</span>
            <span className="text-gray-600">·</span>
            <span className="text-xs font-bold text-emerald-400">
              {entry.price === 0 ? 'Free' : `$${entry.price.toFixed(2)}`}
            </span>
          </div>

          {/* Description */}
          <p className="text-sm text-gray-300">{entry.description}</p>

          {/* Tags */}
          {entry.tags.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {entry.tags.map(t => (
                <span key={t} className="px-2 py-0.5 bg-gray-700 text-gray-300 text-xs rounded">{t}</span>
              ))}
            </div>
          )}

          {/* Screenshots */}
          {entry.screenshots.length > 0 && (
            <div className="flex gap-2 overflow-x-auto py-1">
              {entry.screenshots.map((s, i) => (
                <img key={i} src={`data:image/png;base64,${s}`} alt={`Screenshot ${i+1}`}
                  className="h-32 rounded border border-gray-700 shrink-0" />
              ))}
            </div>
          )}

          {/* Versions */}
          <div className="text-xs text-gray-500">
            Versions: {entry.versions.join(', ')}
          </div>

          {/* Actions */}
          <div className="flex gap-2">
            <button
              onClick={onInstall}
              className="flex-1 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold rounded"
            >
              Install v{entry.version}
            </button>
            {userId && (
              <button
                onClick={() => setShowReview(!showReview)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm rounded"
              >
                Rate
              </button>
            )}
          </div>

          {/* Review form */}
          {showReview && (
            <div className="bg-gray-800 rounded-lg p-4 flex flex-col gap-2">
              <div className="flex gap-2 items-center">
                <span className="text-xs text-gray-400">Rating:</span>
                <div className="flex gap-1">
                  {[1,2,3,4,5].map(v => (
                    <button key={v} onClick={() => setRatingVal(v)}>
                      <svg width="16" height="16" viewBox="0 0 24 24"
                        fill={v <= ratingVal ? '#f59e0b' : 'none'}
                        stroke="#f59e0b" strokeWidth="2">
                        <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26" />
                      </svg>
                    </button>
                  ))}
                </div>
              </div>
              <textarea
                value={reviewText}
                onChange={e => setReviewText(e.target.value)}
                placeholder="Write a review..."
                rows={2}
                className="bg-gray-700 rounded px-2 py-1 text-xs text-gray-200 resize-y"
              />
              <button
                onClick={() => { onRate(ratingVal, reviewText); setShowReview(false); }}
                className="self-end px-3 py-1 bg-amber-600 hover:bg-amber-500 text-white text-xs rounded"
              >Submit Review</button>
            </div>
          )}

          {/* Reviews list */}
          {entry.reviews.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Reviews</h3>
              <div className="flex flex-col gap-2">
                {entry.reviews.slice(0, 5).map(r => (
                  <div key={r.review_id} className="bg-gray-800 rounded p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <StarRating value={r.rating} size={11} />
                      <span className="text-xs text-gray-500">{r.author_id.slice(0, 8)}</span>
                    </div>
                    {r.text && <p className="text-xs text-gray-300">{r.text}</p>}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

const Marketplace: React.FC<MarketplaceProps> = ({
  baseUrl = 'https://physicore-hybrid-mpc-production.up.railway.app',
  userId = '',
  installDir = './plugins',
}) => {
  const [plugins, setPlugins]         = useState<PluginEntry[]>([]);
  const [query, setQuery]             = useState('');
  const [category, setCategory]       = useState('all');
  const [freeOnly, setFreeOnly]       = useState(false);
  const [verifiedOnly, setVerifiedOnly] = useState(false);
  const [selected, setSelected]       = useState<PluginEntry | null>(null);
  const [activeTab, setActiveTab]     = useState<'browse' | 'installed'>('browse');
  const [installed, setInstalled]     = useState<string[]>([]);
  const [loading, setLoading]         = useState(false);
  const [toast, setToast]             = useState<string | null>(null);
  const [error, setError]             = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  };

  const search = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ q: query });
      if (category !== 'all') params.set('category', category);
      if (freeOnly) params.set('free_only', '1');
      if (verifiedOnly) params.set('verified_only', '1');
      const res = await fetch(`${baseUrl}/api/marketplace/search?${params}`);
      if (!res.ok) throw new Error(await res.text());
      setPlugins(await res.json());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [baseUrl, query, category, freeOnly, verifiedOnly]);

  useEffect(() => { search(); }, [category, freeOnly, verifiedOnly]);

  const handleInstall = async (entry: PluginEntry) => {
    try {
      const res = await fetch(`${baseUrl}/api/marketplace/${entry.plugin_id}/install`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ version: entry.version, target_dir: installDir }),
      });
      if (!res.ok) throw new Error(await res.text());
      setInstalled(prev => [...new Set([...prev, entry.plugin_id])]);
      showToast(`✓ ${entry.name} installed`);
    } catch (e) {
      setError(String(e));
    }
  };

  const handleRate = async (entry: PluginEntry, rating: number, text: string) => {
    if (!userId) return;
    try {
      await fetch(`${baseUrl}/api/marketplace/${entry.plugin_id}/rate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rating, text, author_id: userId }),
      });
      showToast('Review submitted!');
      search();
    } catch (e) {
      setError(String(e));
    }
  };

  const installedPlugins = plugins.filter(p => installed.includes(p.plugin_id));

  return (
    <div className="flex flex-col gap-4 text-sm text-gray-200 relative">
      {/* Toast */}
      {toast && (
        <div className="fixed top-4 right-4 z-50 bg-emerald-800 text-emerald-100 px-4 py-2 rounded shadow-lg text-sm">
          {toast}
        </div>
      )}

      {/* Tab bar */}
      <div className="flex gap-2">
        {(['browse', 'installed'] as const).map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            className={`px-4 py-1.5 rounded text-xs font-semibold capitalize transition-colors
              ${activeTab === t ? 'bg-indigo-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
          >
            {t === 'installed' ? `Installed (${installed.length})` : 'Browse'}
          </button>
        ))}
      </div>

      {/* Search + filters */}
      {activeTab === 'browse' && (
        <div className="flex flex-col gap-2">
          <div className="flex gap-2">
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && search()}
              placeholder="Search plugins…"
              className="flex-1 bg-gray-800 rounded px-3 py-2 text-sm text-gray-200"
            />
            <button onClick={search}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-sm font-semibold">
              Search
            </button>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            {CATEGORIES.map(cat => (
              <button key={cat} onClick={() => setCategory(cat)}
                className={`px-2.5 py-1 rounded text-xs font-semibold capitalize transition-colors
                  ${category === cat ? 'bg-indigo-700 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
              >{cat}</button>
            ))}
            <label className="flex items-center gap-1.5 text-xs text-gray-400 cursor-pointer ml-auto">
              <input type="checkbox" checked={freeOnly} onChange={e => setFreeOnly(e.target.checked)} className="accent-indigo-500" />
              Free only
            </label>
            <label className="flex items-center gap-1.5 text-xs text-gray-400 cursor-pointer">
              <input type="checkbox" checked={verifiedOnly} onChange={e => setVerifiedOnly(e.target.checked)} className="accent-indigo-500" />
              Verified
            </label>
          </div>
        </div>
      )}

      {/* Error */}
      {error && <p className="text-red-400 text-xs">{error}</p>}

      {/* Plugin grid */}
      {loading ? (
        <div className="text-gray-400 text-xs py-8 text-center">Loading marketplace…</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {(activeTab === 'browse' ? plugins : installedPlugins).map(entry => (
            <div key={entry.plugin_id} className="relative">
              <PluginCard
                entry={entry}
                onSelect={() => setSelected(entry)}
                onInstall={() => handleInstall(entry)}
              />
              {installed.includes(entry.plugin_id) && (
                <div className="absolute top-2 right-2 bg-emerald-700 text-emerald-100 text-xs px-1.5 py-0.5 rounded font-semibold">
                  Installed
                </div>
              )}
            </div>
          ))}
          {(activeTab === 'browse' ? plugins : installedPlugins).length === 0 && (
            <div className="col-span-full text-center text-gray-500 py-12 text-sm">
              {activeTab === 'installed' ? 'No plugins installed yet.' : 'No plugins found.'}
            </div>
          )}
        </div>
      )}

      {/* Detail modal */}
      {selected && (
        <PluginModal
          entry={selected}
          userId={userId}
          onClose={() => setSelected(null)}
          onInstall={() => { handleInstall(selected); setSelected(null); }}
          onRate={(rating, text) => handleRate(selected, rating, text)}
        />
      )}
    </div>
  );
};

export default Marketplace;

