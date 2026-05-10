import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

// ── Types ────────────────────────────────────────────────────────────────────

export type ToastType = "success" | "warning" | "error";

export interface Toast {
  id: string;
  message: string;
  type: ToastType;
  /** true while the toast is animating out */
  dismissing: boolean;
}

export interface ToastContextValue {
  addToast: (message: string, type: ToastType) => void;
}

// ── Context ──────────────────────────────────────────────────────────────────

const ToastContext = createContext<ToastContextValue | null>(null);

export const useToast = (): ToastContextValue => {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error("useToast must be used inside <ToastProvider>");
  }
  return ctx;
};

// ── Constants ────────────────────────────────────────────────────────────────

const DISMISS_AFTER_MS = 4000;
const FADE_DURATION_MS = 300;
const MAX_TOASTS = 5;

// ── Individual Toast item ────────────────────────────────────────────────────

interface ToastItemProps {
  toast: Toast;
  onRemove: (id: string) => void;
}

const borderClass: Record<ToastType, string> = {
  success: "border-green",
  warning: "border-amber",
  error: "border-red",
};

const bgClass: Record<ToastType, string> = {
  success: "bg-green/10",
  warning: "bg-amber/10",
  error: "bg-red/10",
};

const textClass: Record<ToastType, string> = {
  success: "text-green",
  warning: "text-amber",
  error: "text-red",
};

const labelMap: Record<ToastType, string> = {
  success: "OK",
  warning: "WARN",
  error: "ERR",
};

const ToastItem: React.FC<ToastItemProps> = ({ toast, onRemove }) => {
  return (
    <div
      className={[
        "flex items-start gap-2 px-3 py-2 rounded border",
        "font-mono text-[10px] text-white",
        "shadow-lg min-w-[220px] max-w-[340px]",
        bgClass[toast.type],
        borderClass[toast.type],
        "transition-all duration-300",
        toast.dismissing
          ? "opacity-0 translate-x-4 pointer-events-none"
          : "opacity-100 translate-x-0",
      ].join(" ")}
      role="alert"
      aria-live="assertive"
    >
      <span className={`font-bold shrink-0 ${textClass[toast.type]}`}>
        [{labelMap[toast.type]}]
      </span>
      <span className="flex-1 break-words">{toast.message}</span>
      <button
        onClick={() => onRemove(toast.id)}
        className="shrink-0 text-textDim hover:text-white transition-colors leading-none"
        aria-label="Dismiss"
      >
        ×
      </button>
    </div>
  );
};

// ── Provider ─────────────────────────────────────────────────────────────────

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [toasts, setToasts] = useState<Toast[]>([]);
  // Map of toast id → auto-dismiss timer id
  const timers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  /** Begin the dismiss animation, then remove from state after fade */
  const startDismiss = useCallback((id: string) => {
    setToasts((prev) =>
      prev.map((t) => (t.id === id ? { ...t, dismissing: true } : t))
    );
    const removeTimer = setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, FADE_DURATION_MS);
    timers.current.set(`remove-${id}`, removeTimer);
  }, []);

  const addToast = useCallback(
    (message: string, type: ToastType) => {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

      setToasts((prev) => {
        let next = [...prev, { id, message, type, dismissing: false }];
        // If over the limit, start dismissing the oldest non-dismissing toasts
        while (next.filter((t) => !t.dismissing).length > MAX_TOASTS) {
          const oldest = next.find((t) => !t.dismissing);
          if (!oldest) break;
          next = next.map((t) =>
            t.id === oldest.id ? { ...t, dismissing: true } : t
          );
          // Schedule removal of overflow toast
          const overflowTimer = setTimeout(() => {
            setToasts((p) => p.filter((t) => t.id !== oldest.id));
          }, FADE_DURATION_MS);
          timers.current.set(`remove-${oldest.id}`, overflowTimer);
          // Clear auto-dismiss for the overflowed toast
          const existingTimer = timers.current.get(oldest.id);
          if (existingTimer !== undefined) {
            clearTimeout(existingTimer);
            timers.current.delete(oldest.id);
          }
        }
        return next;
      });

      // Auto-dismiss after DISMISS_AFTER_MS
      const autoTimer = setTimeout(() => {
        startDismiss(id);
        timers.current.delete(id);
      }, DISMISS_AFTER_MS);
      timers.current.set(id, autoTimer);
    },
    [startDismiss]
  );

  // Cleanup all timers on unmount
  useEffect(() => {
    const currentTimers = timers.current;
    return () => {
      currentTimers.forEach((timer) => clearTimeout(timer));
    };
  }, []);

  const handleRemove = useCallback(
    (id: string) => {
      // Cancel the auto-dismiss timer if it hasn't fired yet
      const existing = timers.current.get(id);
      if (existing !== undefined) {
        clearTimeout(existing);
        timers.current.delete(id);
      }
      startDismiss(id);
    },
    [startDismiss]
  );

  return (
    <ToastContext.Provider value={{ addToast }}>
      {children}
      {/* Toast container */}
      <div
        className="fixed top-4 right-4 z-[9999] flex flex-col gap-2 pointer-events-none"
        aria-label="Notifications"
      >
        {toasts.map((toast) => (
          <div key={toast.id} className="pointer-events-auto">
            <ToastItem toast={toast} onRemove={handleRemove} />
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
};
