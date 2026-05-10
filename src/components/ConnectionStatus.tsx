import React from "react";

export interface ConnectionStatusProps {
  status: "connected" | "reconnecting" | "disconnected";
  endpoint?: string;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  status,
  endpoint,
}) => {
  const dotColor =
    status === "connected"
      ? "bg-green"
      : status === "reconnecting"
        ? "bg-amber"
        : "bg-red";

  const textColor =
    status === "connected"
      ? "text-green"
      : status === "reconnecting"
        ? "text-amber"
        : "text-red";

  const label =
    status === "connected"
      ? "CONNECTED"
      : status === "reconnecting"
        ? "RECONNECTING..."
        : "DISCONNECTED";

  const shouldPulse = status === "reconnecting" || status === "disconnected";

  return (
    <div className="inline-flex flex-col items-start gap-0.5">
      <div className="flex items-center gap-1.5">
        <span
          className={`inline-block w-1.5 h-1.5 rounded-full ${dotColor} ${shouldPulse ? "animate-pulse" : ""}`}
        />
        <span className={`font-mono text-[9px] font-bold tracking-widest ${textColor}`}>
          {label}
        </span>
      </div>
      {endpoint && (
        <span className="font-mono text-[8px] text-textDim pl-3">{endpoint}</span>
      )}
    </div>
  );
};
