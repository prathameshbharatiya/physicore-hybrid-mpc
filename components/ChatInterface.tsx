
import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '../types';

interface ChatInterfaceProps {
  onQuery: (query: string) => Promise<void>;
  messages: ChatMessage[];
  isAnalyzing: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ onQuery, messages, isAnalyzing }) => {
  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isAnalyzing) return;
    onQuery(input);
    setInput('');
  };

  return (
    <div className="flex flex-col h-full bg-slate-900/30 border border-white/5 rounded-sm p-4 overflow-hidden">
      <div className="flex justify-between items-center mb-3 pb-2 border-b border-white/5">
        <h3 className="text-[10px] font-black text-slate-400 tracking-widest uppercase italic">RIL Conversational Debugger</h3>
        {isAnalyzing && (
          <div className="flex gap-1">
            <div className="w-1 h-3 bg-indigo-600 animate-pulse"></div>
            <div className="w-1 h-3 bg-indigo-600 animate-pulse delay-75"></div>
          </div>
        )}
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scroll mb-4 text-[11px]">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center opacity-30 text-center px-4">
             <div className="w-8 h-8 border border-indigo-500/30 rounded-full animate-ping mb-4"></div>
             <p className="uppercase tracking-widest text-[9px] font-bold">Awaiting User Instruction...</p>
             <p className="mt-2 normal-case font-light italic">"Why is the gripper crushing the berries?"</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] p-3 rounded ${
              m.role === 'user' 
                ? 'bg-indigo-600/20 border border-indigo-500/30 text-indigo-100' 
                : 'bg-slate-800/50 border border-white/5 text-slate-300'
            }`}>
              <div className="text-[8px] uppercase tracking-widest font-bold mb-1 opacity-50">
                {m.role === 'user' ? 'Operator' : 'PhysiCore-RIL'}
              </div>
              <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="relative">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Query physics residuals..."
          disabled={isAnalyzing}
          className="w-full bg-black/40 border border-white/10 rounded p-2 text-[11px] pr-10 focus:outline-none focus:border-indigo-500/50 transition-colors"
        />
        <button 
          type="submit"
          className="absolute right-2 top-1/2 -translate-y-1/2 text-indigo-500 hover:text-indigo-400 disabled:opacity-30"
          disabled={!input.trim() || isAnalyzing}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
