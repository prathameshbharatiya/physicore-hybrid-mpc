
import React from 'react';
import { PhysicalParams } from '../types';

interface ParameterEditorProps {
  params: PhysicalParams;
  onChange: (updates: Partial<PhysicalParams>) => void;
  onReset: () => void;
}

const ParameterEditor: React.FC<ParameterEditorProps> = ({ params, onChange, onReset }) => {
  return (
    <div className="bg-slate-900/80 p-4 rounded border border-white/5 space-y-4 shadow-inner">
      <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Physics Hyperparameters</h3>
      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-[9px] mb-2">
            <span>Mass (kg)</span>
            <span className="text-white font-mono">{params.mass.toFixed(3)}</span>
          </div>
          <input 
            type="range" min="0.1" max="5.0" step="0.1" 
            value={params.mass} 
            onChange={(e) => onChange({ mass: parseFloat(e.target.value) })}
            className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
          />
        </div>
        <div>
          <div className="flex justify-between text-[9px] mb-2">
            <span>Friction (μ)</span>
            <span className="text-white font-mono">{params.friction.toFixed(3)}</span>
          </div>
          <input 
            type="range" min="0" max="1.0" step="0.01" 
            value={params.friction} 
            onChange={(e) => onChange({ friction: parseFloat(e.target.value) })}
            className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
          />
        </div>
        <div>
          <div className="flex justify-between text-[9px] mb-2">
            <span>Textile Stiffness (k)</span>
            <span className="text-white font-mono">{params.textile_k.toFixed(1)}</span>
          </div>
          <input 
            type="range" min="10" max="1000" step="10" 
            value={params.textile_k} 
            onChange={(e) => onChange({ textile_k: parseFloat(e.target.value) })}
            className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-pink-500"
          />
        </div>
      </div>
      <button 
        onClick={onReset}
        className="w-full py-2 border border-slate-800 text-slate-600 text-[9px] font-bold uppercase hover:border-slate-700 hover:text-slate-400 transition-colors rounded mt-4"
      >
        Restore Beliefs
      </button>
    </div>
  );
};

export default ParameterEditor;
