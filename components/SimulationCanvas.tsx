
import React, { useEffect, useRef } from 'react';
import { SimMode, SimState, StateVector, ControlInput } from '../types';
// Fix: Use stepDynamicsRK4 instead of non-existent stepDynamics
import { stepDynamicsRK4 } from '../services/physicsLogic';

declare const Matter: any;

interface SimulationCanvasProps {
  mode: SimMode;
  onStateUpdate: (state: SimState) => void;
  target: [number, number];
  controlAction: ControlInput;
  physicsPriors: { mass: number; friction: number; gravity: number };
}

const SimulationCanvas: React.FC<SimulationCanvasProps> = ({ mode, onStateUpdate, target, controlAction, physicsPriors }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<any>(null);
  const robotRef = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current || !canvasRef.current) return;

    const engine = Matter.Engine.create();
    engineRef.current = engine;
    engine.world.gravity.y = 0; // Handled manually in dynamics

    const render = Matter.Render.create({
      element: containerRef.current,
      canvas: canvasRef.current,
      engine: engine,
      options: {
        width: containerRef.current.clientWidth,
        height: containerRef.current.clientHeight,
        wireframes: false,
        background: 'transparent',
      }
    });

    const { width, height } = containerRef.current;
    
    const robot = Matter.Bodies.rectangle(width / 2, height / 2, 60, 30, {
      friction: 0,
      airFriction: 0,
      restitution: 0.1,
      render: { fillStyle: '#3b82f6', strokeStyle: '#fff', lineWidth: 2 }
    });
    robotRef.current = robot;

    const ground = Matter.Bodies.rectangle(width / 2, height + 25, width, 50, { isStatic: true, render: { fillStyle: '#0f172a' } });
    Matter.Composite.add(engine.world, [robot, ground]);

    const runner = Matter.Runner.create();
    Matter.Runner.run(runner, engine);
    Matter.Render.run(render);

    const controlLoop = setInterval(() => {
      const b = robotRef.current;
      
      // Apply Force from MPC Controller
      Matter.Body.applyForce(b, b.position, { 
        x: controlAction[0] * 0.001, 
        y: (controlAction[1] + physicsPriors.gravity * 10) * 0.001 
      });

      // Update Parent State
      onStateUpdate({
        current: [b.position.x, b.position.y, b.velocity.x, b.velocity.y, b.angle, b.angularVelocity],
        target: target,
        estimatedParams: physicsPriors,
        predictionError: 0, // Calculated in App
        controlEffort: Math.abs(controlAction[0]) + Math.abs(controlAction[1]),
        stability: 100 - Math.min(100, (Math.abs(b.angularVelocity) * 100)),
        time: Date.now(),
        controlAction: controlAction
      });
    }, 16);

    const drawOverlay = () => {
      const ctx = canvasRef.current?.getContext('2d');
      if (!ctx || !robotRef.current) return;

      // Draw Projected Rollout (Ground Truth Visualization)
      let tempX: StateVector = [
        robotRef.current.position.x,
        robotRef.current.position.y,
        robotRef.current.velocity.x,
        robotRef.current.velocity.y,
        robotRef.current.angle,
        robotRef.current.angularVelocity
      ];
      
      ctx.beginPath();
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
      ctx.moveTo(tempX[0], tempX[1]);
      
      for (let i = 0; i < 20; i++) {
        // Fix: Use stepDynamicsRK4
        tempX = stepDynamicsRK4(tempX, controlAction, physicsPriors);
        ctx.lineTo(tempX[0], tempX[1]);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Target
      ctx.beginPath();
      ctx.arc(target[0], target[1], 8, 0, Math.PI * 2);
      ctx.strokeStyle = '#f43f5e';
      ctx.stroke();

      requestAnimationFrame(drawOverlay);
    };
    drawOverlay();

    return () => {
      clearInterval(controlLoop);
      Matter.Engine.clear(engine);
      Matter.Render.stop(render);
    };
  }, [mode]);

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-hidden bg-slate-950 border border-slate-800">
      <canvas ref={canvasRef} className="block w-full h-full" />
    </div>
  );
};

export default SimulationCanvas;
