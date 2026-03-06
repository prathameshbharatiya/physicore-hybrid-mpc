
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Suppress recharts defaultProps warning in React 18/19
const originalError = console.error;
console.error = (...args: any[]) => {
  if (typeof args[0] === 'string' && args[0].includes('defaultProps')) return;
  originalError(...args);
};

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
