import '@testing-library/jest-dom/vitest';

// Polyfill fetch if older jsdom is used; modern jsdom includes it.
if (typeof globalThis.fetch !== 'function') {
  globalThis.fetch = async () => {
    throw new Error('fetch is not available in this test environment');
  };
}

// jsdom doesn't implement ResizeObserver; recharts' ResponsiveContainer needs it.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}
