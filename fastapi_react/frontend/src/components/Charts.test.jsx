import { describe, it, expect, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { ScatterPanel, LinePanel, BarPanel } from './Charts.jsx';

beforeEach(() => {
  // ResponsiveContainer measures its parent; jsdom gives it 0x0 by default
  // which makes Recharts charts render as empty SVGs in tests. We mock
  // getBoundingClientRect on the container's parent so charts have a size.
  Object.defineProperty(HTMLElement.prototype, 'getBoundingClientRect', {
    configurable: true,
    value: () => ({ width: 800, height: 400, top: 0, left: 0, right: 800, bottom: 400, x: 0, y: 0 }),
  });
});

describe('ScatterPanel', () => {
  it('returns null for empty rows', () => {
    const { container } = render(<ScatterPanel rows={[]} x="a" y="b" />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders a card with title for non-empty rows', () => {
    render(<ScatterPanel title="Scatter" rows={[{ a: 1, b: 2 }]} x="a" y="b" />);
    expect(document.querySelector('.card')).toBeInTheDocument();
  });
});

describe('LinePanel', () => {
  it('returns null for empty rows', () => {
    const { container } = render(<LinePanel rows={[]} x="a" y="b" />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders a chart container for non-empty rows', () => {
    render(<LinePanel title="Line" rows={[{ a: 1, b: 2 }]} x="a" y="b" />);
    expect(document.querySelector('.card')).toBeInTheDocument();
  });
});

describe('BarPanel', () => {
  it('returns null for empty rows', () => {
    const { container } = render(<BarPanel rows={[]} x="a" y="b" />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders a chart container for non-empty rows', () => {
    render(<BarPanel title="Bar" rows={[{ a: 'x', b: 5 }]} x="a" y="b" />);
    expect(document.querySelector('.card')).toBeInTheDocument();
  });
});
