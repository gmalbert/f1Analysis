import { describe, it, expect, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import {
  ScatterPanel, LinePanel, BarPanel, MultiBarPanel, RegressionPanel, MultiLinePanel, PiePanel,
} from './Charts.jsx';

beforeEach(() => {
  Object.defineProperty(HTMLElement.prototype, 'getBoundingClientRect', {
    configurable: true,
    value: () => ({ width: 800, height: 400, top: 0, left: 0, right: 800, bottom: 400, x: 0, y: 0 }),
  });
});

describe('chart panels', () => {
  it.each([
    ['scatter', ScatterPanel, { title: 'Scatter', rows: [], x: 'a', y: 'b' }],
    ['line', LinePanel, { title: 'Line', rows: [], x: 'a', y: 'b' }],
    ['bar', BarPanel, { title: 'Bar', rows: [], x: 'a', y: 'b' }],
    ['multi bar', MultiBarPanel, { title: 'Multi Bar', rows: [], x: 'a', ys: ['b', 'c'] }],
    ['regression', RegressionPanel, { title: 'Regression', points: [], fit: [], x: 'a', y: 'b', xLabel: 'A', yLabel: 'B' }],
    ['multi line', MultiLinePanel, { title: 'Multi Line', rows: [], x: 'a', y: 'b', series: 'series' }],
    ['pie', PiePanel, { title: 'Pie', rows: [], nameKey: 'name', valueKey: 'value' }],
  ])('returns null for empty %s data', (_name, Component, props) => {
    const { container } = render(<Component {...props} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders all supported chart types with accessible chart containers', () => {
    const { rerender } = render(<ScatterPanel title="Scatter" rows={[{ a: 1, b: 2 }, { a: 3, b: 4 }]} x="a" y="b" />);
    expect(document.querySelector('[role="img"]')).toHaveAttribute('aria-label', expect.stringContaining('Scatter'));

    rerender(<LinePanel title="Line" rows={[{ a: 1, b: 2 }]} x="a" y="b" />);
    expect(document.querySelector('.card')).toBeInTheDocument();

    rerender(<BarPanel title="Bar" rows={[{ a: 'x', b: 5 }]} x="a" y="b" />);
    expect(document.querySelector('.card')).toBeInTheDocument();

    rerender(<MultiBarPanel title="Multi Bar" rows={[{ a: 'x', b: 5, c: 7 }]} x="a" ys={['b', 'c']} />);
    expect(document.querySelector('[aria-label*="2 series"]')).toBeInTheDocument();

    rerender(<RegressionPanel title="Regression" points={[{ a: 1, b: 2 }, { a: 3, b: 4 }]} fit={[{ a: 1, b: 2 }, { a: 3, b: 4 }]} x="a" y="b" xLabel="A" yLabel="B" />);
    expect(document.querySelector('[aria-label*="Regression fit"]')).toBeInTheDocument();

    rerender(<MultiLinePanel title="Multi Line" rows={[
      { year: 2024, value: 1, driver: 'A' },
      { year: 2024, value: 2, driver: 'B' },
      { year: 2025, value: 3, driver: 'A' },
    ]} x="year" y="value" series="driver" />);
    expect(document.querySelector('[aria-label*="2 series"]')).toBeInTheDocument();

    rerender(<PiePanel title="Pie" rows={[{ name: 'A', value: 2 }, { name: 'B', value: 3 }]} nameKey="name" valueKey="value" />);
    expect(document.querySelector('[aria-label*="2 categories"]')).toBeInTheDocument();
  });
});
