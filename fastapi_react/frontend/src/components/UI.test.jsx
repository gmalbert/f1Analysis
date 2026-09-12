import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Card, Status, DataTable, JsonBlock, Metric, Tabs } from './UI.jsx';

describe('Card', () => {
  it('renders children inside a card', () => {
    render(<Card>hello</Card>);
    expect(screen.getByText('hello')).toBeInTheDocument();
  });

  it('renders title as h3 when provided', () => {
    render(<Card title="My Title">body</Card>);
    const heading = screen.getByRole('heading', { level: 3, name: 'My Title' });
    expect(heading).toBeInTheDocument();
  });
});

describe('Status', () => {
  it('shows loading state with aria-busy and aria-live', () => {
    render(<Status loading>kid</Status>);
    const node = screen.getByRole('status');
    expect(node).toHaveAttribute('aria-busy', 'true');
    expect(node).toHaveAttribute('aria-live', 'polite');
    expect(node).toHaveTextContent(/Loading/i);
  });

  it('shows error state with role=alert and assertive live region', () => {
    render(<Status error={new Error('boom')}>kid</Status>);
    const node = screen.getByRole('alert');
    expect(node).toHaveAttribute('aria-live', 'assertive');
    expect(node).toHaveTextContent('boom');
  });

  it('shows children when neither loading nor error', () => {
    render(<Status>kid</Status>);
    expect(screen.getByText('kid')).toBeInTheDocument();
  });
});

describe('DataTable', () => {
  it('shows empty message when no rows', () => {
    render(<DataTable rows={[]} />);
    expect(screen.getByText(/No rows available/i)).toBeInTheDocument();
  });

  it('renders columns and rows from props', () => {
    render(
      <DataTable
        rows={[{ a: 1, b: 2 }]}
        columns={['a', 'b']}
      />
    );
    expect(screen.getByRole('columnheader', { name: 'a' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'b' })).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
  });

  it('infers columns from rows when columns not provided', () => {
    render(<DataTable rows={[{ a: 1, b: 2 }, { a: 3, b: 4 }]} />);
    expect(screen.getByRole('columnheader', { name: 'a' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'b' })).toBeInTheDocument();
  });

  it('formats numeric cells', () => {
    render(<DataTable rows={[{ x: 1.5 }]} />);
    expect(screen.getByText('1.5')).toBeInTheDocument();
  });
});

describe('JsonBlock', () => {
  it('renders JSON pretty-printed', () => {
    render(<JsonBlock value={{ a: 1 }} />);
    expect(screen.getByText(/"a":\s*1/)).toBeInTheDocument();
  });
});

describe('Metric', () => {
  it('shows label and value', () => {
    render(<Metric label="MAE" value="1.5" />);
    expect(screen.getByText('MAE')).toBeInTheDocument();
    expect(screen.getByText('1.5')).toBeInTheDocument();
  });

  it('uses em dash for null values', () => {
    render(<Metric label="x" value={null} />);
    expect(screen.getByText('—')).toBeInTheDocument();
  });
});

describe('Tabs', () => {
  it('renders all tabs and marks active', () => {
    const onChange = vi.fn();
    render(<Tabs tabs={['one', 'two', 'three']} active="two" onChange={onChange} />);
    const buttons = screen.getAllByRole('button');
    expect(buttons).toHaveLength(3);
    expect(buttons[1]).toHaveClass('active');
  });

  it('invokes onChange when a tab is clicked', () => {
    const onChange = vi.fn();
    render(<Tabs tabs={['one', 'two']} active="one" onChange={onChange} />);
    fireEvent.click(screen.getByRole('button', { name: 'two' }));
    expect(onChange).toHaveBeenCalledWith('two');
  });
});
