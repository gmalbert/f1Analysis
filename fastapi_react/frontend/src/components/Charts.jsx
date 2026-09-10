import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  LineChart, Line, BarChart, Bar, Legend
} from "recharts";
import { Card } from "./UI";

function numericExtent(rows, key) {
  const vals = rows.map(r => Number(r[key])).filter(Number.isFinite);
  return vals.length ? [Math.min(...vals), Math.max(...vals)] : ["auto", "auto"];
}

export function ScatterPanel({ title, rows = [], x, y }) {
  if (!rows.length) return null;
  return (
    <Card title={title}>
      <div className="chart">
        <ResponsiveContainer width="100%" height={320}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 25, left: 15 }}>
            <CartesianGrid />
            <XAxis dataKey={x} name={x} type="number" domain={numericExtent(rows, x)} label={{ value: x, position: "insideBottom", offset: -15 }} />
            <YAxis dataKey={y} name={y} type="number" domain={numericExtent(rows, y)} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} />
            <Scatter data={rows} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

export function LinePanel({ title, rows = [], x, y }) {
  if (!rows.length) return null;
  return (
    <Card title={title}>
      <div className="chart">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={rows}>
            <CartesianGrid />
            <XAxis dataKey={x} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey={y} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

export function BarPanel({ title, rows = [], x, y }) {
  if (!rows.length) return null;
  return (
    <Card title={title}>
      <div className="chart">
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={rows}>
            <CartesianGrid />
            <XAxis dataKey={x} interval={0} angle={-30} textAnchor="end" height={90} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey={y} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}
