import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  LineChart, Line, BarChart, Bar, Legend, ComposedChart, PieChart, Pie, Cell
} from "recharts";
import { Card } from "./UI";

/** @type {Record<string, string>} */
const axisLabels = {
  averagePracticePosition: "Average Practice Position",
  averageStopTime: "Avg. Stop Time",
  grandPrixYear: "Year",
  positionsGained: "Positions Gained",
  resultsFinalPositionNumber: "Final Position",
  resultsStartingGridPositionNumber: "Starting Position",
  short_date: "Date",
  yearsActive: "Years Active",
};

/** @param {string} title @param {Array<Record<string, any>>} rows @param {string} x @param {string} y */
function chartLabel(title, rows, x, y) {
  return `${title}. ${rows.length} data points. Horizontal axis: ${axisLabels[x] || x}. Vertical axis: ${axisLabels[y] || y}.`;
}

/** @param {Array<Record<string, any>>} rows @param {string} key */
function numericExtent(rows, key) {
  const vals = rows.map(r => Number(r[key])).filter(Number.isFinite);
  return vals.length ? [Math.min(...vals), Math.max(...vals)] : ["auto", "auto"];
}

/** @param {{ title: string, rows?: Array<Record<string, any>>, x: string, y: string, xLabel?: string, yLabel?: string }} props */
export function ScatterPanel({ title, rows = [], x, y, xLabel = axisLabels[x] || x, yLabel = axisLabels[y] || y }) {
  if (!rows.length) return null;
  return (
    <Card title={title}>
      <div className="chart" role="img" aria-label={chartLabel(title, rows, x, y)}>
        <ResponsiveContainer width="100%" height={320}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 25, left: 15 }}>
            <CartesianGrid />
            <XAxis dataKey={x} name={xLabel} type="number" domain={numericExtent(rows, x)} label={{ value: xLabel, position: "insideBottom", offset: -15 }} />
            <YAxis dataKey={y} name={yLabel} type="number" domain={numericExtent(rows, y)} label={{ value: yLabel, angle: -90, position: "insideLeft" }} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} />
            <Scatter data={rows} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

/** @param {{ title: string, rows?: Array<Record<string, any>>, x: string, y: string, xLabel?: string, yLabel?: string }} props */
export function LinePanel({ title, rows = [], x, y, xLabel = axisLabels[x] || x, yLabel = axisLabels[y] || y }) {
  if (!rows.length) return null;
  return (
    <Card title={title}>
      <div className="chart" role="img" aria-label={chartLabel(title, rows, x, y)}>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={rows}>
            <CartesianGrid />
            <XAxis dataKey={x} label={{ value: xLabel, position: "insideBottom", offset: -15 }} />
            <YAxis label={{ value: yLabel, angle: -90, position: "insideLeft" }} />
            <Tooltip />
            <Line type="monotone" dataKey={y} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

/** @param {{ title: string, rows?: Array<Record<string, any>>, x: string, y: string, xLabel?: string, yLabel?: string }} props */
export function BarPanel({ title, rows = [], x, y, xLabel = axisLabels[x] || x, yLabel = axisLabels[y] || y }) {
  if (!rows.length) return null;
  return (
    <Card title={title}>
      <div className="chart" role="img" aria-label={chartLabel(title, rows, x, y)}>
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={rows}>
            <CartesianGrid />
            <XAxis dataKey={x} interval={0} angle={-30} textAnchor="end" height={90} label={{ value: xLabel, position: "insideBottom", offset: -5 }} />
            <YAxis label={{ value: yLabel, angle: -90, position: "insideLeft" }} />
            <Tooltip />
            <Legend />
            <Bar dataKey={y} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

/** @param {{ title: string, points?: Array<Record<string, any>>, fit?: Array<Record<string, any>>, x: string, y: string, xLabel: string, yLabel: string }} props */
export function RegressionPanel({ title, points = [], fit = [], x, y, xLabel, yLabel }) {
  if (!points.length) return null;
  return (
    <Card title={title}>
      <div className="chart" role="img" aria-label={`${chartLabel(title, points, x, y)} Regression fit shown as a line.`}>
        <ResponsiveContainer width="100%" height={340}>
          <ComposedChart margin={{ top: 10, right: 20, bottom: 25, left: 15 }}>
            <CartesianGrid />
            <XAxis dataKey={x} type="number" domain={numericExtent(points, x)} label={{ value: xLabel, position: "insideBottom", offset: -15 }} />
            <YAxis dataKey={y} type="number" domain={numericExtent(points, y)} label={{ value: yLabel, angle: -90, position: "insideLeft" }} />
            <Tooltip />
            <Scatter data={points} fill="#e10600" />
            <Line data={fit} dataKey={y} type="linear" stroke="#f4c542" dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}


const SERIES_COLORS = [
  "#0068c9", "#ff4b4b", "#00a86b", "#7d3cff", "#f0a202", "#2a9d8f",
  "#e76f51", "#264653", "#8d99ae", "#9b5de5", "#00bbf9", "#f15bb5",
];

/** @param {{ title: string, rows?: Array<Record<string, any>>, x: string, y: string, series: string, xLabel?: string, yLabel?: string }} props */
export function MultiLinePanel({ title, rows = [], x, y, series, xLabel = axisLabels[x] || x, yLabel = axisLabels[y] || y }) {
  if (!rows.length) return null;
  const seriesNames = [...new Set(rows.map(row => String(row[series] ?? "")).filter(Boolean))];
  const xValues = [...new Set(rows.map(row => row[x]))];
  const byX = xValues.map(xValue => {
    /** @type {Record<string, any>} */
    const point = { [x]: xValue };
    for (const row of rows.filter(item => item[x] === xValue)) {
      point[String(row[series])] = row[y];
    }
    return point;
  });
  return (
    <Card title={title}>
      <div className="chart" role="img" aria-label={`${title}. ${seriesNames.length} series across ${xValues.length} x-axis values.`}>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={byX}>
            <CartesianGrid />
            <XAxis dataKey={x} label={{ value: xLabel, position: "insideBottom", offset: -15 }} />
            <YAxis label={{ value: yLabel, angle: -90, position: "insideLeft" }} />
            <Tooltip />
            <Legend />
            {seriesNames.map((name, index) => (
              <Line key={name} type="monotone" dataKey={name} stroke={SERIES_COLORS[index % SERIES_COLORS.length]} dot={false} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

/** @param {{ title: string, rows?: Array<Record<string, any>>, nameKey: string, valueKey: string }} props */
export function PiePanel({ title, rows = [], nameKey, valueKey }) {
  if (!rows.length) return null;
  return (
    <Card title={title}>
      <div className="chart" role="img" aria-label={`${title}. Pie chart with ${rows.length} categories.`}>
        <ResponsiveContainer width="100%" height={400}>
          <PieChart>
            <Pie data={rows} dataKey={valueKey} nameKey={nameKey} outerRadius={140} label>
              {rows.map((row, index) => <Cell key={String(row[nameKey] ?? index)} fill={SERIES_COLORS[index % SERIES_COLORS.length]} />)}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}
