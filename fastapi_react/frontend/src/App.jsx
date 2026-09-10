import React, { useEffect, useState } from "react";
import { api } from "./api";
import DataExplorer from "./pages/DataExplorer";
import Analytics from "./pages/Analytics";
import CurrentSeason from "./pages/CurrentSeason";
import NextRace from "./pages/NextRace";
import Models from "./pages/Models";
import RawData from "./pages/RawData";
import BettingResearch from "./pages/BettingResearch";

const pages = {
  "Data Explorer": DataExplorer,
  "Analytics": Analytics,
  "Current Season": CurrentSeason,
  "Next Race": NextRace,
  "Predictive Models": Models,
  "Raw Data": RawData,
  "Betting Research": BettingResearch,
};

const icons = {
  "Data Explorer": "▦",
  "Analytics": "⌁",
  "Current Season": "◷",
  "Next Race": "🏁",
  "Predictive Models": "◆",
  "Raw Data": "≡",
  "Betting Research": "📐",
};

export default function App() {
  const [active, setActive] = useState("Data Explorer");
  const [health, setHealth] = useState(null);

  useEffect(() => {
    api.get("/api/health").then(setHealth).catch(() => {});
    const hash = decodeURIComponent(location.hash.replace("#/", ""));
    if (pages[hash]) setActive(hash);
  }, []);

  function navigate(page) {
    setActive(page);
    location.hash = `/${encodeURIComponent(page)}`;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  const Page = pages[active];
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">F1</div>
          <div><strong>F1 Analysis</strong><small>FastAPI + React</small></div>
        </div>
        <nav>
          {Object.keys(pages).map(page => (
            <button key={page} className={active === page ? "active" : ""} onClick={() => navigate(page)}>
              <span>{icons[page]}</span>{page}
            </button>
          ))}
        </nav>
        <div className="runtime">
          <span className={health?.status === "ok" ? "dot ok" : "dot"} />
          <div>
            <strong>{health?.status === "ok" ? "API connected" : "API…"}</strong>
            <small>{health?.rss_mb ? `${health.rss_mb} MB RSS` : "checking"}</small>
          </div>
        </div>
      </aside>
      <main className="content">
        <Page />
        <footer>F1 Analysis · React presentation layer · FastAPI analytical backend</footer>
      </main>
    </div>
  );
}
