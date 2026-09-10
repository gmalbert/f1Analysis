import { useEffect, useState } from 'react'
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
  "Data Explorer": "\u25A6",
  "Analytics": "\u2301",
  "Current Season": "\u25F7",
  "Next Race": "\uD83C\uDFC1",
  "Predictive Models": "\u25C6",
  "Raw Data": "\u2261",
  "Betting Research": "\uD83D\uDCD0",
};

const BASE_TITLE = "F1 Analysis";

export default function App() {
  const [active, setActive] = useState("Data Explorer");
  const [health, setHealth] = useState(null);

  useEffect(() => {
    api.get("/api/health").then(setHealth).catch(() => {});
    const hash = decodeURIComponent(location.hash.replace("#/", ""));
    if (pages[hash]) setActive(hash);
  }, []);

  useEffect(() => {
    document.title = `${active} \u2014 ${BASE_TITLE}`;
  }, [active]);

  function navigate(page) {
    setActive(page);
    location.hash = `/${encodeURIComponent(page)}`;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  const Page = pages[active];
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <aside className="sidebar" aria-label="Primary">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">F1</div>
          <div><strong>F1 Analysis</strong><small>FastAPI + React</small></div>
        </div>
        <nav aria-label="Sections">
          {Object.keys(pages).map(page => (
            <button
              key={page}
              className={active === page ? "active" : ""}
              onClick={() => navigate(page)}
              aria-current={active === page ? "page" : undefined}
            >
              <span aria-hidden="true">{icons[page]}</span>{page}
            </button>
          ))}
        </nav>
        <div className="runtime" aria-live="polite">
          <span className={health?.status === "ok" ? "dot ok" : "dot"} aria-hidden="true" />
          <div>
            <strong>{health?.status === "ok" ? "API connected" : "API\u2026"}</strong>
            <small>{health?.rss_mb ? `${health.rss_mb} MB RSS` : "checking"}</small>
          </div>
        </div>
      </aside>
      <main className="content" id="main-content" tabIndex={-1}>
        <Page />
        <footer>F1 Analysis \u00B7 React presentation layer \u00B7 FastAPI analytical backend</footer>
      </main>
    </div>
  );
}
