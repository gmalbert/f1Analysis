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
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("f1analysis.theme") === "light" ? "light" : "dark"; }
    catch { return "dark"; }
  });

  useEffect(() => {
    api.get("/api/health").then(setHealth).catch(() => {});
    const hash = decodeURIComponent(location.hash.replace("#/", ""));
    if (pages[hash]) setActive(hash);
  }, []);

  useEffect(() => {
    document.title = `${active} \u2014 ${BASE_TITLE}`;
  }, [active]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("f1analysis.theme", theme); } catch { /* storage is optional */ }
  }, [theme]);

  function navigate(page) {
    setActive(page);
    location.hash = `/${encodeURIComponent(page)}`;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  const Page = pages[active];
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <header className="site-header">
        <div className="site-brand">
          <img src="/api/brand/logo" alt="Gridlocked" />
          <div className="site-title">F1 Races from 2016 to {new Date().getFullYear()}</div>
        </div>
        <div className="runtime" aria-live="polite" role="status">
          <span className={health?.status === "ok" ? "dot ok" : "dot"} aria-hidden="true" />
          <div>
            <strong>{health?.status === "ok" ? "API connected" : "API…"}</strong>
            <small>{health?.rss_mb ? `${health.rss_mb} MB RSS` : "checking"}</small>
          </div>
        </div>
        <label className="theme-toggle">
          <input aria-label="Use light theme" type="checkbox" checked={theme === "light"} onChange={event => setTheme(event.target.checked ? "light" : "dark")} />
          Light theme
        </label>
      </header>
      <nav className="section-nav" aria-label="Sections">
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
      <main className="content" id="main-content" tabIndex={-1}>
        <Page />
        <footer className="site-footer">
          <span>Powered by</span>
          <a href="https://www.betting-oracle.com" target="_blank" rel="noreferrer">Betting Oracle</a>
          <span>Sports Prediction Analytics</span>
          <small>All content is for informational purposes only and does not constitute betting advice. Wager responsibly.</small>
        </footer>
      </main>
    </div>
  );
}
