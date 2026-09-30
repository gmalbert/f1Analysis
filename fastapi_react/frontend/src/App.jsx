import { useEffect, useState } from "react";
import { api } from "./api";
import DataExplorer from "./pages/DataExplorer";
import Analytics from "./pages/Analytics";
import CurrentSeason from "./pages/CurrentSeason";
import NextRace from "./pages/NextRace";
import Models from "./pages/Models";
import RawData from "./pages/RawData";
import BettingResearch from "./pages/BettingResearch";

const pages = [
  ["Data Explorer", "📊 Data Explorer", DataExplorer],
  ["Analytics", "📈 Analytics & Visualizations", Analytics],
  ["Current Season", "🏎️ Schedule", CurrentSeason],
  ["Next Race", "🏁 Next Race", NextRace],
  ["Predictive Models", "🤖 Predictive Models", Models],
  ["Raw Data", "💾 Data & Debug", RawData],
  ["Betting Research", "📐 Betting Research", BettingResearch],
];

const BASE_TITLE = "Gridlocked - Formula 1 Betting & Analytics";

export default function App() {
  const [active, setActive] = useState("Data Explorer");
  const [meta, setMeta] = useState(null);

  useEffect(() => {
    api.get("/api/meta").then(setMeta).catch(() => {});
    const hash = decodeURIComponent(location.hash.replace("#/", ""));
    if (pages.some(([key]) => key === hash)) setActive(hash);
  }, []);

  useEffect(() => {
    const current = pages.find(([key]) => key === active);
    document.title = current ? `${current[1].replace(/^\S+\s/, "")} — ${BASE_TITLE}` : BASE_TITLE;
  }, [active]);

  function navigate(page) {
    setActive(page);
    location.hash = `/${encodeURIComponent(page)}`;
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  const Page = pages.find(([key]) => key === active)?.[2] || DataExplorer;
  const startYear = meta?.race_start_year ?? 2016;
  const currentYear = meta?.current_year ?? new Date().getFullYear();

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <main className="streamlit-main" id="main-content" tabIndex={-1}>
        <div className="block-container">
          <header className="streamlit-hero">
            <img src="/api/brand/logo" alt="Gridlocked" className="brand-logo" />
            <h1>F1 Races from {startYear} to {currentYear}</h1>
            <p className="caption">Last updated: {meta?.last_updated || "Loading…"}</p>
            <p className="caption">Code deployed at: {meta?.code_deployed_at || "Loading…"}</p>
          </header>

          <nav className="streamlit-tabs" aria-label="Main sections">
            {pages.map(([key, label]) => (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={active === key}
                className={active === key ? "active" : ""}
                onClick={() => navigate(key)}
              >
                {label}
              </button>
            ))}
          </nav>

          <section className="page-content">
            <Page />
          </section>

          <footer className="site-footer">
            <p>Powered by <a href="https://www.betting-oracle.com" target="_blank" rel="noreferrer"><strong>Betting Oracle</strong></a></p>
            <p className="footer-subtitle">Sports Prediction Analytics</p>
            <a href="https://www.betting-oracle.com" target="_blank" rel="noreferrer">
              <img src="https://raw.githubusercontent.com/gmalbert/betting-oracle/main/data_files/logo.png" alt="Betting Oracle Logo" />
            </a>
          </footer>
        </div>
      </main>
    </div>
  );
}
