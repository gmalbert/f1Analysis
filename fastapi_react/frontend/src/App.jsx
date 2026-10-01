import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import DataExplorer from "./pages/DataExplorer";
import Analytics from "./pages/Analytics";
import CurrentSeason from "./pages/CurrentSeason";
import NextRace from "./pages/NextRace";
import Models from "./pages/Models";
import RawData from "./pages/RawData";
import BettingResearch from "./pages/BettingResearch";
import FilterSidebar from "./components/FilterSidebar";

const pages = [
  { key: "Data Explorer", label: "📊 Data Explorer", Component: DataExplorer },
  { key: "Analytics", label: "📈 Analytics & Visualizations", Component: Analytics },
  { key: "Current Season", label: "🏎️ Schedule", Component: CurrentSeason },
  { key: "Next Race", label: "🏁 Next Race", Component: NextRace },
  { key: "Predictive Models", label: "🤖 Predictive Models", Component: Models },
  { key: "Raw Data", label: "💾 Data & Debug", Component: RawData },
  { key: "Betting Research", label: "📐 Betting Research", Component: BettingResearch },
];

const BASE_TITLE = "Gridlocked - Formula 1 Betting & Analytics";

export default function App() {
  const [active, setActive] = useState("Data Explorer");
  const [meta, setMeta] = useState(null);
  const [filterRevision, setFilterRevision] = useState(0);\n  const tabStripRef = useRef(null);
  const [filtersActive, setFiltersActive] = useState(() => {
    try { return Boolean(JSON.parse(sessionStorage.getItem("f1analysis.filters") || "null")?.applied); }
    catch { return false; }
  });

  useEffect(() => {
    api.get("/api/meta").then(setMeta).catch(() => {});
    const hash = decodeURIComponent(location.hash.replace("#/", ""));
    if (pages.some(page => page.key === hash)) setActive(hash);

    const syncFilters = () => {
      try { setFiltersActive(Boolean(JSON.parse(sessionStorage.getItem("f1analysis.filters") || "null")?.applied)); }
      catch { setFiltersActive(false); }
      setFilterRevision(value => value + 1);
    };
    window.addEventListener("f1analysis:filters-changed", syncFilters);
    return () => window.removeEventListener("f1analysis:filters-changed", syncFilters);
  }, []);

  useEffect(() => {
    document.title = BASE_TITLE;
  }, [active]);

  function navigate(page) {
    setActive(page);
    location.hash = `/${encodeURIComponent(page)}`;
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  const Page = pages.find(page => page.key === active)?.Component || DataExplorer;
  const startYear = meta?.race_start_year ?? 2016;
  const currentYear = meta?.current_year ?? new Date().getFullYear();

  return (
    <div className={`app-shell ${filtersActive ? "filters-active" : ""}`}>
      <a className="skip-link" href="#main-content">Skip to main content</a>
      {filtersActive && <FilterSidebar />}
      <main className="streamlit-main" id="main-content" tabIndex={-1}>
        <div className="block-container">
          <header className="streamlit-hero">
            <img src="/api/brand/logo" alt="Gridlocked" className="brand-logo" />
            <h1>F1 Races from {startYear} to {currentYear}</h1>
            <p className="caption">Last updated: {meta?.last_updated || "Loading…"}</p>
            <p className="caption">Code deployed at: {meta?.code_deployed_at || "Loading…"}</p>
          </header>

          <nav ref={tabStripRef} className="streamlit-tabs" aria-label="Main sections">
            <div className="streamlit-tablist" role="tablist" aria-label="Main sections">
              {pages.map(({ key, label }) => (
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
            </div>
          </nav>

          <section className="page-content">
            <Page key={`${active}-${filterRevision}`} />
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
