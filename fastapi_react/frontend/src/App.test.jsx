import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

vi.mock("./api", () => ({
  api: { get: vi.fn().mockResolvedValue({
    race_start_year: 2016,
    current_year: 2026,
    last_updated: "2026-09-29 09:00 PM",
    code_deployed_at: "2026-09-30 01:00:00 UTC",
  }) },
}));
vi.mock("./pages/DataExplorer", () => ({ default: () => <h1>Explorer page</h1> }));
vi.mock("./pages/Analytics", () => ({ default: () => <h1>Analytics page</h1> }));
vi.mock("./pages/CurrentSeason", () => ({ default: () => <h1>Season page</h1> }));
vi.mock("./pages/NextRace", () => ({ default: () => <h1>Next race page</h1> }));
vi.mock("./pages/Models", () => ({ default: () => <h1>Models page</h1> }));
vi.mock("./pages/RawData", () => ({ default: () => <h1>Raw data page</h1> }));
vi.mock("./pages/BettingResearch", () => ({ default: () => <h1>Betting page</h1> }));

describe("application shell", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/");
    Object.defineProperty(window, "scrollTo", { configurable: true, value: vi.fn() });
    document.title = "";
  });

  it("renders the Streamlit reference brand, title, captions, and seven tabs", async () => {
    render(<App />);
    expect(screen.getByRole("img", { name: "Gridlocked" })).toHaveAttribute("src", "/api/brand/logo");
    expect(await screen.findByText("F1 Races from 2016 to 2026")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Main sections" })).toBeInTheDocument();
    expect(screen.getAllByRole("tab")).toHaveLength(7);
    expect(screen.getByRole("tab", { name: "📊 Data Explorer" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "💾 Data & Debug" })).toBeInTheDocument();
  });

  it("switches sections from the Streamlit-style tab row and preserves the Streamlit page title", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "📈 Analytics & Visualizations" }));
    expect(await screen.findByRole("heading", { name: "Analytics page" })).toBeInTheDocument();
    await waitFor(() => expect(document.title).toBe("Gridlocked - Formula 1 Betting & Analytics"));
    expect(window.location.hash).toBe("#/Analytics");
  });
});
