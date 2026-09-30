import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

vi.mock("./api", () => ({
  api: { get: vi.fn().mockResolvedValue({ status: "ok", rss_mb: 120 }) },
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

  it("renders the reference brand, section navigation, and API status", async () => {
    render(<App />);
    expect(screen.getByRole("img", { name: "Gridlocked" })).toHaveAttribute("src", "/api/brand/logo");
    expect(screen.getByText(/F1 Races from 2016 to/)).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Sections" })).toBeInTheDocument();
    expect(await screen.findByText("API connected")).toBeInTheDocument();
  });

  it("switches sections from the horizontal navigation and updates the title", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Analytics" }));
    expect(await screen.findByRole("heading", { name: "Analytics page" })).toBeInTheDocument();
    await waitFor(() => expect(document.title).toBe("Analytics — F1 Analysis"));
    expect(window.location.hash).toBe("#/Analytics");
  });

  it("persists the light theme toggle", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("checkbox", { name: "Use light theme" }));
    await waitFor(() => expect(document.documentElement).toHaveAttribute("data-theme", "light"));
    expect(localStorage.getItem("f1analysis.theme")).toBe("light");
  });
});