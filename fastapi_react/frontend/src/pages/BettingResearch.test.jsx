import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const apiMock = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock("../api.js", () => ({ api: apiMock }));

import BettingResearch from "./BettingResearch.jsx";

beforeEach(() => {
  apiMock.get.mockReset();
  apiMock.post.mockReset();
  apiMock.post.mockImplementation((url) => {
    if (url === "/api/betting/value") {
      return Promise.resolve({
        market_probability: 0.5,
        raw_ev: 0.05,
        adjusted_probability: 0.45,
        stake: 50,
        reason_code: "positive_ev",
      });
    }
    return Promise.resolve({});
  });
});

describe("BettingResearch page", () => {
  it("matches the Streamlit value-and-stake defaults and fixed $10k bankroll", async () => {
    render(<BettingResearch />);
    expect(screen.getByText("Probability & Betting Research")).toBeInTheDocument();
    await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
      "/api/betting/value",
      expect.objectContaining({
        model_probability: 0.25,
        decimal_odds: 2.1,
        opposing_odds: 1.8,
        uncertainty: 0.02,
        bankroll: 10000,
      }),
    ));
    expect(await screen.findByText("50.00%")).toBeInTheDocument();
    expect(screen.getByText("Paper stake on $10k")).toBeInTheDocument();
  });

  it("offers the exact field template flow and selected simulation count", async () => {
    apiMock.post.mockImplementation((url) => {
      if (url === "/api/betting/simulate") {
        return Promise.resolve({
          rows: [{ driver_id: "driver-a", win_probability: 0.5 }],
          columns: ["driver_id", "win_probability"],
        });
      }
      return Promise.resolve({
        market_probability: 0.5,
        raw_ev: 0.05,
        adjusted_probability: 0.45,
        stake: 50,
        reason_code: "positive_ev",
      });
    });
    render(<BettingResearch />);
    fireEvent.click(screen.getByRole("tab", { name: "Field simulation" }));
    const template = screen.getByRole("link", { name: "Download input template" });
    expect(template).toHaveAttribute("download", "f1_field_simulation_template.csv");
    fireEvent.change(screen.getByRole("slider", { name: "Simulations" }), { target: { value: "12000" } });
    fireEvent.click(screen.getByRole("button", { name: "Run coherent field simulation" }));
    await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
      "/api/betting/simulate",
      expect.objectContaining({ simulations: 12000 }),
    ));
    expect(await screen.findByText("driver-a")).toBeInTheDocument();
  });

  it("shows Streamlit's no-ledger message until a replay CSV is uploaded", () => {
    render(<BettingResearch />);
    fireEvent.click(screen.getByRole("tab", { name: "Paper replay" }));
    expect(screen.getByText("No odds ledger is bundled, so profitability is intentionally not estimated.")).toBeInTheDocument();
  });

  it("shows Streamlit's calibration requirements before upload", () => {
    render(<BettingResearch />);
    fireEvent.click(screen.getByRole("tab", { name: "Calibration" }));
    expect(screen.getByText("Required columns: probability and outcome. Optional columns: market and stage.")).toBeInTheDocument();
  });
});
