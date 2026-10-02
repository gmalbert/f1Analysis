import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const apiMock = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock("../api.js", () => ({ api: apiMock }));

import RawData from "./RawData.jsx";

beforeEach(() => {
  apiMock.get.mockReset();
  apiMock.post.mockReset();
});

describe("RawData page", () => {
  it("matches the Streamlit Data & Debug tab structure", async () => {
    apiMock.get.mockResolvedValue({ status: "ok", expensive_tools_enabled: false });
    render(<RawData />);
    expect(screen.getByText("Data & Debug Tools")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Raw Data" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Temporal Leakage Audit" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Hyperparameter Tuning" })).toBeInTheDocument();
  });

  it("shows the complete unfiltered dataset on demand", async () => {
    apiMock.get.mockResolvedValue({ status: "ok", expensive_tools_enabled: false });
    apiMock.post.mockResolvedValue({ total: 2, columns: ["grandPrixYear"], rows: [{ grandPrixYear: 2025 }, { grandPrixYear: 2026 }] });
    render(<RawData />);
    fireEvent.click(screen.getByRole("checkbox", { name: "Show Raw Data" }));
    await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith("/api/data-explorer/query", expect.objectContaining({ filters: [] })));
    expect(await screen.findByText("Total number of results: 2")).toBeInTheDocument();
  });

  it("keeps the leakage audit gated in hosted mode", async () => {
    apiMock.get.mockResolvedValue({ status: "ok", expensive_tools_enabled: false });
    render(<RawData />);
    fireEvent.click(screen.getByRole("tab", { name: "Temporal Leakage Audit" }));
    expect(await screen.findByRole("button", { name: "Run Leakage Audit" })).toBeDisabled();
    expect(screen.getByText("Research controls are disabled in hosted mode.")).toBeInTheDocument();
  });
});
