import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const apiMock = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock("../api.js", () => ({ api: apiMock }));

import DataExplorer from "./DataExplorer.jsx";

beforeEach(() => {
  apiMock.get.mockReset();
  apiMock.post.mockReset();
  sessionStorage.clear();
});

describe("DataExplorer page", () => {
  it("matches the reference unchecked filter state until users opt in", async () => {
    apiMock.get.mockResolvedValueOnce({ filters: [{ column: "grandPrixYear", label: "Year", kind: "range", min: 2016, max: 2026 }] });
    render(<DataExplorer />);
    expect(screen.getByText("Data Explorer")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Filter Results" })).toBeInTheDocument();
    await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith("/api/data-explorer/schema"));
    expect(apiMock.post).not.toHaveBeenCalled();
  });

  it("enabling Filter Results renders all schema controls and loads the unfiltered table", async () => {
    apiMock.get.mockResolvedValueOnce({ filters: [
      { column: "grandPrixYear", label: "Year", kind: "range", min: 2016, max: 2026 },
      { column: "DNF", label: "DNF", kind: "boolean" },
    ] });
    apiMock.post.mockResolvedValue({ total: 12, columns: ["grandPrixYear"], rows: [{ grandPrixYear: 2026 }] });
    render(<DataExplorer />);
    fireEvent.click(await screen.findByRole("checkbox", { name: "Filter Results" }));
    expect(await screen.findByText("Select filters to apply:")).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "Year minimum" })).toBeInTheDocument();
    await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
    expect(await screen.findByText("Number of filtered results: 12")).toBeInTheDocument();
  });

  it("updates shared filter state when a Streamlit-style control changes", async () => {
    apiMock.get.mockResolvedValueOnce({ filters: [
      { column: "grandPrixYear", label: "Year", kind: "range", min: 2016, max: 2026 },
    ] });
    apiMock.post.mockResolvedValue({ total: 10, columns: ["grandPrixYear"], rows: [] });
    render(<DataExplorer />);
    fireEvent.click(await screen.findByRole("checkbox", { name: "Filter Results" }));
    fireEvent.change(await screen.findByRole("spinbutton", { name: "Year minimum" }), { target: { value: "2020" } });
    await waitFor(() => {
      expect(JSON.parse(sessionStorage.getItem("f1analysis.filters"))).toMatchObject({
        applied: true,
        filters: [{ column: "grandPrixYear", kind: "range", value: [2020, 2026] }],
      });
    });
  });
});
