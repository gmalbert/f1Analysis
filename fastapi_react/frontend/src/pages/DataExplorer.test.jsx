import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const apiMock = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock("../api.js", () => ({ api: apiMock }));

import FilterSidebar from "../components/FilterSidebar.jsx";
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

  it("enabling Filter Results stores Streamlit defaults and loads the unfiltered table", async () => {
    apiMock.get.mockResolvedValueOnce({ filters: [
      { column: "grandPrixYear", label: "Year", kind: "range", min: 2016, max: 2026 },
      { column: "DNF", label: "DNF", kind: "boolean" },
    ] });
    apiMock.post.mockResolvedValue({ total: 12, columns: ["grandPrixYear"], rows: [{ grandPrixYear: 2026 }] });
    render(<DataExplorer />);
    fireEvent.click(await screen.findByRole("checkbox", { name: "Filter Results" }));
    await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
    expect(JSON.parse(sessionStorage.getItem("f1analysis.filters"))).toMatchObject({
      applied: true,
      filters: [{ column: "grandPrixYear", kind: "range", value: [2016, 2026] }],
    });
    expect(await screen.findByText("Number of filtered results: 12")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Data" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Data & Debug" })).toBeInTheDocument();
  });
});

describe("persistent filter sidebar", () => {
  it("renders all schema controls and updates shared filter state", async () => {
    sessionStorage.setItem("f1analysis.filters", JSON.stringify({
      applied: true,
      filters: [{ column: "grandPrixYear", kind: "range", value: [2016, 2026] }],
      values: { grandPrixYear: [2016, 2026] },
    }));
    apiMock.get.mockResolvedValueOnce({ filters: [
      { column: "grandPrixYear", label: "Year", kind: "range", min: 2016, max: 2026 },
      { column: "DNF", label: "DNF", kind: "boolean" },
    ] });
    render(<FilterSidebar />);
    expect(await screen.findByText("Select filters to apply:")).toBeInTheDocument();
    fireEvent.change(screen.getByRole("slider", { name: "Year minimum" }), { target: { value: "2020" } });
    await waitFor(() => {
      expect(JSON.parse(sessionStorage.getItem("f1analysis.filters"))).toMatchObject({
        applied: true,
        filters: [{ column: "grandPrixYear", kind: "range", value: [2020, 2026] }],
      });
    });
  });
});
