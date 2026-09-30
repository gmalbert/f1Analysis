import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

const apiMock = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock("../api.js", () => ({ api: apiMock }));

import CurrentSeason from "./CurrentSeason.jsx";

beforeEach(() => {
  apiMock.get.mockReset();
  apiMock.post.mockReset();
});

describe("CurrentSeason page", () => {
  it("renders the ten-column Streamlit schedule and next-race highlight", async () => {
    apiMock.get.mockResolvedValueOnce({
      year: 2026,
      rows: [
        { round: 1, fullName: "Australian Grand Prix", date: "2026-03-08", time: "05:00", circuitType: "Race", courseLength: 5.3, laps: 58, turns: 14, distance: 307, totalRacesHeld: 40, seasonStatus: "Completed" },
        { round: 2, fullName: "Chinese Grand Prix", date: "2026-03-15", time: "07:00", circuitType: "Race", courseLength: 5.4, laps: 56, turns: 16, distance: 305, totalRacesHeld: 20, seasonStatus: "Next Race" },
      ],
    });
    render(<CurrentSeason />);
    await waitFor(() => expect(screen.getByText("Australian Grand Prix")).toBeInTheDocument());
    expect(screen.getByText("Total number of races: 2")).toBeInTheDocument();
    expect(screen.getByText("Chinese Grand Prix").closest("tr")).toHaveClass("next-race-row");
  });
});
