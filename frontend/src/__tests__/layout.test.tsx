import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "../App";

vi.mock("../api/client", () => ({
  default: {
    get: vi.fn(),
  },
}));

vi.mock("../api/converter", () => ({
  fetchLutList: vi.fn().mockResolvedValue({ luts: [] }),
  convertPreview: vi.fn(),
  convertGenerate: vi.fn(),
  getFileUrl: vi.fn(),
}));

import apiClient from "../api/client";

describe("EditorLayout (Phase 4)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.get).mockResolvedValue({
      data: { status: "ok", version: "2.0", uptime_seconds: 100 },
    });
  });

  it("renders left panel (aside) with LeftPanel component", () => {
    render(<App />);
    const panel = screen.getByTestId("left-panel");
    expect(panel).toBeInTheDocument();
    expect(panel.tagName).toBe("ASIDE");
  });

  it("renders right canvas area (section)", () => {
    render(<App />);
    const canvas = screen.getByTestId("canvas-area");
    expect(canvas).toBeInTheDocument();
    expect(canvas.tagName).toBe("SECTION");
  });

  it('preserves "Lumina Studio 2.0" header', () => {
    render(<App />);
    expect(screen.getByText("Lumina Studio 2.0")).toBeInTheDocument();
  });

  it("left panel contains control panel content", () => {
    render(<App />);
    const panel = screen.getByTestId("left-panel");
    // LeftPanel now renders real controls instead of placeholder text
    expect(panel).toBeInTheDocument();
    // Verify it contains actual UI elements (e.g., the image upload area)
    expect(panel.querySelector("input[type='file']")).toBeInTheDocument();
  });
});
