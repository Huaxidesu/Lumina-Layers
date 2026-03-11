import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import App from "../App";

// Mock apiClient to prevent real HTTP calls (health check)
vi.mock("../api/client", () => ({
  default: {
    get: vi.fn().mockResolvedValue({ data: { status: "ok" } }),
    post: vi.fn(),
  },
}));

// Mock Scene3D to avoid Three.js rendering in jsdom
vi.mock("../components/Scene3D", () => ({
  default: ({ modelUrl }: { modelUrl?: string }) => (
    <div data-testid="scene3d-mock">{modelUrl ?? "no-model"}</div>
  ),
}));

// Mock calibration API to prevent CalibrationPanel network calls
vi.mock("../api/calibration", () => ({
  calibrationGenerate: vi.fn(),
}));

// Mock converter API to prevent LeftPanel's fetchLutList call
vi.mock("../api/converter", () => ({
  fetchLutList: vi.fn().mockResolvedValue({ luts: [] }),
  convertPreview: vi.fn(),
  convertGenerate: vi.fn(),
  getFileUrl: vi.fn((id: string) => `/api/files/${id}`),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

describe("App Tab Navigation", () => {
  it("defaults to Converter tab active", () => {
    render(<App />);

    const converterTab = screen.getByTestId("tab-converter");
    const calibrationTab = screen.getByTestId("tab-calibration");

    expect(converterTab).toHaveAttribute("aria-selected", "true");
    expect(calibrationTab).toHaveAttribute("aria-selected", "false");
  });

  it("renders LeftPanel when Converter tab is active", () => {
    render(<App />);

    expect(screen.getByTestId("left-panel")).toBeInTheDocument();
    expect(screen.queryByTestId("calibration-panel")).not.toBeInTheDocument();
  });

  it("switches to CalibrationPanel when Calibration tab is clicked", () => {
    render(<App />);

    fireEvent.click(screen.getByTestId("tab-calibration"));

    expect(screen.getByTestId("calibration-panel")).toBeInTheDocument();
    expect(screen.queryByTestId("left-panel")).not.toBeInTheDocument();

    const calibrationTab = screen.getByTestId("tab-calibration");
    expect(calibrationTab).toHaveAttribute("aria-selected", "true");
  });

  it("switches back to LeftPanel when Converter tab is clicked", () => {
    render(<App />);

    // Switch to Calibration first
    fireEvent.click(screen.getByTestId("tab-calibration"));
    expect(screen.getByTestId("calibration-panel")).toBeInTheDocument();

    // Switch back to Converter
    fireEvent.click(screen.getByTestId("tab-converter"));
    expect(screen.getByTestId("left-panel")).toBeInTheDocument();
    expect(screen.queryByTestId("calibration-panel")).not.toBeInTheDocument();

    const converterTab = screen.getByTestId("tab-converter");
    expect(converterTab).toHaveAttribute("aria-selected", "true");
  });

  it("renders CalibrationPanel when Calibration tab is active", () => {
    render(<App />);

    fireEvent.click(screen.getByTestId("tab-calibration"));

    const calibrationTab = screen.getByTestId("tab-calibration");
    const converterTab = screen.getByTestId("tab-converter");

    expect(calibrationTab).toHaveAttribute("aria-selected", "true");
    expect(converterTab).toHaveAttribute("aria-selected", "false");
    expect(screen.getByTestId("calibration-panel")).toBeInTheDocument();
  });
});
