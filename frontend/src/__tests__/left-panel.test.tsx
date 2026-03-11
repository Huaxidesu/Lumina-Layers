import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import LeftPanel from "../components/LeftPanel";
import { useConverterStore } from "../stores/converterStore";
import { ModelingMode } from "../api/types";

vi.mock("../api/converter", () => ({
  fetchLutList: vi.fn().mockResolvedValue({
    luts: [{ name: "test-lut", color_mode: "4-Color", path: "/test" }],
  }),
  convertPreview: vi.fn(),
  convertGenerate: vi.fn(),
  getFileUrl: vi.fn((id: string) => `/api/files/${id}`),
}));

beforeEach(() => {
  useConverterStore.setState({
    imageFile: null,
    imagePreviewUrl: null,
    aspectRatio: null,
    lut_name: "",
    modeling_mode: ModelingMode.HIGH_FIDELITY,
    enable_relief: false,
    enable_outline: false,
    enable_cloisonne: false,
    isLoading: false,
    error: null,
    lutList: [],
    lutListLoading: false,
  });
});

describe("LeftPanel", () => {
  it("renders LeftPanel with basic settings", () => {
    render(<LeftPanel />);
    expect(screen.getByTestId("left-panel")).toBeInTheDocument();
  });

  it("disables outline checkbox in vector mode", () => {
    useConverterStore.setState({ modeling_mode: ModelingMode.VECTOR });
    render(<LeftPanel />);

    // Expand the 描边设置 accordion
    fireEvent.click(screen.getByText("描边设置"));

    const checkbox = screen.getByLabelText("启用描边");
    expect(checkbox).toBeDisabled();
  });

  it("disables cloisonne checkbox in vector mode", () => {
    useConverterStore.setState({ modeling_mode: ModelingMode.VECTOR });
    render(<LeftPanel />);

    // Expand the 掐丝珐琅 accordion
    fireEvent.click(screen.getByText("掐丝珐琅"));

    const checkbox = screen.getByLabelText("启用掐丝珐琅");
    expect(checkbox).toBeDisabled();
  });

  it("hides relief slider when enable_relief is false", () => {
    useConverterStore.setState({ enable_relief: false });
    render(<LeftPanel />);

    // Expand the 浮雕设置 accordion
    fireEvent.click(screen.getByText("浮雕设置"));

    expect(screen.queryByText("最大高度")).not.toBeInTheDocument();
  });

  it("shows relief slider when enable_relief is true", () => {
    useConverterStore.setState({ enable_relief: true });
    render(<LeftPanel />);

    // Expand the 浮雕设置 accordion
    fireEvent.click(screen.getByText("浮雕设置"));

    expect(screen.getByText("最大高度")).toBeInTheDocument();
  });

  it("disables generate button when no image uploaded", () => {
    useConverterStore.setState({ imageFile: null, lut_name: "" });
    render(<LeftPanel />);

    const generateButton = screen.getByRole("button", { name: "生成" });
    expect(generateButton).toBeDisabled();
  });

  it("loads LUT list on mount", async () => {
    const { fetchLutList } = await import("../api/converter");
    render(<LeftPanel />);

    expect(fetchLutList).toHaveBeenCalled();
  });
});
