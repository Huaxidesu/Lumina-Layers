import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, cleanup } from "@testing-library/react";
import fc from "fast-check";
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

describe("Feature: frontend-scaffold, Property 2: 非 'ok' 状态显示失败徽章", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    cleanup();
  });

  /**
   * Validates: Requirements 5.3
   * For any non-"ok" status string, the App should render a red (fail) badge.
   */
  it("renders red badge for any non-ok status", async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.string().filter((s) => s !== "ok"),
        async (status) => {
          vi.clearAllMocks();
          cleanup();

          vi.mocked(apiClient.get).mockResolvedValueOnce({
            data: { status, version: "2.0", uptime_seconds: 0 },
          });

          render(<App />);

          await waitFor(() => {
            expect(screen.getByTestId("health-badge-fail")).toBeInTheDocument();
          });

          expect(
            screen.queryByTestId("health-badge-ok")
          ).not.toBeInTheDocument();
        }
      ),
      { numRuns: 100 }
    );
  });
});
