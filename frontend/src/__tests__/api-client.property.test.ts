import { describe, it, expect } from "vitest";
import fc from "fast-check";
import apiClient from "../api/client";

describe("Feature: frontend-scaffold, Property 1: API 客户端 baseURL 前缀", () => {
  /**
   * Validates: Requirements 3.4
   * For any relative path string, the constructed URL should start with the baseURL.
   */
  it("apiClient.getUri({ url: path }) always starts with baseURL", () => {
    const baseURL = "http://localhost:8000/api";
    const pathChars = "/abcdefghijklmnopqrstuvwxyz0123456789-_".split("");
    const pathArb = fc
      .array(fc.constantFrom(...pathChars), { minLength: 1, maxLength: 50 })
      .map((chars) => chars.join(""));

    fc.assert(
      fc.property(
        pathArb,
        (path) => {
          const uri = apiClient.getUri({ url: path });
          expect(uri).toMatch(
            new RegExp(
              `^${baseURL.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`
            )
          );
        }
      ),
      { numRuns: 100 }
    );
  });
});
