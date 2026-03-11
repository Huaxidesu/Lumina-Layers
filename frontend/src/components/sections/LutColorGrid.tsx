import { useState, useMemo, useEffect } from "react";
import { useConverterStore } from "../../stores/converterStore";
import type { LutColorEntry } from "../../api/types";
import Accordion from "../ui/Accordion";

export type HueCategory =
  | "all"
  | "red"
  | "orange"
  | "yellow"
  | "green"
  | "cyan"
  | "blue"
  | "purple"
  | "neutral";

const HUE_FILTERS: { key: HueCategory; label: string; dot: string }[] = [
  { key: "all", label: "全部", dot: "" },
  { key: "red", label: "红", dot: "#e53935" },
  { key: "orange", label: "橙", dot: "#fb8c00" },
  { key: "yellow", label: "黄", dot: "#fdd835" },
  { key: "green", label: "绿", dot: "#43a047" },
  { key: "cyan", label: "青", dot: "#00acc1" },
  { key: "blue", label: "蓝", dot: "#1e88e5" },
  { key: "purple", label: "紫", dot: "#8e24aa" },
  { key: "neutral", label: "中性", dot: "#9e9e9e" },
];

export function classifyHue(r: number, g: number, b: number): HueCategory {
  const rf = r / 255;
  const gf = g / 255;
  const bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const d = max - min;
  const s = max === 0 ? 0 : d / max;
  const v = max;

  if (s < 0.15 || v < 0.1) return "neutral";

  let h = 0;
  if (d !== 0) {
    if (max === rf) h = ((gf - bf) / d) % 6;
    else if (max === gf) h = (bf - rf) / d + 2;
    else h = (rf - gf) / d + 4;
  }
  h = ((h * 60) + 360) % 360;

  if (h < 15 || h >= 345) return "red";
  if (h < 40) return "orange";
  if (h < 70) return "yellow";
  if (h < 160) return "green";
  if (h < 195) return "cyan";
  if (h < 260) return "blue";
  if (h < 345) return "purple";
  return "neutral";
}

/**
 * Check if a LUT color entry matches a search query.
 * 检查 LUT 颜色条目是否匹配搜索查询。
 *
 * Supports hex substring matching and RGB exact matching (e.g. "255,0,0" or "rgb(255,0,0)").
 * 支持 hex 子串匹配和 RGB 精确匹配（如 "255,0,0" 或 "rgb(255,0,0)"）。
 */
export function matchesSearch(entry: LutColorEntry, query: string): boolean {
  const q = query.toLowerCase().trim();
  if (!q) return true;

  // Hex match
  const hexNoHash = entry.hex.replace("#", "").toLowerCase();
  if (hexNoHash.includes(q.replace("#", ""))) return true;

  // RGB match: extract three numbers from query
  const rgbMatch = q.match(/(\d{1,3})\s*[,\s]\s*(\d{1,3})\s*[,\s]\s*(\d{1,3})/);
  if (rgbMatch) {
    const [, rs, gs, bs] = rgbMatch;
    const [r, g, b] = entry.rgb;
    if (r === Number(rs) && g === Number(gs) && b === Number(bs)) return true;
  }

  return false;
}

function ColorSwatch({
  entry,
  isTarget,
  onClick,
}: {
  entry: LutColorEntry;
  isTarget: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={`替换为颜色 ${entry.hex}`}
      aria-selected={isTarget}
      onClick={onClick}
      className={`flex flex-col items-center gap-0.5 rounded p-1 cursor-pointer transition-colors hover:bg-gray-700/50 ${
        isTarget ? "ring-2 ring-yellow-500" : ""
      }`}
    >
      <span
        className="block w-7 h-7 rounded border border-gray-600"
        style={{ backgroundColor: entry.hex }}
      />
      <span className="text-[9px] text-gray-400 font-mono leading-tight">
        {entry.hex}
      </span>
    </button>
  );
}

function ColorSection({
  title,
  titleColor,
  colors,
  selectedColor,
  colorRemapMap,
  onColorClick,
}: {
  title: string;
  titleColor: string;
  colors: LutColorEntry[];
  selectedColor: string | null;
  colorRemapMap: Record<string, string>;
  onColorClick: (hex: string) => void;
}) {
  if (colors.length === 0) return null;
  return (
    <div>
      <p className={`text-[11px] font-semibold mb-1`} style={{ color: titleColor }}>
        {title}
      </p>
      <div className="grid grid-cols-6 gap-1">
        {colors.map((c) => {
          const hexNoHash = c.hex.replace("#", "");
          const isTarget = selectedColor
            ? colorRemapMap[selectedColor] === hexNoHash
            : false;
          return (
            <ColorSwatch
              key={c.hex}
              entry={c}
              isTarget={isTarget}
              onClick={() => onColorClick(c.hex)}
            />
          );
        })}
      </div>
    </div>
  );
}

export default function LutColorGrid() {
  const palette = useConverterStore((s) => s.palette);
  const selectedColor = useConverterStore((s) => s.selectedColor);
  const applyColorRemap = useConverterStore((s) => s.applyColorRemap);
  const setSelectedColor = useConverterStore((s) => s.setSelectedColor);
  const colorRemapMap = useConverterStore((s) => s.colorRemapMap);
  const lutColors = useConverterStore((s) => s.lutColors);
  const lutColorsLoading = useConverterStore((s) => s.lutColorsLoading);
  const lut_name = useConverterStore((s) => s.lut_name);
  const fetchLutColors = useConverterStore((s) => s.fetchLutColors);

  const [hueFilter, setHueFilter] = useState<HueCategory>("all");
  const [searchText, setSearchText] = useState("");

  // Fetch LUT colors when lut_name changes
  useEffect(() => {
    if (lut_name) {
      fetchLutColors(lut_name);
    }
  }, [lut_name, fetchLutColors]);

  // Build set of palette hex values (colors used in image)
  const usedHexSet = useMemo(() => {
    const s = new Set<string>();
    for (const e of palette) {
      s.add(`#${e.matched_hex}`.toLowerCase());
    }
    return s;
  }, [palette]);

  // Classify and filter colors
  const { usedColors, otherColors, visibleCount } = useMemo(() => {
    const used: LutColorEntry[] = [];
    const other: LutColorEntry[] = [];

    for (const c of lutColors) {
      const hex = c.hex.toLowerCase();
      const [r, g, b] = c.rgb;

      // Hue filter
      if (hueFilter !== "all" && classifyHue(r, g, b) !== hueFilter) continue;

      // Search filter (using matchesSearch pure function)
      if (searchText && !matchesSearch(c, searchText)) continue;

      if (usedHexSet.has(hex)) {
        used.push(c);
      } else {
        other.push(c);
      }
    }

    return { usedColors: used, otherColors: other, visibleCount: used.length + other.length };
  }, [lutColors, hueFilter, searchText, usedHexSet]);

  const handleColorClick = (clickedHex: string) => {
    if (!selectedColor) return;
    const hexNoHash = clickedHex.replace("#", "");
    applyColorRemap(selectedColor, hexNoHash);
    setSelectedColor(null);
  };

  return (
    <Accordion title="LUT 颜色网格">
      {lutColorsLoading ? (
        <p className="text-xs text-gray-500 py-2">加载 LUT 颜色中...</p>
      ) : lutColors.length === 0 ? (
        <p className="text-xs text-gray-500 py-2">
          请先选择 LUT 以加载可用颜色
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {/* Status line */}
          <p className="text-xs text-gray-400">
            共 {lutColors.length} 色，显示 {visibleCount} 色
            {selectedColor && (
              <>
                {" · 已选中 "}
                <span
                  className="inline-block w-3 h-3 rounded-sm border border-gray-600 align-middle"
                  style={{ backgroundColor: `#${selectedColor}` }}
                />
                <span className="font-mono"> #{selectedColor}</span>
              </>
            )}
          </p>

          {/* Search */}
          <input
            type="text"
            placeholder="搜索 HEX / RGB 颜色..."
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            className="w-full px-2 py-1 text-xs rounded border border-gray-600 bg-gray-800 text-gray-200 outline-none focus:border-blue-500"
          />

          {/* Hue filter bar */}
          <div className="flex flex-wrap gap-1">
            {HUE_FILTERS.map((f) => (
              <button
                key={f.key}
                type="button"
                onClick={() => setHueFilter(f.key)}
                className={`flex items-center gap-1 px-2 py-0.5 text-[10px] rounded-full border transition-colors ${
                  hueFilter === f.key
                    ? "bg-gray-200 text-gray-900 border-gray-400"
                    : "bg-gray-800 text-gray-400 border-gray-600 hover:border-gray-400"
                }`}
              >
                {f.dot && (
                  <span
                    className="inline-block w-2 h-2 rounded-full"
                    style={{ backgroundColor: f.dot }}
                  />
                )}
                {f.label}
              </button>
            ))}
          </div>

          {/* Color grid */}
          <div className="max-h-72 overflow-y-auto flex flex-col gap-3" role="listbox" aria-label="LUT 可用颜色列表">
            <ColorSection
              title={`图中使用 (${usedColors.length})`}
              titleColor="#4CAF50"
              colors={usedColors}
              selectedColor={selectedColor}
              colorRemapMap={colorRemapMap}
              onColorClick={handleColorClick}
            />
            <ColorSection
              title={usedColors.length > 0 ? `其他可用 (${otherColors.length})` : `全部可用 (${otherColors.length})`}
              titleColor="#888"
              colors={otherColors}
              selectedColor={selectedColor}
              colorRemapMap={colorRemapMap}
              onColorClick={handleColorClick}
            />
            {visibleCount === 0 && (
              <p className="text-xs text-gray-500 py-2">无匹配颜色</p>
            )}
          </div>
        </div>
      )}
    </Accordion>
  );
}
