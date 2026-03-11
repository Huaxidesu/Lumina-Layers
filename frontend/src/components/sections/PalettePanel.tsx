import { useConverterStore } from "../../stores/converterStore";
import type { PaletteEntry } from "../../api/types";
import Accordion from "../ui/Accordion";
import Slider from "../ui/Slider";
import Button from "../ui/Button";

// ========== PaletteItem ==========

interface PaletteItemProps {
  entry: PaletteEntry;
  isSelected: boolean;
  remappedHex: string | undefined;
  heightMm: number | undefined;
  showHeightSlider: boolean;
  maxHeight: number;
  onSelect: () => void;
  onHeightChange: (h: number) => void;
}

function PaletteItem({
  entry,
  isSelected,
  remappedHex,
  heightMm,
  showHeightSlider,
  maxHeight,
  onSelect,
  onHeightChange,
}: PaletteItemProps) {
  const displayHex = remappedHex ?? entry.matched_hex;
  const isRemapped = !!remappedHex;

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={`颜色 #${entry.matched_hex}，占比 ${entry.percentage.toFixed(1)}%${isRemapped ? `，已替换为 #${remappedHex}` : ""}`}
      aria-pressed={isSelected}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
      className={`flex flex-col gap-2 rounded-md px-3 py-2 cursor-pointer transition-colors ${
        isSelected
          ? "ring-2 ring-blue-500 bg-gray-700/60"
          : "hover:bg-gray-700/40"
      }`}
    >
      <div className="flex items-center gap-3">
        {/* Dual color swatches */}
        <div className="flex items-center gap-1 shrink-0">
          <span
            className="inline-block w-5 h-5 rounded border border-gray-600"
            style={{ backgroundColor: `#${entry.quantized_hex}` }}
            title={`原色 #${entry.quantized_hex}`}
          />
          <span className="text-gray-500 text-xs">→</span>
          <span
            className={`inline-block w-5 h-5 rounded border ${isRemapped ? "border-yellow-500" : "border-gray-600"}`}
            style={{ backgroundColor: `#${displayHex}` }}
            title={isRemapped ? `替换色 #${remappedHex}` : `匹配色 #${entry.matched_hex}`}
          />
        </div>

        {/* Hex label + percentage */}
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <span className="text-xs text-gray-300 font-mono truncate">
            #{displayHex}
          </span>
          {isRemapped && (
            <span className="text-[10px] text-yellow-400 shrink-0">已替换</span>
          )}
        </div>
        <span className="text-xs text-gray-400 tabular-nums shrink-0">
          {entry.percentage.toFixed(1)}%
        </span>
      </div>

      {/* Height slider (relief mode only) */}
      {showHeightSlider && (
        <div
          onClick={(e) => e.stopPropagation()}
          onKeyDown={(e) => e.stopPropagation()}
        >
          <Slider
            label="深度"
            value={heightMm ?? maxHeight * 0.5}
            min={0.08}
            max={maxHeight}
            step={0.04}
            unit="mm"
            onChange={onHeightChange}
          />
        </div>
      )}
    </div>
  );
}

// ========== PalettePanel ==========

export default function PalettePanel() {
  const palette = useConverterStore((s) => s.palette);
  const selectedColor = useConverterStore((s) => s.selectedColor);
  const setSelectedColor = useConverterStore((s) => s.setSelectedColor);
  const enable_relief = useConverterStore((s) => s.enable_relief);
  const color_height_map = useConverterStore((s) => s.color_height_map);
  const updateColorHeight = useConverterStore((s) => s.updateColorHeight);
  const colorRemapMap = useConverterStore((s) => s.colorRemapMap);
  const remapHistory = useConverterStore((s) => s.remapHistory);
  const undoColorRemap = useConverterStore((s) => s.undoColorRemap);
  const clearAllRemaps = useConverterStore((s) => s.clearAllRemaps);
  const heightmap_max_height = useConverterStore((s) => s.heightmap_max_height);
  const replacePreviewLoading = useConverterStore((s) => s.replacePreviewLoading);
  const submitReplacePreview = useConverterStore((s) => s.submitReplacePreview);

  const hasRemaps = Object.keys(colorRemapMap).length > 0;
  const hasHistory = remapHistory.length > 0;

  const handleSelect = (hex: string) => {
    setSelectedColor(selectedColor === hex ? null : hex);
  };

  return (
    <Accordion title="调色板 / 颜色映射">
      {palette.length === 0 ? (
        <p className="text-xs text-gray-500 py-2">
          暂无调色板数据，请先完成预览
        </p>
      ) : (
        <div className="flex flex-col gap-1">
          {/* Undo / Clear / Apply buttons */}
          <div className="flex gap-2 mb-2">
            <Button
              label="撤销"
              variant="secondary"
              onClick={undoColorRemap}
              disabled={!hasHistory}
            />
            <Button
              label="清空替换"
              variant="secondary"
              onClick={clearAllRemaps}
              disabled={!hasRemaps}
            />
            <Button
              label="应用替换到预览"
              variant="primary"
              onClick={() => void submitReplacePreview()}
              disabled={!hasRemaps || replacePreviewLoading}
              loading={replacePreviewLoading}
            />
          </div>

          {/* Palette items */}
          <div
            className="flex flex-col gap-1 max-h-80 overflow-y-auto"
            role="listbox"
            aria-label="调色板颜色列表"
          >
            {palette.map((entry) => (
              <PaletteItem
                key={entry.matched_hex}
                entry={entry}
                isSelected={selectedColor === entry.matched_hex}
                remappedHex={colorRemapMap[entry.matched_hex]}
                heightMm={color_height_map[entry.matched_hex]}
                showHeightSlider={enable_relief}
                maxHeight={heightmap_max_height}
                onSelect={() => handleSelect(entry.matched_hex)}
                onHeightChange={(h) => updateColorHeight(entry.matched_hex, h)}
              />
            ))}
          </div>
        </div>
      )}
    </Accordion>
  );
}
