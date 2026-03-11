import { useEffect } from "react";
import { useLutManagerStore } from "../stores/lutManagerStore";
import Dropdown from "./ui/Dropdown";
import Slider from "./ui/Slider";
import Button from "./ui/Button";

export default function LutManagerPanel() {
  const {
    lutList,
    lutListLoading,
    primaryName,
    primaryInfo,
    primaryLoading,
    secondaryNames,
    secondaryInfos,
    filteredSecondaryOptions,
    dedupThreshold,
    merging,
    mergeResult,
    error,
    fetchLutList,
    selectPrimary,
    setSecondaryNames,
    setDedupThreshold,
    executeMerge,
    clearError,
  } = useLutManagerStore();

  useEffect(() => {
    void fetchLutList();
  }, [fetchLutList]);

  const allDisabled = merging;

  const primaryOptions = lutList.map((lut) => ({
    label: lut.name,
    value: lut.name,
  }));

  const isPrimaryModeInvalid =
    primaryInfo !== null &&
    !primaryInfo.color_mode.startsWith("6-Color") &&
    !primaryInfo.color_mode.startsWith("8-Color");

  const mergeDisabled =
    merging ||
    !primaryName ||
    secondaryNames.length === 0 ||
    isPrimaryModeInvalid;

  const handleSecondaryToggle = (name: string) => {
    if (allDisabled) return;
    const updated = secondaryNames.includes(name)
      ? secondaryNames.filter((n) => n !== name)
      : [...secondaryNames, name];
    setSecondaryNames(updated);
  };

  return (
    <aside
      data-testid="lut-manager-panel"
      className="w-[350px] h-full overflow-y-auto bg-gray-800 p-4 flex flex-col gap-4"
    >
      <div>
        <h2 className="text-lg font-semibold text-gray-100">LUT Merge Tool</h2>
        <p className="text-xs text-gray-400 mt-1">
          将多个 LUT 合并为一个，支持 Delta-E 去重。Primary LUT 必须为 6-Color 或 8-Color 模式。
        </p>
      </div>

      {/* Primary LUT 选择 */}
      <div data-testid="primary-dropdown">
        <Dropdown
          label="Primary LUT"
          value={primaryName}
          options={primaryOptions}
          onChange={(v) => void selectPrimary(v)}
          disabled={allDisabled || lutListLoading}
          placeholder="选择主 LUT..."
        />
        {primaryLoading && (
          <p data-testid="loading-indicator" className="text-xs text-gray-400 mt-1">
            加载中...
          </p>
        )}
        {primaryInfo && (
          <p className="text-xs text-gray-400 mt-1">
            Mode: {primaryInfo.color_mode} ({primaryInfo.color_count} colors)
          </p>
        )}
        {isPrimaryModeInvalid && (
          <p className="text-xs text-yellow-400 mt-1">
            主 LUT 必须为 6-Color 或 8-Color 模式
          </p>
        )}
      </div>

      {/* Secondary LUT 多选 */}
      <div data-testid="secondary-list" className="flex flex-col gap-1">
        <label className="text-sm text-gray-300">Secondary LUTs</label>
        <div className="max-h-40 overflow-y-auto rounded-md border border-gray-600 bg-gray-700 p-2 flex flex-col gap-1">
          {filteredSecondaryOptions.length === 0 ? (
            <p className="text-xs text-gray-500">
              {primaryName ? "无可用的 Secondary LUT" : "请先选择 Primary LUT"}
            </p>
          ) : (
            filteredSecondaryOptions.map((name) => {
              const info = secondaryInfos.get(name);
              return (
                <label
                  key={name}
                  className="flex items-center gap-2 text-xs text-gray-200 cursor-pointer hover:bg-gray-600 rounded px-1 py-0.5"
                >
                  <input
                    type="checkbox"
                    checked={secondaryNames.includes(name)}
                    onChange={() => handleSecondaryToggle(name)}
                    disabled={allDisabled}
                    className="accent-blue-500"
                  />
                  <span className="truncate">{name}</span>
                  {info && (
                    <span className="text-gray-400 ml-auto shrink-0">
                      {info.color_mode} ({info.color_count})
                    </span>
                  )}
                </label>
              );
            })
          )}
        </div>
      </div>

      {/* Dedup Threshold 滑块 */}
      <div>
        <Slider
          label="Dedup Threshold"
          value={dedupThreshold}
          min={0}
          max={20}
          step={0.5}
          onChange={setDedupThreshold}
          disabled={allDisabled}
        />
        <p className="text-xs text-gray-500 mt-1">
          0 = 仅精确去重，值越大去除越多相近色
        </p>
      </div>

      {/* Merge & Save 按钮 */}
      <Button
        label="Merge & Save"
        variant="primary"
        onClick={() => void executeMerge()}
        disabled={mergeDisabled}
        loading={merging}
      />

      {/* 合并结果 */}
      {mergeResult && (
        <div data-testid="merge-result" className="rounded-md bg-green-900/30 border border-green-700 p-3 text-xs text-green-300 flex flex-col gap-1">
          <p>✓ 合并成功！</p>
          <p>
            合并前: {mergeResult.stats.total_before} 色 → 合并后: {mergeResult.stats.total_after} 色
          </p>
          <p>
            精确去重: {mergeResult.stats.exact_dupes} | 相近色去除: {mergeResult.stats.similar_removed}
          </p>
          <p>文件: {mergeResult.filename}</p>
        </div>
      )}

      {/* 错误消息 */}
      {error && (
        <div data-testid="error-message" className="rounded-md bg-red-900/30 border border-red-700 p-3 text-xs text-red-300 flex items-start gap-2">
          <span className="shrink-0">✗</span>
          <span>{error}</span>
          <button
            onClick={clearError}
            className="ml-auto text-red-400 hover:text-red-200 shrink-0"
            aria-label="关闭错误"
          >
            ×
          </button>
        </div>
      )}
    </aside>
  );
}
