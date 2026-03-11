import { useCalibrationStore } from "../stores/calibrationStore";
import { CalibrationColorMode, BackingColor } from "../api/types";
import Dropdown from "./ui/Dropdown";
import Slider from "./ui/Slider";
import Button from "./ui/Button";

const colorModeOptions = Object.values(CalibrationColorMode).map((v) => ({
  label: v,
  value: v,
}));

const backingColorOptions = Object.values(BackingColor).map((v) => ({
  label: v,
  value: v,
}));

export default function CalibrationPanel() {
  const {
    color_mode,
    block_size,
    gap,
    backing,
    isLoading,
    error,
    downloadUrl,
    previewImageUrl,
    statusMessage,
    setColorMode,
    setBlockSize,
    setGap,
    setBacking,
    submitGenerate,
  } = useCalibrationStore();

  const isEightColor = color_mode === CalibrationColorMode.EIGHT_COLOR;
  const isFiveColorExt = color_mode === CalibrationColorMode.FIVE_COLOR_EXT;
  const isSixColor = color_mode === CalibrationColorMode.SIX_COLOR;
  const blockSizeDisabled = isEightColor || isFiveColorExt;
  const gapDisabled = isEightColor || isFiveColorExt;
  const backingDisabled = isEightColor || isFiveColorExt || isSixColor;

  return (
    <aside
      data-testid="calibration-panel"
      className="w-[350px] h-full overflow-y-auto bg-gray-800 p-4 flex flex-col gap-4"
    >
      <Dropdown
        label="颜色模式"
        value={color_mode}
        options={colorModeOptions}
        onChange={(v) => setColorMode(v as CalibrationColorMode)}
      />

      <Slider
        label="色块尺寸"
        value={block_size}
        min={3}
        max={10}
        step={0.5}
        unit="mm"
        onChange={setBlockSize}
        disabled={blockSizeDisabled}
      />

      <Slider
        label="色块间距"
        value={gap}
        min={0.4}
        max={2.0}
        step={0.01}
        unit="mm"
        onChange={setGap}
        disabled={gapDisabled}
      />

      <Dropdown
        label="底板颜色"
        value={backing}
        options={backingColorOptions}
        onChange={(v) => setBacking(v as BackingColor)}
        disabled={backingDisabled}
      />

      <Button
        label="生成校准板"
        variant="primary"
        onClick={() => void submitGenerate()}
        disabled={isLoading}
        loading={isLoading}
      />

      {statusMessage && (
        <p data-testid="status-message" className="text-xs text-green-400">
          {statusMessage}
        </p>
      )}

      {error && (
        <p data-testid="error-message" className="text-xs text-red-400">
          {error}
        </p>
      )}

      {downloadUrl && (
        <a
          data-testid="download-link"
          href={downloadUrl}
          download
          className="text-sm text-blue-400 underline hover:text-blue-300"
        >
          下载 3MF 文件
        </a>
      )}

      {previewImageUrl && (
        <img
          data-testid="preview-image"
          src={previewImageUrl}
          alt="校准板预览"
          className="w-full rounded-md border border-gray-700"
        />
      )}
    </aside>
  );
}
