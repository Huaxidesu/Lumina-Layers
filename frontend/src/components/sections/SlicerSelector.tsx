import { useEffect } from "react";
import { useSlicerStore } from "../../stores/slicerStore";
import Dropdown from "../ui/Dropdown";
import Button from "../ui/Button";

interface SlicerSelectorProps {
  filePath: string;
}

export default function SlicerSelector({ filePath }: SlicerSelectorProps) {
  const slicers = useSlicerStore((s) => s.slicers);
  const selectedSlicerId = useSlicerStore((s) => s.selectedSlicerId);
  const isDetecting = useSlicerStore((s) => s.isDetecting);
  const isLaunching = useSlicerStore((s) => s.isLaunching);
  const launchMessage = useSlicerStore((s) => s.launchMessage);
  const error = useSlicerStore((s) => s.error);
  const detectSlicers = useSlicerStore((s) => s.detectSlicers);
  const setSelectedSlicerId = useSlicerStore((s) => s.setSelectedSlicerId);
  const launchSlicer = useSlicerStore((s) => s.launchSlicer);
  const clearMessage = useSlicerStore((s) => s.clearMessage);

  useEffect(() => {
    void detectSlicers();
  }, [detectSlicers]);

  // Auto-clear messages after 5 seconds
  useEffect(() => {
    if (!launchMessage && !error) return;
    const timer = setTimeout(clearMessage, 5000);
    return () => clearTimeout(timer);
  }, [launchMessage, error, clearMessage]);

  const hasSlicers = slicers.length > 0;

  const options = slicers.map((s) => ({
    label: s.display_name,
    value: s.id,
  }));

  const handleLaunch = () => {
    void launchSlicer(filePath);
  };

  const handleDownload = () => {
    const link = document.createElement("a");
    link.href = filePath;
    link.download = "";
    link.click();
  };

  return (
    <div className="flex flex-col gap-2">
      {hasSlicers ? (
        <div className="flex items-end gap-2">
          <div className="flex-1">
            <Dropdown
              label="切片软件"
              value={selectedSlicerId ?? ""}
              options={options}
              onChange={setSelectedSlicerId}
              disabled={isDetecting || isLaunching}
              placeholder={isDetecting ? "检测中..." : "选择切片软件..."}
            />
          </div>
          <Button
            label="打开切片"
            variant="primary"
            onClick={handleLaunch}
            disabled={!selectedSlicerId || isLaunching}
            loading={isLaunching}
          />
        </div>
      ) : (
        <div className="flex items-end gap-2">
          <p className="text-xs text-gray-400">
            {isDetecting ? "正在检测切片软件..." : "未检测到切片软件"}
          </p>
          {!isDetecting && (
            <Button
              label="下载 3MF"
              variant="secondary"
              onClick={handleDownload}
            />
          )}
        </div>
      )}

      {launchMessage && (
        <p className="text-xs text-green-400">{launchMessage}</p>
      )}
      {error && (
        <p className="text-xs text-red-400">{error}</p>
      )}
    </div>
  );
}
