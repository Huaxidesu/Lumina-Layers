import { useConverterStore } from "../../stores/converterStore";
import Accordion from "../ui/Accordion";
import Slider from "../ui/Slider";
import Checkbox from "../ui/Checkbox";

export default function AdvancedSettings() {
  const {
    quantize_colors,
    bg_tol,
    auto_bg,
    enable_cleanup,
    separate_backing,
    setQuantizeColors,
    setBgTol,
    setAutoBg,
    setEnableCleanup,
    setSeparateBacking,
  } = useConverterStore();

  return (
    <Accordion title="高级设置">
      <div className="flex flex-col gap-4">
        <Slider
          label="量化颜色数"
          value={quantize_colors}
          min={8}
          max={256}
          step={8}
          onChange={setQuantizeColors}
        />

        <Slider
          label="背景容差"
          value={bg_tol}
          min={0}
          max={150}
          step={1}
          onChange={setBgTol}
        />

        <Checkbox
          label="自动背景"
          checked={auto_bg}
          onChange={setAutoBg}
        />

        <Checkbox
          label="启用清理"
          checked={enable_cleanup}
          onChange={setEnableCleanup}
        />

        <Checkbox
          label="分离底板"
          checked={separate_backing}
          onChange={setSeparateBacking}
        />
      </div>
    </Accordion>
  );
}
