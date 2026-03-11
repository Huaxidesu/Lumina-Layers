import { useConverterStore } from "../../stores/converterStore";
import Accordion from "../ui/Accordion";
import Checkbox from "../ui/Checkbox";
import Slider from "../ui/Slider";

export default function CoatingSettings() {
  const {
    enable_coating,
    coating_height_mm,
    setEnableCoating,
    setCoatingHeightMm,
  } = useConverterStore();

  return (
    <Accordion title="涂层设置">
      <div className="flex flex-col gap-4">
        <Checkbox
          label="启用涂层"
          checked={enable_coating}
          onChange={setEnableCoating}
        />

        {enable_coating && (
          <Slider
            label="涂层高度"
            value={coating_height_mm}
            min={0.04}
            max={0.12}
            step={0.04}
            unit="mm"
            onChange={setCoatingHeightMm}
          />
        )}
      </div>
    </Accordion>
  );
}
