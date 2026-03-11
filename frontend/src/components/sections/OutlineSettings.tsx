import { useConverterStore } from "../../stores/converterStore";
import { ModelingMode } from "../../api/types";
import Accordion from "../ui/Accordion";
import Checkbox from "../ui/Checkbox";
import Slider from "../ui/Slider";

export default function OutlineSettings() {
  const {
    enable_outline,
    outline_width,
    modeling_mode,
    setEnableOutline,
    setOutlineWidth,
  } = useConverterStore();

  const isVector = modeling_mode === ModelingMode.VECTOR;

  return (
    <Accordion title="描边设置">
      <div className="flex flex-col gap-4">
        <Checkbox
          label="启用描边"
          checked={enable_outline}
          onChange={setEnableOutline}
          disabled={isVector}
        />

        {enable_outline && (
          <Slider
            label="描边宽度"
            value={outline_width}
            min={0.5}
            max={10.0}
            step={0.5}
            unit="mm"
            onChange={setOutlineWidth}
          />
        )}
      </div>
    </Accordion>
  );
}
