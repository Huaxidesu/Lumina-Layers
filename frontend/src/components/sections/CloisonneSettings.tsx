import { useConverterStore } from "../../stores/converterStore";
import { ModelingMode } from "../../api/types";
import Accordion from "../ui/Accordion";
import Checkbox from "../ui/Checkbox";
import Slider from "../ui/Slider";

export default function CloisonneSettings() {
  const {
    enable_cloisonne,
    wire_width_mm,
    wire_height_mm,
    modeling_mode,
    setEnableCloisonne,
    setWireWidthMm,
    setWireHeightMm,
  } = useConverterStore();

  const isVector = modeling_mode === ModelingMode.VECTOR;

  return (
    <Accordion title="掐丝珐琅">
      <div className="flex flex-col gap-4">
        <Checkbox
          label="启用掐丝珐琅"
          checked={enable_cloisonne}
          onChange={setEnableCloisonne}
          disabled={isVector}
        />

        {enable_cloisonne && (
          <>
            <Slider
              label="金属丝宽度"
              value={wire_width_mm}
              min={0.2}
              max={1.2}
              step={0.1}
              unit="mm"
              onChange={setWireWidthMm}
            />

            <Slider
              label="金属丝高度"
              value={wire_height_mm}
              min={0.04}
              max={1.0}
              step={0.04}
              unit="mm"
              onChange={setWireHeightMm}
            />
          </>
        )}
      </div>
    </Accordion>
  );
}
