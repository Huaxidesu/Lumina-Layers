import { useConverterStore } from "../../stores/converterStore";
import Accordion from "../ui/Accordion";
import Checkbox from "../ui/Checkbox";
import Slider from "../ui/Slider";

export default function KeychainLoopSettings() {
  const {
    add_loop,
    loop_width,
    loop_length,
    loop_hole,
    setAddLoop,
    setLoopWidth,
    setLoopLength,
    setLoopHole,
  } = useConverterStore();

  return (
    <Accordion title="挂件环">
      <div className="flex flex-col gap-4">
        <Checkbox
          label="添加挂件环"
          checked={add_loop}
          onChange={setAddLoop}
        />

        {add_loop && (
          <>
            <Slider
              label="环宽度"
              value={loop_width}
              min={2}
              max={10}
              step={0.5}
              unit="mm"
              onChange={setLoopWidth}
            />

            <Slider
              label="环长度"
              value={loop_length}
              min={4}
              max={15}
              step={0.5}
              unit="mm"
              onChange={setLoopLength}
            />

            <Slider
              label="环孔直径"
              value={loop_hole}
              min={1}
              max={5}
              step={0.25}
              unit="mm"
              onChange={setLoopHole}
            />
          </>
        )}
      </div>
    </Accordion>
  );
}
