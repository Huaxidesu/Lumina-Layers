import { useConverterStore } from "../../stores/converterStore";
import Dropdown from "../ui/Dropdown";

export default function BedSizeSelector() {
  const { bed_label, bedSizes, bedSizesLoading, setBedLabel } =
    useConverterStore();

  const options = bedSizes.map((bed) => ({
    label: bed.label,
    value: bed.label,
  }));

  return (
    <Dropdown
      label="热床尺寸"
      value={bed_label}
      options={options}
      onChange={setBedLabel}
      disabled={bedSizesLoading}
      placeholder={bedSizesLoading ? "加载中..." : "选择热床尺寸..."}
    />
  );
}
