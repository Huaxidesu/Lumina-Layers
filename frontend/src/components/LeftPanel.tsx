import { useEffect } from "react";
import { useConverterStore } from "../stores/converterStore";
import BasicSettings from "./sections/BasicSettings";
import AdvancedSettings from "./sections/AdvancedSettings";
import ReliefSettings from "./sections/ReliefSettings";
import OutlineSettings from "./sections/OutlineSettings";
import CloisonneSettings from "./sections/CloisonneSettings";
import CoatingSettings from "./sections/CoatingSettings";
import KeychainLoopSettings from "./sections/KeychainLoopSettings";
import PalettePanel from "./sections/PalettePanel";
import LutColorGrid from "./sections/LutColorGrid";
import ActionBar from "./sections/ActionBar";

export default function LeftPanel() {
  const fetchLutList = useConverterStore((s) => s.fetchLutList);
  const fetchBedSizes = useConverterStore((s) => s.fetchBedSizes);

  useEffect(() => {
    void fetchLutList();
    void fetchBedSizes();
  }, [fetchLutList, fetchBedSizes]);

  return (
    <aside data-testid="left-panel" className="w-[350px] h-full overflow-y-auto bg-gray-800 p-4 flex flex-col gap-4">
      <BasicSettings />
      <AdvancedSettings />
      <ReliefSettings />
      <PalettePanel />
      <LutColorGrid />
      <OutlineSettings />
      <CloisonneSettings />
      <CoatingSettings />
      <KeychainLoopSettings />
      <ActionBar />
    </aside>
  );
}
