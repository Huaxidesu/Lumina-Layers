import { useSettingsStore } from "../stores/settingsStore";

export function LanguageToggle() {
  const language = useSettingsStore((s) => s.language);
  const setLanguage = useSettingsStore((s) => s.setLanguage);

  const handleToggle = () => {
    setLanguage(language === "zh" ? "en" : "zh");
  };

  return (
    <button
      onClick={handleToggle}
      aria-label="Toggle language"
      className="px-3 py-1 rounded text-sm font-medium bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
    >
      {language === "zh" ? "EN" : "中"}
    </button>
  );
}
