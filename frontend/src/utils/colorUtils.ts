import type { PaletteEntry, ColorReplacementItem } from '../api/types';

/**
 * 计算 RGB 颜色的感知亮度 (ITU-R BT.601)。
 * @param hex - 6 位 hex 颜色字符串（无 # 前缀）
 * @returns 亮度值 0-255
 */
export function hexToLuminance(hex: string): number {
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/**
 * 根据调色板颜色亮度自动分配浮雕高度。
 * @param palette - 调色板条目列表
 * @param mode - 自动高度模式 ('darker-higher' or 'lighter-higher')
 * @param maxHeight - 最大高度 (mm)
 * @param minHeight - 最小高度 (mm)，默认 0.08
 * @returns color_height_map { hex: heightMm }
 */
export function computeAutoHeightMap(
  palette: PaletteEntry[],
  mode: 'darker-higher' | 'lighter-higher',
  maxHeight: number,
  minHeight: number = 0.08,
): Record<string, number> {
  const result: Record<string, number> = {};
  const range = maxHeight - minHeight;

  for (const entry of palette) {
    const lum = hexToLuminance(entry.matched_hex);
    const ratio = lum / 255;

    if (mode === 'darker-higher') {
      // luminance 0 → maxHeight, luminance 255 → minHeight
      result[entry.matched_hex] = maxHeight - ratio * range;
    } else {
      // luminance 0 → minHeight, luminance 255 → maxHeight
      result[entry.matched_hex] = minHeight + ratio * range;
    }
  }

  return result;
}

/**
 * 将前端 colorRemapMap 转换为后端 replacement_regions 格式。
 * @param remapMap - 前端颜色替换映射 { origHex: newHex }，origHex 是 matched_hex
 * @param palette - 调色板条目
 * @returns ColorReplacementItem[] 后端格式（hex 值带 # 前缀）
 */
export function colorRemapToReplacementRegions(
  remapMap: Record<string, string>,
  palette: PaletteEntry[],
): ColorReplacementItem[] {
  const result: ColorReplacementItem[] = [];

  for (const [origHex, newHex] of Object.entries(remapMap)) {
    const entry = palette.find((p) => p.matched_hex === origHex);
    if (!entry) continue;

    // Backend expects #rrggbb format per API contract
    const ensureHash = (h: string) => (h.startsWith('#') ? h : `#${h}`);

    result.push({
      quantized_hex: ensureHash(entry.quantized_hex),
      matched_hex: ensureHash(entry.matched_hex),
      replacement_hex: ensureHash(newHex),
    });
  }

  return result;
}
