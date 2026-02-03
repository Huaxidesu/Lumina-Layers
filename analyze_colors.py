import numpy as np

# ==========================================
# 🚑 紧急修复: 给 colormath 库打补丁
# numpy 1.20+ 移除了 asscalar，这里手动加回去
# ==========================================
def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)

# 补丁打完后再引入 colormath
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import itertools

# ================= 配置区域 (已更新拓竹数据) =================

# 打印参数
LAYER_HEIGHT = 0.08  # 层高
LAYERS = 5           # 混色层数
BACKING_COLOR = np.array([255, 255, 255]) # 底板颜色 (白色)

# 耗材定义 (基于拓竹 PLA Basic 官方色卡 + 实测 TD 值)
# 格式: [R, G, B, TD值]
FILAMENTS = {
    0: {"name": "White (Jade)", "rgb": [255, 255, 255], "td": 5.0},   # 对应色卡"白色" #FFFFFF, TD=5.0
    1: {"name": "Cyan",         "rgb": [0, 134, 214],   "td": 3.5},   # 对应色卡"青色" #0086D6, TD=3.5
    2: {"name": "Magenta",      "rgb": [236, 0, 140],   "td": 3.0},   # 对应色卡"品红色" #EC008C, TD=3.0
    3: {"name": "Green",        "rgb": [0, 174, 66],    "td": 2.0},   # 对应色卡"拓竹绿" #00AE42, TD=2.0
    4: {"name": "Yellow",       "rgb": [244, 238, 42],  "td": 6.0},   # 对应色卡"黄色" #F4EE2A, TD=6.0
    5: {"name": "Black",        "rgb": [0, 0, 0],       "td": 0.6},   # 对应色卡"黑色" #000000, TD=0.6
    6: {"name": "Red",          "rgb": [255, 0, 0],     "td": 4.0},   # 示例：红色，请修改为实际TD
    7: {"name": "Blue",         "rgb": [0, 0, 255],     "td": 4.0},   # 示例：蓝色，请修改为实际TD
}

# 色差阈值 (Delta E)
# < 1.0: 肉眼无法分辨
# 1.0 - 2.0: 仔细对比可分辨
# > 2.0: 明显不同 (我们设为 2.5，过滤掉极其相似的颜色)
THRESHOLD_DELTA_E = 2.5 

# ===========================================

def calculate_alpha(td_value, layer_height):
    """
    根据 TD 值计算单层的覆盖能力 (Alpha)
    公式推导: BD = TD / 10
    如果 BD 是完全遮盖厚度，那么单层贡献的覆盖率 alpha ≈ layer_height / BD
    """
    blending_distance = td_value / 10.0
    if blending_distance <= 0: return 1.0
    
    # 计算 Alpha (0.0 = 全透, 1.0 = 全遮盖)
    alpha = layer_height / blending_distance
    return min(max(alpha, 0.0), 1.0)

def mix_colors(stack):
    """
    模拟从下往上的颜色混合
    stack: list of filament_ids [底层 ... 顶层]
    """
    # 初始颜色是底板
    current_rgb = BACKING_COLOR.astype(float)
    
    # 逐层叠加
    for fid in stack:
        fil = FILAMENTS[fid]
        f_rgb = np.array(fil["rgb"])
        f_alpha = calculate_alpha(fil["td"], LAYER_HEIGHT)
        
        # Alpha Blending 算法: New = Source * Alpha + BG * (1 - Alpha)
        current_rgb = f_rgb * f_alpha + current_rgb * (1.0 - f_alpha)
        
    return current_rgb.astype(np.uint8)

def rgb_to_lab(rgb):
    """辅助函数：RGB转LAB"""
    rgb_obj = sRGBColor(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
    return convert_color(rgb_obj, LabColor)

def main():
# 这里的 8 对应 FILAMENTS 中的颜色数量
    COLOR_COUNT = 8 
    
    print(f"🔄 开始模拟 {COLOR_COUNT}色 {LAYERS}层 全排列 ({COLOR_COUNT**LAYERS} 种组合)...")
    print(f"📏 色差阈值 (Delta E): {THRESHOLD_DELTA_E}")
    
    # 1. 生成并计算所有组合的颜色
    all_combinations = []
    
    # 生成 8^5 全排列 (32768 种组合)
    permutations = itertools.product(range(COLOR_COUNT), repeat=LAYERS)
    
    for stack in permutations:
        # 这里的 stack 是从底层到顶层
        final_rgb = mix_colors(stack)
        all_combinations.append({
            "stack": stack,
            "rgb": final_rgb,
            "lab": rgb_to_lab(final_rgb)
        })
        
    print(f"✅ 计算完成，共 {len(all_combinations)} 个原始数据。")
    print("🧹 开始执行视觉去重筛选 (这可能需要几秒钟)...")
    
    # 2. 贪婪筛选法去重
    unique_colors = []
    
    # 进度条辅助
    total = len(all_combinations)
    
    for i, candidate in enumerate(all_combinations):
        is_distinct = True
        
        # 与已选出的颜色对比
        for existing in unique_colors:
            # 计算色差
            delta_e = delta_e_cie2000(candidate["lab"], existing["lab"])
            
            if delta_e < THRESHOLD_DELTA_E:
                is_distinct = False
                break
        
        if is_distinct:
            unique_colors.append(candidate)
            
        if i % 1000 == 0:
            print(f"   处理进度: {i}/{total} | 当前保留: {len(unique_colors)}")

    total_combinations = COLOR_COUNT ** LAYERS  # 8^5 = 32768

    print("-" * 30)
    print(f"🎉 最终结果: 在 {total_combinations} 种组合中")
    print(f"💎 肉眼可见的独立颜色数量: {len(unique_colors)}")
    print(f"📉 冗余率: {(1 - len(unique_colors)/total_combinations)*100:.1f}%")
    
    # 3. 打印一些统计建议
    if len(unique_colors) <= 1024:
        print("💡 结论: 1024 个色块完全足够覆盖所有颜色变化！")
        print("   建议：直接生成 1024 色校准板，不需要打印 7776 个。")
    else:
        print(f"💡 结论: 颜色变化丰富，建议筛选出最具代表性的 {min(1024, len(unique_colors))} 个。")

if __name__ == "__main__":
    main()