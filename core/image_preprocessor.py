"""
Lumina Studio - Image Preprocessor

Handles image cropping and format conversion before main processing.
Independent module that doesn't modify existing image_processing.py.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class CropRegion:
    """Crop region data model"""
    x: int = 0
    y: int = 0
    width: int = 100
    height: int = 100

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, w, h) tuple"""
        return (self.x, self.y, self.width, self.height)

    def clamp(self, img_width: int, img_height: int) -> 'CropRegion':
        """Clamp crop region to image boundaries"""
        x = max(0, min(self.x, img_width - 1))
        y = max(0, min(self.y, img_height - 1))
        w = max(1, min(self.width, img_width - x))
        h = max(1, min(self.height, img_height - y))
        return CropRegion(x, y, w, h)


@dataclass
class ImageInfo:
    """Image information data model"""
    original_path: str
    processed_path: str
    width: int
    height: int
    original_format: str
    was_converted: bool


class ImagePreprocessor:
    """
    Image preprocessor - handles cropping and format conversion.
    
    This is a standalone module that processes images before they
    enter the main conversion pipeline.
    """

    # Supported formats
    SUPPORTED_FORMATS = {'JPEG', 'JPG', 'PNG', 'GIF', 'BMP', 'WEBP'}
    
    @staticmethod
    def detect_format(image_path: str) -> str:
        """
        Detect image format.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Format string (e.g., 'JPEG', 'PNG')
            
        Raises:
            ValueError: If file cannot be read or format unsupported
        """
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        try:
            with Image.open(image_path) as img:
                fmt = img.format
                if fmt is None:
                    # Try to detect from extension
                    ext = os.path.splitext(image_path)[1].upper().lstrip('.')
                    if ext in ('JPG', 'JPEG'):
                        return 'JPEG'
                    elif ext == 'PNG':
                        return 'PNG'
                    raise ValueError(f"Cannot detect image format: {image_path}")
                return fmt.upper()
        except Exception as e:
            raise ValueError(f"Cannot read image file: {e}")

    @staticmethod
    def get_image_dimensions(image_path: str) -> Tuple[int, int]:
        """
        Get image dimensions.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (width, height)
            
        Raises:
            ValueError: If file cannot be read
        """
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        try:
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            raise ValueError(f"Cannot read image dimensions: {e}")

    @staticmethod
    def convert_to_png(image_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert image to PNG format.
        
        Args:
            image_path: Path to source image
            output_path: Optional output path. If None, creates temp file.
            
        Returns:
            Path to PNG file
            
        Raises:
            ValueError: If conversion fails
        """
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        try:
            with Image.open(image_path) as img:
                # Check if already PNG
                if img.format == 'PNG':
                    return image_path
                
                # Convert to RGBA to preserve transparency
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')
                
                # Generate output path if not provided
                if output_path is None:
                    fd, output_path = tempfile.mkstemp(suffix='.png')
                    os.close(fd)
                
                # Save as PNG
                img.save(output_path, 'PNG')
                return output_path
                
        except Exception as e:
            raise ValueError(f"Cannot convert image to PNG: {e}")

    @staticmethod
    def crop_image(image_path: str, x: int, y: int, 
                   width: int, height: int,
                   output_path: Optional[str] = None) -> str:
        """
        Crop image to specified region.
        
        Args:
            image_path: Path to source image
            x: X offset (left)
            y: Y offset (top)
            width: Crop width
            height: Crop height
            output_path: Optional output path. If None, creates temp file.
            
        Returns:
            Path to cropped image (PNG format)
            
        Raises:
            ValueError: If crop fails
        """
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
                
                # Validate and clamp crop region
                region = CropRegion(x, y, width, height)
                region = region.clamp(img_w, img_h)
                
                # Calculate crop box (left, upper, right, lower)
                box = (
                    region.x,
                    region.y,
                    region.x + region.width,
                    region.y + region.height
                )
                
                # Crop image
                cropped = img.crop(box)
                
                # Convert to RGBA if needed
                if cropped.mode in ('RGBA', 'LA') or (cropped.mode == 'P' and 'transparency' in img.info):
                    cropped = cropped.convert('RGBA')
                else:
                    cropped = cropped.convert('RGB')
                
                # Generate output path if not provided
                if output_path is None:
                    fd, output_path = tempfile.mkstemp(suffix='.png')
                    os.close(fd)
                
                # Save as PNG
                cropped.save(output_path, 'PNG')
                return output_path
                
        except Exception as e:
            raise ValueError(f"Cannot crop image: {e}")

    @staticmethod
    def validate_crop_region(img_width: int, img_height: int,
                            x: int, y: int,
                            crop_w: int, crop_h: int) -> Tuple[int, int, int, int]:
        """
        Validate and correct crop region to fit within image boundaries.
        
        Args:
            img_width: Image width
            img_height: Image height
            x: Requested X offset
            y: Requested Y offset
            crop_w: Requested crop width
            crop_h: Requested crop height
            
        Returns:
            Tuple of valid (x, y, width, height)
        """
        region = CropRegion(x, y, crop_w, crop_h)
        clamped = region.clamp(img_width, img_height)
        return clamped.to_tuple()

    @classmethod
    def process_upload(cls, image_path: str) -> ImageInfo:
        """
        Process uploaded image: detect format, convert if needed.
        
        Args:
            image_path: Path to uploaded image
            
        Returns:
            ImageInfo with processing results
            
        Raises:
            ValueError: If processing fails
        """
        # Detect format
        fmt = cls.detect_format(image_path)
        
        # Get dimensions
        width, height = cls.get_image_dimensions(image_path)
        
        # Convert to PNG if JPEG
        was_converted = False
        if fmt in ('JPEG', 'JPG'):
            processed_path = cls.convert_to_png(image_path)
            was_converted = True
        else:
            processed_path = image_path
        
        return ImageInfo(
            original_path=image_path,
            processed_path=processed_path,
            width=width,
            height=height,
            original_format=fmt,
            was_converted=was_converted
        )


    @staticmethod
    def analyze_recommended_colors(image_path: str, target_width_mm: float = 60.0) -> dict:
        """
        分析图片，推荐最佳量化颜色数。
        
        算法原理：
        1. 根据目标打印宽度缩放图片（模拟实际打印效果）
        2. 使用多种指标综合判断图片复杂度：
           - 色彩分布的集中度（主色占比）
           - 色系数量（HSV色相分布）
           - 边缘复杂度
        3. 基于综合复杂度推荐合适的量化颜色数
        
        Args:
            image_path: 图片路径
            target_width_mm: 目标打印宽度（毫米），默认 60mm
            
        Returns:
            dict: {
                'recommended': 推荐颜色数,
                'max_safe': 最大安全颜色数（超过会有噪点）,
                'unique_colors': 独特颜色数,
                'complexity_score': 复杂度评分 (0-100)
            }
        """
        import time
        from collections import Counter
        total_start = time.time()
        
        if not image_path or not os.path.exists(image_path):
            return {'recommended': 64, 'max_safe': 128, 'unique_colors': 0, 'complexity_score': 50}
        
        try:
            # 加载图片
            t0 = time.time()
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return {'recommended': 64, 'max_safe': 128, 'unique_colors': 0, 'complexity_score': 50}
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_h, original_w = img_rgb.shape[:2]
            print(f"[ColorAnalysis] 加载图片: {time.time() - t0:.2f}s, 原始尺寸: {original_w}x{original_h}")
            
            # 根据目标打印宽度计算缩放后的尺寸
            ANALYSIS_PX_PER_MM = 5
            target_width_px = int(target_width_mm * ANALYSIS_PX_PER_MM)
            target_width_px = min(target_width_px, 600)  # 限制最大分析尺寸
            
            scale = target_width_px / original_w
            target_height_px = int(original_h * scale)
            
            # 缩放图片到分析尺寸
            t0 = time.time()
            if scale != 1.0:
                img_rgb = cv2.resize(img_rgb, (target_width_px, target_height_px), interpolation=cv2.INTER_AREA)
                print(f"[ColorAnalysis] 缩放到分析尺寸: {time.time() - t0:.2f}s, 新尺寸: {target_width_px}x{target_height_px}")
            
            h, w = img_rgb.shape[:2]
            pixel_count = w * h
            print(f"[ColorAnalysis] 分析尺寸: {w}x{h}, 像素数: {pixel_count:,}")
            
            # ========== 指标1: 独特颜色数（粗量化） ==========
            t0 = time.time()
            quantized_coarse = (img_rgb // 8) * 8
            pixels_coarse = quantized_coarse.reshape(-1, 3)
            unique_colors_coarse = len(np.unique(pixels_coarse, axis=0))
            print(f"[ColorAnalysis] 独特颜色数（粗量化32级）: {unique_colors_coarse}, 耗时: {time.time() - t0:.2f}s")
            
            # ========== 指标2: 色系数量（HSV色相分布） ==========
            t0 = time.time()
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            
            # 过滤掉低饱和度（灰色/白色/黑色）的像素
            saturation = img_hsv[:, :, 1].flatten()
            value = img_hsv[:, :, 2].flatten()
            hue = img_hsv[:, :, 0].flatten()
            
            # 只考虑有颜色的像素（饱和度>30，亮度在20-235之间）
            color_mask = (saturation > 30) & (value > 20) & (value < 235)
            colored_hues = hue[color_mask]
            
            if len(colored_hues) > 100:  # 至少需要100个有色像素才有意义
                # 将色相量化到 12 个区间（每30度一个色系）
                hue_bins = colored_hues // 15  # 0-11 共12个色系
                hue_counts = np.bincount(hue_bins.astype(int), minlength=12)
                # 只计算占比超过5%的色系
                significant_hues = np.sum(hue_counts > len(colored_hues) * 0.05)
                unique_hue_bins = significant_hues
                colored_ratio = len(colored_hues) / pixel_count
            else:
                unique_hue_bins = 1
                colored_ratio = 0
            
            print(f"[ColorAnalysis] 色系数量: {unique_hue_bins}/12, 有色像素占比: {colored_ratio:.2%}, 耗时: {time.time() - t0:.2f}s")
            
            # ========== 指标3: 主色占比（色彩集中度） ==========
            t0 = time.time()
            quantized_very_coarse = (img_rgb // 4) * 4
            pixels_vc = [tuple(p) for p in quantized_very_coarse.reshape(-1, 3)]
            color_counts = Counter(pixels_vc)
            total_pixels = len(pixels_vc)
            
            top_colors = color_counts.most_common(16)
            top16_ratio = sum(c[1] for c in top_colors) / total_pixels
            top8_ratio = sum(c[1] for c in top_colors[:8]) / total_pixels
            top4_ratio = sum(c[1] for c in top_colors[:4]) / total_pixels
            
            print(f"[ColorAnalysis] 主色占比: top4={top4_ratio:.2%}, top8={top8_ratio:.2%}, top16={top16_ratio:.2%}, 耗时: {time.time() - t0:.2f}s")
            
            # ========== 指标4: 边缘复杂度 ==========
            t0 = time.time()
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / pixel_count
            print(f"[ColorAnalysis] 边缘占比: {edge_ratio:.2%}, 耗时: {time.time() - t0:.2f}s")
            
            # ========== 综合评分 ==========
            # 复杂度评分 (0-100)
            # - 色系数量贡献 (0-35分) - 最重要的指标
            # - 主色集中度贡献 (0-35分) - 集中度高=简单
            # - 独特颜色数贡献 (0-20分)
            # - 边缘复杂度贡献 (0-10分)
            
            # 色系数量评分：色系越多越复杂
            # 只计算显著色系（占比>5%的色系）
            if unique_hue_bins <= 2:
                hue_score = 0
            elif unique_hue_bins <= 3:
                hue_score = 7
            elif unique_hue_bins <= 4:
                hue_score = 14
            elif unique_hue_bins <= 6:
                hue_score = 21
            elif unique_hue_bins <= 8:
                hue_score = 28
            else:
                hue_score = 35
            
            # 如果有色像素占比很低（<40%），说明图片主要是单色/灰度，降低色系评分
            if colored_ratio < 0.30:
                hue_score = 0  # 有色像素<30%，色系评分为0
            elif colored_ratio < 0.50:
                hue_score = min(hue_score, 7)  # 最多7分
            
            # 主色集中度评分：集中度越高越简单
            if top8_ratio > 0.90:
                concentration_score = 0
            elif top8_ratio > 0.80:
                concentration_score = 7
            elif top8_ratio > 0.65:
                concentration_score = 14
            elif top8_ratio > 0.50:
                concentration_score = 21
            elif top8_ratio > 0.35:
                concentration_score = 28
            else:
                concentration_score = 35
            
            # 独特颜色评分
            if unique_colors_coarse < 100:
                color_score = 0
            elif unique_colors_coarse < 300:
                color_score = 5
            elif unique_colors_coarse < 600:
                color_score = 10
            elif unique_colors_coarse < 1000:
                color_score = 15
            else:
                color_score = 20
            
            # 边缘复杂度评分
            if edge_ratio < 0.03:
                edge_score = 0
            elif edge_ratio < 0.06:
                edge_score = 3
            elif edge_ratio < 0.10:
                edge_score = 6
            else:
                edge_score = 10
            
            complexity_score = hue_score + concentration_score + color_score + edge_score
            print(f"[ColorAnalysis] 复杂度评分: {complexity_score} (色系={hue_score}, 集中度={concentration_score}, 颜色={color_score}, 边缘={edge_score})")
            
            # ========== 基于复杂度评分推荐颜色数 ==========
            # 评分范围 0-100
            # 0-20: 非常简单 -> 16
            # 20-40: 简单 -> 24 (马图片39分应该在这里)
            # 40-55: 较简单 -> 48
            # 55-70: 中等 -> 96 (洛琪希61分，需要更高)
            # 70-85: 较复杂 -> 128
            # 85+: 非常复杂 -> 192-256
            
            if complexity_score < 20:
                base_recommended = 16
                base_max_safe = 24
            elif complexity_score < 40:
                base_recommended = 24
                base_max_safe = 32
            elif complexity_score < 55:
                base_recommended = 48
                base_max_safe = 64
            elif complexity_score < 70:
                base_recommended = 96
                base_max_safe = 128
            elif complexity_score < 85:
                base_recommended = 128
                base_max_safe = 192
            else:
                base_recommended = 192
                base_max_safe = 256
            
            # ========== 根据打印宽度调整 ==========
            # 基准宽度 60mm，宽度越大需要更多颜色
            # 宽度因子: sqrt(target_width / 60) - 使用平方根避免过度增长
            # 60mm -> 1.0x, 120mm -> 1.41x, 240mm -> 2.0x
            width_factor = (target_width_mm / 60.0) ** 0.5
            width_factor = max(0.8, min(width_factor, 2.5))  # 限制在 0.8x - 2.5x
            
            recommended = int(base_recommended * width_factor)
            max_safe = int(base_max_safe * width_factor)
            
            # 对齐到常用值: 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256
            common_values = [8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256]
            recommended = min(common_values, key=lambda x: abs(x - recommended))
            max_safe = min(common_values, key=lambda x: abs(x - max_safe))
            
            # 确保 max_safe >= recommended
            if max_safe < recommended:
                max_safe = recommended
            
            print(f"[ColorAnalysis] 宽度因子: {width_factor:.2f} (基于 {target_width_mm}mm)")
            
            total_time = time.time() - total_start
            print(f"[ColorAnalysis] ✅ 完成! 总耗时: {total_time:.2f}s")
            print(f"[ColorAnalysis] 结果: 复杂度={complexity_score}, 推荐={recommended}, 最大安全={max_safe}")
            
            return {
                'recommended': recommended,
                'max_safe': max_safe,
                'unique_colors': unique_colors_coarse,
                'complexity_score': complexity_score
            }
            
        except Exception as e:
            print(f"Color analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {'recommended': 64, 'max_safe': 128, 'unique_colors': 0, 'complexity_score': 50}
