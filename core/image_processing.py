"""
Lumina Studio - Image Processing Core

Handles image loading, preprocessing, color quantization and matching.
"""

import numpy as np
import cv2
from PIL import Image
from scipy.spatial import KDTree

from config import PrinterConfig

# SVG support (optional dependency)
try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    HAS_SVG = True
except ImportError:
    HAS_SVG = False
    print("⚠️ [SVG] svglib/reportlab not installed. SVG support disabled.")


class LuminaImageProcessor:
    """
    Image processor class.
    
    Handles LUT loading, image processing, and color matching.
    """
    
    def __init__(self, lut_path, color_mode):
        """
        Initialize image processor.
        
        Args:
            lut_path: LUT file path (.npy)
            color_mode: Color mode string (CMYW/RYBW/6-Color)
        """
        self.color_mode = color_mode
        self.lut_rgb = None
        self.ref_stacks = None
        self.kdtree = None
        
        self._load_lut(lut_path)
    
    def _load_svg(self, svg_path, target_width_mm):
        """
        [New] Render SVG to ultra-high-resolution numpy array
        [Updated] Increased resolution from 10 px/mm to 20 px/mm for smoother curves
        
        Target resolution: 20 pixels/mm (Ultra-High-Fidelity)
        - Standard High-Fidelity: 10 px/mm (0.1mm precision)
        - Ultra High-Fidelity: 20 px/mm (0.05mm precision)
        
        This super-sampling approach eliminates jagged edges on curves
        without needing blur filters, preserving vector crispness.
        
        Args:
            svg_path: Path to SVG file
            target_width_mm: Target width in millimeters
        
        Returns:
            numpy.ndarray: RGBA image array
        """
        if not HAS_SVG:
            raise ImportError("Please install 'svglib' and 'reportlab' to process SVG files.")
        
        print(f"[SVG] Rasterizing vector file: {svg_path}")
        
        # 1. Read SVG
        drawing = svg2rlg(svg_path)
        
        # 2. Calculate scale for Ultra-High-Fidelity (20 px/mm)
        # [UPGRADED] Doubled resolution for smoother curves
        # Old: 10.0 px/mm (0.1mm) - visible jaggies on curves
        # New: 20.0 px/mm (0.05mm) - ultra-smooth, exceeds most FDM printer limits
        pixels_per_mm = 20.0
        target_width_px = int(target_width_mm * pixels_per_mm)
        scale_factor = target_width_px / drawing.width
        
        print(f"[SVG] Ultra-High-Fidelity mode: {pixels_per_mm} px/mm (0.05mm precision)")
        
        # 3. Scale drawing
        drawing.scale(scale_factor, scale_factor)
        drawing.width = int(drawing.width * scale_factor)
        drawing.height = int(drawing.height * scale_factor)
        
        # 4. Render to PIL Image
        # use bg=0xffffff to handle transparency correctly if needed, or maintain alpha
        pil_img = renderPM.drawToPIL(drawing, bg=None, configPIL={'transparent': True})
        
        print(f"[SVG] Rendered resolution: {drawing.width}x{drawing.height} px")
        
        # 5. Convert to RGBA numpy array
        return np.array(pil_img.convert('RGBA'))
    
    def _load_lut(self, lut_path):
        """
        Load and validate LUT file (Supports 4-Color and 6-Color).
        
        Automatically detects LUT type based on size:
        - 1024 colors: 4-Color Standard (CMYW/RYBW)
        - 1296 colors: 6-Color Smart 1296
        """
        try:
            lut_grid = np.load(lut_path)
            measured_colors = lut_grid.reshape(-1, 3)
            total_colors = measured_colors.shape[0]
        except Exception as e:
            raise ValueError(f"❌ LUT file corrupted: {e}")
        
        valid_rgb = []
        valid_stacks = []
        
        print(f"[IMAGE_PROCESSOR] Loading LUT with {total_colors} points...")
        
        # Branch 1: 6-Color Smart 1296
        if "6-Color" in self.color_mode or total_colors == 1296:
            print("[IMAGE_PROCESSOR] Detected 6-Color Smart 1296 mode")
            
            from core.calibration import get_top_1296_colors
            
            # Retrieve 1296 intelligent stacking order (must match calibration.py logic)
            # Note: generate_smart_board uses padding to fill 38x38,
            # but extractor extracts the border-removed 36x36 (1296 cells).
            # So we directly get the original stacking data here.
            
            smart_stacks = get_top_1296_colors()
            
            # Reverse stacking order to make it (Top -> Bottom)
            # Original smart_stacks is [Bottom, ..., Top] (simulation data order)
            # But Converter's Face Down logic prints Z=0 as Index=0
            # So we need to reverse to [Top, ..., Bottom], making Z=0 the viewing surface
            smart_stacks = [tuple(reversed(s)) for s in smart_stacks]
            print("[IMAGE_PROCESSOR] Stacks reversed for Face-Down printing compatibility.")
            
            if len(smart_stacks) != total_colors:
                print(f"⚠️ Warning: Stacks count ({len(smart_stacks)}) != LUT count ({total_colors})")
                min_len = min(len(smart_stacks), total_colors)
                smart_stacks = smart_stacks[:min_len]
                measured_colors = measured_colors[:min_len]
            
            # No "Base Blue" filtering in 6-color mode (colors too complex)
            self.lut_rgb = measured_colors
            self.ref_stacks = np.array(smart_stacks)
            
            print(f"✅ LUT loaded: {len(self.lut_rgb)} colors (6-Color mode)")
        
        # Branch 2: 4-Color Standard (1024)
        else:
            print("[IMAGE_PROCESSOR] Detected 4-Color Standard mode")
            
            # Keep original outlier filtering logic (Blue Check)
            base_blue = np.array([30, 100, 200])
            dropped = 0
            
            for i in range(1024):
                if i >= total_colors:
                    break
                
                # Rebuild 4-base stacking (0..1023)
                digits = []
                temp = i
                for _ in range(5):
                    digits.append(temp % 4)
                    temp //= 4
                stack = digits[::-1]
                
                real_rgb = measured_colors[i]
                
                # Filter outliers: close to blue but doesn't contain blue
                dist = np.linalg.norm(real_rgb - base_blue)
                if dist < 60 and 3 not in stack:  # 3 is Blue in RYBW/CMYW
                    dropped += 1
                    continue
                
                valid_rgb.append(real_rgb)
                valid_stacks.append(stack)
            
            self.lut_rgb = np.array(valid_rgb)
            self.ref_stacks = np.array(valid_stacks)
            
            print(f"✅ LUT loaded: {len(self.lut_rgb)} colors (filtered {dropped} outliers)")
        
        # Build KD-Tree
        self.kdtree = KDTree(self.lut_rgb)
    
    def process_image(self, image_path, target_width_mm, modeling_mode,
                     quantize_colors, auto_bg, bg_tol,
                     blur_kernel=0, smooth_sigma=10):
        """
        Main image processing method
        
        Args:
            image_path: Image file path
            target_width_mm: Target width (millimeters)
            modeling_mode: Modeling mode ("high-fidelity", "pixel")
            quantize_colors: K-Means quantization color count
            auto_bg: Whether to auto-remove background
            bg_tol: Background tolerance
            blur_kernel: Median filter kernel size (0=disabled, recommended 0-5)
            smooth_sigma: Bilateral filter sigma value (recommended 5-20)
        
        Returns:
            dict: Dictionary containing processing results
                - matched_rgb: (H, W, 3) Matched RGB array
                - material_matrix: (H, W, Layers) Material index matrix
                - mask_solid: (H, W) Solid mask
                - dimensions: (width, height) Pixel dimensions
                - pixel_scale: mm/pixel ratio
                - mode_info: Mode information dictionary
                - debug_data: Debug data (high-fidelity mode only)
        """
        # Normalize modeling mode
        mode_str = str(modeling_mode).lower()
        use_high_fidelity = "high-fidelity" in mode_str or "高保真" in mode_str
        use_pixel = "pixel" in mode_str or "像素" in mode_str
        
        # Determine mode name
        if use_high_fidelity:
            mode_name = "High-Fidelity"
        elif use_pixel:
            mode_name = "Pixel Art"
        else:
            # Default to High-Fidelity if mode is unclear
            mode_name = "High-Fidelity"
            use_high_fidelity = True
        
        print(f"[IMAGE_PROCESSOR] Mode: {mode_name}")
        print(f"[IMAGE_PROCESSOR] Filter settings: blur_kernel={blur_kernel}, smooth_sigma={smooth_sigma}")
        
        # ========== Image Loading Logic Branch ==========
        is_svg = image_path.lower().endswith('.svg')
        
        if is_svg:
            print("[IMAGE_PROCESSOR] SVG detected - Engaging Ultra-High-Fidelity Vector Mode")
            img_arr = self._load_svg(image_path, target_width_mm)
            # SVG reset to PIL object to reuse subsequent logic (e.g., get dimensions)
            img = Image.fromarray(img_arr)
            
            # [CRITICAL] SVG is also a type of High-Fidelity, but it doesn't need denoising
            # Force override filter parameters, because vector graphics have no noise, no need to blur
            # 
            # [SUPER-SAMPLING STRATEGY]
            # We render at 20 px/mm (2x standard), which physically eliminates jaggies
            # through super-sampling. This is superior to blur-based anti-aliasing
            # because it preserves sharp edges while making curves smooth.
            blur_kernel = 0
            smooth_sigma = 0
            print("[IMAGE_PROCESSOR] SVG Mode: Filters disabled (Vector source is clean)")
            print("[IMAGE_PROCESSOR] Super-sampling at 20 px/mm eliminates jagged edges naturally")
            
            # Recalculate target_w/h (based on rendered dimensions)
            target_w, target_h = img.size
            pixel_to_mm_scale = 0.05  # 20 px/mm (1/20) - Ultra-High-Fidelity
        else:
            # [Original Logic] Bitmap loading
            # Load image
            img = Image.open(image_path).convert('RGBA')
            
            # DEBUG: Check original image properties
            print(f"[IMAGE_PROCESSOR] Original image: {image_path}")
            print(f"[IMAGE_PROCESSOR] Image mode: {Image.open(image_path).mode}")
            print(f"[IMAGE_PROCESSOR] Image size: {Image.open(image_path).size}")
            
            # Check if image has transparency
            original_img = Image.open(image_path)
            has_alpha = original_img.mode in ('RGBA', 'LA') or (original_img.mode == 'P' and 'transparency' in original_img.info)
            print(f"[IMAGE_PROCESSOR] Has alpha channel: {has_alpha}")
            
            if has_alpha:
                # Check alpha channel statistics
                if original_img.mode != 'RGBA':
                    original_img = original_img.convert('RGBA')
                alpha_data = np.array(original_img)[:, :, 3]
                print(f"[IMAGE_PROCESSOR] Alpha stats: min={alpha_data.min()}, max={alpha_data.max()}, mean={alpha_data.mean():.1f}")
                print(f"[IMAGE_PROCESSOR] Transparent pixels (alpha<10): {np.sum(alpha_data < 10)}")
            
            # Calculate target resolution
            if use_high_fidelity:
                # High-precision mode: 10 pixels/mm
                PIXELS_PER_MM = 10
                target_w = int(target_width_mm * PIXELS_PER_MM)
                pixel_to_mm_scale = 1.0 / PIXELS_PER_MM  # 0.1 mm per pixel
                print(f"[IMAGE_PROCESSOR] High-res mode: {PIXELS_PER_MM} px/mm")
            else:
                # Pixel mode: Based on nozzle width
                target_w = int(target_width_mm / PrinterConfig.NOZZLE_WIDTH)
                pixel_to_mm_scale = PrinterConfig.NOZZLE_WIDTH
                print(f"[IMAGE_PROCESSOR] Pixel mode: {1.0/pixel_to_mm_scale:.2f} px/mm")
            
            target_h = int(target_w * img.height / img.width)
            print(f"[IMAGE_PROCESSOR] Target: {target_w}×{target_h}px ({target_w*pixel_to_mm_scale:.1f}×{target_h*pixel_to_mm_scale:.1f}mm)")
        
        # ========== End of Image Loading Logic Branch ==========
        
        # ========== CRITICAL FIX: Use NEAREST for both modes ==========
        # REASON: LANCZOS anti-aliasing creates light transition pixels at edges.
        # These light pixels map to stacks with WHITE bases (Layer 1),
        # causing the mesh to "float" above the build plate.
        # 
        # SOLUTION: Use NEAREST to preserve hard edges and ensure dark pixels
        # map to solid dark stacks from Layer 1 upwards.
        print(f"[IMAGE_PROCESSOR] Using NEAREST interpolation (no anti-aliasing)")
        img = img.resize((target_w, target_h), Image.Resampling.NEAREST)
        
        img_arr = np.array(img)
        rgb_arr = img_arr[:, :, :3]
        alpha_arr = img_arr[:, :, 3]
        
        # CRITICAL FIX: Identify transparent pixels BEFORE color processing
        # This prevents transparent areas from being matched to LUT colors
        mask_transparent_initial = alpha_arr < 10
        print(f"[IMAGE_PROCESSOR] Found {np.sum(mask_transparent_initial)} transparent pixels (alpha<10)")
        
        # Color processing and matching
        debug_data = None
        if use_high_fidelity:
            matched_rgb, material_matrix, bg_reference, debug_data = self._process_high_fidelity_mode(
                rgb_arr, target_h, target_w, quantize_colors, blur_kernel, smooth_sigma
            )
        else:
            matched_rgb, material_matrix, bg_reference = self._process_pixel_mode(
                rgb_arr, target_h, target_w
            )
        
        # Background removal - combine alpha transparency with optional auto-bg
        mask_transparent = mask_transparent_initial.copy()
        if auto_bg:
            bg_color = bg_reference[0, 0]
            diff = np.sum(np.abs(bg_reference - bg_color), axis=-1)
            mask_transparent = np.logical_or(mask_transparent, diff < bg_tol)
        
        # Apply transparency mask to material matrix
        material_matrix[mask_transparent] = -1
        mask_solid = ~mask_transparent
        
        result = {
            'matched_rgb': matched_rgb,
            'material_matrix': material_matrix,
            'mask_solid': mask_solid,
            'dimensions': (target_w, target_h),
            'pixel_scale': pixel_to_mm_scale,
            'mode_info': {
                'name': mode_name,
                'use_high_fidelity': use_high_fidelity,
                'use_pixel': use_pixel
            }
        }
        
        # Add debug data (high-fidelity mode only)
        if debug_data is not None:
            result['debug_data'] = debug_data
        
        return result

    
    def _process_high_fidelity_mode(self, rgb_arr, target_h, target_w, quantize_colors,
                                    blur_kernel, smooth_sigma):
        """
        High-fidelity mode image processing
        Includes configurable filtering, K-Means quantization and color matching
        
        Args:
            rgb_arr: Input RGB array
            target_h: Target height
            target_w: Target width
            quantize_colors: K-Means color count
            blur_kernel: Median filter kernel size (0=disabled)
            smooth_sigma: Bilateral filter sigma value
        
        Returns:
            tuple: (matched_rgb, material_matrix, quantized_image, debug_data)
        """
        print(f"[IMAGE_PROCESSOR] Starting edge-preserving processing...")
        
        # Step 1: Bilateral filter (edge-preserving smoothing)
        if smooth_sigma > 0:
            print(f"[IMAGE_PROCESSOR] Applying bilateral filter (sigma={smooth_sigma})...")
            rgb_processed = cv2.bilateralFilter(
                rgb_arr.astype(np.uint8), 
                d=9,  # Larger neighborhood for better smoothing
                sigmaColor=smooth_sigma, 
                sigmaSpace=smooth_sigma
            )
        else:
            print(f"[IMAGE_PROCESSOR] Bilateral filter disabled (sigma=0)")
            rgb_processed = rgb_arr.astype(np.uint8)
        
        # Step 2: Optional median filter (remove salt-and-pepper noise)
        if blur_kernel > 0:
            # Ensure kernel size is odd
            kernel_size = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            print(f"[IMAGE_PROCESSOR] Applying median blur (kernel={kernel_size})...")
            rgb_processed = cv2.medianBlur(rgb_processed, kernel_size)
        else:
            print(f"[IMAGE_PROCESSOR] Median blur disabled (kernel=0)")
        
        # Step 3: Skip sharpening to prevent noise amplification
        # Sharpening creates high-contrast noise in flat color areas
        print(f"[IMAGE_PROCESSOR] Skipping sharpening to reduce noise...")
        rgb_sharpened = rgb_processed
        
        # Step 4: K-Means quantization
        print(f"[IMAGE_PROCESSOR] K-Means quantization to {quantize_colors} colors...")
        h, w = rgb_sharpened.shape[:2]
        pixels = rgb_sharpened.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        flags = cv2.KMEANS_PP_CENTERS
        
        _, labels, centers = cv2.kmeans(
            pixels, quantize_colors, None, criteria, 10, flags
        )
        
        centers = centers.astype(np.uint8)
        quantized_pixels = centers[labels.flatten()]
        quantized_image = quantized_pixels.reshape(h, w, 3)
        
        # [CRITICAL FIX] Post-Quantization Cleanup
        # Removes isolated "salt-and-pepper" noise pixels that survive quantization
        print(f"[IMAGE_PROCESSOR] Applying post-quantization cleanup (Denoising)...")
        quantized_image = cv2.medianBlur(quantized_image, 3)  # Kernel size 3 is optimal for detail preservation
        
        print(f"[IMAGE_PROCESSOR] Quantization complete!")
        
        # Find unique colors
        unique_colors = np.unique(quantized_image.reshape(-1, 3), axis=0)
        print(f"[IMAGE_PROCESSOR] Found {len(unique_colors)} unique colors")
        
        # Match to LUT
        print(f"[IMAGE_PROCESSOR] Matching colors to LUT...")
        _, unique_indices = self.kdtree.query(unique_colors.astype(float))
        
        # Build color mapping
        color_to_stack = {}
        color_to_rgb = {}
        for i, color in enumerate(unique_colors):
            color_key = tuple(color)
            color_to_stack[color_key] = self.ref_stacks[unique_indices[i]]
            color_to_rgb[color_key] = self.lut_rgb[unique_indices[i]]
        
        # Map back to full image
        print(f"[IMAGE_PROCESSOR] Mapping to full image...")
        matched_rgb = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        material_matrix = np.zeros((target_h, target_w, PrinterConfig.COLOR_LAYERS), dtype=int)
        
        for y in range(target_h):
            for x in range(target_w):
                color_key = tuple(quantized_image[y, x])
                matched_rgb[y, x] = color_to_rgb[color_key]
                material_matrix[y, x] = color_to_stack[color_key]
        
        print(f"[IMAGE_PROCESSOR] Color matching complete!")
        
        # Prepare debug data
        debug_data = {
            'quantized_image': quantized_image.copy(),  # K-Means quantized image
            'num_colors': len(unique_colors),
            'bilateral_filtered': rgb_processed.copy(),  # Filtered image
            'sharpened': rgb_sharpened.copy(),  # Sharpened image
            'filter_settings': {
                'blur_kernel': blur_kernel,
                'smooth_sigma': smooth_sigma
            }
        }
        
        return matched_rgb, material_matrix, quantized_image, debug_data
    
    def _process_pixel_mode(self, rgb_arr, target_h, target_w):
        """
        Pixel art mode image processing
        Direct pixel-level color matching, no smoothing
        """
        print(f"[IMAGE_PROCESSOR] Direct pixel-level matching (Pixel Art mode)...")
        
        flat_rgb = rgb_arr.reshape(-1, 3)
        _, indices = self.kdtree.query(flat_rgb)
        
        matched_rgb = self.lut_rgb[indices].reshape(target_h, target_w, 3)
        material_matrix = self.ref_stacks[indices].reshape(
            target_h, target_w, PrinterConfig.COLOR_LAYERS
        )
        
        print(f"[IMAGE_PROCESSOR] Direct matching complete!")
        
        return matched_rgb, material_matrix, rgb_arr
