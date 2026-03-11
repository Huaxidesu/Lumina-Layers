import { Suspense, useRef, useState, useEffect, useCallback } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import ModelViewer from "./ModelViewer";
import InteractiveModelViewer from "./InteractiveModelViewer";
import BedPlatform from "./BedPlatform";
import KeychainRing3D from "./KeychainRing3D";
import { useConverterStore } from "../stores/converterStore";
import { useI18n } from "../i18n/context";

interface Scene3DProps {
  modelUrl?: string;
}

/**
 * Helper component rendered inside <Canvas> to expose the gl context
 * for screenshot functionality via a callback ref.
 */
function ScreenshotHelper({
  onGlReady,
}: {
  onGlReady: (gl: THREE.WebGLRenderer) => void;
}) {
  const { gl } = useThree();
  useEffect(() => {
    onGlReady(gl);
  }, [gl, onGlReady]);
  return null;
}

function Scene3D({ modelUrl }: Scene3DProps) {
  const { t } = useI18n();
  const containerRef = useRef<HTMLDivElement>(null);
  const glRef = useRef<THREE.WebGLRenderer | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const previewGlbUrl = useConverterStore((s) => s.previewGlbUrl);
  const colorRemapMap = useConverterStore((s) => s.colorRemapMap);
  const colorHeightMap = useConverterStore((s) => s.color_height_map);
  const selectedColor = useConverterStore((s) => s.selectedColor);
  const baseHeight = useConverterStore((s) => s.spacer_thick);
  const enableRelief = useConverterStore((s) => s.enable_relief);
  const isLoading = useConverterStore((s) => s.isLoading);
  const setSelectedColor = useConverterStore((s) => s.setSelectedColor);

  // Keychain ring params
  const addLoop = useConverterStore((s) => s.add_loop);
  const loopWidth = useConverterStore((s) => s.loop_width);
  const loopLength = useConverterStore((s) => s.loop_length);
  const loopHole = useConverterStore((s) => s.loop_hole);
  const modelBounds = useConverterStore((s) => s.modelBounds);

  // Listen to fullscreenchange event
  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", handler);
    return () => document.removeEventListener("fullscreenchange", handler);
  }, []);

  const handleGlReady = useCallback((gl: THREE.WebGLRenderer) => {
    glRef.current = gl;
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      containerRef.current.requestFullscreen().catch(() => {
        // Fullscreen API not available, fail silently
      });
    }
  }, []);

  const takeScreenshot = useCallback(() => {
    const gl = glRef.current;
    if (!gl) return;
    // Render one frame with preserveDrawingBuffer behavior
    try {
      const dataUrl = gl.domElement.toDataURL("image/png");
      const link = document.createElement("a");
      link.download = `lumina-3d-screenshot-${Date.now()}.png`;
      link.href = dataUrl;
      link.click();
    } catch (err) {
      console.warn("Screenshot failed:", err);
    }
  }, []);

  const handleColorClick = useCallback(
    (hex: string | null) => {
      setSelectedColor(hex);
    },
    [setSelectedColor],
  );

  // Check if Fullscreen API is available
  const fullscreenSupported = typeof document.fullscreenElement !== "undefined";

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full"
      data-testid="scene3d-container"
    >
      {/* Toolbar buttons */}
      <div className="absolute top-2 right-2 z-10 flex gap-1">
        {fullscreenSupported && (
          <button
            onClick={toggleFullscreen}
            className="px-2 py-1 rounded text-xs font-medium bg-gray-700/80 text-gray-200 hover:bg-gray-600 transition-colors backdrop-blur-sm"
            aria-label={isFullscreen ? t("viewer_exit_fullscreen") : t("viewer_fullscreen")}
            title={isFullscreen ? t("viewer_exit_fullscreen") : t("viewer_fullscreen")}
          >
            {isFullscreen ? "⛶" : "⛶"} {isFullscreen ? t("viewer_exit_fullscreen") : t("viewer_fullscreen")}
          </button>
        )}
        <button
          onClick={takeScreenshot}
          className="px-2 py-1 rounded text-xs font-medium bg-gray-700/80 text-gray-200 hover:bg-gray-600 transition-colors backdrop-blur-sm"
          aria-label={t("viewer_screenshot")}
          title={t("viewer_screenshot")}
        >
          {t("viewer_screenshot")}
        </button>
      </div>

      {/* Loading indicator overlay (Req 1.4) */}
      {isLoading && (
        <div
          className="absolute inset-0 z-20 flex items-center justify-center bg-black/40 backdrop-blur-sm"
          data-testid="loading-overlay"
        >
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-gray-300 border-t-blue-500" />
        </div>
      )}

      <Canvas
        camera={{ position: [0, 200, 400], fov: 45 }}
        gl={{ preserveDrawingBuffer: true }}
        onPointerMissed={() => setSelectedColor(null)}
        onCreated={({ gl }) => {
          gl.setClearColor("#1e1e26");
          const canvas = gl.domElement;
          canvas.addEventListener("webglcontextlost", (e) => {
            e.preventDefault();
          });
          canvas.addEventListener("webglcontextrestored", () => {
            gl.setSize(canvas.clientWidth, canvas.clientHeight);
          });
        }}
      >
        <ScreenshotHelper onGlReady={handleGlReady} />
        {/* 天空/地面半球光模拟自然环境光 */}
        <hemisphereLight args={["#ddeeff", "#303030", 1.0]} />
        {/* 主光源：模拟阳光，从右上前方照射 */}
        <directionalLight position={[300, 500, 300]} intensity={1.2} color="#fff5e6" />
        {/* 补光：从左后方填充阴影区域 */}
        <directionalLight position={[-200, 300, -200]} intensity={0.4} color="#e6f0ff" />
        {/* 底部微弱反射光 */}
        <directionalLight position={[0, -100, 0]} intensity={0.15} color="#ffffff" />
        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.1}
          minDistance={10}
          maxDistance={2000}
        />
        <BedPlatform />
        {modelUrl ? (
          <Suspense fallback={null}>
            <ModelViewer url={modelUrl} />
          </Suspense>
        ) : previewGlbUrl ? (
          <Suspense fallback={null}>
            <InteractiveModelViewer
              url={previewGlbUrl}
              colorRemapMap={colorRemapMap}
              colorHeightMap={colorHeightMap}
              selectedColor={selectedColor}
              baseHeight={baseHeight}
              enableRelief={enableRelief}
              onColorClick={handleColorClick}
            />
          </Suspense>
        ) : null}
        {addLoop && modelBounds && (
          <KeychainRing3D
            enabled={addLoop}
            width={loopWidth}
            length={loopLength}
            hole={loopHole}
            modelBounds={modelBounds}
          />
        )}
      </Canvas>
    </div>
  );
}

export default Scene3D;
