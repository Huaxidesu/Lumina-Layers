import { useMemo, useEffect } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useConverterStore } from "../stores/converterStore";
import { computeFitDistance } from "./ModelViewer";

/**
 * Create a textured print bed mesh matching the backend's PEI dark style.
 * Uses a canvas-generated texture with grid lines.
 */
function createBedTexture(widthMm: number, heightMm: number): THREE.CanvasTexture {
  const scale = 2; // pixels per mm for texture
  const texW = widthMm * scale;
  const texH = heightMm * scale;

  const canvas = document.createElement("canvas");
  canvas.width = texW;
  canvas.height = texH;
  const ctx = canvas.getContext("2d")!;

  // Dark PEI base
  ctx.fillStyle = "#26262c";
  ctx.fillRect(0, 0, texW, texH);

  // Inner area
  const margin = 4;
  const radius = 16;
  ctx.fillStyle = "#3a3a42";
  ctx.beginPath();
  ctx.roundRect(margin, margin, texW - margin * 2, texH - margin * 2, radius);
  ctx.fill();

  // Fine grid (10mm)
  ctx.strokeStyle = "#2a2a30";
  ctx.lineWidth = 1;
  const step10 = 10 * scale;
  for (let x = 0; x < texW; x += step10) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, texH);
    ctx.stroke();
  }
  for (let y = 0; y < texH; y += step10) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(texW, y);
    ctx.stroke();
  }

  // Bold grid (50mm)
  ctx.strokeStyle = "#5a5a64";
  ctx.lineWidth = 2;
  const step50 = 50 * scale;
  for (let x = 0; x < texW; x += step50) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, texH);
    ctx.stroke();
  }
  for (let y = 0; y < texH; y += step50) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(texW, y);
    ctx.stroke();
  }

  // Border
  ctx.strokeStyle = "#2d2d34";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.roundRect(margin, margin, texW - margin * 2, texH - margin * 2, radius);
  ctx.stroke();

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export default function BedPlatform() {
  const bed_label = useConverterStore((s) => s.bed_label);
  const bedSizes = useConverterStore((s) => s.bedSizes);
  const modelUrl = useConverterStore((s) => s.modelUrl);
  const { camera, controls } = useThree();

  // Find current bed dimensions
  const bedDims = useMemo(() => {
    const found = bedSizes.find((b) => b.label === bed_label);
    return found ? { w: found.width_mm, h: found.height_mm } : { w: 256, h: 256 };
  }, [bed_label, bedSizes]);

  // Create bed geometry + material
  const bedMesh = useMemo(() => {
    const geo = new THREE.PlaneGeometry(bedDims.w, bedDims.h);
    const texture = createBedTexture(bedDims.w, bedDims.h);
    const mat = new THREE.MeshStandardMaterial({ map: texture, roughness: 0.8 });
    const mesh = new THREE.Mesh(geo, mat);
    // Plane is XY by default, rotate to lie flat on XZ
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(0, -0.1, 0);
    return mesh;
  }, [bedDims]);

  // Auto-fit camera when bed changes and no model is loaded
  useEffect(() => {
    if (modelUrl) return; // Don't override camera when model is present

    const radius = Math.max(bedDims.w, bedDims.h) / 2;
    const perspCam = camera as THREE.PerspectiveCamera;
    const dist = computeFitDistance(radius, perspCam.fov);

    camera.position.set(dist * 0.3, dist * 0.5, dist * 0.8);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();

    if (controls) {
      const oc = controls as unknown as {
        target: THREE.Vector3;
        maxDistance: number;
        minDistance: number;
        update: () => void;
      };
      oc.target.set(0, 0, 0);
      oc.maxDistance = dist * 5;
      oc.minDistance = dist * 0.1;
      oc.update();
    }
  }, [bedDims, modelUrl, camera, controls]);

  return <primitive object={bedMesh} />;
}
