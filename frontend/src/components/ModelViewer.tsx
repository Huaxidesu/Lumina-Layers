import { useMemo, useEffect } from "react";
import { useThree } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";

/**
 * Compute the offset needed to center a bounding box at the origin.
 * Pure function, independently testable.
 */
export function computeCenterOffset(
  min: [number, number, number],
  max: [number, number, number],
): [number, number, number] {
  return [
    -(min[0] + max[0]) / 2,
    -(min[1] + max[1]) / 2,
    -(min[2] + max[2]) / 2,
  ];
}

/**
 * Compute camera distance so the model fits in view.
 * Returns the distance from the origin along the camera's forward axis.
 */
export function computeFitDistance(
  boundingSphereRadius: number,
  fovDeg: number,
): number {
  const halfFovRad = (fovDeg * Math.PI) / 360;
  return (boundingSphereRadius / Math.sin(halfFovRad)) * 1.2;
}

interface ModelViewerProps {
  url: string;
}

function ModelViewer({ url }: ModelViewerProps) {
  const { scene } = useGLTF(url);
  const { camera, controls } = useThree();

  const preparedScene = useMemo(() => {
    const clone = scene.clone(true);

    // Remove any baked-in bed mesh from old GLB files
    const toRemove: THREE.Object3D[] = [];
    clone.traverse((child) => {
      if (child.name.toLowerCase() === "bed") {
        toRemove.push(child);
      }
    });
    toRemove.forEach((obj) => obj.removeFromParent());

    // Trimesh exports Z-up, Three.js is Y-up → rotate -90° around X
    clone.rotation.x = -Math.PI / 2;
    clone.updateMatrixWorld(true);

    // Compute bounding box after rotation
    const box = new THREE.Box3().setFromObject(clone);

    // Center on XZ plane, sit on Y=0 (on top of bed)
    const center = new THREE.Vector3();
    box.getCenter(center);
    clone.position.set(-center.x, -box.min.y, -center.z);

    return clone;
  }, [scene]);

  // Auto-fit camera to model after load
  useEffect(() => {
    // Need a wrapper to get correct world bounds after position offset
    const wrapper = new THREE.Group();
    wrapper.add(preparedScene.clone(true));
    wrapper.updateMatrixWorld(true);

    const box = new THREE.Box3().setFromObject(wrapper);
    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);

    const perspCam = camera as THREE.PerspectiveCamera;
    const dist = computeFitDistance(sphere.radius, perspCam.fov);

    camera.position.set(dist * 0.3, dist * 0.5, dist * 0.8);
    camera.lookAt(sphere.center);
    camera.updateProjectionMatrix();

    if (controls) {
      const oc = controls as unknown as {
        target: THREE.Vector3;
        maxDistance: number;
        minDistance: number;
        update: () => void;
      };
      oc.target.copy(sphere.center);
      oc.maxDistance = dist * 5;
      oc.minDistance = dist * 0.1;
      oc.update();
    }

    wrapper.clear();
  }, [preparedScene, camera, controls]);

  return <primitive object={preparedScene} />;
}

export default ModelViewer;
