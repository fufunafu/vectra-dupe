import * as THREE from 'three';
import { OrbitControls } from './vendor/controls/OrbitControls.js';
import { GLTFLoader } from './vendor/loaders/GLTFLoader.js';

const $ = id => document.getElementById(id);
const host = $('view');
const status = $('status');
try {
  const response = await fetch('report.json');
  if (!response.ok) throw new Error('Could not read the processing report.');
  const report = await response.json();
  for (const [label, value] of [
    ['Reconstruction', report.mode === 'depth' ? 'LiDAR depth' : 'Photo stereo'],
    ['Captured photographs', report.photos ?? 0],
    ['Surface triangles', Number(report.triangles).toLocaleString()],
    ['Photo coverage', `${Math.round(report.texture_coverage * 100)}%`],
    ['Capture type', report.demo ? 'Synthetic demo' : 'Real capture'],
  ]) {
    const line = document.createElement('div'); line.className = 'fact';
    const key = document.createElement('span'); key.textContent = label;
    const val = document.createElement('strong'); val.textContent = String(value);
    line.append(key, val); $('facts').appendChild(line);
  }
  $('facts').firstChild?.nodeType === Node.TEXT_NODE && $('facts').firstChild.remove();
  for (const warning of report.warnings ?? []) {
    const item = document.createElement('li'); item.textContent = warning; $('notes').appendChild(item);
  }
  $('mode').textContent = report.demo ? 'SYNTHETIC DEMO' : report.mode === 'photos' ? 'EXPERIMENTAL PHOTO STEREO' : 'LIDAR CAPTURE';
  let renderer;
  try { renderer = new THREE.WebGLRenderer({antialias:true}); }
  catch { throw new Error('This browser cannot display 3D. You can still download the GLB or PLY and open it in a 3D viewer.'); }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#f0f2ed');
  const camera = new THREE.PerspectiveCamera(36, 1, .001, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8d9a88, 2.1));
  const light = new THREE.DirectionalLight(0xffffff, 2); light.position.set(-1, 2, 3); scene.add(light);
  const asset = await new GLTFLoader().loadAsync('model.glb');
  scene.add(asset.scene);
  const bounds = new THREE.Box3().setFromObject(asset.scene);
  if (bounds.isEmpty()) throw new Error('The exported model contains no displayable geometry.');
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  function reset() {
    const distance = Math.max(size.x, size.y, size.z) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.4;
    camera.position.copy(center).add(new THREE.Vector3(0, 0, distance));
    controls.target.copy(center); controls.update();
  }
  reset();
  const materials = [];
  asset.scene.traverse(obj => {
    if (obj.isMesh) {
      const list = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const material of list) materials.push({material, map:material.map, color:material.color.clone()});
    }
  });
  $('texture').onchange = () => {
    for (const {material,map,color} of materials) {
      material.map = $('texture').checked ? map : null;
      material.color.copy($('texture').checked ? color : new THREE.Color('#c2cbbd'));
      material.needsUpdate = true;
    }
  };
  $('wireframe').onchange = () => { for (const {material} of materials) material.wireframe = $('wireframe').checked; };
  $('reset').onclick = reset;
  new ResizeObserver(() => {
    const {width,height} = host.getBoundingClientRect();
    renderer.setSize(width,height); camera.aspect = width/height; camera.updateProjectionMatrix();
  }).observe(host);
  status.hidden = true;
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene,camera); });
} catch(error) {
  status.hidden = false;
  status.textContent = error.message || 'The model could not be opened. Keep faceMap Desktop running and try opening the viewer again.';
}
