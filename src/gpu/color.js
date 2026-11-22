/**
 * Gray image of a field, row j of the grid in row j of the image.
 * The same map is used by the GPU draw and the CPU fallback.
 * @param {ArrayLike<number>} field
 * @param {number} nx
 * @param {number} ny
 * @param {number} scale
 */
export function fieldToImage(field, nx, ny, scale) {
  const out = new Uint8ClampedArray(nx * ny * 4);
  const s = scale === 0 ? 1 : scale;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const t = Math.min(1, Math.max(0, 0.5 + 0.5 * (field[j * nx + i] / s)));
      const b = Math.round(t * 255);
      const o = (j * nx + i) * 4;
      out[o] = b;
      out[o + 1] = b;
      out[o + 2] = b;
      out[o + 3] = 255;
    }
  }
  return out;
}
