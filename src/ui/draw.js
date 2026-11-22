const buffers = new WeakMap();

export function decimatePeaks(data, buckets) {
  const n = data?.length ?? 0;
  if (n === 0 || buckets < 1) return new Float64Array(0);
  if (n <= buckets) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = Number.isFinite(data[i]) ? data[i] : 0;
    return out;
  }
  const out = new Float64Array(buckets);
  const span = n / buckets;
  for (let b = 0; b < buckets; b++) {
    const start = Math.floor(b * span);
    const end = Math.max(start + 1, Math.min(n, Math.floor((b + 1) * span)));
    let peak = 0;
    for (let i = start; i < end; i++) {
      const v = data[i];
      if (Number.isFinite(v) && Math.abs(v) > Math.abs(peak)) peak = v;
    }
    out[b] = peak;
  }
  return out;
}

export function fieldRgb(v, maxAbs) {
  const t = Math.max(-1, Math.min(1, v / (maxAbs || 1)));
  if (t >= 0) {
    return [Math.round(8 + t * 28), Math.round(18 + t * 210), Math.round(28 + t * 170)];
  }
  const a = -t;
  return [Math.round(32 + a * 220), Math.round(18 + a * 110), Math.round(16 + a * 20)];
}

export function fitCanvas(canvas) {
  const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const cssW = rect.width || canvas.clientWidth || 320;
  const cssH = rect.height || canvas.clientHeight || 180;
  const width = Math.max(1, cssW);
  const height = Math.max(1, cssH);
  const w = Math.max(1, Math.round(width * dpr));
  const h = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function fieldBuffer(canvas, nx, ny) {
  let entry = buffers.get(canvas);
  if (!entry || entry.nx !== nx || entry.ny !== ny) {
    const off = document.createElement('canvas');
    off.width = nx;
    off.height = ny;
    entry = { off, nx, ny };
    buffers.set(canvas, entry);
  }
  return entry.off;
}

function drawChamber(ctx, width, height, pml, nx, ny) {
  ctx.fillStyle = '#070b10';
  ctx.fillRect(0, 0, width, height);
  const px = (pml / nx) * width;
  const py = (pml / ny) * height;
  ctx.fillStyle = '#0e1722';
  ctx.fillRect(px, py, Math.max(0, width - 2 * px), Math.max(0, height - 2 * py));
  ctx.strokeStyle = 'rgba(46, 230, 199, 0.45)';
  ctx.lineWidth = 1;
  ctx.strokeRect(px + 0.5, py + 0.5, Math.max(0, width - 2 * px - 1), Math.max(0, height - 2 * py - 1));
  ctx.strokeStyle = 'rgba(146, 170, 196, 0.16)';
  ctx.beginPath();
  const cells = 8;
  for (let g = 1; g < cells; g++) {
    const x = (g / cells) * width;
    const y = (g / cells) * height;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
  }
  ctx.stroke();
}

function cellBox(rect, nx, ny, width, height) {
  return {
    x: (rect.x0 / nx) * width,
    y: (rect.y0 / ny) * height,
    w: ((rect.x1 - rect.x0 + 1) / nx) * width,
    h: ((rect.y1 - rect.y0 + 1) / ny) * height,
  };
}

function paintOverlays(ctx, view, width, height) {
  const { nx, ny, scene, drag } = view;
  for (const material of scene.materials) {
    const box = cellBox(material, nx, ny, width, height);
    ctx.fillStyle = material.pec ? 'rgba(186, 198, 214, 0.38)' : 'rgba(46, 230, 199, 0.22)';
    ctx.strokeStyle = material.pec ? '#d5deea' : '#2ee6c7';
    ctx.fillRect(box.x, box.y, box.w, box.h);
    ctx.strokeRect(box.x + 0.5, box.y + 0.5, Math.max(0, box.w - 1), Math.max(0, box.h - 1));
  }
  if (drag) {
    const box = cellBox(drag.rect, nx, ny, width, height);
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = drag.tool === 'metal' ? '#d5deea' : '#2ee6c7';
    ctx.strokeRect(box.x + 0.5, box.y + 0.5, Math.max(0, box.w - 1), Math.max(0, box.h - 1));
    ctx.restore();
  }
  if (scene.source) {
    const x = ((scene.source.i + 0.5) / nx) * width;
    const y = ((scene.source.j + 0.5) / ny) * height;
    ctx.strokeStyle = '#ffb020';
    ctx.fillStyle = '#ffb020';
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillRect(x - 0.5, y - 7, 1, 14);
    ctx.fillRect(x - 7, y - 0.5, 14, 1);
  }
  scene.probes.forEach((probe, index) => {
    const x = ((probe.i + 0.5) / nx) * width;
    const y = ((probe.j + 0.5) / ny) * height;
    ctx.fillStyle = view.probeColors[index % view.probeColors.length];
    ctx.beginPath();
    ctx.moveTo(x, y - 6);
    ctx.lineTo(x + 5, y);
    ctx.lineTo(x, y + 6);
    ctx.lineTo(x - 5, y);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = '#e7eef6';
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText(String(index + 1), x + 7, y + 3);
  });
}

export function drawField(canvas, view) {
  const { ctx, width, height } = fitCanvas(canvas);
  const { nx, ny, pml, field, solverReady } = view;
  let warning = '';
  const usable = field && field.length === nx * ny;
  if (field && field.length !== nx * ny) warning = 'snapshot size does not match the grid';
  if (!usable) {
    drawChamber(ctx, width, height, pml, nx, ny);
  } else {
    let maxAbs = 0;
    for (let i = 0; i < field.length; i++) {
      const v = field[i];
      if (Number.isFinite(v) && Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    }
    if (!(maxAbs > 0)) {
      drawChamber(ctx, width, height, pml, nx, ny);
    } else {
      const off = fieldBuffer(canvas, nx, ny);
      const offCtx = off.getContext('2d');
      const image = offCtx.createImageData(nx, ny);
      const data = image.data;
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const v = field[j * nx + i];
          const [r, g, b] = fieldRgb(Number.isFinite(v) ? v : 0, maxAbs);
          const p = (j * nx + i) * 4;
          data[p] = r;
          data[p + 1] = g;
          data[p + 2] = b;
          data[p + 3] = 255;
        }
      }
      offCtx.putImageData(image, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, width, height);
      const px = (pml / nx) * width;
      const py = (pml / ny) * height;
      ctx.strokeStyle = 'rgba(46, 230, 199, 0.45)';
      ctx.strokeRect(px + 0.5, py + 0.5, width - 2 * px - 1, height - 2 * py - 1);
    }
  }
  paintOverlays(ctx, view, width, height);
  if (!solverReady) {
    const label = view.holdLabel || 'coming soon';
    ctx.fillStyle = 'rgba(8, 11, 16, 0.78)';
    ctx.fillRect(10, 10, Math.min(width - 20, 16 + label.length * 7), 22);
    ctx.fillStyle = '#ffd38a';
    ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(label, 18, 25);
  }
  return warning;
}

function plotFrame(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#070b10';
  ctx.fillRect(0, 0, width, height);
  return { left: 32, right: Math.max(40, width - 8), top: 8, bottom: Math.max(16, height - 16) };
}

function emptyLabel(ctx, box, text, warn) {
  ctx.fillStyle = warn ? '#ffd38a' : '#93a3b5';
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText(text, box.left, (box.top + box.bottom) / 2);
}

export function drawSeries(canvas, data, { color = '#2ee6c7', empty = 'no samples', warn = false } = {}) {
  const { ctx, width, height } = fitCanvas(canvas);
  const box = plotFrame(ctx, width, height);
  if (!data || data.length === 0) {
    emptyLabel(ctx, box, empty, warn);
    return;
  }
  const series = decimatePeaks(data, Math.max(2, Math.floor(box.right - box.left)));
  let peak = 1e-12;
  for (let i = 0; i < series.length; i++) peak = Math.max(peak, Math.abs(series[i]));
  const plotW = box.right - box.left;
  const plotH = box.bottom - box.top;
  ctx.strokeStyle = '#243244';
  ctx.beginPath();
  ctx.moveTo(box.left, box.top + plotH / 2);
  ctx.lineTo(box.right, box.top + plotH / 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.strokeStyle = color;
  for (let i = 0; i < series.length; i++) {
    const x = box.left + (series.length === 1 ? plotW / 2 : (i / (series.length - 1)) * plotW);
    const y = box.top + plotH / 2 - (series[i] / peak) * (plotH / 2 - 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

export function drawConstellation(canvas, points, { trajectory = false, empty = 'no samples', warn = false } = {}) {
  const { ctx, width, height } = fitCanvas(canvas);
  const box = plotFrame(ctx, width, height);
  if (!points?.length) {
    emptyLabel(ctx, box, empty, warn);
    return;
  }
  let max = 1.2;
  for (const point of points) {
    if (!Number.isFinite(point.i) || !Number.isFinite(point.q)) continue;
    max = Math.max(max, Math.abs(point.i), Math.abs(point.q));
  }
  max *= 1.15;
  const plotW = box.right - box.left;
  const plotH = box.bottom - box.top;
  const scale = Math.min(plotW, plotH) / (2 * max);
  const cx = (box.left + box.right) / 2;
  const cy = (box.top + box.bottom) / 2;
  const X = (v) => cx + v * scale;
  const Y = (v) => cy - v * scale;
  ctx.strokeStyle = '#243244';
  ctx.beginPath();
  ctx.moveTo(box.left, cy);
  ctx.lineTo(box.right, cy);
  ctx.moveTo(cx, box.top);
  ctx.lineTo(cx, box.bottom);
  ctx.stroke();
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(46, 230, 199, 0.35)';
  ctx.arc(cx, cy, scale, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  points.forEach((point, index) => {
    if (!Number.isFinite(point.i) || !Number.isFinite(point.q)) return;
    const x = X(point.i);
    const y = Y(point.q);
    if (trajectory) {
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    } else {
      ctx.moveTo(x + 1.7, y);
      ctx.arc(x, y, 1.7, 0, Math.PI * 2);
    }
  });
  if (trajectory) {
    ctx.strokeStyle = 'rgba(155, 140, 255, 0.95)';
    ctx.stroke();
  } else {
    ctx.fillStyle = 'rgba(255, 176, 32, 0.9)';
    ctx.fill();
  }
}

export function drawSpectrum(canvas, values, { empty = 'no spectrum', warn = false } = {}) {
  const { ctx, width, height } = fitCanvas(canvas);
  const box = plotFrame(ctx, width, height);
  if (!values || values.length === 0) {
    emptyLabel(ctx, box, empty, warn);
    return;
  }
  const plotW = box.right - box.left;
  const plotH = box.bottom - box.top;
  let min = 0;
  let max = -120;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (max <= min) max = min + 1;
  ctx.beginPath();
  ctx.strokeStyle = '#9b8cff';
  for (let i = 0; i < values.length; i++) {
    const v = Number.isFinite(values[i]) ? values[i] : min;
    const x = box.left + (values.length === 1 ? 0 : (i / (values.length - 1)) * plotW);
    const y = box.top + ((max - v) / (max - min)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

export function drawBer(canvas, { theory = [], simulated = [], markerDb = null, note = '', warn = false } = {}) {
  const { ctx, width, height } = fitCanvas(canvas);
  const box = plotFrame(ctx, width, height);
  if (note && theory.length === 0) {
    emptyLabel(ctx, box, note, warn);
    return;
  }
  const x0 = -2;
  const x1 = 12;
  const plotW = box.right - box.left;
  const plotH = box.bottom - box.top;
  const X = (db) => box.left + ((db - x0) / (x1 - x0)) * plotW;
  const Y = (rate) => {
    const clamped = Math.min(1, Math.max(1e-6, Number.isFinite(rate) && rate > 0 ? rate : 1e-6));
    const t = -Math.log10(clamped) / 6;
    return box.top + t * plotH;
  };
  ctx.strokeStyle = '#243244';
  ctx.strokeRect(box.left + 0.5, box.top + 0.5, plotW, plotH);
  if (markerDb != null && Number.isFinite(markerDb)) {
    ctx.strokeStyle = 'rgba(255, 176, 32, 0.35)';
    ctx.beginPath();
    ctx.moveTo(X(markerDb), box.top);
    ctx.lineTo(X(markerDb), box.bottom);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.strokeStyle = '#2ee6c7';
  theory.forEach((point, index) => {
    const x = X(point.ebn0dB);
    const y = Y(point.ber);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = '#ffb020';
  for (const point of simulated) {
    ctx.beginPath();
    ctx.arc(X(point.ebn0dB), Y(point.ber), 2.4, 0, Math.PI * 2);
    ctx.fill();
  }
  if (note) {
    ctx.fillStyle = '#93a3b5';
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText(note, box.left, box.top + 10);
  }
}
