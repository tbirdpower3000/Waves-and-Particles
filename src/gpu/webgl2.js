import { accumulateSource } from "../fdtd.js";

const VERT = `#version 300 es
layout(location=0) in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
`;

const HX = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uField;
uniform sampler2D uHx;
uniform sampler2D uPsi;
uniform sampler2D uA;
uniform sampler2D uB;
uniform int uNy;
uniform int uPeriodic;
uniform int uHx0;
uniform int uHx1;
uniform float uInvDy;
uniform float uCh;
layout(location=0) out float oHx;
layout(location=1) out float oPsi;
void main() {
  ivec2 ij = ivec2(gl_FragCoord.xy);
  int j = ij.y;
  float cur = texelFetch(uHx, ij, 0).r;
  float psi0 = texelFetch(uPsi, ij, 0).r;
  if (uPeriodic == 0 && j >= uNy - 1) {
    oHx = cur;
    oPsi = psi0;
    return;
  }
  int up = j + 1;
  if (up >= uNy) up = 0;
  float d = (texelFetch(uField, ivec2(ij.x, up), 0).r - texelFetch(uField, ij, 0).r) * uInvDy;
  bool pml = uPeriodic == 0 && (j < uHx0 || j >= uHx1);
  if (!pml) {
    oHx = cur - uCh * d;
    oPsi = psi0;
    return;
  }
  float a = texelFetch(uA, ivec2(j, 0), 0).r;
  float b = texelFetch(uB, ivec2(j, 0), 0).r;
  float psi = b * psi0 + a * d;
  oPsi = psi;
  oHx = cur - uCh * (d + psi);
}
`;

const HY = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uField;
uniform sampler2D uHy;
uniform sampler2D uPsi;
uniform sampler2D uA;
uniform sampler2D uB;
uniform int uNx;
uniform int uHy0;
uniform int uHy1;
uniform float uInvDx;
uniform float uCh;
layout(location=0) out float oHy;
layout(location=1) out float oPsi;
void main() {
  ivec2 ij = ivec2(gl_FragCoord.xy);
  int i = ij.x;
  float cur = texelFetch(uHy, ij, 0).r;
  float psi0 = texelFetch(uPsi, ij, 0).r;
  if (i >= uNx - 1) {
    oHy = cur;
    oPsi = psi0;
    return;
  }
  float d = (texelFetch(uField, ivec2(i + 1, ij.y), 0).r - texelFetch(uField, ij, 0).r) * uInvDx;
  bool pml = i < uHy0 || i >= uHy1;
  if (!pml) {
    oHy = cur + uCh * d;
    oPsi = psi0;
    return;
  }
  float a = texelFetch(uA, ivec2(i, 0), 0).r;
  float b = texelFetch(uB, ivec2(i, 0), 0).r;
  float psi = b * psi0 + a * d;
  oPsi = psi;
  oHy = cur + uCh * (d + psi);
}
`;

const EZ = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uField;
uniform sampler2D uHx;
uniform sampler2D uHy;
uniform sampler2D uPsiX;
uniform sampler2D uPsiY;
uniform sampler2D uCa;
uniform sampler2D uCb;
uniform sampler2D uPec;
uniform sampler2D uSrc;
uniform sampler2D uAEx;
uniform sampler2D uBEx;
uniform sampler2D uAEy;
uniform sampler2D uBEy;
uniform int uNx;
uniform int uNy;
uniform int uPeriodic;
uniform int uI0;
uniform int uI1;
uniform int uJ0;
uniform int uJ1;
uniform float uInvDx;
uniform float uInvDy;
layout(location=0) out float oField;
layout(location=1) out float oPsiX;
layout(location=2) out float oPsiY;
void main() {
  ivec2 ij = ivec2(gl_FragCoord.xy);
  int i = ij.x;
  int j = ij.y;
  float e = texelFetch(uField, ij, 0).r;
  float psx = texelFetch(uPsiX, ij, 0).r;
  float psy = texelFetch(uPsiY, ij, 0).r;
  float pec = texelFetch(uPec, ij, 0).r;
  if (pec > 0.5) {
    oField = 0.0;
    oPsiX = psx;
    oPsiY = psy;
    return;
  }
  bool interiorX = i > 0 && i < uNx - 1;
  bool interiorY = uPeriodic == 1 || (j > 0 && j < uNy - 1);
  if (interiorX && interiorY) {
    int below = j - 1;
    if (below < 0) below = uNy - 1;
    float dHy = (texelFetch(uHy, ij, 0).r - texelFetch(uHy, ivec2(i - 1, j), 0).r) * uInvDx;
    float dHx = (texelFetch(uHx, ij, 0).r - texelFetch(uHx, ivec2(i, below), 0).r) * uInvDy;
    bool edge = i < uI0 || i > uI1 || (uPeriodic == 0 && (j < uJ0 || j > uJ1));
    float ca = texelFetch(uCa, ij, 0).r;
    float cb = texelFetch(uCb, ij, 0).r;
    if (!edge) {
      e = ca * e + cb * (dHy - dHx);
    } else {
      float px = texelFetch(uBEx, ivec2(i, 0), 0).r * psx + texelFetch(uAEx, ivec2(i, 0), 0).r * dHy;
      float py = texelFetch(uBEy, ivec2(j, 0), 0).r * psy + texelFetch(uAEy, ivec2(j, 0), 0).r * dHx;
      psx = px;
      psy = py;
      e = ca * e + cb * (dHy + px - (dHx + py));
    }
  }
  oField = e + texelFetch(uSrc, ij, 0).r;
  oPsiX = psx;
  oPsiY = psy;
}
`;

const PROBE = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uField;
uniform int uI[64];
uniform int uJ[64];
layout(location=0) out float o;
void main() {
  int q = int(gl_FragCoord.x);
  o = texelFetch(uField, ivec2(uI[q], uJ[q]), 0).r;
}
`;

const COLOR = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uField;
uniform float uScale;
layout(location=0) out vec4 o;
void main() {
  ivec2 ij = ivec2(gl_FragCoord.xy);
  float v = texelFetch(uField, ij, 0).r;
  float t = clamp(0.5 + 0.5 * v / uScale, 0.0, 1.0);
  o = vec4(t, t, t, 1.0);
}
`;

function f32(src) {
  const out = new Float32Array(src.length);
  out.set(src);
  return out;
}

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(log || "shader compile failed");
  }
  return shader;
}

function program(gl, fragment) {
  const prog = gl.createProgram();
  const vs = compile(gl, gl.VERTEX_SHADER, VERT);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fragment);
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(prog) || "program link failed");
  }
  return prog;
}

function makeTexture(gl, w, h, data) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, data);
  return tex;
}

function makeTarget(gl, textures) {
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  const bufs = [];
  for (let k = 0; k < textures.length; k++) {
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0 + k,
      gl.TEXTURE_2D,
      textures[k],
      0,
    );
    bufs.push(gl.COLOR_ATTACHMENT0 + k);
  }
  gl.drawBuffers(bufs);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error(`framebuffer incomplete (${status})`);
  }
  return { fb, textures };
}

function bind(gl, unit, tex) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
}

const SOFTWARE_RENDERER = /swiftshader|llvmpipe|softpipe|lavapipe|software|basic render/i;
const HARDWARE_RENDERER =
  /\b(nvidia|geforce|quadro|\brtx\b|radeon|\bamd\b|intel\(r\)|\biris\b|uhd graphics|hd graphics|apple gpu|apple m\d|\bmali\b|adreno|powervr)\b/i;

/**
 * Masked or unrecognized WebGL renderers are not hardware. Hardware requires
 * the unmasked renderer string and a known non-software GPU name.
 * @param {WebGL2RenderingContext} gl
 */
export function classifyWebGLRenderer(gl) {
  const ext = gl.getExtension("WEBGL_debug_renderer_info");
  const maskedVendor = String(gl.getParameter(gl.VENDOR) ?? "");
  const maskedRenderer = String(gl.getParameter(gl.RENDERER) ?? "");
  let vendor = maskedVendor;
  let renderer = maskedRenderer;
  let unmasked = false;
  if (ext) {
    const rawVendor = gl.getParameter(ext.UNMASKED_VENDOR_WEBGL);
    const rawRenderer = gl.getParameter(ext.UNMASKED_RENDERER_WEBGL);
    if (rawRenderer) {
      vendor = String(rawVendor ?? "");
      renderer = String(rawRenderer);
      unmasked = true;
    }
  }
  const blob = `${vendor} ${renderer}`;
  const software = SOFTWARE_RENDERER.test(blob);
  const hardware = unmasked && !software && HARDWARE_RENDERER.test(blob);
  return {
    vendor,
    architecture: software ? "swiftshader" : "",
    device: "",
    description: renderer,
    isFallbackAdapter: software,
    software,
    hardware,
    unmasked,
  };
}

export function createWebGL2Stepper(grid) {
  if (typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  canvas.width = grid.nx;
  canvas.height = grid.ny;
  const gl = canvas.getContext("webgl2", {
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
  });
  if (!gl) return null;
  if (!gl.getExtension("EXT_color_buffer_float")) return null;
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
  gl.viewport(0, 0, grid.nx, grid.ny);

  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  const { nx, ny } = grid;
  const n = nx * ny;
  const pec = new Float32Array(n);
  for (let i = 0; i < n; i++) pec[i] = grid.pec[i] ? 1 : 0;

  const allocPair = (count, seed) => {
    const a = [];
    const b = [];
    for (let k = 0; k < count; k++) a.push(makeTexture(gl, nx, ny, seed[k]));
    for (let k = 0; k < count; k++) b.push(makeTexture(gl, nx, ny, null));
    return [makeTarget(gl, a), makeTarget(gl, b)];
  };

  let field = allocPair(3, [f32(grid.field), f32(grid.psiEzx), f32(grid.psiEzy)]);
  let hx = allocPair(2, [f32(grid.Hx), f32(grid.psiHxy)]);
  let hy = allocPair(2, [f32(grid.Hy), f32(grid.psiHyx)]);
  const ca = makeTexture(gl, nx, ny, f32(grid.ca));
  const cb = makeTexture(gl, nx, ny, f32(grid.cb));
  const pecTex = makeTexture(gl, nx, ny, pec);
  const src = makeTexture(gl, nx, ny, new Float32Array(n));
  const aEx = makeTexture(gl, grid.aEx.length, 1, f32(grid.aEx));
  const bEx = makeTexture(gl, grid.bEx.length, 1, f32(grid.bEx));
  const aEy = makeTexture(gl, grid.aEy.length, 1, f32(grid.aEy));
  const bEy = makeTexture(gl, grid.bEy.length, 1, f32(grid.bEy));
  const aHx = makeTexture(gl, Math.max(1, grid.aHx.length), 1, f32(grid.aHx));
  const bHx = makeTexture(gl, Math.max(1, grid.bHx.length), 1, f32(grid.bHx));
  const aHy = makeTexture(gl, Math.max(1, grid.aHy.length), 1, f32(grid.aHy));
  const bHy = makeTexture(gl, Math.max(1, grid.bHy.length), 1, f32(grid.bHy));

  const progHx = program(gl, HX);
  const progHy = program(gl, HY);
  const progEz = program(gl, EZ);

  const setUniforms = (progId, pairs) => {
    gl.useProgram(progId);
    for (const [name, value, kind] of pairs) {
      const loc = gl.getUniformLocation(progId, name);
      if (loc == null) throw new Error(`missing uniform ${name}`);
      if (kind === "f") gl.uniform1f(loc, value);
      else gl.uniform1i(loc, value | 0);
    }
  };
  setUniforms(progHx, [
    ["uField", 0], ["uHx", 1], ["uPsi", 2], ["uA", 3], ["uB", 4],
    ["uNy", ny], ["uPeriodic", grid.periodicY ? 1 : 0],
    ["uHx0", grid.hx0], ["uHx1", grid.hx1],
    ["uInvDy", grid.invDy, "f"], ["uCh", grid.ch, "f"],
  ]);
  setUniforms(progHy, [
    ["uField", 0], ["uHy", 1], ["uPsi", 2], ["uA", 3], ["uB", 4],
    ["uNx", nx], ["uHy0", grid.hy0], ["uHy1", grid.hy1],
    ["uInvDx", grid.invDx, "f"], ["uCh", grid.ch, "f"],
  ]);
  setUniforms(progEz, [
    ["uField", 0], ["uHx", 1], ["uHy", 2], ["uPsiX", 3], ["uPsiY", 4],
    ["uCa", 5], ["uCb", 6], ["uPec", 7], ["uSrc", 8],
    ["uAEx", 9], ["uBEx", 10], ["uAEy", 11], ["uBEy", 12],
    ["uNx", nx], ["uNy", ny], ["uPeriodic", grid.periodicY ? 1 : 0],
    ["uI0", grid.iInt0], ["uI1", grid.iInt1], ["uJ0", grid.jInt0], ["uJ1", grid.jInt1],
    ["uInvDx", grid.invDx, "f"], ["uInvDy", grid.invDy, "f"],
  ]);

  const sources = grid.sources.map((s) => ({ ...s, acc: s.acc ?? 0 }));
  const probes = grid.probes.map((p) => ({ i: p.i, j: p.j, samples: p.samples.slice() }));
  if (sources.length > 8) throw new Error("WebGL2 stepper supports at most 8 sources");
  if (probes.length > 64) throw new Error("WebGL2 stepper supports at most 64 probes");
  const delta = new Float32Array(n);
  const caCpu = f32(grid.ca);
  const cbCpu = f32(grid.cb);
  const pecCpu = pec.slice();
  let time = grid.time;
  let materialBytesUploaded = 0;
  let fullFieldReads = 0;
  const RING = 4096;
  let head = 0;
  let drained = 0;
  let histTex = null;
  let histFb = null;
  let progProbe = null;
  if (probes.length) {
    histTex = makeTexture(gl, probes.length, RING, null);
    histFb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, histFb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, histTex, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    progProbe = program(gl, PROBE);
    gl.useProgram(progProbe);
    gl.uniform1i(gl.getUniformLocation(progProbe, "uField"), 0);
    const probeI = new Int32Array(64);
    const probeJ = new Int32Array(64);
    probes.forEach((p, q) => {
      probeI[q] = p.i;
      probeJ[q] = p.j;
    });
    gl.uniform1iv(gl.getUniformLocation(progProbe, "uI"), probeI);
    gl.uniform1iv(gl.getUniformLocation(progProbe, "uJ"), probeJ);
  }

  function draw(progId, target) {
    gl.useProgram(progId);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fb);
    gl.viewport(0, 0, nx, ny);
    gl.drawBuffers(target.textures.map((_, k) => gl.COLOR_ATTACHMENT0 + k));
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  function uploadRuns(ids) {
    let p = 0;
    while (p < ids.length) {
      const id0 = ids[p];
      const j = (id0 / nx) | 0;
      let i0 = id0 - j * nx;
      let i1 = i0;
      while (p + 1 < ids.length && ids[p + 1] === ids[p] + 1 && ((ids[p + 1] / nx) | 0) === j) {
        p += 1;
        i1 += 1;
      }
      const w = i1 - i0 + 1;
      const start = j * nx + i0;
      const put = (tex, cpu) => {
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, i0, j, w, 1, gl.RED, gl.FLOAT, cpu.subarray(start, start + w));
      };
      put(ca, caCpu);
      put(cb, cbCpu);
      put(pecTex, pecCpu);
      materialBytesUploaded += w * 4 * 3;
      p += 1;
    }
  }

  function syncMaterials(next = grid) {
    const bits = new Float32Array(1);
    const asF32 = (value) => {
      bits[0] = value;
      return bits[0];
    };
    const changed = [];
    for (let id = 0; id < n; id++) {
      const flag = next.pec[id] ? 1 : 0;
      const caNext = asF32(next.ca[id]);
      const cbNext = asF32(next.cb[id]);
      if (caCpu[id] !== caNext || cbCpu[id] !== cbNext || pecCpu[id] !== flag) {
        caCpu[id] = caNext;
        cbCpu[id] = cbNext;
        pecCpu[id] = flag;
        changed.push(id);
      }
    }
    uploadRuns(changed);
    return changed.length;
  }

  function runSteps(count) {
    const nStep = count | 0;
    syncMaterials();
    for (let s = 0; s < nStep; s++) {
      time += grid.dt;
      delta.fill(0);
      for (let q = 0; q < sources.length; q++) {
        const src = sources[q];
        const id = src.j * nx + src.i;
        if (grid.pec[id]) continue;
        delta[id] += accumulateSource(src, time);
      }
      gl.bindTexture(gl.TEXTURE_2D, src);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, nx, ny, gl.RED, gl.FLOAT, delta);

      bind(gl, 0, field[0].textures[0]);
      bind(gl, 1, hx[0].textures[0]);
      bind(gl, 2, hx[0].textures[1]);
      bind(gl, 3, aHy);
      bind(gl, 4, bHy);
      draw(progHx, hx[1]);

      bind(gl, 0, field[0].textures[0]);
      bind(gl, 1, hy[0].textures[0]);
      bind(gl, 2, hy[0].textures[1]);
      bind(gl, 3, aHx);
      bind(gl, 4, bHx);
      draw(progHy, hy[1]);

      bind(gl, 0, field[0].textures[0]);
      bind(gl, 1, hx[1].textures[0]);
      bind(gl, 2, hy[1].textures[0]);
      bind(gl, 3, field[0].textures[1]);
      bind(gl, 4, field[0].textures[2]);
      bind(gl, 5, ca);
      bind(gl, 6, cb);
      bind(gl, 7, pecTex);
      bind(gl, 8, src);
      bind(gl, 9, aEx);
      bind(gl, 10, bEx);
      bind(gl, 11, aEy);
      bind(gl, 12, bEy);
      draw(progEz, field[1]);

      field.reverse();
      hx.reverse();
      hy.reverse();

      if (probes.length) {
        const row = head % RING;
        gl.bindFramebuffer(gl.FRAMEBUFFER, histFb);
        gl.viewport(0, row, probes.length, 1);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
        bind(gl, 0, field[0].textures[0]);
        gl.useProgram(progProbe);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.viewport(0, 0, nx, ny);
        head += 1;
      }
    }
  }

  function drainProbes() {
    if (!probes.length || drained === head) return;
    const pending = head - drained;
    const readRange = (row, count) => {
      const buf = new Float32Array(count * probes.length);
      gl.bindFramebuffer(gl.FRAMEBUFFER, histFb);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.readPixels(0, row, probes.length, count, gl.RED, gl.FLOAT, buf);
      for (let s = 0; s < count; s++) {
        for (let q = 0; q < probes.length; q++) probes[q].samples.push(buf[s * probes.length + q]);
      }
    };
    let left = pending;
    let row = drained % RING;
    while (left > 0) {
      const count = Math.min(left, RING - row);
      readRange(row, count);
      left -= count;
      row = 0;
    }
    drained = head;
  }

  function step(count = 1) {
    runSteps(count | 0);
    drainProbes();
    return Promise.resolve();
  }

  function kick(count = 1) {
    runSteps(count | 0);
    return Promise.resolve();
  }

  function idle() {
    drainProbes();
    gl.finish();
    return Promise.resolve();
  }

  function readField() {
    fullFieldReads += 1;
    const out = new Float32Array(n);
    gl.bindFramebuffer(gl.FRAMEBUFFER, field[0].fb);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, nx, ny, gl.RED, gl.FLOAT, out);
    return Promise.resolve(out);
  }

  function createView(viewCanvas, opts = {}) {
    const scale = opts.scale ?? 1;
    const stepsPerFrame = opts.stepsPerFrame ?? 1;
    const colorTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, colorTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, nx, ny, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const colorFb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, colorFb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colorTex, 0);
    const progColor = program(gl, COLOR);
    gl.useProgram(progColor);
    gl.uniform1i(gl.getUniformLocation(progColor, "uField"), 0);
    gl.uniform1f(gl.getUniformLocation(progColor, "uScale"), scale);
    if (viewCanvas) {
      viewCanvas.width = nx;
      viewCanvas.height = ny;
    }

    function present() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, colorFb);
      gl.viewport(0, 0, nx, ny);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      bind(gl, 0, field[0].textures[0]);
      gl.useProgram(progColor);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      if (viewCanvas && viewCanvas !== canvas) {
        const ctx = viewCanvas.getContext("2d");
        if (ctx) ctx.drawImage(canvas, 0, 0);
      }
    }

    function frame(count) {
      runSteps(count == null ? stepsPerFrame : count | 0);
      present();
    }

    function readDrawn() {
      const pixels = new Uint8Array(nx * ny * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, colorFb);
      gl.readPixels(0, 0, nx, ny, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      return Promise.resolve(new Uint8ClampedArray(pixels));
    }

    return {
      backend: "webgl2",
      canvas: viewCanvas || canvas,
      stepsPerFrame,
      scale,
      get fullFieldReads() {
        return fullFieldReads;
      },
      frame,
      idle,
      readDrawn,
      readProbes: () => probes.map((p) => ({ i: p.i, j: p.j, samples: p.samples })),
      destroy() {},
    };
  }

  const adapter = classifyWebGLRenderer(gl);
  return {
    backend: "webgl2",
    adapter,
    probeStream: null,
    get fullFieldReads() {
      return fullFieldReads;
    },
    get materialBytesUploaded() {
      return materialBytesUploaded;
    },
    step,
    kick,
    idle,
    syncMaterials,
    createView,
    readField,
    readProbes: () => probes.map((p) => ({ i: p.i, j: p.j, samples: p.samples })),
    destroy() {
      const lose = gl.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();
    },
  };
}
