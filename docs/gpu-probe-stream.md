# GPU probe stream

`createGpuStepper` keeps the sampled probes in a GPU buffer so a later `createGpuSignal` can read them without copying the trace back to the CPU. This note is the whole contract. Nothing in `src/gpu-signal/` is required to exist yet, and this repo does not import it.

The object is `stepper.probeStream`. It is `null` when the grid has no probes, and `null` on the WebGL2 and CPU paths, which have no `GPUBuffer`.

```js
{
  buffer,          // GPUBuffer, float32, STORAGE | COPY_SRC
  head,            // GPUBuffer, the stepper clock
  headByteOffset,  // 8; a u32 count of completed steps
  stride,          // floats per step, one per probe, probe order
  capacity,        // steps in the ring (4096)
  probeCount,      // same as stride
  format,          // "f32"
}
```

The sample for completed step `s` and probe `q` is

```text
buffer[((s % capacity) * stride + q) * 4]
```

`head` at `headByteOffset` is the next step index, not wrapped. The ring is double-buffered on the way to the CPU: the frame submit copies a finished span into a staging `GPUBuffer` and maps that copy with `mapAsync`. The frame call itself does not wait. Do not destroy `buffer` or `head`; the stepper owns them.

Fields, materials, and CPML profiles stay in their own GPU buffers. `stepper.syncMaterials(grid)` uploads only the material cells whose `ca`, `cb`, or PEC flag changed.
