# GPU probe stream

`createGpuStepper` keeps sampled probes in a GPU buffer so `createGpuSignal` can read the trace without copying it back to the CPU. `createGpuSignal` is implemented in `src/gpu-signal/` and takes the `GPUDevice` that owns that buffer.

The supported path is `createGpuSignal(stepper.device)`. `stepper.device` is the read-only `GPUDevice` from the WebGPU stepper. Do not destroy it; `stepper.destroy()` does. WebGL2 and the CPU fallback have no device, so `stepper.device` is null there and `probeStream` is null too.

`stepper.probeStream` is also null when the grid has no probes.

```js
{
  buffer,          // GPUBuffer, float32, STORAGE | COPY_SRC, same device as stepper.device
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

`head` at `headByteOffset` is the next step index, not wrapped. The ring is double-buffered on the way to the CPU: a finished span is copied into a staging `GPUBuffer` and mapped with `mapAsync`. At most two frames are submitted ahead of that drain. Do not destroy `buffer` or `head`; the stepper owns them.

Fields, materials, and CPML profiles stay in their own GPU buffers. `stepper.syncMaterials(grid)` uploads only the material cells whose `ca`, `cb`, or PEC flag changed.
