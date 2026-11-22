# fdtd-wasm

Float64 2D FDTD compiled to WebAssembly. The arithmetic matches `src/fdtd.js`. The site does not load this module; `src/wasm/index.js` is the Node and test wrapper, and it falls back to the JS solver when the module fails to instantiate.

## Toolchain

- Rust 1.83.0, target `wasm32-unknown-unknown` (`rust-toolchain.toml`)
- wasm-bindgen 0.2.100, commit `2405ec2b4`, the `x86_64-unknown-linux-musl` release binary
- No wasm-pack and no extra crates beyond `wasm-bindgen`

`ci.sh` downloads that binary, checks it with `sha256sum -c`, and refuses a different 0.2.100.

## Rebuild

From the repo root:

```bash
bash rust/fdtd-wasm/ci.sh
```

That runs `build.sh`, which maps `CARGO_HOME` and the repo root out of panic paths (`--remap-path-prefix`) and builds with `cargo build --release --locked --target wasm32-unknown-unknown`. The committed files are `src/wasm/pkg/fdtd_wasm.js` and `src/wasm/pkg/fdtd_wasm_bg.wasm`. A second run must leave them unchanged.

`WAP_WASM_PERF=1 node --test tests/fdtd-wasm.test.js` enforces the local A9 speed gate. CI only reports the median ratio and its variance.
