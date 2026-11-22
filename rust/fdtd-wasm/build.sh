#!/bin/bash
# Rebuild the browser package into src/wasm/pkg.
# Rust 1.83.0 and wasm-bindgen-cli 0.2.100. No wasm-pack.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root/rust/fdtd-wasm"
if ! command -v wasm-bindgen >/dev/null 2>&1 || ! wasm-bindgen --version | grep -q '0.2.100'; then
  echo "wasm-bindgen 0.2.100 is required" >&2
  exit 1
fi
cargo build --release --target wasm32-unknown-unknown
out="$root/src/wasm/pkg"
mkdir -p "$out"
wasm-bindgen \
  --target web \
  --out-dir "$out" \
  target/wasm32-unknown-unknown/release/fdtd_wasm.wasm
rm -f "$out/.gitignore" "$out/package.json" "$out"/*.d.ts
echo "wrote $out"
