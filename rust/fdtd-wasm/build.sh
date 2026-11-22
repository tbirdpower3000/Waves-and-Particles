#!/bin/bash
# Rebuild the browser package into src/wasm/pkg.
# Rust 1.83.0 and wasm-bindgen-cli 0.2.100. No wasm-pack.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root/rust/fdtd-wasm"
# Same release binary ci.sh installs: wasm-bindgen 0.2.100, commit 2405ec2b4.
# sha256 of that musl executable (the tarball digest is checked in ci.sh).
wb_bin="$(command -v wasm-bindgen || true)"
wb_hash="5172892fb0253966d74be4d24f181b2b5fdaa0d20604fc41998d813f08633152"
if [[ -z "$wb_bin" ]] || ! "$wb_bin" --version | grep -qx 'wasm-bindgen 0.2.100'; then
  echo "wasm-bindgen 0.2.100 commit 2405ec2b4 is required" >&2
  exit 1
fi
if ! grep -a -q '2405ec2b' "$wb_bin"; then
  echo "wasm-bindgen 0.2.100 is not commit 2405ec2b4" >&2
  exit 1
fi
wb_dir="$(cd "$(dirname "$wb_bin")" && pwd)"
wb_base="$(basename "$wb_bin")"
wb_sum="$(mktemp)"
printf '%s  %s\n' "$wb_hash" "$wb_base" > "$wb_sum"
if ! (cd "$wb_dir" && sha256sum -c "$wb_sum"); then
  rm -f "$wb_sum"
  echo "wasm-bindgen checksum does not match the 0.2.100 release binary" >&2
  exit 1
fi
rm -f "$wb_sum"

# Panic paths must not keep this machine's cargo home or checkout, or the
# CI rebuild will not match the committed package.
cargo_home="${CARGO_HOME:-${HOME}/.cargo}"
cargo_home="$(cd "$cargo_home" && pwd)"
export RUSTFLAGS="--remap-path-prefix=${cargo_home}=/cargo --remap-path-prefix=${root}=/src${RUSTFLAGS:+ ${RUSTFLAGS}}"

cargo build --release --locked --target wasm32-unknown-unknown
out="$root/src/wasm/pkg"
mkdir -p "$out"
wasm-bindgen \
  --target web \
  --out-dir "$out" \
  target/wasm32-unknown-unknown/release/fdtd_wasm.wasm
rm -f "$out/.gitignore" "$out/package.json" "$out"/*.d.ts
echo "wrote $out"
