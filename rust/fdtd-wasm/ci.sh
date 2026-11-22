#!/bin/bash
# CI: install the pinned wasm-bindgen binary if needed, rebuild, and
# fail if src/wasm/pkg does not match. The committed package is what
# GitHub Pages serves; this job checks it was produced by Rust 1.83.0.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
if ! command -v wasm-bindgen >/dev/null 2>&1 || ! wasm-bindgen --version | grep -q '0.2.100'; then
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/wb.tgz" \
    https://github.com/rustwasm/wasm-bindgen/releases/download/0.2.100/wasm-bindgen-0.2.100-x86_64-unknown-linux-musl.tar.gz
  tar -xzf "$tmp/wb.tgz" -C "$tmp"
  mkdir -p "$HOME/.local/bin"
  cp "$tmp"/wasm-bindgen-*/wasm-bindgen "$HOME/.local/bin/wasm-bindgen"
  chmod +x "$HOME/.local/bin/wasm-bindgen"
  export PATH="$HOME/.local/bin:$PATH"
fi
bash "$root/rust/fdtd-wasm/build.sh"
git -C "$root" diff --exit-code -- src/wasm/pkg
echo "wasm package matches the commit"
