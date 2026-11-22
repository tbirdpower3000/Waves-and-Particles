#!/bin/bash
# CI: install the pinned wasm-bindgen binary, rebuild, and fail if
# src/wasm/pkg does not match. The committed package is what GitHub Pages
# serves; this job checks it was produced by Rust 1.83.0.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
# Release asset for wasm-bindgen 0.2.100, x86_64-unknown-linux-musl.
# Published checksum (same digest):
# https://github.com/rustwasm/wasm-bindgen/releases/download/0.2.100/wasm-bindgen-0.2.100-x86_64-unknown-linux-musl.tar.gz.sha256sum
# that file's line is "<hash>  gh-release/wasm-bindgen-0.2.100-x86_64-unknown-linux-musl.tar.gz"
file="wasm-bindgen-0.2.100-x86_64-unknown-linux-musl.tar.gz"
hash="63d6a38deb65bd7023c02bdf382ab66b0d2c0241c8582fd3413b5a808b8aeb5b"
curl -fsSL -o "$tmp/$file" \
  "https://github.com/rustwasm/wasm-bindgen/releases/download/0.2.100/$file"
printf '%s  %s\n' "$hash" "$file" > "$tmp/$file.sha256"
(cd "$tmp" && sha256sum -c "$file.sha256")
tar -xzf "$tmp/$file" -C "$tmp"
install -m 755 "$tmp"/wasm-bindgen-*/wasm-bindgen "$tmp/wasm-bindgen"
export PATH="$tmp:$PATH"
if ! wasm-bindgen --version | grep -q '0.2.100'; then
  echo "pinned wasm-bindgen is not 0.2.100" >&2
  wasm-bindgen --version >&2
  exit 1
fi
bash "$root/rust/fdtd-wasm/build.sh"
git -C "$root" diff --exit-code -- src/wasm/pkg
echo "wasm package matches the commit"
