#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
  echo 'This project targets Ubuntu Linux x86_64. Do not run camera builds in Windows.' >&2
  exit 1
fi
command -v cmake >/dev/null || { echo 'Install build-essential cmake libopencv-dev usbutils first.' >&2; exit 1; }
printf '%s  %s\n' 'cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57' \
  'third_party/onnxruntime-linux-x64-1.31.0.tgz' | sha256sum -c -
tar -xzf third_party/onnxruntime-linux-x64-1.31.0.tgz -C third_party
sha256sum -c models/SHA256SUMS
echo 'Local CPU C++ runtime ready. No Python/CUDA installation is needed.'
