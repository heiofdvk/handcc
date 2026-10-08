#!/bin/zsh
# Build HandCC.app next to this script. The app is only a wrapper: its
# Info.plist carries a camera-usage description so macOS shows the camera
# permission prompt, and its launcher runs handcc.py with this folder's .venv.
# Rebuilding changes the ad-hoc signature, so macOS may ask for the camera again.
set -e
cd "${0:A:h}"
rm -rf HandCC.app
mkdir -p HandCC.app/Contents/MacOS
cp app/Info.plist HandCC.app/Contents/Info.plist
cp app/handcc HandCC.app/Contents/MacOS/handcc
chmod +x HandCC.app/Contents/MacOS/handcc
codesign --force --sign - HandCC.app
echo "Built HandCC.app (ad-hoc signed). Start it with: open HandCC.app"
