#!/bin/bash
set -ex

# Create a symlink for safetensors-lib
ln -sf ../../safetensors safetensors-lib
# Modify cargo.toml to include this symlink
sed -i 's/\.\.\/\.\.\/safetensors/\.\/safetensors-lib/' Cargo.toml
# Build the source distribution
python setup.py sdist
