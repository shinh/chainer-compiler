#!/bin/bash

set -eux

. .chainerci/run_onnx_setup.sh

python3 -m pip list -v