#!/bin/bash

rm -r ./build/

python setup.py build_ext
python setup.py install

rm -r ./build/