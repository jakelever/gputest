#!/bin/bash
set -ex

for i in $(seq 100); do
	python run.py > logs.$i
done

