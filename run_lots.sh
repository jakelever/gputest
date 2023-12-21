#!/bin/bash
set -ex

num_epochs=16

for learning_rate in 1e-3 5e-4 1e-4 5e-5 1e-5; do
	for batch_size in 8 16 32; do
		python run_tokenclassification.py --learning_rate $learning_rate --batch_size $batch_size --num_epochs $num_epochs > log.$learning_rate.$batch_size.$num_epochs
	done
done

