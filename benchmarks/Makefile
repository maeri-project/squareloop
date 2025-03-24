clean:
	rm -rf results

timeloop_dse:
	export PYTHONPATH="./script:$PYTHONPATH"
	python3 ./script/timeloop_mapper_all_arch.py results_bw_layout_modeling

layoutloop_dse:
	export PYTHONPATH="./script:$PYTHONPATH"
	python3 ./script/layoutloop_mapper_all_arch.py results_precise_layout_modeling