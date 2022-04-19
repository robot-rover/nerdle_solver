
.PHONY: all
all: python

cuda/cuda_lib.obj: cuda/cuda_lib.cu
	nvcc -rdc=true -c $< -o $@

cuda/cuda_lib_rt.obj: cuda/cuda_lib.obj
	nvcc -dlink --lib $^ -lcudart -o $@

cuda/cuda_lib_rt.lib: cuda/cuda_lib_rt.obj cuda/cuda_lib.obj
	lib -nologo -out:$@ $^

.PHONY: python
python: cuda/cuda_lib_rt.lib cuda/python_lib.c
	rm -rf cuda/build
	cd cuda && python setup.py build
	cd cuda && pip install .

.PHONY: clean
clean:
	rm -rf cuda/*.obj cuda/*.exp cuda/*.lib nerdle_lib cuda/build cuda/*.egg-info