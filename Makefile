
.PHONY: all
all: nerdle_lib

cuda/cuda_lib.obj: cuda/cuda_lib.cu
	nvcc -rdc=true -c $< -o $@

cuda/cuda_lib_rt.obj: cuda/cuda_lib.obj
	nvcc -dlink --lib $^ -lcudart -o $@

cuda/cuda_lib_rt.lib: cuda/cuda_lib_rt.obj cuda/cuda_lib.obj
	lib -nologo -out:$@ $^

python: cuda/cuda_lib_rt.lib
	cd cuda && python setup.py build
	cd cuda && python setup.py install

nerdle_lib: cuda/lib.o
	nvcc $^ -o $@

.PHONY: clean
clean:
	rm -f cuda/*.obj cuda/*.exp cuda/*.lib nerdle_lib