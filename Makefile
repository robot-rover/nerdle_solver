
.PHONY: all
all: python

cuda/nerdle_cuda/cuda_lib_sh.dll: build/cuda_lib_sh.dll
	cp $< $@

build/cuda_lib.obj: cuda/cuda_lib.cu cuda/cuda_lib.h
	nvcc -rdc=true -c $< -o $@

build/cuda_lib_rt.obj: build/cuda_lib.obj
	nvcc -dlink --lib $^ -lcudart -o $@

build/cuda_lib_st.lib: build/cuda_lib_rt.obj build/cuda_lib.obj
	lib -nologo $^ -out:$@

build/cuda_lib_sh.dll: build/cuda_lib_rt.obj build/cuda_lib.obj
	link /LIBPATH:'${CUDA_PATH}\lib\x64\' /dll /out:$(subst /,\,$@) $(subst /,\,$^) cudart.lib

.PHONY: python
python: cuda/nerdle_cuda/cuda_lib_sh.dll
	cd cuda && pip install .


.PHONY: python_ext
python_ext: build/cuda_lib_st.lib cuda/python_lib.c
	rm -rf cuda/build
	cd cuda && python setup.py build
	cd cuda && pip install .

.PHONY: clean
clean:
	rm -rf cuda/build cuda/*.egg-info build/