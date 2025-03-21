NVCC        = nvcc
NVCC_FLAGS  = -O3 -arch=sm_75 --compiler-options -Wall
LD_FLAGS    = -lcudart
EXE	        = vector_addition
OBJ	        = vector_addition.o

default: $(EXE)

vector_addition.o: vector_addition.cu
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $< -o $@ $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)