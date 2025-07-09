Pytorch optimizing 

Runtime optimization : faster 

- 최대한 병렬화를 진행하는 것이 당연히 좋지만, 불가능한 경우는 존재한다.

Memory Optimization : smaller 

Focus on : How pytorch optimizing runtime

### multi_tensor_apply

```python
# Tensor operation, python code 
add(Tensor self, Tesnsor oter, *, Scalar alpha = ) -> Tensor 

#CUDA C++ example 
__device__ void add_kernel(float* self, float* other, float* rest, float alpha=1) 

# How about Tensor List Operation, python code 
_foreach_add(Tensor[] self, Tensor[] other, *, Scalar alpha = 1) -> Tensor[]
```

attempt1, std::vector 

```cpp
__device__ void _foreach_add_kernle(std::vector<float*> self, std::vector<float*> other, std::vector<float*>. res, float alpha = 1)

```

CUDA Doset work, coudn't recog vector

Attempt2, float** 

```cpp
__device__ void _foreach_add_kernle(float** self, float** other, float** res, float alpha = 1)

```

Illegal Memory Access (IMA), outer * is a CPU Address
	# Tensor live on CUDA, float*은 GPU Memory 안에 존재, float**는 CPU memory안에 존재
	# tensor list address exist on CPU. So parameter address on CPU move to GPU 

![스크린샷 2025-06-17 오후 1.59.51.png](attachment:c47038d7-62f4-4317-a7f7-42f5919ee09d:스크린샷_2025-06-17_오후_1.59.51.png)

Attemp3 : pass by chonky boi ( not reference )

```cpp
Struct TenstorListMetadata{ 
	const float* addresses[3][NUM_TENSOR]; 
}; 

// <add all the address into the struct> 
__device__ void _foreach_add_kernel(TensorListMetadta tlm, float alpha=1){...}

// CI Code 
params = [torch.rand(2,3,device="cuda") for _ in range(N)] 
torch._foreach_norm(params,ord=1)
torch.cuda.synchronize()
// boob start on N = 424
```

it worksm but has some problem. It's work utill NUM_TENSOR < 424 
CUDA Kerneal argument space have a max limit of 4KB 

> To resolve them. we need to make more trips, Batch
> 

memcpy the list of address to CUDA 

```cpp
__device__ void _foreach_add_kernel(float** self, float** other, float** resl, float alpha = 1) {...}

/// memcpy from CPU to GPU. but it. is expensive 
/// Tensor can be 100~1000. it can be long, if lot o

// Conclusion : we will be doing a mix of struct + memcpy 
m// memcpy once, and distribute several time. 
```

There are lot’s of optimization List Tensor… up to developer

unified memory : memory map used in GPU.

Page falting / load storage access 

### torch.compile()

optimizer = torch.optim.AdamW(param) 

@torch.compile(fullgraph=False)

def compiled_step()