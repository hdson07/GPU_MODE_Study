# lecture8 : CUDA Performance Gatcha

DRAM vs SRAM 

- DRAM : 1 transistor, 1 capacistor
- SRAM : 6 transistors , SRAM is **faster but, more expensive.**

Performance Checklist 

- Coalesced Global Memory Access
- Maximize occupancy : block size, threads num…
- Understand if memory or compute bound
- Minimize control divergence : scheduling threads, warp
- Tiling of reused data
- Privatization
- Thread Coarsening
- Rewrite your algorithm using better math

## Memory latencies

Global memory >  L2 cache > L1 cache > shared memory 의 cycle 고려

[latency, Stupid](http://www.stuartcheshire.org/rants/latency.html), TP is easy, Latency is not 

많은 Thread를 이용하여 동시에 여러개를 수행하는 것은 쉽지만, 1개의 Thread의 수행 시간을 줄이는 것은 쉽지 않음. 

Memory Coalescing

연산 각각의 latency를 줄일 수 있지만, 최적화 기법을 통해 load 뒤로 숨길 수 있음.

How ofen hit DRAM? how long to load data? L1 cache TP

[coalesce.cu](lecturenote/coalesce.cu) 

메모리 접근 패턴이 성능에 미치는 영향. ( coalesced vs non-coaleced ) 

- 연속/병합된 메모리 접근과 불연속/부작위 패턴의 메모리 접근 비교

## Occupancy

Block Size에 따라서 Occupancy가 바뀔 수 있으며, 최적의 Block Size를 찾아야 한다. 

Poor Occupancy를 만드는 구조. 

- Tile quantization: matrix dimensions are not divisible by the thread block tile size.
- Wave quantization: total number of tiles is not divisible by number of SM on GPU

cudaOccupancyMaxPotentialBlockSize 함수를 통해, 최적의 block size를 찾아서 사용하는 것이 효율적. 

[occupancy.cu](lecturenote/occupancy.cu) 

GPU 리소스를 최대한 효율적으로 활용하기 위한 최적화 방법 

- Occupancy = (활성 warp 수) / (최대 가능한 warp 수), SM에서 처리할 수 있는 warp 비율
- cudaOccupancyMaxPotentialBlockSize 를 이용하여 최적화된 block size 계산
- 레지스터 사용량, 메모리 사용량, 하드웨어 제약를 고려하여 최적화 값 계산

## Arithmetic Intensity

Math limited if : FLOPS / bytes > math TP / memory BW 

ReLU : f(x) = max(0, x) , 대표적인 load 중심 연산 → 최대한 다른 연산과 묶어줘야함. 

- FP32 cases
    - 1 read : 1 ops : 1 write  = 4bytes : 1ops : 4bytes
    - Arithmetic Intensity = 1ops / 8 bytes
- FP16 cases
    - 1 read : 1 ops : 1 write  = 2bytes : 1ops : 2bytes
    - Arithmetic Intensity = 1ops / 4 bytes
    - HW 스펙 상 Best boud가 1/4 라면, FP16 case가 최적

Matmul ,C[M,K] = A[M,N] * B[N,K]

- FLOPS : M * K * 2N ( M, K 행에 대한 N 번의 곱, N번의 합 )
- Bytes = M*N + N*K
- AI : 2MNK / ( MN + NK + MK )

# TL;DR

- Bandwidth Bound Kernels: Fuse, quantize, compile
- Compute Bound Kernels: Write a better algorithm

---

## Minimize control divergence

single Thread의 경우 큰 영향이 없지만, Thread가 많아 질 수록 Branch의 영향은 매우 커짐 

[divergence.cu](lecturenote/divergence.cu) 

Warp Divergence 문제, SIMT(Single Instruction, Multiple Threads)에서 주로 발생 

- Warp Divergence : Warp(32 Threads)내에 모든 Threads가 같은 명령어를 실행해야 하지만, 분기가 발생하여 Serialization이 발생할 수 있음.
- 분기 조건은 산술 연산으로 변환하여 최적화 가능

## Thread Coarsening
[coarsening.cu](lecturenote/coarsening.cu) 

Thread coarsening 기법을 사용한 벡터 덧셈 최적화, 각 Thread가 최대한 많은 연산 처리 

- Thread 1:1 → 1:2 mapping
- 전체 Thread 수 감소, 명령어 오버헤드 감소, Thread 스케줄링 오버헤드 감소.
- overhead를 방지하기 위한 Bound 연산 필요, Coarsening Factor가 성능에 큰 영향

## Privatization
[privatization.cpp](lecturenote/privatization.cpp) 
Histogram계산에서 Atomic Operation 최적화 

- 전역 메모리에 직접 atomic 연산을 하게되면, 심각한 contention(race)을 유발
- extern __shared__ int private_hist[];   공유 메모리에 private 히스토그램을 넣어 충돌 최소화

[privatization2.cu](lecturenote/privatization2.cu) 

Window sum, moving Average를 공유 메모리를 활용하여 최적화 하는 방법 

- 각 index별로 Window Sum을 계산 할 때 마다, 인접한 메모리를 반복하여 load
- 구간 Data를 shared memory에 저장 후, Thread별로 연산

## Tilling
[Tilling.cu](lecturenote/Tilling.cu) 
Matrix Multiplication 최적화 예제. 

- 행 단위, 순차적인 연산을 타일 단위로 데이터 블록으로 나누어 처리.
- 공유 메모리에 타일 데이터를 올려두고, 재사용.

## softmax(QK^T) V

nomalization factor, 3 memory accesses per element ( 2read, 1store ) 

→ store result of exponent in float64 but not great

## Safe softmax

4 memory access per element ( 3 read, 1store )