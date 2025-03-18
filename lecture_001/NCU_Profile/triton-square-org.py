import numpy as np
import torch
import triton
import triton.language as tl

@triton.jit
def square_kernel(
    x_ptr,  # 입력 배열의 포인터
    output_ptr,  # 출력 배열의 포인터
    n_elements,  # 배열의 요소 수
    BLOCK_SIZE: tl.constexpr,  # 블록 크기
):
    # 프로그램 ID를 통해 각 스레드의 작업 할당
    pid = tl.program_id(axis=0)
    # 각 스레드가 처리할 요소의 시작 인덱스 계산
    block_start = pid * BLOCK_SIZE
    # 각 스레드가 처리할 요소의 오프셋 계산
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 처리할 요소 수를 벗어나는 스레드는 마스킹
    mask = offsets < n_elements
    # 입력 배열에서 데이터 로드
    x = tl.load(x_ptr + offsets, mask=mask)
    # 제곱 연산 수행

    
    # output = x * x
    # # 결과를 출력 배열에 저장
    # tl.store(output_ptr + offsets, output, mask=mask)

    # 1️⃣ Coalesced Memory Access (Offset Alignment)
    aligned_offsets = tl.multiple_of(offsets, 16)

    # 2️⃣ FMA 활용 (x * x 연산 최적화)
    output = tl.fma(x, x, 0)  # x * x + 0

    # 3️⃣ Warp Divergence 최소화 (Mask 활용 개선)
    output = tl.where(mask, output, 0)

    # 4️⃣ Global Memory Store 최적화
    tl.store(output_ptr + aligned_offsets, output)

def square(x):
    """
    입력 텐서의 모든 요소를 제곱하는 함수
    """
    # 출력 텐서 초기화
    output = torch.empty_like(x)
    # 요소 수 계산
    n_elements = x.numel()
    # 블록 크기 설정 (성능 최적화를 위해 조정 가능)
    BLOCK_SIZE = 1024
    # 그리드 크기 계산 (필요한 블록 수)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    # 커널 실행
    square_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    return output

def profile_triton_square():
    """
    Triton 제곱 연산의 성능을 프로파일링하는 함수
    """
    print("Triton Square 연산 프로파일링 시작")
    
    # 다양한 크기의 입력으로 테스트
    sizes = [1024, 8192, 65536, 524288, 4194304]
    
    for size in sizes:
        print(f"\n크기: {size} 요소")
        
        # 입력 데이터 생성
        x = torch.randn(size, device='cuda')
        
        # 워밍업 (JIT 컴파일 및 GPU 초기화)
        for _ in range(10):
            result = square(x)
        
        # 성능 측정
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        iterations = 100
        start.record()
        for _ in range(iterations):
            result = square(x)
        end.record()
        torch.cuda.synchronize()
        
        # 평균 실행 시간 계산 (밀리초)
        elapsed_time = start.elapsed_time(end) / iterations
        print(f"평균 실행 시간: {elapsed_time:.4f} ms")
        
        # 처리 성능 계산 (초당 처리되는 요소 수)
        throughput = size / (elapsed_time / 1000)
        print(f"처리량: {throughput/1e9:.4f} 십억 요소/초")
        
        # 결과 검증
        cpu_result = x.cpu().numpy() ** 2
        gpu_result = result.cpu().numpy()
        max_error = np.max(np.abs(cpu_result - gpu_result))
        print(f"최대 오차: {max_error}")
    
    print("\nTriton Square 연산 프로파일링 완료")

if __name__ == "__main__":
    profile_triton_square()
