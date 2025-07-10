# Lecture 7 : Advanced Quantizatio

![스크린샷 2025-07-09 오후 2.19.14.png](Lecture%207%20Advanced%20Quantizatio%2021d21391dcff808c9163d57720789035/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-07-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.19.14.png)

Dynamic Quantiaztion : 

Float wiehgt → Q → Multiplication > Accu > Rescale 

Float Activation → Q → Multiplication > Accu > Rescale 

어디서 Quntization을 수행하는지, 그로인해 성능을 높일 수 있는지가 중요하다. 

activation / weight 어디서 quntizaztion을 하는가는 메모리, 성능에 크게 영향을 미친다. 

모든 model일 quantization이 가능한가? 더 잘되는 모델이 있는가?? 

- 각각의 model에 수많은 quantizaion 방식이 있고, 여러 방식으로 가능하다.

## Dynamic Quantization

![스크린샷 2025-07-09 오후 2.28.57.png](Lecture%207%20Advanced%20Quantizatio%2021d21391dcff808c9163d57720789035/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-07-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.28.57.png)

`Y = X . W`   : 원래의 부동소수점 행렬 곱셈 

- `X`: 입력 행렬 (활성화값)
- `W`: 가중치 행렬
- `Y`: 출력 행렬

부동소수점의 연산을 Scale factor를 이용하여 Int로 변환하여 연산을 수행 할 수 있다.  

`Y = (Sx * Xint) . (Wint * Sw )`  

- `Xint`: X를 정수로 양자화한 값
- `Wint`: W를 정수로 양자화한 값
- `Sx`: X의 스케일 팩터 (scaling factor)
- `SW`: W의 스케일 팩터

`Y = Sx * (Xint . Wint) * Sw`  

[**Matmul in trition**](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

최적화된 연산을 위해서, BF16의 matMul 연산을 내부적으로, int8로 치환하여 사용하게 됨. Overflow를 대비하여 X.W 값은 Int32 변환 후 rescale param을 이용하려 BF16으로 변환 ( torch.kernel.tuned_fused_int_mm_mul )

![스크린샷 2025-07-09 오후 2.43.55.png](Lecture%207%20Advanced%20Quantizatio%2021d21391dcff808c9163d57720789035/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-07-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.43.55.png)

중간에 연산 결과를 DRAM에 저장 할 필요 없이, SRAM내에 저장이 가능핟. 

GPU 메모리를 고려하여 입력 사이즈를 결정. 

## Int8 Weight Only Quantization

단순하게 DataType을 변환하는 것은 오히려 성능 저하를 만들 수 있음. 

- Scale 연산에 의한 오버에드
- GPU Utilization 문제 : Blocksize limitation
    - 64개의 Block을 처리하도록 제약이 있는 상태에서, Int8을 사용하면 모든 GPU Core를 효율적으로 활용할 수 없게 된다.
    - 
    
    ![스크린샷 2025-07-09 오후 4.29.56.png](Lecture%207%20Advanced%20Quantizatio%2021d21391dcff808c9163d57720789035/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-07-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.29.56.png)
    

Kernel 연산에서의 최적화로 Compile 단에서 최적활르 진행하고. 속도를 개선 할 수 있다. 

## Int4 Weight Only Quantization