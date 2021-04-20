### CUDA

#### 学习资料

[并行程序设计] 机械工业出版社

[CUDA编程入门极简教程] https://zhuanlan.zhihu.com/p/34587739

[HPC CODE SIMPLES]https://github.com/JGU-HPC/parallelprogrammingbook

#### CPU + GPU异构计算

​		GPU并不是一个独立运行的计算平台，而需要与CPU协同工作，可以看成是CPU的协处理器，因此当我们在说GPU并行计算时，其实是指的基于CPU+GPU的异构计算架构。在异构计算架构中，GPU与CPU通过PCIe总线连接在一起来协同工作，CPU所在位置称为为主机端（host），而GPU所在位置称为设备端（device)。

​		GPU中，存在大量的流处理器，可以同时启动很多线程，特别适合数据并行的计算密集型任务。CPU运算核心较少，但可以实现复杂的逻辑运算，适合控制密集型任务。

![img](https://github.com/ddmm2020/HPC/blob/main/imgs/hc.png)

​		主机内存(RAM)和设备显存(VRAM)是物理独分离的，通过PCIe总线进行数据传输。这就要求我们我们需要在两个平台上分别分配数据，然后管理他们之间的数据传输。

![内存](https://github.com/ddmm2020/HPC/blob/main/imgs/meory.png)

​		在C++代码中，可采用以下标识符来指定函数的运行位置。

```c++
__global__:主机端调用，CUDA设备上执行。
__host__:主机端调用，主机上执行。
__device__:设备端调用，设备上执行。
```

```c++
// compile:
nvcc main.cu -O2 -o mian
```



典型的CUDA程序执行流程如下：

1. 分配host内存，并进行数据初始化；
2. 分配device内存，并从host将数据拷贝到device上；
3. 调用CUDA的核函数在device上完成指定的运算；
4. 将device上的运算结果拷贝到host上；
5. 释放device和host上分配的内存；

CUDA的核函数(kernel)就是程序在decive上并行执行的部分。GPU上很多并行化的轻量级线程，kernel在device上执行时实际上是启动很多线程，一个kernel启动的所有线程称为一个**网格**（grid），同一个网格上的线程共享相同的全局内存空间，而网格又可以分为很多**线程块**（block）。

```c++
dim3 grid(3, 2);
dim3 block(5, 3);
kernel_fun<<< grid, block >>>(prams...);
```

这部分代码定义了一个网格，包含$3\times2$个线程块，每个线程块又包含了$5\times3$个线程。层次关系如下图所示。

![img](https://github.com/ddmm2020/HPC/blob/main/imgs/device_struct.png)



### Simple Code

```
#include "stdio.h"

__global__ void kernel(){
    // print grid and block dimensions and identifiers
    printf("Hello from thread (%d %d %d)"
        "in a block of dimensions (%d %d %d)"
        "with block identifier (%d %d %d)"
        "spawned in a grid of shape (%d %d %d) \n",
        threadIdx.x,threadIdx.y,threadIdx.z,
        blockDim.x,blockDim.y,blockDim.z,
        blockIdx.x,blockIdx.y,blockIdx.z,
        gridDim.x,gridDim.y,gridDim.z
    );
}

int main(int argc,char * argv[]){
    // set the id of cuda device
    cudaSetDevice(0);

    // define g grid of 1*2*3 = 6 blocks
    // each containing 4*5*6=120 threads
    dim3 grid_dim(1,2,3);
    dim3 block_dim(4,5,6);
    kernel<<<grid_dim , block_dim>>>();

    // synchronize the GPU preventing premature termination
    cudaDeviceSynchronize();
    return 0;
}
```



### device 和 host之间的内存交换

```
/*
* Using CPU + GPU heterogeneous programming to implement PCA algorithm on 
* CelebFaces Attributes (CelebA) Dataset
* Capitalize the first letter to indicate the data on the GPU(device)
*/

#include "stdio.h"
#include "../include/binary_IO.hpp"
#include "../include/bitmap_IO.hpp"
#include "../include/hpc_helpers.hpp"

#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)

template <
    typename index_t,
    typename value_t> __global__ void compute_mean_kernel(
        value_t * Data,
        value_t * Mean,
        index_t num_entires,
        index_t num_features);

int main(int argc,char * argv[]){
    // set the id of cuda device
    cudaSetDevice(0);

    // 202599 grayscale images each of shape 55 x 45
    constexpr uint64_t imgs = 202599, rows = 55, cols = 45;

    // pointer for data matrix and mean vector
    float * data =nullptr,* mean= nullptr;
    cudaMallocHost(&data,sizeof(float)*imgs*rows*cols);  
    cudaMallocHost(&mean,sizeof(float)*rows*cols); 

    //allocate storage on GPU
    float *Data = nullptr,*Mean = nullptr;
    cudaMalloc(&Data,sizeof(float)*imgs*rows*cols); 
    cudaMalloc(&Mean,sizeof(float)*rows*cols); 

    //load data matrix from disk
    TIMERSTART(read_data_from_disk);
    std::string file_name = "./data/celebA_gray_lowres.202599_55_45_32.bin";
    load_binary(data,imgs*rows*cols,file_name);
    TIMERSTOP(read_data_from_disk);

    // copy data to device and reset Mean
    TIMERSTART(data_H2D);
    cudaMemcpy(Data,data,sizeof(float)*imgs*rows*cols,cudaMemcpyHostToDevice); 
    cudaMemset(Mean,0,sizeof(float)*rows*cols); 
    TIMERSTOP(data_H2D);

    //compute mean
    TIMERSTART(compute_mean_kernel);
    // compute_mean_kernel<<<SDIV(rows*cols,32),32>>>(Data,Mean,imgs,rows*cols); 
    compute_mean_kernel<<<SDIV(rows*cols, 32), 32>>>
                    (Data, Mean, imgs, rows*cols);                     
    TIMERSTOP(compute_mean_kernel);

    // Transfer mean back to host
    TIMERSTART(mean_D2H);
    cudaMemcpy(mean,Mean,sizeof(float)*cols*rows,cudaMemcpyDeviceToHost); 
    TIMERSTOP(mean_D2H);

    // Write mean image to disk
    TIMERSTART(write_mean_images_to_disk);
    dump_bitmap(mean,rows,cols,"./imgs/celebA_mean.bmp");
    TIMERSTOP(write_mean_images_to_disk);

    // synchronize the GPU preventing premature termination
    cudaDeviceSynchronize();
    
    // Get rid of the memory
    cudaFreeHost(data); 
    cudaFreeHost(mean); 
    cudaFree(Data); 
    cudaFree(Mean); 
    
    return 0;
}

template <
    typename index_t,
    typename value_t> __global__
void compute_mean_kernel(
    value_t * Data,
    value_t * Mean,
    index_t num_entries,
    index_t num_features) {

    // Compute global thread identifier
    auto thid = blockDim.x*blockIdx.x + threadIdx.x;

    // Prevent memory access violations
    if( thid < num_features){
        // accumulate in a fast register
        // not in slow global memory

        value_t accum = 0;

        // try unrolling the loop with
        // # pragma unroll 32
        // for some additional performance

        for(index_t entry =0;entry < num_entries ;entry++){
            accum += Data[entry*num_features + thid];
        }

        // Write the register once to global memory
        Mean[thid] = accum / num_entries;
    }
}

```
### 数据中心化和内存访问

图像数据中心化计算公式：

 														<img src="https://latex.codecogs.com/svg.image?\bar{v}_{j}^{(i)}=v_{j}^{(i)}-u_{j}" title="\bar{v}_{j}^{(i)}=v_{j}^{(i)}-u_{j}" />

从不同的数据访问方式来思考，我们有两种并行方法对图像进行中心化处理。

1. 按图像每个像素的索引进行并行化均值校正

2. 按每个图像单独进行均值校正

   并行方式的不同，导致代码访存模式不同，第一种方式，线程个数对应的是图像像素，采用连续访存的方式进行迭代，第二种方式，线程对应的是图像张数，看似批处理的批大小更大，但由于其内存访问方式会造成**缓存行失效**，最终导致代码效率相差近3倍。

   代码如下

   方法1：

   ```
   template<
       typename index_t,
       typename value_t> __global__
   void correction_kernel(
       value_t *Data,
       value_t *Mean,
       index_t num_entries,
       index_t num_features){
       auto thid = blockDim.x*blockIdx.x + threadIdx.x;
   
       if(thid < num_features){
           const value_t value = Mean[thid];
           for(index_t entry =0;entry < num_entries;entry++){
               Data[entry*num_features + thid] -= value;
           }
       }
   }
   
   ```

   运行结果：

![image-20210419213817248](https://github.com/ddmm2020/HPC/blob/main/imgs/method1.png)



方法2：

```
template<
    typename index_t,
    typename value_t> __global__
void correction_kernel_ortho(
    value_t *Data,
    value_t *Mean,
    index_t num_entries,
    index_t num_features){
    auto thid = blockDim.x*blockIdx.x + threadIdx.x;

    if(thid < num_features){
        const value_t value = Mean[thid];
        for(index_t feat =0;feat < num_entries;feat++){
            Data[thid*num_features + feat] -= Mean[feat];
        }
    }
}
```

运行结果：

![image-20210419213703340](https://github.com/ddmm2020/HPC/blob/main/imgs/method2.png)

根据前人经验，通常情况下，本地线程标识符`threadIdx.x`的变化速度比`threadIdx.y`,`threadIdx.z`的变化都要快，因此，依赖threadIdx.x的变量应该总识操作索引方案中的最低有效位。

**凝聚**方案：

```
Data[threadIdx.y*matrix_width + threadIdx.x] = value;
```

通常远快于**非凝聚**方案

```
Data[threadIdx.x*matrix_width + threadIdx.y] = value;
```



### 内存访问速度

​      和C/C++里面的堆内存，栈内存，全局存储区，静态存储区，常量区等相似，CUDA也分为很多不同的内存区。CUDA通具有下面几种内存区。

- 寄存器: 寄存器是GPU上运行速度最快的内存空间，核函数声明的一个没有其他修饰符的自变量，通常存储在寄存器中。

- 共享内存: 核函数中使用 `__shared__`修饰符修饰的变量，存放在共享内存中，它具有更高的带宽和更低的延迟。每一个SM都有一定数量的由线程块分配的共享内存，共享内存是线程之间相互通信的基本方式，一个块内的线程通过共享内存进行合作。可通过`__syncthreads()`进行同步。

- 本地内存：对于每个线程，局部存储器也是私有的。如果寄存器被消耗完，数据将被存储在本地内存中。如果每个线程用了过多的寄存器，或声明了大型结构体或数组，或者编译期无法确定数组的大小，线程的私有数据就有可能被分配到本地内存中。本地内存速度访问也比较慢。

- 常量内存：常量内存保存设备内存中，并在每个SM专用的常量缓存中缓存。使用如下修饰符来修饰：`__constant__`常量变量必须在全局空间内和所有核函数之外进行声明，对所有计算能力的设备只可以声明64KB的内存，对同一编译单元的所有核函数可见。核函数对于常量内存的操作只有读，常量内存需要在主机端使用下面的函数来初始化。

- 全局内存：全局内存是GPU中最大，延迟最高且最常使用的内存。可以被所有SM被访问到，并且贯穿应用程序的整个生命周期。在device代码中，采用`__device__`声明。

​        每一个核函数中的线程都有自己的私有本地内存。每一个线程块有自己的共享内存，并对同一线程块中的所有线程都可见。所有的线程都可以访问全局内存。所有线程都能访问只读内存空间由：常量内存空间核纹理内存。**纹理内存**为各种数据布局提供了不同的寻址模式核滤波模式。

<img src="https://github.com/ddmm2020/HPC/blob/main/imgs/memory_struct.jpg" alt="memory_struct" style="zoom: 200%;" />

