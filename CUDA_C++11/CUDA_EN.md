### CUDA

#### Related Materials

Bertil Schmidt 《Parallel Programming: Concepts and Practice》 

 https://zhuanlan.zhihu.com/p/34587739

https://github.com/JGU-HPC/parallelprogrammingbook

#### CPU + GPU Heterogeneous computing

​		GPU is not an independent computing platform, and needs to work with the CPU. It can be regarded as a coprocessor of the CPU. When we talk about GPU parallel computing, we actually refer to a heterogeneous computing architecture based on CPU+GPU.  In a heterogeneous computing architecture, the GPU and the CPU are connected through a PCIe bus to work together. The location of the CPU is called the host, and the location of the GPU is called the device.

​		  In GPU, there are a large number of stream processors, which can start many threads at the same time, which is especially suitable for data-parallel computing-intensive tasks. The CPU has fewer computing cores, but it can implement complex logic operations and is suitable for control-intensive tasks.

![img](https://github.com/ddmm2020/HPC/blob/main/imgs/hc.png)

  		

​		Host memory (RAM) and device video memory (VRAM) are physically separate, and data transmission is carried out through the PCIe bus. This requires us to allocate data on the two platforms separately, and then manage the data transmission between them.

![内存](https://github.com/ddmm2020/HPC/blob/main/imgs/meory.png)

​		

​			In C++ code, the following identifiers can be used to specify the location of the function.

```c++
__global__: Called on the host, executed on the CUDA device.
__host__: Called on the host, executed on the host.
__device__:Called on the device, executed on the device.
```

```c++
// compile:
nvcc main.cu -O2 -o mian
```



The typical CUDA program execution flow is as follows:

1. Allocate host memory and initialize data;
2. Allocate device memory and copy data from host to device;
3. Call the CUDA kernel function to complete the specified operation on the device;
4. Copy the result of the operation on the device to the host;
5. Release the memory allocated on the device and host;

​    

​		CUDA kernel is the part of the program that executes in parallel on the decive. There are many parallelized lightweight threads on the GPU. When the kernel is executed on the device, many threads are actually started. All threads started by a kernel are called a **grid** , threads on the same grid share the same global memory space, the grid can be divided into many thread **  blocks**.

```c++
dim3 grid(3, 2);
dim3 block(5, 3);
kernel_fun<<< grid, block >>>(prams...);
```

​		

​		This part of the code defines a grid that contains $3\times2$ thread blocks, and each thread block contains $5\times3$ threads. The hierarchical relationship is shown in the figure below.

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



### Memory exchange between device and host

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
### Memory Access Method

Image centralization(Subtract the mean vector of the data from each image vector) :

 														$${\overline{v}_{j}^{(i)}} = {v_{j}^{(i)} - u_{j}}$$

Thinking from different data access methods, we have two parallel methods for centralized processing of images.

1. Parallel mean correction according to the index of each pixel of the image
2. Perform mean correction for each image separately

​         The different parallel methods lead to different code access modes. In the first method, the number of threads corresponds to image pixels, and iteratively **uses continuous memory access**. In the second method, threads correspond to the number of images. It seems The batch size of batch processing is larger, but due to its memory access method, it will cause **cache line invalidation**, which ultimately leads to a nearly 3 times difference in code efficiency.

Code samples

method one：

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

result：

![image-20210419213817248](https://github.com/ddmm2020/HPC/blob/main/imgs/method1.png)



method two：

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

result：

![image-20210419213703340](https://github.com/ddmm2020/HPC/blob/main/imgs/method2.png)

​     According to previous experience, under normal circumstances, the local thread identifier `threadIdx.x` changes faster than `threadIdx.y` and `threadIdx.z`. Therefore, variables that depend on` threadIdx.x should` always be the least significant bit in the operation index scheme.

**condensed** scheme：

```
Data[threadIdx.y*matrix_width + threadIdx.x] = value;
```

Usually faster than the **non-condensed** scheme:

```
Data[threadIdx.x*matrix_width + threadIdx.y] = value;
```



### Memory Access Speed

​      Similar to the heap memory, stack memory, global storage area, static storage area, constant area, etc. in C++. CUDA is also divided into many different memory areas. CUDA has the following memory areas.

- Register: A register is the fastest memory space on the GPU. An argument declared by a kernel function without other modifiers is usually stored in a register.Register memory is usually small.

- Shared memory: Variables modified by the `__shared__` in kernel functions are stored in shared memory, which has higher bandwidth and lower latency. Each SM has a certain amount of shared memory allocated by thread blocks. Shared memory is the basic way for threads to communicate with each other. Threads in a block cooperate through shared memory. It can be synchronized through `__syncthreads()`.

- Local memory: For each thread, local memory is also private. If the registers are consumed, the data will also be stored in local memory. If the thread uses too many registers, or declares a large structure or array, or the size of the array cannot be determined at compile time, the thread's private data may be allocated to local memory. Local memory speed access is also relatively slow.

- Constant memory: The constant memory resides in the device memory and is cached in the constant cache dedicated to each SM. Use the following modifiers to modify: `__constant__` constant variables must be declared in the global space and outside of all core functions. Only 64KB of memory can be declared for all computing power devices, which are visible to all core functions of the same compilation unit. The kernel function can only read the constant memory, and the constant memory needs to be initialized on the host side using the following function.

- Global memory: Global memory is the largest, highest latency, and most commonly used memory in the GPU. It can be accessed by all SMs and runs through the entire life cycle of the application. In the device code, use the `__device__` declaration.

​        Each thread in the kernel function has its own private local memory. Each thread block has its own shared memory and is visible to all threads in the same thread block. All threads can access global memory. All threads can access the read-only memory space by: constant memory space and texture memory. **Texture memory** provides different addressing modes and filtering modes for various data layouts.

<img src="https://github.com/ddmm2020/HPC/blob/main/imgs/memory_struct.jpg" alt="memory_struct" style="zoom: 200%;" />



