/*
* Using CPU + GPU heterogeneous programming to implement PCA algorithm on Minst
* Capitalize the first letter to indicate the data on the GPU(device)
*/

#include "stdio.h"
#include "../include/binary_IO.hpp"
#include "../include/bitmap_IO.hpp"
#include "../include/hpc_helpers.hpp"

#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)
#define COALESCED_ACCESS

template <
    typename index_t,
    typename value_t> __global__ void compute_mean_kernel(
        value_t * Data,
        value_t * Mean,
        index_t num_entries,
        index_t num_features);

template <
    typename index_t,
    typename value_t> __global__ void correction_kernel(
        value_t * Data,
        value_t * Mean,
        index_t num_entries,
        index_t num_features);

template <
    typename index_t,
    typename value_t> __global__ void correction_kernel_ortho(
        value_t * Data,
        value_t * Mean,
        index_t num_entries,
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
    
    compute_mean_kernel<<<SDIV(rows*cols, 32), 32>>>
                    (Data, Mean, imgs, rows*cols);                       
    TIMERSTOP(compute_mean_kernel);


    // correct mean
    TIMERSTART(correction_kernel)
    #ifdef COALESCED_ACCESS
    correction_kernel<<<SDIV(rows*cols, 32), 32>>>
                       (Data, Mean, imgs, rows*cols);
    #else
    correction_kernel_ortho<<<SDIV(imgs, 32), 32>>>
                       (Data, Mean, imgs, rows*cols);
    #endif
    TIMERSTOP(correction_kernel)                    

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

