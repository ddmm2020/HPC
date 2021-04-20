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



template<
    typename index_t,
    typename value_t,
    uint32_t chunk_size = 32> __global__
void shared_covariance_kernel(
    value_t * Data,
    value_t * Cov,
    index_t num_entries,
    index_t num_features){
        // first index in a window of width chunk size
        const index_t base_x = blockIdx.x * chunk_size;
        const index_t base_y = blockIdx.y * chunk_size;

        //local 
        const index_t thid_y = threadIdx.y;
        const index_t thid_x = threadIdx.x;

        // global thread identifiers
        const index_t x = base_x + thid_x;
        const index_t y = base_y + thid_y;

        // optional early exit for tiles above the diagonal
        if(base_x > base_y) return;

        // allocate shared memory
        __shared__ value_t cache_x[chunk_size][chunk_size];
        __shared__ value_t cache_y[chunk_size][chunk_size];

        // compute the number of chunks to be computed
        const index_t num_chunks = SDIV(num_entries,chunk_size);

        // accumulated calue of scalar product
        value_t accum =0;

        // for each chunk
        for (index_t chunk = 0;chunk < num_chunks;chunk++){
            
            //assign thread IDs to rows and columns
            const index_t row = thid_y + chunk*chunk_size;
            const index_t col_x = thid_x + base_x;
            const index_t col_y = thid_x + base_y;

            // check ig valid row or column indices
            const bool valid_row = row < num_entries;
            const bool valid_col_x = col_x < num_features;
            const bool valid_col_y = col_y < num_features;
            
            cache_x[thid_y][thid_x] =  valid_row*valid_col_x?Data[row*num_features + col_x] : 0;
            cache_y[thid_y][thid_x] =  valid_row*valid_col_y?Data[row*num_features + col_y] : 0;

            // Ensure all threads hava finished writing to shared memory
            __syncthreads();

            // optional early exit
            if(x <= y){
                // evaluate the scalar product
                for (index_t entry=0;entry<chunk_size;entry++){
                    accum += cache_y[entry][thid_y] * cache_x[entry][thid_x];
                }
            }

            // ensure shared memory safety be overwritten
             __syncthreads();
        }

        if( y < num_features && x<=y){
            Cov[y*num_features + x] = Cov[x*num_features + y] = accum /num_entries;
        }

    }



int main(int argc,char * argv[]){
    // set the id of cuda device
    cudaSetDevice(0);

    // 202599 grayscale images each of shape 55 x 45
    constexpr uint64_t imgs = 202599, rows = 55, cols = 45;

    // pointer for data matrix and mean vector
    float * data =nullptr,* cov= nullptr;
    cudaMallocHost(&data,sizeof(float)*imgs*rows*cols);  
    cudaMallocHost(&cov,sizeof(float)*rows*cols); 

    //allocate storage on GPU
    float *Data = nullptr,*Mean = nullptr ,* Cov = nullptr;
    cudaMalloc(&Data,sizeof(float)*imgs*rows*cols); 
    cudaMalloc(&Mean,sizeof(float)*rows*cols); 
    cudaMalloc(&Cov,sizeof(float)*rows*cols*rows*cols);

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

    // compute covariance matrix
    TIMERSTART(covariance_kernel)
    dim3 blocks(SDIV(rows*cols, 32), SDIV(rows*cols, 32));
    dim3 threads(32, 32, 1);
    shared_covariance_kernel<<<blocks, threads>>>
                       (Data, Cov, imgs, rows*cols);                      
    TIMERSTOP(covariance_kernel)          

    // Transfer mean back to host
    TIMERSTART(cov_D2H)
    cudaMemcpy(cov, Cov, sizeof(float)*rows*cols*rows*cols,
               cudaMemcpyDeviceToHost);
    TIMERSTOP(cov_D2H)

    // Write mean image to disk
    TIMERSTART(write_mean_images_to_disk);
    dump_bitmap(cov,rows,cols,"./imgs/celebA_cov.bmp");
    TIMERSTOP(write_mean_images_to_disk);

    // synchronize the GPU preventing premature termination
    cudaDeviceSynchronize();
    
    // Get rid of the memory
    cudaFreeHost(data); 
    cudaFreeHost(cov); 
    cudaFree(Data); 
    cudaFree(Mean);
    cudaFree(Cov); 
    
    return 0;
}

