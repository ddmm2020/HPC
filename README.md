## Summary

MPI is  a communication standard, and parallel with GPU is actually complementary. GPU is responsible for parallel computing, and MPI is responsible for communication between multiple GPUs.

In a single-node multi-GPUs or multi-nodes multi-GPUs cluster, CUDA supports mpi to communicate directly between GPUs, without the need to send data back to the host and then to another GPU device, which can effectively shorten the GPU Inter-communication.



## MPI Part

1. blocking communication
2. non-blocking communication
3. broadcast
4. some code-examples



## CUDA Part

1. heterogeneous computing
2. memory access
3. some code-examples based on CelebFaces Attributes (CelebA) Dataset

