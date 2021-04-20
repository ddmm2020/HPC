### Related Materials

https://mpitutorial.com/tutorials/

https://www.mpich.org/static/docs/latest/

Bertil Schmidt 《Parallel Programming: Concepts and Practice》



### Installation

MPI

1. MPI no ch4 netmod selected

   add configuration options：--with-device=ch4:ofi  and -disable-fortran
2. 43858_ recipe for target 'install' failedMakefil
   make install  no enough permissions . Use sudo to solve this problem
3. HYDU_create_process (utils/launch/launch.c:73): execvp error on file a.out (No such file or director）

   execute after compilation ：mpirun -np 4 ./a.out



MPI Compile：

```
mpic++   main.cpp  -o ./test
```

Use -np to specify the number of MPI processes at runtime

````
mpirun -np 4 ./test
````



simple code of MPI

```
#include "stdio.h"
#include "mpi.h"

int main(int argc, char * argv[]){
    int myId, nProcs;
    // Initialize MPI
    MPI::MPI_Init(&argc, &argv);
    // Get the number of processes
    MPI::MPI_Comm_size(MPI_COMM_WORLD,&nProcs);
    // Get the ID of the process
    MPI::MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    printf("rank%d  of%d: Hello,world!\n",myRank, nProcs);
    // Terminate MPI
    MPI::MPI_Finalize();
    return 0;
}

```



### MPI ping-pong communication (blocking communication)

 Implementation of the communication process shown in the figure below

![ping_pong](https://github.com/ddmm2020/HPC/blob/main/imgs/ping_pong.png)

```
#include "stdio.h"
#include "iostream"
#include "mpi.h"

int main(int argc, char * argv[]){
    // Initialize MPI
    MPI::Init(argc, argv);
    // Get the number of processes
    int nProcs = MPI::COMM_WORLD.Get_size();
    // Get the ID of the process
    int my_id = MPI::COMM_WORLD.Get_rank();

    if (argc < 2 ){
        if(!my_id){
            std::cout<<"ERROR: This program needs to set ping_pong_num parameter." <<std::endl;
        };
        MPI::COMM_WORLD.Abort(1);
    }

    if(nProcs %2 != 0 ){
        // only the first process prints the output message
        if(!my_id){
            std::cout<<"ERROR: The nuber of processes must be a multiple of 2"<<std::endl;
        }

        MPI::COMM_WORLD.Abort(1);
    }
    int num_ping_pong = atoi(argv[1]);
    int ping_pong_count = 0;

    int partner_id;
    bool odd = my_id %2;

    partner_id =  odd ? my_id -1: my_id + 1;

    while(ping_pong_count < num_ping_pong){
        // First receive the ping and then send the pong
        ping_pong_count++;

        if(odd){
            // Recv pong
            MPI::COMM_WORLD.Recv(&ping_pong_count,1,MPI::INT,partner_id,0);
            printf("process %d, recv an int \n",my_id);
            // Send ping
            MPI::COMM_WORLD.Send(&ping_pong_count,1,MPI::INT,partner_id,0);
            printf("process %d send an int\n", my_id);
        }else{
            MPI::COMM_WORLD.Send(&ping_pong_count,1,MPI::INT,partner_id,0);
            printf("process %d send an int\n", my_id);
            MPI::COMM_WORLD.Recv(&ping_pong_count,1,MPI::INT,partner_id,0);
            printf("process %d, recv an int \n",my_id);
        }
    }

    // Terminate MPI
    MPI::Finalize();
    return 0;
}

// compile
// mpic++ main.cpp -o ./test
// Run
// mpirun -np 8 ./test  2
```



### Non-blocking communication

 Implementation of the communication process shown in the figure below

![非阻塞通讯](https://github.com/ddmm2020/HPC/blob/main/imgs/non-blocking.png)

```
// Blocking communication causes deadlock
#include "stdio.h"
#include "iostream"
#include "mpi.h"
#include "time.h"

int main(int argc, char * argv[]){
    // Initialize MPI
    MPI::Init(argc, argv);
    // Get the number of processes
    int nProcs = MPI::COMM_WORLD.Get_size();
    // Get the ID of the process
    int my_id = MPI::COMM_WORLD.Get_rank();

    if (argc < 2 ){
        if(!my_id){
            std::cout<<"ERROR: The syntax of the program is ./ping-ping num_ping_pong" <<std::endl;
        };
        MPI::COMM_WORLD.Abort(1);
    }

    if(nProcs %2 != 0 ){
        // only the first process prints the output message
        if(!my_id){
            std::cout<<"ERROR: The nuber of processes must be a multiple of 2"<<std::endl;
        }

        MPI::COMM_WORLD.Abort(1);
    }
    int num_ping_pong = atoi(argv[1]);
    int ping_pong_count = 0;

    // Loop communication
    int next_id = my_id +1 , prev_id = my_id -1;
    if(next_id >= nProcs) next_id =0;
    if(prev_id <0 ) prev_id = nProcs - 1;

    // Thread blocked
    // double start_time,now_time;
    // start_time=clock();


    while(ping_pong_count < num_ping_pong){
        // First receive the ping and then send the pong
        ping_pong_count++;

        // Detection thread blocked 
        // now_time = clock();
        // if( (now_time - start_time)/CLOCKS_PER_SEC > 2) std::cout<<"Time Limit Error! Because of thread block\n";

        // Send the ping
        MPI::COMM_WORLD.Send(&ping_pong_count,1,MPI::INT,next_id,0);
        printf("process %d send ping to process %d\n", my_id,next_id);

        // Wait and receive the ping
        MPI::COMM_WORLD.Recv(&ping_pong_count,1,MPI::INT,prev_id,0);
        printf("process %d, recv ping form process %d \n",my_id,prev_id);

        // Send pong
        MPI::COMM_WORLD.Send(&ping_pong_count,1,MPI::INT,prev_id,0);
        printf("process %d send pong to process %d\n", my_id,prev_id);

        //Wait and receive the pong
        MPI::COMM_WORLD.Recv(&ping_pong_count,1,MPI::INT,next_id,0);
        printf("process %d, recv pong form process %d \n",my_id,next_id);

    }

    // Terminate MPI
    MPI::Finalize();
    return 0;
}

```

When using Send and Recv communication, due to a loop in the communication process, a deadlock occurs.

Use Isend(), Irecv() instead of blocking communication functions Send() and Recv() can solve this problem.

Reason:

Send,Recv. The function signature is as follows:

```
void Send(cosnt void* buf,int count,const Datatype& datatype, int dest,int tag);
void Recv(void* buf,int count,const Datatype& datatype, int source,int tag);
```

Isend,Irecv. The function signature is as follows：

```
MPI::Request Send(cosnt void* buf,int count,const Datatype& datatype, int dest,int tag);
MPI::Request Recv(void* buf,int count,const Datatype& datatype, int source,int tag);
```



Unlike the Send and Receive functions, the Isend and Irecv functions immediately return an MPI::Request object after they are called. This object contains the status information of the message. The function returns immediately after being called without blocking the continued operation of subsequent programs. Use `MPI::Wait()` method to synchronize non-blocking communication, modify the above code to non-blocking communication to eliminate deadlock, the modified code is as follows.

```
#include "stdio.h"
#include "iostream"
#include "mpi.h"
#include "time.h"

int main(int argc, char * argv[]){
    // Initialize MPI
    MPI::Init(argc, argv);
    // Get the number of processes
    int nProcs = MPI::COMM_WORLD.Get_size();
    // Get the ID of the process
    int my_id = MPI::COMM_WORLD.Get_rank();

    if (argc < 2 ){
        if(!my_id){
            std::cout<<"ERROR: The syntax of the program is ./ping-ping num_ping_pong" <<std::endl;
        };
        MPI::COMM_WORLD.Abort(1);
    }

    if(nProcs %2 != 0 ){
        // only the first process prints the output message
        if(!my_id){
            std::cout<<"ERROR: The nuber of processes must be a multiple of 2"<<std::endl;
        }

        MPI::COMM_WORLD.Abort(1);
    }
    int num_ping_pong = atoi(argv[1]);
    int ping_pong_count = 0;

    // Loop communication
    int next_id = my_id +1 , prev_id = my_id -1;
    if(next_id >= nProcs) next_id =0;
    if(prev_id <0 ) prev_id = nProcs - 1;

    MPI::Request rq_send;
    MPI::Request rq_recv;

    while(ping_pong_count < num_ping_pong){
        // First receive the ping and then send the pong
        ping_pong_count++;

        // Send the ping
        rq_send = MPI::COMM_WORLD.Isend(&ping_pong_count,1,MPI::INT,next_id,0);
        printf("process %d send ping to process %d\n", my_id,next_id);

        // Wait and receive the ping
        rq_recv = MPI::COMM_WORLD.Irecv(&ping_pong_count,1,MPI::INT,prev_id,0);
        printf("process %d recv ping form process %d \n",my_id,prev_id);

        // sync Isend and Irecv
        rq_recv.Wait();

        // Send pong
        rq_send = MPI::COMM_WORLD.Isend(&ping_pong_count,1,MPI::INT,prev_id,0);
        printf("process %d send pong to process %d\n", my_id,prev_id);

        //Wait and receive the pong
        rq_recv = MPI::COMM_WORLD.Irecv(&ping_pong_count,1,MPI::INT,next_id,0);
        printf("process %d recv pong form process %d \n",my_id,next_id);

        rq_recv.Wait();

    }

    // Terminate MPI
    MPI::Finalize();
    return 0;
}

```



### Broadcast 

Used to optimize the broadcast of all processes in a communication domain.

![集群通信](..\imgs\bcast.png)

example: Calculate the number of prime numbers between 0 and `n`

`n` is specified at runtime

```
#include "stdio.h"
#include "iostream"
#include "mpi.h"

int main(int argc, char * argv[]){
    // Initialize MPI
    MPI::Init(argc, argv);
    // Get the number of processes
    int nProcs = MPI::COMM_WORLD.Get_size();
    // Get the ID of the process
    int my_id = MPI::COMM_WORLD.Get_rank();

    if (argc < 2 ){
        if(!my_id){
            std::cout<<"ERROR: The syntax of the program is ./ping-ping num_ping_pong" <<std::endl;
        };
        MPI::COMM_WORLD.Abort(1);
    }

    if(nProcs %2 != 0 ){
        // only the first process prints the output message
        if(!my_id){
            std::cout<<"ERROR: The nuber of processes must be a multiple of 2"<<std::endl;
        }

        MPI::COMM_WORLD.Abort(1);
    }

    int n;
    if(!my_id) n = atoi(argv[1]);

    // Barrier to synchronize the processes before measuring time
    MPI::COMM_WORLD.Barrier();

    // Measure the current time
    double start_time = MPI::Wtime();

    // Send the value of n to all process
    // void Bcast(void* buffer, int count,const MPI::Datatype& datatype,int root)
    // root Specify the process that owns the data
    MPI::COMM_WORLD.Bcast(&n,1,MPI::INT,0);

    if(n < 1){
        if(!my_id){
            std::cout<<"ERROR:The parameter 'n' must be higher than 0\n";
        }
        MPI::COMM_WORLD.Abort(1);
    }

    // Perform the computation of the number of primes
    // between 0 and n in parallel
    int my_count = 0;
    int total;
    bool prime;

    // Each process analyzes only part of the numbers below n
    // The distribution is cyclic for better workload balance
    for(int i=2+my_id;i<=n;i=i+nProcs){
        prime = true;
        for(int j=2;j<i;j++){
            if ((i%j) == 0){
                prime = false;
                break;
            }
        }
        my_count += prime;
    }


    // Reduce the partial counts into total in the Process 0
    MPI::COMM_WORLD.Reduce(&my_count,&total,1,MPI::INT,MPI::SUM,0);

    // Measure the current time
    double end_time = MPI::Wtime();

    if(!my_id){
        printf("%d primes between 1 and %d ",total,n);
        printf("Time with %d processes: %f seconds \n",nProcs,end_time - start_time);
    }

    // Terminate MPI
    MPI::Finalize();
    return 0;
}

```

The signature of the broadcast function `Bcast`:

```
//The root parameter specifies the source process that holds the data
void Bcast(void* buffer, int count,const MPI::Datatype& datatype,int root)
```

When the root process calls `Bcast()`, the data will be sent to all other processes. When the receiving process calls `Bcast()`, the data in the root process is copied to the local variables of the receiving process. All processes receive the end of the broadcast. If the processes in the same communication domain do not participate in the broadcast, a deadlock will occur.



Set reduction function `Reduce` function signature

```
void Reduce(const void* sendbuf, void* recvbuf,int count, const MPI::Datapyte& datatype,cosnt MPI::Op& op,int root )
```

The op parameter specifies our operations on the collection data, including summation, average, maximum, minimum or logical operations. Here, the `MPI::SUM` operator is used to sum the prime numbers of each process.

More collection operations can be found in[MPI Docs](https://www.mpich.org/static/docs/latest/)

```
Allreduce() // Combination of reduction and broadcasting, each process gets the same output
Scatter() // Split the root process data block into different parts and send it to different processes
Gather() // Send data from different processes, gather output in the root process
```
