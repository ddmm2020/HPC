### 参考资料

https://mpitutorial.com/tutorials/

https://www.mpich.org/static/docs/latest/

《并行程序设计概念与实践》中国工业出版社 



### Installation

MPI

1. MPI no ch4 netmod selected

   加上配置选项：--with-device=ch4:ofi  和 -disable-fortran
2. 43858_ recipe for target 'install' failedMakefil
   make install 权限不够
3. HYDU_create_process (utils/launch/launch.c:73): execvp error on file a.out (No such file or director）

   编译后运行 ：mpirun -np 4 ./a.out



MPI 编译：

mpic++   main.cpp  -o ./test

运行时采用 -np 指定MPI进程个数

mpirun -np 4 ./test



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



### MPI 点到点通讯（阻塞通讯）

实现下图所示通讯流程

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



### 非阻塞通讯

实现如下通讯流程

![非阻塞通讯](https://github.com/ddmm2020/HPC/blob/main/imgs/non-blocking.png)

```
// 阻塞通讯造成死锁
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

采用Send和Recv通讯时，由于通信过程中出现环路，出现死锁

采用Isend(),Irecv()替代阻塞通讯函数Send()和Recv()

原因：

Send,Recv函数签名如下:

```
void Send(cosnt void* buf,int count,const Datatype& datatype, int dest,int tag);
void Recv(void* buf,int count,const Datatype& datatype, int source,int tag);
```

Isend,Irecv函数签名如下：

```
MPI::Request Send(cosnt void* buf,int count,const Datatype& datatype, int dest,int tag);
MPI::Request Recv(void* buf,int count,const Datatype& datatype, int source,int tag);
```

与Send,Receive函数不同，Isend和Irecv函数调用后，立即返回一个MPI::Request对象，这个对象包含消息的状态信息。函数被调用后立即返回不阻塞后续程序的继续运行。采用`MPI::Wait()`方法来对非阻塞通讯进行同步，将上述代码修改成非阻塞通讯消除死锁，修改后代码如下。

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



### 集合通讯

用于优化对某个通信域的所有进程进行广播

![集群通信](..\imgs\bcast.png)

example：计算0~n之间的素数个数

n由运行参数指定

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

广播函数`Bcast`函数签名：

```
// root参数指定了持有数据的源进程
void Bcast(void* buffer, int count,const MPI::Datatype& datatype,int root)
```

当根进程调用Bcast()时，数据就会被发送到其他所有进程。当接收进程调用Bcast()时，根进程中的数据被复制到接收进程的局部变量中。所有进程接收到数据广播结束。若相同的通信域的进程没有参与广播，死锁就会发生。



集合归约函数`Reduce`函数签名

```
void Reduce(const void* sendbuf, void* recvbuf,int count, const MPI::Datapyte& datatype,cosnt MPI::Op& op,int root )
```

op参数，指定我们对集合数据的操作，包括求和，求均值，最大值，最小值或逻辑运算。这里采用`MPI::SUM`算子，对每个进程的素数个数进行求和。

其他更多的集合操作可以在[MPI函数参考](https://www.mpich.org/static/docs/latest/)

```
Allreduce() // 归约和广播组合，每个进程都获得相同的输出
Scatter() // 把根进程数据块切分成不同部分发送到不同的进程
Gather() // 从不同进程发送数据，在根进程进行聚合输出
```



mpi只是一个通信的标准，与gpu并行其实是互补的关系。gpu负责并行计算，mpi负责多gpu间的通信。

在单节点多gpu或多节点多gpu机群中，cuda支持mpi直接在gpu间进行通信(支持cuda_aware_mpi的gpu)，而无需让数据传回host端再传到另外的gpu中，这可以有效缩短gpu间的通信。

所以gpu和mpi是互补而非对立关系。
