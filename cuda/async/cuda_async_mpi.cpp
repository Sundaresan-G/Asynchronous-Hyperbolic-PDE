// Wave equation solver with constant c. utt - c^2uxx=0

#include <iostream>
#include <fstream>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <chrono>
#include <mpi.h>

#define PI 2*asin(1)

#define C_VEL 1.0
#define X_START 0.0
#define X_END 1.0
#define TIME_START 0.0
#define TIME_END 0.1
#define CFL 0.1
#define NX 1024*1024
#define DELAY 10
// Number of points = NX+1

// CFL = cdt/dx

using namespace std;

// Specify the initial condition value u(x,0)=f(x) in the below function
double initial_condition_function(const double x){
    double answer = sin (2*PI*x);
    return answer;
}



// Specify the initial condition derivative value ut(x,0)=g(x) in the below function
double initial_condition_function_derivative(const double x){
    double answer = 0;
    return answer;
}

double analytical_solution(const double x, const double t){
    double answer = sin (2*PI*x)*cos(2*PI*C_VEL*t);
    return answer;
}

double *device_u_res;

extern "C" void cuda_back_to_host(double **u_res, int last_pos, int nx, double *device_u_res);

extern "C" double* cuda_initialize(double **u_res, int nx, double *device_u_res);

extern "C" void setDeviceProps(int rank, int size);

extern "C" void getDeviceProps(int rank, int size);
extern "C" void cudaDeInitialize(double *device_u_res);
extern "C" void cuda_shift_values(double **u_res, int nx, double *device_u_res);
extern "C" void cuda_compute_values_next_time_step(double **u_res, int nx, double cfl, double *device_u_res);
extern "C" void cuda_get_first_last(double **u_res, int nx, double *device_u_res);
extern "C" void cuda_update_first_and_last(double **u_res, int nx, int pos, double *device_u_res);
extern "C" void async_cuda_update_first_and_last(double **u_res, int nx, int pos, double *device_u_res);
extern "C" void async_cuda_get_first_last(double **u_res, int nx, double *device_u_res);

double compute_error(double **u_res, int nx, double dx, double xstart, double tend, int pos){
    double answer = 0;
    for (int i=1;i<=nx;i++){
        double x = xstart + (i-1)*dx;
        double analytic = analytical_solution(x, tend);
        answer += abs(u_res[pos][i]- analytic);
    }
    return answer;
}


void print_parameters(const int nx, const double cfl){
    cout<<"The given parameters for wave equation with periodic BCs are\n";
    printf("X-start: %lf, X-end: %lf\n", X_START, X_END);
    printf("Time-start: %lf, Time-end: %lf\n", TIME_START, TIME_END);
    printf("Constant c: %lf, CFL: %lf\n", C_VEL, cfl);
    printf("Nx = %d\n", nx);

    double dx = (X_END-X_START)/nx;
    double dt = cfl * dx / C_VEL;
    printf("dx = %.15lf, dt = %.15lf\n", dx, dt);
}

void compute_values_next_time_step(double **u_res, const int nx, const double cfl){


    cuda_compute_values_next_time_step(u_res, nx, cfl, device_u_res);
    
    cuda_shift_values(u_res, nx, device_u_res);
    
    // shift cpu values
    u_res[0][1] = u_res[1][1];
    u_res[0][nx] = u_res[1][nx];

    cuda_get_first_last(u_res, nx, device_u_res);

}

void at_compute_values_next_time_step(double **u_res, const int nx, const double cfl, int k, double **recv_buf){

    if(k==0){
        compute_values_next_time_step(u_res, nx, cfl);
    } else {
        double c[4];

        double k3 = pow(k,3);
        double k2 = pow(k,2);

        c[0] = (k3 + 6*k2 + 11*k + 6) /6;
        c[1] = -(k3 + 5*k2 + 6*k) / 2;
        c[2] = (k3 + 4*k2 + 3*k) / 2;
        c[3] = -(k3 + 3*k2 + 2*k) /6;

        // int rank;
        // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // if(rank==1){
        //     cout<<"\nMy rank is 1"<<endl;
        //     cout<<"\nThe current delay is "<<k<<endl;
        //     cout<<"The coefficients are\n";
        //     printf("%lf %lf %lf %lf, sum= %lf\n", c[0],c[1],c[2],c[3], c[0]+c[1]+c[2]+c[3]);
        // }

        // cout<<"Line 130: before "<<u_res[1][2]<<endl;

        // Get one before last values
        async_cuda_get_first_last(u_res, nx, device_u_res);

        // cout<<"Line 135: after "<<u_res[1][2]<<endl;

        

        // not uxx/Deltax^2 gives approx. double derivative
        double uxx = u_res[1][2] - 2*u_res[1][1];
        for(int i=0; i<4; i++){
            uxx += c[i]*recv_buf[0][3-i];
        }

        // if(rank == 1){
        //     cout<<"The previous time step values for left boundary (from oldest) are\n";
        //     for(int i=0; i<4; i++){
        //         cout<<recv_buf[0][i]<<endl;
        //     }
        //     cout<<"Sum is "<<uxx-(u_res[1][2] - 2*u_res[1][1]);
        // }

        
        
        u_res[2][1] = 2*u_res[1][1] -u_res[0][1]\
            + (cfl*cfl)*(uxx);

        cuda_compute_values_next_time_step(u_res, nx, cfl, device_u_res);

        // recv_buf[0] has older time step values than recv_buf[3]
        uxx = u_res[1][nx-1] - 2*u_res[1][nx];
        for(int i=0; i<4; i++){
            uxx += c[i]*recv_buf[1][3-i];
        }

        u_res[2][nx] = 2*u_res[1][nx] -u_res[0][nx]\
            + (cfl*cfl)*(uxx);

        cuda_shift_values(u_res, nx, device_u_res);

        // CPU shift values
        u_res[0][1] = u_res[1][1];
        u_res[1][1] = u_res[2][1];
        u_res[0][nx]= u_res[1][nx];
        u_res[1][nx]= u_res[2][nx];

        async_cuda_update_first_and_last(u_res, nx, 1, device_u_res);

    }
}

// pos <=2
void print_output_values(double **u_res, int nx, double xstart, \
double tend, double dx, int time_step, int pos){
    cout<<"\nAt time "<<tend<<" and time step "<<time_step<<endl;
    // double dx = (X_END-X_START)/(nx-1);
    char s[500];
    sprintf(s, "%8s %6s", "X", "u");
    cout << s<<endl; 
    
    for(int i=1; i<=nx; i++){
        sprintf(s, "%6lf %6lf", xstart + (i-1)*dx, u_res[pos][i]);
        cout << s<<endl; 
    }
}

void send_and_receive(double **u_res, int nx, int pos, MPI_Comm comm){
    // cout<<"Send and receive started\n";
    int rank, size, src, dest, sendtag, recvtag;
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MPI_Cart_shift(comm, 0, 1,
    &src, &dest);
    sendtag = 1;
    recvtag = 1;
    // if(rank==0){
    //     cout<<"Cart shift done\n";
    //     cout<<"Src "<<src<<endl;
    //     cout<<"dest "<<dest<<endl;
    // }
    MPI_Sendrecv(&u_res[pos][nx], 1, MPI_DOUBLE,
                dest, sendtag,
                &u_res[pos][0], 1, MPI_DOUBLE,
                src, recvtag, comm, MPI_STATUS_IGNORE);
    
    // if(rank==0){
    //     cout<<"Sendrecv done\n";
    // }

    MPI_Cart_shift(comm, 0, -1,
    &src, &dest);
    sendtag = 0;
    recvtag = 0;
    MPI_Sendrecv(&u_res[pos][1], 1, MPI_DOUBLE,
                dest, sendtag,
                &u_res[pos][nx+1], 1, MPI_DOUBLE,
                src, recvtag, comm, MPI_STATUS_IGNORE);
    
}

// numbers indicate the number of data points to be sent
// recv_buf[0] for left and recv_buf[1] for right
void async_send_and_receive(double **u_res, int nx, int pos, int numbers,\
double **recv_buf, int recv_buf_pos, MPI_Comm comm){
    // cout<<"Send and receive started\n";
    int rank, size, src, dest, sendtag, recvtag;
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MPI_Cart_shift(comm, 0, 1,
    &src, &dest);
    sendtag = 1;
    recvtag = 1;
    // if(rank==0){
    //     cout<<"Cart shift done\n";
    //     cout<<"Src "<<src<<endl;
    //     cout<<"dest "<<dest<<endl;
    // }
    MPI_Sendrecv(&u_res[pos][nx-numbers+1], numbers, MPI_DOUBLE,
                dest, sendtag,
                &recv_buf[0][recv_buf_pos], numbers, MPI_DOUBLE,
                src, recvtag, comm, MPI_STATUS_IGNORE);
    
    // if(rank==0){
    //     cout<<"Sendrecv done\n";
    // }

    MPI_Cart_shift(comm, 0, -1,
    &src, &dest);
    sendtag = 0;
    recvtag = 0;
    MPI_Sendrecv(&u_res[pos][1], numbers, MPI_DOUBLE,
                dest, sendtag,
                &recv_buf[1][recv_buf_pos], numbers, MPI_DOUBLE,
                src, recvtag, comm, MPI_STATUS_IGNORE);

    // The below ones are for continuing synchronous scheme as is 
    u_res[pos][0]=recv_buf[0][recv_buf_pos];
    u_res[pos][nx+1]=recv_buf[1][recv_buf_pos];

    if(pos!=0){
        cuda_update_first_and_last(u_res, nx, pos, device_u_res);
    }
    
}

void buf_send_and_receive(double **u_res, int nx, double **send_buf, \
double **recv_buf, MPI_Comm comm){
    // cout<<"Send and receive started\n";
    int rank, size, src, dest, sendtag, recvtag;
    int numbers = 4;
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MPI_Cart_shift(comm, 0, 1,
    &src, &dest);
    sendtag = 1;
    recvtag = 1;
    // if(rank==0){
    //     cout<<"Cart shift done\n";
    //     cout<<"Src "<<src<<endl;
    //     cout<<"dest "<<dest<<endl;
    // }
    MPI_Sendrecv(&send_buf[1][0], numbers, MPI_DOUBLE,
                dest, sendtag,
                &recv_buf[0][0], numbers, MPI_DOUBLE,
                src, recvtag, comm, MPI_STATUS_IGNORE);
    
    // if(rank==0){
    //     cout<<"Sendrecv done\n";
    // }

    MPI_Cart_shift(comm, 0, -1,
    &src, &dest);
    sendtag = 0;
    recvtag = 0;
    MPI_Sendrecv(&send_buf[0][0], numbers, MPI_DOUBLE,
                dest, sendtag,
                &recv_buf[1][0], numbers, MPI_DOUBLE,
                src, recvtag, comm, MPI_STATUS_IGNORE);

    // The below ones are for continuing synchronous scheme as is 
    u_res[1][0]=recv_buf[0][3];
    u_res[1][nx+1]=recv_buf[1][3];

    cuda_update_first_and_last(u_res, nx, 1, device_u_res);
    
}

void shift_send_buf(double **send_buf){
    for(int i=0; i<3; i++){
        send_buf[0][i] = send_buf[0][i+1];
        send_buf[1][i] = send_buf[1][i+1];
    }
}

// Command line arguments Nx, CFL
int main(int argc, char *argv[]){

    MPI_Init(&argc,&argv);

    int time_step=0;

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double xstart = X_START, xend = X_END;
    double tstart = TIME_START, tend = TIME_END;
    double c = C_VEL;

    double cfl = CFL;

    int globalnx=NX;
    
    // read Nx if given
    if(argc>1){
        globalnx = stoi(argv[1]);
        if (globalnx<=0){
            cout<<"Incorrect grid size\n";
            return -1;
        }
    }

    if(globalnx % size != 0){
        if (rank==0){
            cout<<"Please specify MPI processes to be a divisor of "<<globalnx<<endl;
        }
        return -1;
    } 

    // read cfl if given
    if(argc>2){
        cfl = stod(argv[2]);
        if (cfl<=0 || cfl>1){
            cout<<"Incorrect CFL value (need between 0 and 1(inclusive))\n";
            return -1;
        }
    }

    double dx = (X_END-X_START)/globalnx;

    // cout<< "dx is "<<dx<<endl;

    double dt = cfl * dx / C_VEL;

    if(rank == 0){
        print_parameters(globalnx, cfl);
        cout<<"\nMPI Procs are "<<size<<endl;
        cout<<"\nThe explicit CD2 solver starts \n";
    }

    // Divide grids (not points) equally among processes
    int nx = globalnx/size;

    // Create cartesian communicator peridic for SendRecv operations
    MPI_Comm comm_cart;
    int ndims = 1;
    int dims[] = {0};
    MPI_Dims_create(size, ndims, dims);
    int periods[] = {1};
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims,
                    periods, 1, &comm_cart);

    MPI_Barrier(comm_cart);

    // Since ranks might change
    MPI_Comm_rank(comm_cart, &rank);

    setDeviceProps(rank, size);
    getDeviceProps(rank, size);

    // To store 3 time step results
    double *u_res[3];
    
    for(int i=0; i<3; i++){
        // +2 extra is given for left and right points which shall be received
        u_res[i] = new double[nx+2]{0};
    }

    // 0 for left and 1 for right boundary values
    double *recv_buf[2];
    for(int i=0; i<2; i++){
        recv_buf[i] = new double[4]{0};
    }

    // 0 for left and 1 for right boundary values
    double *send_buf[2];
    for(int i=0; i<2; i++){
        send_buf[i] = new double[4]{0};
    }

    double *print_res[3];
    for(int i=0;i<3;i++){
        print_res[i] = new double[nx+2]{0};
    }

    /* Initialization started */

    auto start = std::chrono::high_resolution_clock::now();

    // The last proc. need not have that extra point as we assume periodic boundary conditions

    // if(rank == size-1)
    //     nx++; // last proc. gets one extra point

    // update the starting point for all processes
    xstart += rank * nx * dx;

    // Array index for each starts from one for their points
    // 0 is for left ghost and nx+1 is for right ghost point
    for(int i=1; i<=nx; i++){
        double x = xstart + (i-1)*dx;
        u_res[0][i] = initial_condition_function(x);
    }

    // if(rank == 0){
    //     cout<<"Initial values\n";
    //     print_output_values((double **)u_res, nx, xstart, tstart, dx, time_step, 0);    
    // }

    time_step++;
        
    
    // For the first time step we use first order accurate time

    int last_pos = 2; 
    //This is for printing the result at first time step if 
    // below if is satisfied 

    if(dt > tend - tstart){
        dt = tend - tstart;
        cfl = c*dt/dx;
        last_pos=1;
    }

    // Send the initial values to others for next time step
    send_and_receive(u_res, nx, 0, comm_cart);

    for(int i=1; i<=nx; i++){
        double x = xstart + (i-1)*dx;
        u_res[1][i] = u_res[0][i] + (dt)*initial_condition_function_derivative(x)\
        + (cfl*cfl/2)*(u_res[0][i-1] - 2*u_res[0][i] + u_res[0][i+1]);
    }

    // if(rank==0){
    //     cout<<"First time step values\n";
    //     print_output_values((double **)u_res, nx, xstart, tstart+dt, dx, time_step, 1);
    // }

    time_step++;

    device_u_res = cuda_initialize(u_res, nx, device_u_res);

    int nt = tend/dt;

    // The below variable keeps count of delay and ensures < DELAY
    int k =0; 

    // Do synchronous for the first 4 steps
    for(int j=2; j<=nt && k < 4; j++){
        
        // 1 numbers are sent for this AT scheme
        async_send_and_receive((double **)u_res, nx, 1, 1, (double **)recv_buf, k, comm_cart);

        compute_values_next_time_step((double **)u_res, nx, cfl);  

        // cout<<"Line 508 u_res "<<u_res[1][nx]<<endl;      

        // at_compute_values_next_time_step((double **)u_res, nx, cfl, 0, (double **)recv_buf); 

        k++;

        // if(rank==0){
        //     double temp[2] = {u_res[2][1], u_res[2][nx]};
        //     cuda_back_to_host(print_res, 2, nx, device_u_res);
        //     if(DELAY!=1){
        //         // print_res[2][1] = temp[0];
        //         // print_res[2][nx] = temp[1];
        //     }             
        //     print_output_values((double **)print_res, nx, xstart, tstart+j*dt, dx, time_step, 2);       
        // }

        time_step++;
    }

    k = 1;

    // for delay==1
    send_buf[0][3] = u_res[1][1];
    send_buf[1][3] = u_res[1][nx];

    // cout<<"Line 551 send buf\n";
    // for(int i=0;i<4;i++){
    //     cout<<send_buf[1][i]<<endl;
    // }

    for(int j=6; j<=nt; j++){
        
        if(k%DELAY==0){
            // 4 numbers are sent for this AT scheme
            buf_send_and_receive((double **)u_res, nx,\
            (double **)send_buf, (double **)recv_buf, comm_cart);
            // cout<<"Line 526 j value "<<j<<endl;
        }
        

        at_compute_values_next_time_step((double **)u_res, nx, cfl, k%DELAY, (double **)recv_buf); 
        // cout<<"Line 570 u_res "<<u_res[1][nx]<<endl;

        // cout<<"Line 556: "<<u_res[1][nx];

        // To shift one time step
        shift_send_buf(send_buf);

        send_buf[0][3] = u_res[1][1];
        send_buf[1][3] = u_res[1][nx];

        // cout<<"Line 579 send buf\n";
        // for(int i=0;i<4;i++){
        //     cout<<send_buf[1][i]<<endl;
        // }

        k++;

        // if(rank==0){
        //     double temp[2] = {u_res[2][1], u_res[2][nx]};
        //     cuda_back_to_host(print_res, 2, nx, device_u_res);
        //     if(DELAY!=1){
        //         print_res[2][1] = temp[0];
        //         print_res[2][nx] = temp[1];
        //     }             
        //     print_output_values((double **)print_res, nx, xstart, tstart+j*dt, dx, time_step, 2);       
        // }

        // if(rank==0){

        //     cout<<"Line 581\n";

        //     print_output_values((double **)u_res, nx, xstart, (time_step-1)*dt, dx, time_step, last_pos);

        //     cout<<"\nAverage error at time "<<(time_step-1)*dt<<" is "<<total2_error<<endl;

        // }

        // if(rank==0){
        //     double temp[2] = {u_res[2][1], u_res[2][nx]};

        //     cuda_back_to_host(print_res, 2, nx, device_u_res);

        //     u_res[2][1] = temp[0];
        //     u_res[2][nx] = temp[1];
        //     print_output_values((double **)print_res, nx, xstart, tstart+j*dt, dx, time_step, 2);     
        // }
        // if(j==11)
        //     return 0;

        time_step++;
    }

    // if(rank==0){
    //     double temp[2] = {u_res[2][1], u_res[2][nx]};

    //     cuda_back_to_host(u_res, 2, nx, device_u_res);

    //     u_res[2][1] = temp[0];
    //     u_res[2][nx] = temp[1];
    //     print_output_values((double **)u_res, nx, xstart, tstart+time_step*dt, dx, time_step, 2);     
    // }

    // For final step
    if (nt*dt < tend){
        dt = tend - nt*dt;
        cfl = C_VEL *dt /dx;
        if(k%DELAY==0)        
            send_and_receive(u_res, nx, 1, comm_cart);
        if(rank==0)
            cout<<"Last step CFL value is "<<cfl<<endl;
        at_compute_values_next_time_step((double **)u_res, nx, cfl, k%DELAY, (double **)recv_buf);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();

    // duration is in microseconds
    double duration = chrono::duration_cast<chrono::microseconds>(end - start).count();

    double temp[2] = {u_res[2][1], u_res[2][nx]};

    cuda_back_to_host(u_res, last_pos, nx, device_u_res);

    if(DELAY!=1){
        u_res[2][1] = temp[0];
        u_res[2][nx] = temp[1];
    }        

    cudaDeInitialize(device_u_res);

    double local_error = compute_error(u_res, nx, dx, xstart, tend, 2);

    double total_error;

    MPI_Reduce(&local_error, &total_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    total_error /= globalnx;

    if(rank==0){
        cout<<"\nThe total time consumed is "<<duration/1000<<" ms\nThe results are\n";

        // print_output_values((double **)u_res, nx, xstart, tend, dx, time_step, last_pos);

        cout<<"\nAverage error at time "<<tend<<" is "<<total_error<<endl;

    }
    

    MPI_Finalize();
     

    return 0;

}