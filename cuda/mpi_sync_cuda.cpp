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

double *device_u_res;

double analytical_solution(const double x, const double t){
    double answer = sin (2*PI*x)*cos(2*PI*C_VEL*t);
    return answer;
}

extern "C" void cuda_back_to_host(double **u_res, int last_pos, int nx, double *device_u_res);

extern "C" double* cuda_initialize(double **u_res, int nx, double *device_u_res);

extern "C" void setDeviceProps(int rank, int size);

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

extern "C" void getDeviceProps(int rank, int size);
extern "C" void cudaDeInitialize(double *device_u_res);
extern "C" void cuda_shift_values(double **u_res, int nx, double *device_u_res);
extern "C" void cuda_compute_values_next_time_step(double **u_res, int nx, double cfl, double *device_u_res);
extern "C" void cuda_get_first_last(double **u_res, int nx, double *device_u_res);

void compute_values_next_time_step(double **u_res, const int nx, const double cfl){


    cuda_compute_values_next_time_step(u_res, nx, cfl, device_u_res);
    
    cuda_shift_values(u_res, nx, device_u_res);

    cuda_get_first_last(u_res, nx, device_u_res);

    
    // Shift values
    // for(int i=0; i<=nx+1; i++){
    //     u_res[0][i] = u_res[1][i];
    //     u_res[1][i] = u_res[2][i]; 
    // }

}

// pos <=2
void print_output_values(double **u_res, int nx, double xstart, \
double tend, double dx, int pos){
    cout<<"At time "<<tend<<endl;
    // double dx = (X_END-X_START)/(nx-1);
    char s[500];
    sprintf(s, "%8s %6s", "X", "u");
    cout << s<<endl; 
    
    for(int i=1; i<=nx; i++){
        sprintf(s, "%6lf %6lf", xstart + (i-1)*dx, u_res[pos][i]);
        cout << s<<endl; 
    }
}

extern "C" void cuda_update_first_and_last(double **u_res, int nx, int pos, double *device_u_res);

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

    if(pos!=0){
        cuda_update_first_and_last(u_res, nx, pos, device_u_res);
    }
    
    
}

// Command line arguments Nx, CFL
int main(int argc, char *argv[]){

    MPI_Init(&argc,&argv);

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

    /* Initialization started */

    auto start = std::chrono::high_resolution_clock::now();

    // The last proc. need not have that extra point as we assume periodic boundary conditions

    // if(rank == size-1)
    //     nx++; // last proc. gets one extra point

    // To store 3 time step results
    double *u_res[3];
    
    for(int i=0; i<3; i++){
        // +2 extra is given for left and right points which shall be received
        u_res[i] = new double[nx+2]{0};
    }

    // update the starting point for all processes
    xstart += rank * nx * dx;

    // Array index for each starts from one for their points
    // 0 is for left ghost and nx+1 is for right ghost point
    for(int i=1; i<=nx; i++){
        double x = xstart + (i-1)*dx;
        u_res[0][i] = initial_condition_function(x);
    }

    // cout<<"Line 236\n";

    // if(rank == 0){
    //     cout<<"Initial values\n";
    //     print_output_values((double **)u_res, nx, xstart, tstart, dx, 0);    
    // }
        
    
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

    // u_res[1][0] = u_res[0][0] + (dt)*initial_condition_function_derivative(xstart)\
    //     + (cfl*cfl/2)*(u_res[0][nx-2] - 2*u_res[0][0] + u_res[0][1]);

    for(int i=1; i<=nx; i++){
        double x = xstart + (i-1)*dx;
        u_res[1][i] = u_res[0][i] + (dt)*initial_condition_function_derivative(x)\
        + (cfl*cfl/2)*(u_res[0][i-1] - 2*u_res[0][i] + u_res[0][i+1]);
    }

    // if(rank==0){
    //     cout<<"First time step values\n";
    //     print_output_values((double **)u_res, nx, xstart, tstart+dt, dx, 1);
    // }

    // The below includes copying too
    
    device_u_res = cuda_initialize(u_res, nx, device_u_res);

    int nt = tend/dt;
    int time_step=1;

    for(int j=2; j<=nt; j++){
        
        send_and_receive(u_res, nx, 1, comm_cart);

        compute_values_next_time_step((double **)u_res, nx, cfl); 

        time_step++;

        // if(rank==0)
        //     cout<<time_step<<endl;

        // if(rank==0){
        //     cuda_back_to_host(u_res, 2, nx, device_u_res);
        //     print_output_values((double **)u_res, nx, xstart, tstart+j*dt, dx, 2);       
        // }
    }

    // For final step
    if (nt*dt < tend){
        dt = tend - nt*dt;
        cfl = C_VEL *dt /dx;
        
        send_and_receive(u_res, nx, 1, comm_cart);
        if(rank==0)
            cout<<"Last step CFL value is "<<cfl<<endl;
        compute_values_next_time_step((double **)u_res, nx, cfl);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();

    cuda_back_to_host(u_res, last_pos, nx, device_u_res);

    // duration is in microseconds
    double duration = chrono::duration_cast<chrono::microseconds>(end - start).count();


    double local_error = compute_error(u_res, nx, dx, xstart, tend, 2);

    double total_error;

    MPI_Reduce(&local_error, &total_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    total_error /= globalnx;

    if(rank==0){
        cout<<"\nThe total time consumed is "<<duration/1000<<" ms\nThe results are\n";

        // print_output_values((double **)u_res, nx, xstart, tstart, dx, last_pos);

        cout<<"\nAverage error at time "<<tend<<" is "<<total_error<<endl;

    }

    
    cudaDeInitialize(device_u_res);

    MPI_Finalize();
     

    return 0;

}