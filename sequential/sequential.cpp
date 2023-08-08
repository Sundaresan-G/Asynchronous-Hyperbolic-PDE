// Wave equation solver with constant c. utt - c^2uxx=0

#include <iostream>
#include <fstream>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <chrono>

#define PI 2*asin(1)

#define C_VEL 1.0
#define X_START 0.0
#define X_END 1.0
#define TIME_START 0.0
#define TIME_END 0.1
#define CFL 0.1
#define NX 256
// Number of points = NX+1

// CFL = cdt/dx

using namespace std;

// Specify the initial condition value u(x,0)=f(x) in the below function
double initial_condition_function(double x){
    double answer = sin (2*PI*x);
    return answer;
}

// Specify the initial condition derivative value ut(x,0)=g(x) in the below function
double initial_condition_function_derivative(double x){
    double answer = 0;
    return answer;
}

double analytical_solution(const double x, const double t){
    double answer = sin (2*PI*x)*cos(2*PI*C_VEL*t);
    return answer;
}

double compute_error(double **u_res, int nx, double dx, double tend, int pos){
    double answer = 0;
    for (int i=0;i<nx;i++){
        double x = X_START + (i)*dx;
        double analytic = analytical_solution(x, tend);
        answer += abs(u_res[pos][i]- analytic);
    }
    answer/=nx;
    return answer;
}


void print_parameters(int nx, double cfl){
    cout<<"The given parameters for wave equation with periodic BCs are\n";
    printf("X-start: %lf, X-end: %lf\n", X_START, X_END);
    printf("Time-start: %lf, Time-end: %lf\n", TIME_START, TIME_END);
    printf("Constant c: %lf, CFL: %lf\n", C_VEL, cfl);
    printf("Nx = %d\n", nx);
    double dx = (X_END-X_START)/nx;
    double dt = cfl * dx / C_VEL;
    int nt = (TIME_END-TIME_START)/dt;
    printf("Nt = %d\n", nt);    
    printf("dx = %.10lf, dt = %.10lf\n", dx, dt);
}

void compute_values_next_time_step(double **u_res, int nx, double cfl){
    
    u_res[2][0] = 2*u_res[1][0] -u_res[0][0] \
        + (cfl*cfl)*(u_res[1][nx-2] - 2*u_res[1][0] + u_res[1][1]);

    for(int i=1; i<nx-1; i++){
        u_res[2][i] = 2*u_res[1][i] -u_res[0][i]\
        + (cfl*cfl)*(u_res[1][i-1] - 2*u_res[1][i] + u_res[1][i+1]);
    }

    u_res[2][nx-1] = u_res[2][0]; // periodic BC

    // Shift values
    for(int i=0; i<nx; i++){
        u_res[0][i] = u_res[1][i];
        u_res[1][i] = u_res[2][i]; 
    }

}

// pos <=2
void print_output_values(double **u_res, int nx, double tend, int pos){
    cout<<"At time "<<tend<<endl;
    double dx = (X_END-X_START)/(nx-1);
    char s[500];
    sprintf(s, "%8s %6s", "X", "u");
    cout << s<<endl; 
    
    for(int i=0; i<nx; i++){
        sprintf(s, "%6lf %6lf", X_START + i*dx, u_res[pos][i]);
        cout << s<<endl; 
    }
}

// Command line arguments Nx, CFL
int main(int argc, char *argv[]){

    double xstart = X_START, xend = X_END;
    double tstart = TIME_START, tend = TIME_END;
    double c = C_VEL;

    double cfl = CFL;

    int nx=NX;
    
    // read Nx if given
    if(argc>1){
        nx = stoi(argv[1]);
        if (nx<=0){
            cout<<"Incorrect grid size\n";
            return -1;
        }
    }

    // read cfl if given
    if(argc>2){
        cfl = stod(argv[2]);
        if (cfl<=0 || cfl>1){
            cout<<"Incorrect CFL value (need between 0 and 1(inclusive))\n";
            return -1;
        }
    }

    double dx = (X_END-X_START)/nx;
    double dt = cfl * dx / C_VEL;

    print_parameters(nx, cfl);

    /* Initialization completed */

    cout<<"\nThe explicit CD2 solver starts \n";

    auto start = std::chrono::high_resolution_clock::now();

    nx++; // add an extra point for array storage

    // To store 3 time step results
    double *u_res[3];
    
    for(int i=0; i<3; i++){
        u_res[i] = new double[nx]{0};
    }
      

    for(int i=0; i<nx; i++){
        double x = xstart + i*dx;
        u_res[0][i] = initial_condition_function(x);
    }

    // print_output_values((double **)u_res, nx, tstart, 0);

    // For the first time step we use first order accurate time

    if(dt > tend - tstart){
        dt = tend - tstart;
        cfl = c*dt/dx;
    }

    u_res[1][0] = u_res[0][0] + (dt)*initial_condition_function_derivative(xstart)\
        + (cfl*cfl/2)*(u_res[0][nx-2] - 2*u_res[0][0] + u_res[0][1]);

    for(int i=1; i<nx-1; i++){
        double x = xstart + i*dx;
        u_res[1][i] = u_res[0][i] + (dt)*initial_condition_function_derivative(x)\
        + (cfl*cfl/2)*(u_res[0][i-1] - 2*u_res[0][i] + u_res[0][i+1]);
    }

    u_res[1][nx-1] = u_res[1][0]; // periodic BC

    // print_output_values((double **)u_res, nx, tstart+dt, 1);

    int nt = tend/dt;

    for(int j=2; j<=nt; j++){
        compute_values_next_time_step((double **)u_res, nx, cfl); 
        // print_output_values((double **)u_res, nx, tstart+j*dt, 2);       
    }

    // For final step
    if (nt*dt < tend){
        dt = tend - nt*dt;
        cfl = C_VEL *dt /dx;

        cout<<"Last step CFL value is "<<cfl<<endl;
        compute_values_next_time_step((double **)u_res, nx, cfl);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // duration is in microseconds
    double duration = chrono::duration_cast<chrono::microseconds>(end - start).count();

    cout<<"\nThe total time consumed is "<<duration/1000<<" ms\nThe results are\n";

    print_output_values((double **)u_res, nx, tend, 2);

    double error = compute_error((double **)u_res, nx, dx, tend, 2);
    
    cout<<"\nAverage error at time "<<tend<<" is "<<error<<endl;
     

    return 0;

}