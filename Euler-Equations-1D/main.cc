/*
 * main.cc
 *      Author: sunder
 */

#include "hype.h"
#include "pde.h"

void solveLinearConvectionEquation(); 
void solveEulerEquations(); 

int main() {
    
    solveEulerEquations();    

    return 0; 
}



void solveEulerEquations() {

    double gamma = 1.4; 
    EulerEquations Euler(gamma, EulerEquations::blast_wave);
    double xmax,xmin,tend;  
    int bcond_l, bcond_r;
    Euler.getProblemParameters(xmin,xmax,tend,bcond_l,bcond_r); 
    int nCells = 256;
    HyPE_1D<EulerEquations> Sol(Euler,xmin,xmax,nCells,tend, bcond_l, bcond_r);
    Sol.run();
    printf("Max Error = %.6e\n", Sol.maxError(0)); 

    Sol.plot("sol.csv"); 
}


