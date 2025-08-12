/*
 * euler_equations.cc
 *      Author: sunder
 */

#include "pde.h"

/* Function definitions for Euler equations */

//----------------------------------------------------------------------------
// Convert a conserved variable to a primitive variable
// (Density,momentum,total energy) -> (Density,velocity,pressure)
//----------------------------------------------------------------------------

void EulerEquations::cons2Prim(const multi_array<double,1>& Q, multi_array<double,1>& V) const {
    double p = (GAMMA-1.0)*( Q[2] - 0.5*Q[1]*Q[1]/Q[0] );    // fluid pressure
    V[0] = Q[0];             // fluid density
    V[1] = Q[1]/Q[0];        // fluid velocity
    V[2] = p;                // fluid pressure
}

//----------------------------------------------------------------------------
// Convert a primitive variable to conserved variable
// (Density,velocity,pressure) -> (Density,momentum,total energy)
//----------------------------------------------------------------------------

void EulerEquations::prim2Cons(const multi_array<double,1>& V, multi_array<double,1>& Q) const {
    Q[0] = V[0];           // fluid density
    Q[1] = V[0]*V[1];      // momentum
    Q[2] = V[2]/(GAMMA-1.0) + 0.5*V[0]*V[1]*V[1];  // total energy = internal energy + kinetic energy
}

//----------------------------------------------------------------------------
// Conservative part of the flux. The input should be conserved variable.
//----------------------------------------------------------------------------

void EulerEquations::flux(const multi_array<double,1>& Q, multi_array<double,1>& F) const {
    
    double p = (GAMMA-1.0)*( Q[2] - 0.5*Q[1]*Q[1]/Q[0] );    // fluid pressure
    double irho = 1./Q[0];

    F[0] = Q[1];
    F[1] = irho*Q[1]*Q[1] + p;
    F[2] = irho*Q[1]*(Q[2] + p);
}

//----------------------------------------------------------------------------
// Find the eigenvalues
//----------------------------------------------------------------------------

void EulerEquations::eigenValues(const multi_array<double,1>& Q, multi_array<double,1>& L) const {

    double p = (GAMMA-1.0)*( Q[2] - 0.5*Q[1]*Q[1]/Q[0] );    // fluid pressure
    double irho = 1./Q[0];
    double u = Q[1]*irho;


    if (Q[0] < rho_floor) {
        std::cerr << "Negative density, rho = " << Q[0] << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (p < prs_floor) {
        std::cerr << "Negative pressure, p = " << p << std::endl;
        std::exit(EXIT_FAILURE);
    }

    double c = std::sqrt(GAMMA*p*irho);

    L[0] = u-c;
    L[1] = u;
    L[2] = u+c;
}

//----------------------------------------------------------------------------
// Check PAD
//----------------------------------------------------------------------------

bool EulerEquations::checkPAD(const multi_array<double,1>& Q) const {

    double rho = Q[0];
    double p = (GAMMA-1.0)*( Q[2] - 0.5*Q[1]*Q[1]/Q[0] );    // fluid pressure

    if (rho < rho_floor)
        return false;
    else if (p < prs_floor)
        return false;
    else
        return true;
}

//----------------------------------------------------------------------------
// Project conserved variable to charecteristic space
//----------------------------------------------------------------------------

void EulerEquations::project2CharacteristicSpace(const multi_array<double,1>& Qbar, const multi_array<double,1>& Q, multi_array<double,1>& C) const {

    
    int i,j; 

    double L[_nVar][_nVar];

    double rho  = Qbar[0];             // fluid density
    double irho = 1.0/rho;
    double u    = Qbar[1]*irho;        // fluid velocity
    double E    = Qbar[2];
    double p    = (GAMMA-1.0)*( E - 0.5*rho*u*u );    // fluid pressure


    if (rho < rho_floor) {
        std::cerr << "Negative density, rho = " << rho << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (p < prs_floor) {
        std::cerr << "Negative pressure, p = " << p << std::endl;
        std::exit(EXIT_FAILURE);
    }

    double c = std::sqrt(GAMMA*p*irho);
    double ic = 1.0/c;
    double B1 = (GAMMA-1.0)*ic*ic;
    double B2 = 0.5*B1*u*u;

    L[0][0] = 0.5*(B2+u*ic); L[0][1] = -0.5*(B1*u+ic); L[0][2] = 0.5*B1;
    L[1][0] = 1.0 - B2;      L[1][1] = B1*u;           L[1][2] = -B1;
    L[2][0] = 0.5*(B2-u*ic); L[2][1] = -0.5*(B1*u-ic); L[2][2] = 0.5*B1;

    for (i = 0; i < _nVar; ++i) {
        C[i] = 0.0;
        for (j = 0; j < _nVar; ++j) {
            C[i] += L[i][j]*Q[j];
        }
    }
}

//----------------------------------------------------------------------------
// Project charecteristic variable to conservative variable
//----------------------------------------------------------------------------

void EulerEquations::project2RealSpace(const multi_array<double,1>& Qbar, const multi_array<double,1>& C, multi_array<double,1>& Q) const {

    int i, j; 

    double R[_nVar][_nVar];
    double rho = Qbar[0];             // fluid density
    double irho = 1.0/rho;
    double u = Qbar[1]*irho;        // fluid velocity
    double E = Qbar[2];
    double p = (GAMMA-1.0)*( E - 0.5*rho*u*u );    // fluid pressure

    if (rho < rho_floor) {
        std::cerr << "Negative density, rho = " << rho << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (p < prs_floor) {
        std::cerr << "Negative pressure, p = " << p << std::endl;
        std::exit(EXIT_FAILURE);
    }

    double c = std::sqrt(GAMMA*p*irho);
    double H = (E+p)*irho;

    R[0][0] = 1.0; R[0][1] = 1.0; R[0][2] = 1.0;
    R[1][0] = u-c;      R[1][1] = u;           R[1][2] = u+c;
    R[2][0] = H-c*u; R[2][1] = 0.5*u*u; R[2][2] = H+c*u;

    for (i = 0; i < _nVar; ++i) {
        Q[i] = 0.0;
        for (j = 0; j < _nVar; ++j) {
            Q[i] += R[i][j]*C[j];
        }
    }
}

//----------------------------------------------------------------------------
// Set boundary conditions
//----------------------------------------------------------------------------

void EulerEquations::getRightState(const multi_array<double,1>& QL, int bcond, multi_array<double,1>& QR) const {

    if (bcond == 2) { // Reflective
        QR[0] = QL[0];
        QR[1] = -QL[1];
        QR[2] = QL[2];
    }

    else { // Transmissive (default)
        QR[0] = QL[0];
        QR[1] = QL[1];
        QR[2] = QL[2];
    }
}

//----------------------------------------------------------------------------
// Periodic sine wave 
//----------------------------------------------------------------------------

void EulerEquations::sineWave(double x, multi_array<double,1>& V0) const {

    V0[0] = 1.0 + 0.2*std::sin(M_PI*x); 
    V0[1] = 1.0; 
    V0[2] = 1.0;  
}

//----------------------------------------------------------------------------
// Sod Shock Tube Problem 
//----------------------------------------------------------------------------

void EulerEquations::sodShockTube(double x, multi_array<double,1>& V0) const {

    if (x < 0.0) {
        V0[0] = 1.0; 
        V0[1] = 0.0; 
        V0[2] = 1.0;  
    }
        
    else  {
        V0[0] = 0.125;
        V0[1] = 0.0; 
        V0[2] = 0.1;      
    }
}

//----------------------------------------------------------------------------
// Lax Shock Tube Problem 
//----------------------------------------------------------------------------

void EulerEquations::laxShockTube(double x, multi_array<double,1>& V0) const {

    if (x < 0.0) {
        V0[0] = 0.445; 
        V0[1] = 0.698; 
        V0[2] = 3.528;  
    }
        
    else  {
        V0[0] = 0.5;
        V0[1] = 0.0; 
        V0[2] = 0.571;      
    }
}

//----------------------------------------------------------------------------
// Shu-Osher Problem 
//----------------------------------------------------------------------------

void EulerEquations::shuOsher(double x, multi_array<double,1>& V0) const {

    if (x < -4.0) {
        V0[0] = 3.857143; 
        V0[1] = 2.629369; 
        V0[2] = 10.33333;  
    }
        
    else  {
        V0[0] = 1.0 + 0.2*std::sin(5.0*x);
        V0[1] = 0.0; 
        V0[2] = 1.0;      
    }
}

//----------------------------------------------------------------------------
// Titarev-Toro Problem 
//----------------------------------------------------------------------------

void EulerEquations::titarevToro(double x, multi_array<double,1>& V0) const {

    if (x < -1.5) {
        V0[0] = 1.515695; 
        V0[1] = 0.523346; 
        V0[2] = 1.805000;  
    }
        
    else  {
        V0[0] = 1.0 + 0.1*std::sin(20.0*M_PI*x);
        V0[1] = 0.0; 
        V0[2] = 1.0;      
    }
}

//----------------------------------------------------------------------------
// Blast Wave Problem 
//----------------------------------------------------------------------------

void EulerEquations::blastWave(double x, multi_array<double,1>& V0) const {

    if (x < 0.1) {
        V0[0] = 1.0; 
        V0[1] = 0.0; 
        V0[2] = 1000.0;  
    }
        
    else if (x >= 0.1 && x < 0.9) {
        V0[0] = 1.0;
        V0[1] = 0.0; 
        V0[2] = 1./100.;      
    }

    else {

        V0[0] = 1.0;
        V0[1] = 0.0; 
        V0[2] = 100.;  
    }
}

//----------------------------------------------------------------------------
// Initial condition 
//----------------------------------------------------------------------------

void EulerEquations::initCond(double x, multi_array<double,1>& Q0) const {

	multi_array<double,1> V0(extents[_nVar]);

    if (ic == EulerEquations::sine_wave)
        sineWave(x,V0); 
    else if (ic == EulerEquations::sod_shock_tube)
        sodShockTube(x,V0); 
    else if (ic == EulerEquations::lax_shock_tube)
        laxShockTube(x,V0); 
    else if (ic == EulerEquations::shu_osher)
        shuOsher(x,V0); 
    else if (ic == EulerEquations::titarev_toro)
        titarevToro(x,V0); 
    else if (ic == EulerEquations::blast_wave)
        blastWave(x,V0); 
    else {
        V0[0] = 1.0; 
        V0[1] = 0.0; 
        V0[2] = 1.0;     
    }

    prim2Cons(V0,Q0); 
}
