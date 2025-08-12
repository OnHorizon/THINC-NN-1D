/*
 * linear_convection.cc
 *      Author: sunder
 */

#include "pde.h"

/* Function definitions for linear convection equation */

//----------------------------------------------------------------------------
// Convert a conserved variable to a primitive variable 
// Both are same in this case 
//----------------------------------------------------------------------------

void LinearConvection::cons2Prim(const multi_array<double,1>& Q, multi_array<double,1>& V) const {
    V[0] = Q[0];   
}

//----------------------------------------------------------------------------
// Convert a primitive variable to conserved variable
// Both are same for linear convection)
//----------------------------------------------------------------------------

void LinearConvection::prim2Cons(const multi_array<double,1>& V, multi_array<double,1>& Q) const {
    Q[0] = V[0];        
}

//----------------------------------------------------------------------------
// Conservative part of the flux. The input should be conserved variable.
//----------------------------------------------------------------------------

void LinearConvection::flux(const multi_array<double,1>& Q, multi_array<double,1>& F) const {
    
    F[0] = a*Q[0];
}

//----------------------------------------------------------------------------
// Find the eigenvalues
//----------------------------------------------------------------------------

void LinearConvection::eigenValues(const multi_array<double,1>& Q, multi_array<double,1>& L) const {
    (void) Q; 
    L[0] = a; 
}

//----------------------------------------------------------------------------
// Check PAD (No physical admissibility conditions)
//----------------------------------------------------------------------------

bool LinearConvection::checkPAD(const multi_array<double,1>& Q) const {
    (void) Q; 
    return true; 
}

//----------------------------------------------------------------------------
// Project conserved variable to charecteristic space
//----------------------------------------------------------------------------

void LinearConvection::project2CharacteristicSpace(const multi_array<double,1>& Qbar, const multi_array<double,1>& Q, multi_array<double,1>& C) const {
    (void) Qbar; 
    C[0] = Q[0]; 
}

//----------------------------------------------------------------------------
// Project charecteristic variable to conservative variable
//----------------------------------------------------------------------------

void LinearConvection::project2RealSpace(const multi_array<double,1>& Qbar, const multi_array<double,1>& C, multi_array<double,1>& Q) const {
    (void) Qbar; 
    Q[0] = C[0]; 
}

//----------------------------------------------------------------------------
// Set boundary conditions
//----------------------------------------------------------------------------

void LinearConvection::getRightState(const multi_array<double,1>& QL, int bcond, multi_array<double,1>& QR) const {
    (void) bcond; 
    QR[0] = QL[0];
    
}

//----------------------------------------------------------------------------
// Periodic sine wave 
//----------------------------------------------------------------------------

void LinearConvection::sineWave(double x, multi_array<double,1>& V0) const {

    V0[0] = std::sin(M_PI*x); 
}

//----------------------------------------------------------------------------
// Gauss wave
//----------------------------------------------------------------------------

void LinearConvection::gaussWave(double x, multi_array<double,1>& V0) const {
    V0[0] = std::exp(-10*x*x);  
}

//----------------------------------------------------------------------------
// Square Wave
//----------------------------------------------------------------------------

void LinearConvection::squareWave(double x, multi_array<double,1>& V0) const {

    if (std::abs(x) < 0.5)
        V0[0] = 1.0;
    else
        V0[0] = 0.0;
}

//----------------------------------------------------------------------------
// Jiang-Shu multi-wave problem 
//----------------------------------------------------------------------------

double LinearConvection::F(double x, double alpha, double a) const {
    return std::sqrt( std::max(  1.0 - alpha*alpha*(x-a)*(x-a), 0.0  )); 
}

double LinearConvection::G(double x, double beta, double z) const {
    return std::exp(-beta*(x-z)*(x-z)); 
}

void LinearConvection::jiangShuWave(double x, multi_array<double,1>& V0) const {

    double a = 0.5; 
    double z = -0.7; 
    double delta = 0.005; 
    double alpha  = 10.0;
    double beta = std::log(2.0)/(36.0*delta*delta); 

    if (x >= -0.8 && x < -0.6) 
        V0[0] = (1./6.)*(G(x,beta,z-delta) + G(x,beta,z+delta) + 4.0*G(x,beta,z) ); 
    else if (x >= -0.4 && x < -0.2)
        V0[0] = 1.0; 
    else if (x >= 0.0 && x < 0.2)
        V0[0] = 1.0 - std::abs(10.0*(x-0.1)); 
    else if (x >= 0.4 && x < 0.6) 
        V0[0] = (1./6.)*(F(x,alpha,a-delta) + F(x,alpha,a+delta) + 4.0*F(x,alpha,a) ); 
    else
        V0[0]  = 0.0; 
    
}


//----------------------------------------------------------------------------
// Initial condition 
//----------------------------------------------------------------------------

void LinearConvection::initCond(double x, multi_array<double,1>& Q0) const {

	multi_array<double,1> V0(extents[_nVar]);

    if (ic == LinearConvection::sine_wave)
        sineWave(x,V0); 
    else if (ic == LinearConvection::gauss_wave)
        gaussWave(x,V0); 
    else if (ic == LinearConvection::square_wave)
        squareWave(x,V0); 
    else if (ic == LinearConvection::jiang_shu_wave)
        jiangShuWave(x,V0); 
    else {
        V0[0] = 1.0;      
    }

    prim2Cons(V0,Q0); 
}
