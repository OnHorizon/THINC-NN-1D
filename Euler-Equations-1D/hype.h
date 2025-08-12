/*
 * hype.h
 *      Author: sunder
 */

#ifndef HYPE_H_
#define HYPE_H_

#include<iostream>
#include<fstream>
#include<string>
#include <cmath>
#include <sstream>
#include <chrono>
#include <stdio.h>
#include <string.h> 
#define BOOST_DISABLE_ASSERTS
#include "boost/multi_array.hpp"

using namespace boost;

const double small_num = 1.0e-12;
const int NGP = 5; // Quadrature points 
const double xGP[] = {-0.453089922969332,-0.26923465505284155,0.0,0.26923465505284155,0.453089922969332};
const double wGP[] = {0.11846344252809471,0.2393143352496831,0.2844444444444445,0.2393143352496831,0.11846344252809471};

//----------------------------------------------------------------------------
// Various types of boundary conditions
//----------------------------------------------------------------------------

enum bndry_type{inflow, periodic, reflective, transmissive};


template <typename T>
class HyPE_1D {

    const int N_ph = 3;
    const double CFL = 0.6;  //0.9 previously

    T PDE;               // PDE System we want to so
    double xmin, xmax;   // Domain size
    int nCells;          // Number of cells in the domain 
    double tend;         // Final time of the simulation 
    int left,right; // Boundary conditions on the lef tand the right 

    multi_array<double,2> qh;   // Conserved variables at cells
    multi_array<double,2> qhi;  // Initial solution for checking errors 
    multi_array<double,2> F;    // Upwind Conservative fluxes at cell faces
    multi_array<double,2> dqh;  // RHS term for each face
    multi_array<double,3> qbnd; // Value of conserved variable at cell left face
    multi_array<double,1> x;    // Cell centers

    // Additional storage for time stepping

    multi_array<double,2> qh0;
    multi_array<double,2> qh2;
    multi_array<double,2> qh3;
    multi_array<double,2> dqh3;

    double dx;
    double dt;
    double time;
    int time_step;

    // Reconstruction 

    inline double pow4(double a) const {double a2 = a*a; return a2*a2;}

    void minmod(const multi_array<double,1>&, double&, double&) const;
    void mclim(const multi_array<double,1>&, double&, double&) const;
    void THINC(const multi_array<double,1>&, const double&, double&, double&) const;
    void WENO5(const multi_array<double,1>&, double&, double&) const;

    void applyBoundaryConditions();
    void computeRHS();
    void step_SSPRK33();

public:
    HyPE_1D(T, double,double,int,double,int,int);
    double maxError(int) const;
    void run();
    void plot(const std::string) const;
};

//----------------------------------------------------------------------------
// Constructor - Allocate memory and initialize solution
// using initial conditions
//----------------------------------------------------------------------------

template <typename T>
HyPE_1D<T>::HyPE_1D(T _PDE, double _xmin, double _xmax, int _nCells, double _tend, int _left, int _right) :
    PDE(_PDE), 
    xmin(_xmin),
    xmax(_xmax),
    nCells(_nCells),
    tend(_tend),
    left(_left),
    right(_right),
    qh(extents[multi_array_types::extent_range(-N_ph,nCells+N_ph)][_PDE.nVar()]),
    qhi(extents[nCells][_PDE.nVar()]),
    F(extents[nCells + 1][_PDE.nVar()]),
    dqh(extents[nCells][_PDE.nVar()]),
    qbnd(extents[multi_array_types::extent_range(-N_ph,nCells+N_ph)][2][_PDE.nVar()]), // 2 is for two faces per cell
    x(extents[nCells]),
    qh0(extents[nCells][_PDE.nVar()]),
    qh2(extents[nCells][_PDE.nVar()]),
    qh3(extents[nCells][_PDE.nVar()]),
    dqh3(extents[nCells][_PDE.nVar()]),
    dt(0.0),
    time(0.0),
    time_step(0)
{

    int i, c, q; 

    // Initialize the grid

    dx = (xmax - xmin)/static_cast<double>(nCells);

    for (int i = 0; i < nCells; ++i)
        x[i] = xmin + ((i+1)-0.5)*dx;


    multi_array<double,1> Q0(extents[PDE.nVar()]);

    // Loop through all the cells

    for (i = 0; i < nCells; ++i) {

        for (c= 0; c < PDE.nVar(); ++c)
            qh[i][c] = 0.0;

        for (q = 0; q < NGP; ++q) {

            PDE.initCond(x[i] + dx*xGP[q],Q0);

            for (c = 0; c < PDE.nVar(); ++c)
                qh[i][c] += wGP[q]*Q0[c];
        }
        
        for (c = 0; c < PDE.nVar(); ++c)
            qhi[i][c] = qh[i][c]; 
    }
}

//----------------------------------------------------------------------------
// Apply boundary conditions
//----------------------------------------------------------------------------

template <typename T>
void HyPE_1D<T>::applyBoundaryConditions() {

    int i,c,oned_begin, oned_end, ilhs, irhs;
    multi_array<double,1> QL(extents[PDE.nVar()]), QR(extents[PDE.nVar()]);

    // ---------------------- Left boundary ----------------------

    oned_begin = 0; oned_end = nCells-1;

    for (i = 0; i < N_ph; ++i) {

        ilhs = oned_begin - i - 1;

        // Periodic boundary condition

        if (left == 0) {

            irhs = oned_end - i;

            for (c = 0; c < PDE.nVar(); ++c)
                qh[ilhs][c] = qh[irhs][c];

        }

        // Other boundary condition

        else {

            irhs = oned_begin + i;

            for (c = 0; c < PDE.nVar(); ++c)
                QL[c] = qh[irhs][c];

            PDE.getRightState(QL,left,QR);

            for (c = 0; c < PDE.nVar(); ++c)
                qh[ilhs][c] = QR[c];
        }
    }

    // ---------------------- Right boundary ----------------------

    for (i = 0; i < N_ph; ++i) {

        ilhs = oned_end + i + 1;

        // Periodic boundary condition

        if (right == 0) {

            irhs = oned_begin + i;

            for (c = 0; c < PDE.nVar(); ++c)
                qh[ilhs][c] = qh[irhs][c];
        }

        // Other boundary condition

        else {

            irhs = oned_end - i;

            for (c = 0; c < PDE.nVar(); ++c)
                QL[c] = qh[irhs][c];

            PDE.getRightState(QL,right,QR);

            for (c = 0; c < PDE.nVar(); ++c)
                qh[ilhs][c] = QR[c];
        }
    }
}

//----------------------------------------------------------------------------
// Minmod Reconstruction
//----------------------------------------------------------------------------

template <typename T>
void HyPE_1D<T>::minmod(const multi_array<double,1>& Q, double& qL, double& qR) const {

    double a = Q[1] - Q[0];
    double b = Q[0] - Q[-1]; 
    double slope; 
    
    if (a*b < 0.0)
        slope = 0.0; 
    else {
        if (std::abs(a) < std::abs(b))
            slope = a; 
        else
            slope = b;     
    }    

    qL = Q[0] - 0.5*slope; 
    qR = Q[0] + 0.5*slope; 
}





//----------------------------------------------------------------------------
// THINC Reconstruction
//----------------------------------------------------------------------------


double calc_delta(double q_im1,double q_i,double q_ip1){
    double d1, d2, M;
    d1 = std::abs(q_i-q_im1);
    d2 = std::abs(q_ip1-q_i);
    M  = std::max(d1,d2) + small_num;
    
    return std::min(d1,d2)/M;
}

double beta_linear(double q_im1, double q_i, double q_ip1){

    double delta = calc_delta(q_im1, q_i, q_ip1);
  
                                                             
    double delta1 = 0.617714274258482;   
    double delta2 = 0.886534319811552;
                                      
                                                                
    if(delta < delta1)
        return 1.6;
    else if(delta >= delta1 && delta<= delta2){
        double m = (1.6-std::log(3.))/(delta1 - delta2);
        
        return 1.6 + m*(delta - delta1);
    }
    else{
        return std::log(3.);
    
    }
}


double sigmoid(double x) {
    return 1./(1. + std::exp(-x));
}



double beta_sigm(double q_im1, double q_i, double q_ip1){

    double delta = calc_delta(q_im1, q_i, q_ip1);
  
                                                             
    //double a = 1.6, b = std::log(3.), k = 30.9405, d = -0.415834;  //x0 = 0.47 for data generation
    
    double a = 1.6, b = std::log(3.), k = 54.3893, d = -0.752935;  //x0 = 0.47 for data generation

                
                                                                
   double beta_sigmoid = a + (b-a)*1./(1.+std::exp(-k*(delta+d)));
   
   return beta_sigmoid;
   
}


double beta_NN(double q_im1, double q_i, double q_ip1) {
    
    double b0[] = {2.619666337966918945e+00, -2.924051284790039062e+00, -2.807695865631103516e+00};
    double b1[] = {-5.585313439369201660e-01, -1.093235015869140625e+00};
    double b2[] = {6.471831321716308594e+00};
    
    double W0[3][1] = {
        {-3.743696928024291992e+00},
        {4.118163108825683594e+00},
        {3.971714735031127930e+00}
    };
    
    double W1[2][3] = {
        {-8.126420021057128906e+00, 3.645585536956787109e+00, 3.279516220092773438e+00},
        {-7.788772106170654297e+00, 3.839841127395629883e+00, 4.147911071777343750e+00}
        
    };
    
    double W2[1][2] = {
        {-7.096570491790771484e+00, -8.373348236083984375e+00}
    };
    
    
    

    
    double input[1] = {calc_delta(q_im1, q_i, q_ip1)};
    
    double y[3],z[2], beta_[1];
    
    for (int i = 0; i < 3; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < 1; ++j) {
            y[i] += (W0[i][j] * input[j]);
            
        }
        y[i] += b0[i];
    }
    
    for (int i = 0; i < 3; ++i) {
        y[i] = sigmoid(y[i]); 
        
        
    }
    
    
    for (int i = 0; i < 2; ++i) {
        z[i] = 0.0;
        for (int j = 0; j < 3; ++j) {
            z[i] = z[i] + (W1[i][j] * y[j]);
            
        }
        z[i] = z[i] + b1[i];
        
    }
    
    for (int i = 0; i < 2; ++i) {
        z[i] = sigmoid(z[i]);
        
    }
     
   for (int i = 0; i < 1; ++i) {
        beta_[i] = 0.0;
        for (int j = 0; j < 2; ++j) {
            beta_[i] += (W2[i][j] * z[j]);
        }
        
        beta_[i] += b2[i];
    }
    
    for (int i = 0; i < 1; ++i)
        beta_[i] = (1.6 - std::log(3.))*sigmoid(beta_[i]) + std::log(3.);
    
    
    
    return beta_[0];
}


template <typename T>
void HyPE_1D<T>::THINC(const multi_array<double,1>& Q, const double& beta, double& qL, double& qR) const {



        if ((Q[1]-Q[0])*(Q[0]-Q[-1]) < 0.0 ) {
            qL = Q[0];
            qR = Q[0];   
        }

        else {
            

            double min, max, gamma; 
            
            min = std::min(Q[-1],Q[1]);
            max = std::max(Q[-1],Q[1]); 
            
            if (Q[-1] < Q[1])
                gamma = 1.0; 
            else
                gamma = -1.0; 

            double C = (Q[0] - min + 1.0e-20)/(max-min+1.0e-20); 
            double B = std::exp(gamma*beta*(2.*C-1.)); 
            double A = (B/std::cosh(beta)  - 1.)/std::tanh(beta);
            double D = (std::tanh(beta) + A)/(1. + A*std::tanh(beta));

            qL = min + 0.5*(max-min)*(1. + gamma*A);
            qR = min + 0.5*(max-min)*(1. + gamma*D); 
        
        }

 
     
}


//----------------------------------------------------------------------------
// Monotized-Central limiter
//----------------------------------------------------------------------------

template <typename T>
void HyPE_1D<T>::mclim(const multi_array<double,1>& Q, double& qL, double& qR) const {

  

        double a = Q[0]-Q[-1];
        double b = Q[1]-Q[ 0];

        double slope;

        if (a*b < small_num)
            slope = 0.0;     
        
        else {

            double BETA= 2.0;         
            slope = std::min(0.5*std::abs(a+b), std::min(BETA*std::abs(a), BETA*std::abs(b)));

            if (a < 0.0)
                slope *= -1.; 
        }


        qL = Q[0] - 0.5*slope;
        qR = Q[0] + 0.5*slope;  

    
}


//----------------------------------------------------------------------------
// WENO3 reconstruction
//----------------------------------------------------------------------------

template <typename T>
void HyPE_1D<T>::WENO5(const multi_array<double,1>& Q, double& qL, double& qR) const {

    int i; 

    const double small_num = 1.0e-12;
    const double gammaHi = 0.85;
    const double gammaLo = 0.85;

    double u_xR3[3], u_xxR3[3], IS_R3[3], w_R5, w_R3[3], gammaR3[3], total;

    double gammaR5 = gammaHi;
    gammaR3[0] = (1.0-gammaHi)*gammaLo;
    gammaR3[1] = 0.5*(1.0-gammaHi)*(1.0-gammaLo);
    gammaR3[2] = 0.5*(1.0-gammaHi)*(1.0-gammaLo);

    double wt_ratio;

    // Fifth order stencil

    double u_0   = Q[0];
    double u_xR5  = (-82.*Q[-1] + 11.*Q[-2] + 82.*Q[1] - 11.*Q[2])/120.;;
    double u_x2R5 = (40.*Q[-1] - 3.*Q[-2] - 74.*Q[0] + 40.*Q[1] - 3.*Q[2])/56.;
    double u_x3R5 = (2.*Q[-1] - Q[-2] - 2.*Q[1] + Q[2])/12.;
    double u_x4R5 = (-4.*Q[-1] + Q[-2] + 6.*Q[0] - 4.*Q[1] + Q[2])/24.;
    double tempa = u_xR5 + 0.1*u_x3R5;
    double tempb = u_x2R5 + (123./455.)*u_x4R5;
    double IS_R5 = tempa*tempa + (13./3.)*tempb*tempb + (781./20.)*u_x3R5*u_x3R5 + (1421461./2275.)*u_x4R5*u_x4R5;

    // Central third

    u_xR3[0] = 0.5*(Q[1] - Q[-1]);
    u_xxR3[0] = 0.5*(Q[-1] - 2.0*Q[0] + Q[1]);

    // Left stencil

    u_xR3[1]  = -2.0*Q[-1] + 0.5*Q[-2] + 1.5*Q[0];
    u_xxR3[1] =  0.5*(Q[-2] - 2.0*Q[-1] + Q[0]);

    // Right stencil

    u_xR3[2]  = -1.5*u_0 + 2.0*Q[1] - 0.5*Q[2];
    u_xxR3[2] = 0.5*(u_0 - 2.0*Q[1] + Q[2]);

    for (i = 0; i < 3; ++i)
        IS_R3[i] = u_xR3[i]*u_xR3[i] + (13./3.)*u_xxR3[i]*u_xxR3[i];

    double tau = (1./3.)*(std::abs(IS_R5 - IS_R3[0]) + std::abs(IS_R5 - IS_R3[1]) + std::abs(IS_R5 - IS_R3[2])); 
    
    tau = tau*tau; 

    w_R5    = gammaR5*(1. + tau/((IS_R5+small_num)*(IS_R5+small_num)));          total  = w_R5; 
    w_R3[0] = gammaR3[0]*(1. + tau/((IS_R3[0]+small_num)*(IS_R3[0]+small_num))); total += w_R3[0]; 
    w_R3[1] = gammaR3[1]*(1. + tau/((IS_R3[1]+small_num)*(IS_R3[1]+small_num))); total += w_R3[1];  
    w_R3[2] = gammaR3[2]*(1. + tau/((IS_R3[2]+small_num)*(IS_R3[2]+small_num))); total += w_R3[2];

    w_R5 = w_R5/total; w_R3[0] = w_R3[0]/total; w_R3[1] = w_R3[1]/total; w_R3[2] = w_R3[2]/total;

    wt_ratio = w_R5/gammaR5;

    double u_x    = wt_ratio*(u_xR5 - gammaR3[0]*u_xR3[0] - gammaR3[1]*u_xR3[1] - gammaR3[2]*u_xR3[2]) +  w_R3[0]*u_xR3[0] + w_R3[1]*u_xR3[1] + w_R3[2]*u_xR3[2];
    double u_xx   = wt_ratio*(u_x2R5 - gammaR3[0]*u_xxR3[0] - gammaR3[1]*u_xxR3[1] - gammaR3[2]*u_xxR3[2]) +  w_R3[0]*u_xxR3[0] + w_R3[1]*u_xxR3[1] + w_R3[2]*u_xxR3[2];
    double u_xxx  = wt_ratio*u_x3R5;
    double u_xxxx = wt_ratio*u_x4R5;

    qL = u_0 - u_x/2. + u_xx/6. - u_xxx/20. + u_xxxx/70.;
    qR = u_0 + u_x/2. + u_xx/6. + u_xxx/20. + u_xxxx/70.;
}

//----------------------------------------------------------------------------
// Compute the RHS in each cell
//----------------------------------------------------------------------------

template <typename T>
void HyPE_1D<T>::computeRHS() {

    int i, j, c; 
    double beta;
    double beta_s = std::log(3.), beta_l = 1.6;
    double qiph_L_A, qiph_R_A, qiph_L_B, qiph_R_B;
    double qimh_L_A, qimh_R_A, qimh_L_B, qimh_R_B;
    double W, TBV_A = 0.0, TBV_B = 0.0;
    
    
    
    const double r1_dx = 1./dx;
    multi_array<double,1> Flux(extents[PDE.nVar()]);
    multi_array<double,1> Qbar(extents[PDE.nVar()]),C0(extents[PDE.nVar()]),Q0(extents[PDE.nVar()]);
    multi_array<double,1> QL(extents[PDE.nVar()]), QR(extents[PDE.nVar()]);
    multi_array<double,1> CL(extents[PDE.nVar()]), CR(extents[PDE.nVar()]);

    int end_point = (3+1)/2; // (order + 1)/2
    multi_array<double,1> stencil(extents[multi_array_types::extent_range(-end_point,end_point+1)]);
    multi_array<double,1> stencil_L(extents[multi_array_types::extent_range(-1,2)]);
    multi_array<double,1> stencil_R(extents[multi_array_types::extent_range(-1,2)]);
    
    multi_array<double,2> Q(extents[multi_array_types::extent_range(-end_point,end_point+1)][PDE.nVar()]);
    multi_array<double,2> C(extents[multi_array_types::extent_range(-end_point,end_point+1)][PDE.nVar()]);

    bool PAD, is_corrupt;

    applyBoundaryConditions();

    for (i = -1; i < nCells+1; ++i) {

        is_corrupt = false;

        for (c = 0; c < PDE.nVar(); ++c) {

            for (j = -end_point; j < end_point+1; ++j)
                Q[j][c] = qh[i+j][c];

            Qbar[c] = qh[i][c];
        }


        for (j = -end_point; j < end_point+1; ++j) {
            for (c = 0; c < PDE.nVar(); ++c)
                Q0[c] = Q[j][c];

            PDE.project2CharacteristicSpace(Qbar,Q0,C0);

            for (c = 0; c < PDE.nVar(); ++c)
                C[j][c] = C0[c];
        }


        for (c = 0; c < PDE.nVar(); ++c) {

            for (j = -end_point; j < end_point+1; ++j)
                stencil[j] = C[j][c];

            //WENO5(stencil,CL[c],CR[c]);
            beta = beta_NN(stencil[-1], stencil[0], stencil[1]);
            THINC(stencil, beta, CL[c], CR[c]);
            //minmod(stencil,CL[c], CR[c]);
            //mclim(stencil,CL[c], CR[c]);
            
            
           /*        // TBV here
                   
       stencil_L[-1] = stencil[-2];
       stencil_L[0]  = stencil[-1];
       stencil_L[1]  = stencil[0];
       
       stencil_R[-1] = stencil[0];
       stencil_R[0]  = stencil[1];
       stencil_R[1]  = stencil[2];
       
       
       
        //Scheme A : THINC with beta_l = 1.6
        THINC(stencil_L,   beta_l, W, qimh_L_A);
        THINC(stencil, beta_l, qimh_R_A, qiph_L_A);
        THINC(stencil_R, beta_l, qiph_R_A, W);
        
        //Scheme B : THINC with beta_s = log(3) or MC
        
        
        THINC(stencil_L, beta_s, W, qimh_L_B);
        THINC(stencil, beta_s, qimh_R_B, qiph_L_B);
        THINC(stencil_R, beta_s, qiph_R_B, W);
        
        //mclim(stencil_L, W, qimh_L_B);
        //mclim(stencil, qimh_R_B, qiph_L_B);
        //mclim(stencil_R,  qiph_R_B, W);
        
        
        TBV_A = std::min( {std::abs(qimh_L_A - qimh_R_A) + std::abs(qiph_L_A - qiph_R_A),
        std::abs(qimh_L_A - qimh_R_A) + std::abs(qiph_L_A - qiph_R_B),
        std::abs(qimh_L_B - qimh_R_A) + std::abs(qiph_L_A - qiph_R_A),
        std::abs(qimh_L_B - qimh_R_A) + std::abs(qiph_L_A - qiph_R_B) }  );
        

        
        TBV_B = std::min( {std::abs(qimh_L_A - qimh_R_B) + std::abs(qiph_L_B - qiph_R_A),
        std::abs(qimh_L_A - qimh_R_B) + std::abs(qiph_L_B - qiph_R_B),
        std::abs(qimh_L_B - qimh_R_B) + std::abs(qiph_L_B - qiph_R_A),
        std::abs(qimh_L_B - qimh_R_B) + std::abs(qiph_L_B - qiph_R_B) }  );

        if(TBV_A < TBV_B) //choose Scheme A 
            THINC(stencil, beta_l, CL[c], CR[c]);
        else
            THINC(stencil, beta_s, CL[c], CR[c]);
        
             // TBV here*/
             
            
            
        }

        PDE.project2RealSpace(Qbar,CL,QL);
        PDE.project2RealSpace(Qbar,CR,QR);

        PAD = PDE.checkPAD(QL);

        if (PAD == false)
            is_corrupt = true;

        PAD = PDE.checkPAD(QR);

        if (PAD == false)
            is_corrupt = true;

        if (is_corrupt) {
            for (c = 0; c < PDE.nVar(); ++c) {
                QL[c] = Qbar[c];
                QR[c] = Qbar[c];
            }
        }

        for (c = 0; c < PDE.nVar(); ++c) {
            qbnd[i][0][c] = QL[c];
            qbnd[i][1][c] = QR[c];
        }

    } // cell loop

    // Find upwind flux

    for (i = 0; i < nCells + 1; ++i) {

        for (c = 0; c < PDE.nVar(); ++c) {
            QL[c] = qbnd[i-1][1][c];
            QR[c] = qbnd[i][0][c];
        }

        PDE.riemannSolver(QL,QR,Flux);

        for (c = 0; c < PDE.nVar(); ++c)
            F[i][c] = Flux[c];
    }

    // Find RHS

    for (i = 0; i < nCells; ++i)
        for (c = 0; c < PDE.nVar(); ++c)
            dqh[i][c] = -r1_dx*(F[i+1][c] - F[i][c]);
}

//----------------------------------------------------------------------------
// Advance solution through one time step (SSPRK54)
//----------------------------------------------------------------------------

template <typename T>
void HyPE_1D<T>::step_SSPRK33() {


//for SSPRK3
    
    int i, c;
    double old_time;

    old_time = time;

    // Stage 1

    computeRHS();

    dt = CFL*dx/PDE.maxEigenValue(); 
    
    if (dt+time>tend)
        dt = tend-time; 

    PDE.resetMaxEigenValue();

    for (i = 0; i < nCells; ++i) {
        for (c = 0; c < PDE.nVar(); ++c) {
            qh0[i][c] = qh[i][c];
            qh[i][c] = qh0[i][c]  + dt*dqh[i][c]; 
        }
    }
    
     computeRHS();
        
        //Second stage
        for (int i = 0; i < nCells; ++i)
        {
            for (int c = 0; c < PDE.nVar(); ++c)
            {
                qh[i][c] = 0.25*(qh[i][c] + 3.0*qh0[i][c] + dt * dqh[i][c]);
            }
        }
        
     computeRHS();
        
        //Third Stage
        for (int i = 0; i < nCells; ++i)
        {
            for (int c = 0; c < PDE.nVar(); ++c)
            {
                qh[i][c] = (2.0/3.0)*qh[i][c] + (1.0/3.0)*qh0[i][c] + (2.0/3.0)*dt * dqh[i][c];
            }
        }
        
       
    


//SSPRK(5,4)
    /*int i, c;
    double old_time;

    old_time = time;

    // Stage 1

    computeRHS();

    dt = CFL*dx/PDE.maxEigenValue(); 
    
    if (dt+time>tend)
        dt = tend-time; 

    PDE.resetMaxEigenValue();

    for (i = 0; i < nCells; ++i) {
        for (c = 0; c < PDE.nVar(); ++c) {
            qh0[i][c] = qh[i][c];
            qh[i][c] = qh0[i][c]  + dt*0.391752226571890*dqh[i][c]; // v1
        }
    }


    // Stage 2

    time = old_time + 0.39175222700392*dt;

    computeRHS();

    for (i = 0; i < nCells; ++i) {

        for (c = 0; c < PDE.nVar(); ++c) {

            qh[i][c] = 0.444370493651235*qh0[i][c] + 0.555629506348765*qh[i][c]  + 0.368410593050371*dt*dqh[i][c]; // v2;
            qh2[i][c]= qh[i][c];

        }
    }

    // Stage 3

    time = old_time + 0.58607968896780*dt;

    computeRHS();

    for (i = 0; i < nCells; ++i) {
        for (c = 0; c < PDE.nVar(); ++c) {
            qh[i][c] = 0.620101851488403*qh0[i][c] + 0.379898148511597*qh[i][c] + 0.251891774271694*dt*dqh[i][c]; // v3
            qh3[i][c] = qh[i][c];
        }
    }
       
    // Stage 4

    time = old_time + 0.47454236302687*dt;

    computeRHS();

    for (i = 0; i < nCells; ++i) {
        for (c = 0; c < PDE.nVar(); ++c) {
            dqh3[i][c] = dqh[i][c];
            qh[i][c] = 0.178079954393132*qh0[i][c] + 0.821920045606868*qh[i][c] + 0.544974750228521*dt*dqh[i][c]; // v4
        }
    }

    // Stage 5

    time = old_time + 0.93501063100924*dt;

    computeRHS();

    for (i = 0; i < nCells; ++i) {
        for (c = 0; c < PDE.nVar(); ++c) {
            qh[i][c] = 0.517231671970585*qh2[i][c] + 0.096059710526147*qh3[i][c] + 0.386708617503269*qh[i][c]  +
                                    dt*(0.063692468666290*dqh3[i][c] + 0.226007483236906*dqh[i][c]); // n+1
        }
    }

    time = old_time;*/
}

//----------------------------------------------------------------------------
// Compute max error norm at the end of simulation. Works only for
// periodic test cases where end of solution and initial condition are same
//----------------------------------------------------------------------------

template <typename T>
double HyPE_1D<T>::maxError(int c) const {
    
    int i;     
    double error, max_error = 0.0; 
    for (i = 0; i < nCells; ++i) {
        error = std::abs( qh[i][c] - qhi[i][c] ); 
        if (error > max_error)
            max_error = error; 
    }
    
    return max_error; 
}

//----------------------------------------------------------------------------
// Plot solution as csv file
//----------------------------------------------------------------------------

template <typename T>
void HyPE_1D<T>::plot(const std::string filename) const {

    int i, c; 
    multi_array<double,1> Q(extents[PDE.nVar()]), V(extents[PDE.nVar()]);
    
    std::ofstream out_data;
    out_data.open (filename);
    out_data.flags( std::ios::dec | std::ios::scientific );
    out_data.precision(6);

    
    out_data << "x,";
    
    for (c = 0; c < PDE.nVar(); ++c) {
        if (c == PDE.nVar()-1) 
            out_data << PDE.getFieldName(c) << std::endl;     
        else
            out_data << PDE.getFieldName(c) <<  ",";    
    }   
    
    for (i = 0; i < nCells; ++i) {
        for (c = 0; c < PDE.nVar(); ++c) 
            Q[c] = qh[i][c]; 
        PDE.cons2Prim(Q,V); 
        out_data << x[i] << ","; 
        
        for (c = 0; c < PDE.nVar(); ++c) {
            if (c == PDE.nVar()-1) 
                out_data << V[c] << std::endl;     
            else
                out_data << V[c] <<  ",";       
        }

    }

    out_data.close();
}

//----------------------------------------------------------------------------
// Put everything together and run the problem
//----------------------------------------------------------------------------

template <typename T>
void HyPE_1D<T>::run() {

    auto start = std::chrono::system_clock::now();

    //-------------------------------------

    computeRHS();

   
    while (time < tend) {

        if (time_step%10 == 0)
            printf ("t = %4.2e\n", time);

        step_SSPRK33();

        time += dt;
        time_step++;        
    }
   
    printf ("t = %4.2e\n", time);
    std::cout << "No. of times steps = " << time_step << std::endl; 

    //-------------------------------------

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    std::cout << "CPU time (s) = " << elapsed_seconds.count() << std::endl;
}

#endif /* HYPE_H_ */
