#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#define BOOST_DISABLE_ASSERTS
#include "boost/multi_array.hpp"

using namespace boost;

const double adv = 1.0;   // advection coefficient
const double alpha = 0.0; // diffusion coefficient
const double small_num = 1.0e-12; 

//----------------------------------------------------------------------------
// Gauss Quadrature points and weights
//----------------------------------------------------------------------------

const int NGP = 5;
const double xGP[] = {-0.453089922969332,-0.26923465505284155,0.0,0.26923465505284155,0.453089922969332};
const double wGP[] = {0.11846344252809471,0.2393143352496831,0.2844444444444445,0.2393143352496831,0.11846344252809471};


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

double beta_sigm(double q_im1, double q_i, double q_ip1){

    double delta = calc_delta(q_im1, q_i, q_ip1);
  
                                                             
    //double a = 1.6, b = std::log(3.), k = 54.38931363, d = -0.75293469;  //x0 = 0.15 for data generation
    double a = 1.6, b = std::log(3.), k = 30.94046122, d = -0.41583446;  //x0 = 0.15 for data generation
                
                                                                
   double beta_sigmoid = a + (b-a)*1./(1.+std::exp(-k*(delta+d)));
   
   return beta_sigmoid;
   
}


double beta_binary(double q_im1, double q_i, double q_ip1){

    double delta = calc_delta(q_im1, q_i, q_ip1);
  
                                                             
    double delta0 = 0.755;   
   
                                   
                                                                
    if(delta < delta0)
        return 1.6;
   
    else{
        return std::log(3.);
    
    }
}

double calc_eta(double q_im1,double q_i,double q_ip1){
    
    double r;
    
    r = (q_i-q_im1)/(q_ip1-q_i + 1.0e-12);
    
    return 4.*std::pow(r,4.)/std::pow((1.0 +  std::pow(r,4)),2 ) ;

}

double beta_eta_linear(double q_im1, double q_i, double q_ip1){

    double eta = calc_eta(q_im1, q_i, q_ip1);
  
    
    double eta1 = 0.46269816939068;  
    double eta2 = 0.9555970707617626;
    
    if(eta < eta1)
        return 1.6;
    else if(eta >= eta1 && eta<= eta2){
        double m = (1.6-std::log(3.))/(eta1 - eta2);
        
        return 1.6 + m*(eta - eta1);
    }
    else{
        return std::log(3.);
    
    }
}

//----------------------------------------------------------------------------------
// Flux for advection diffusion equation
//----------------------------------------------------------------------------------

double advFlux(double uL, double uR) {
    
    
    return 0.5*(adv*(uL+uR) - std::abs(adv)*(uR-uL));
    
    
}

double diffFlux (double grad_uL, double grad_uR) {
    return -0.5*alpha*(grad_uL+grad_uR);
}

double F(double x, double alpha, double a) {
    return std::sqrt( std::max(  1.0 - alpha*alpha*(x-a)*(x-a), 0.0  ));
}

double G(double x, double beta, double z) {
    return std::exp(-beta*(x-z)*(x-z));
}


double jiangShuWave(double x) {

    double a = 0.5;
    double z = -0.7;
    double delta = 0.005;
    double alpha  = 10.0;
    double beta = std::log(2.0)/(36.0*delta*delta);

    if (x >= -0.8 && x < -0.6)
        return (1./6.)*(G(x,beta,z-delta) + G(x,beta,z+delta) + 4.0*G(x,beta,z) );
    else if (x >= -0.4 && x < -0.2)
        return 1.0;
    else if (x >= 0.0 && x < 0.2)
        return 1.0 - std::abs(10.0*(x-0.1));
    else if (x >= 0.4 && x < 0.6)
        return (1./6.)*(F(x,alpha,a-delta) + F(x,alpha,a+delta) + 4.0*F(x,alpha,a) );
    else
        return 0.0;

}

double semiEllipse(double x){
    double a = 0.5;
    double z = -0.7;
    double delta = 0.005;
    double alpha  = 10.0;
    double beta = std::log(2.0)/(36.0*delta*delta);
    
    if (x >= 0.4 && x < 0.6)
        return (1./6.)*(F(x,alpha,a-delta) + F(x,alpha,a+delta) + 4.0*F(x,alpha,a) );
    else
        return 0.0;


}
//----------------------------------------------------------------------------
// Main Class 
//----------------------------------------------------------------------------

class Thinc {
    const int N = 9; // Degree of approximation
    const int N_ph = 6;
    const double CFL = 0.2;

    double xmin, xmax;
    int IMAX;
    double tend;

    multi_array<double,1> qh;   // Conserved variables at cells
    multi_array<double,1> qhe;   // Exact Solution(only for periodic problems at the end of integer cycles)
    multi_array<double,1> F;    // Upwind Conservative fluxes at cell faces
    multi_array<double,1> dqh;  // RHS term for each face
    multi_array<double,2> qbnd; // Value of conserved variable at cell faces
    multi_array<double,1> x;    // Cell centers
    
    
    
    multi_array<double,1> betas;
    multi_array<double,1> deltas;
    multi_array<double,1> track;
    
    // Additional storage for time stepping

    multi_array<double,1> qh0;
    multi_array<double,1> qh2;
    multi_array<double,1> qh3;
    multi_array<double,1> dqh1;
    multi_array<double,1> dqh2;
    multi_array<double,1> dqh3;
    multi_array<double,1> dqh4;
    multi_array<double,1> dqh5;
    multi_array<double,1> dqh6;
    multi_array<double,1> dqh7;
    multi_array<double,1> dqh8;
    multi_array<double,1> dqh9;
    multi_array<double,1> dqh10;
    multi_array<double,1> dqh11;
    multi_array<double,1> dqh12;
    multi_array<double,1> dqh13;
    
    


    double dx;
    double dt;
    double time;
    int time_step;
    
    double initialCondition(double) const; 
    void reconstructTVD(const multi_array<double,1>&,double&,double&);
    void reconstructWENO(const multi_array<double,1>&,double&,double&);
    void reconstructTHINC(const multi_array<double,1>&,double&,double&);
    void applyBoundaryConditions();
    void calcTimeStep();
    void computeRHS();
    void updateSolution();
    void track_beta();
    
public:
    Thinc(double,double,int,double);
    void errorNorms() const;
    void run();
    void plot() const;
};

//----------------------------------------------------------------------------------
// Tanh-Profile
//----------------------------------------------------------------------------------

double tanh1d(double x, double x0, double h, double L, double R) {

    return 0.5*((L+R) + (R-L)*std::tanh((x-x0)/(h)));
}


//----------------------------------------------------------------------------------
// Square Wave
//----------------------------------------------------------------------------------

double squareWave(double x) {

    if(std::abs(x) < 0.5 )
        return 1.0;
    else
        return 0.0;

    
}



//----------------------------------------------------------------------------------
// Initial Condition
//----------------------------------------------------------------------------------




double Thinc::initialCondition(double xx) const {//icond
      
    
    double eps = 1.0e-8;    
    double x0 = 0.5;
    double smear = 2.0*dx; 
    double y;
    //y = (tanh1d(xx, x0, smear, 1.0-eps, eps) + tanh1d(xx, -x0, smear, eps, 1.-eps)) -(1.0-eps);
    //y = std::sin(M_PI*xx);
    //y = jiangShuWave(xx);
    //y = semiEllipse(xx);
    y = squareWave(xx);
    
    return y;
    
 

}





//----------------------------------------------------------------------------
// Constructor - Allocate memory and initialize solution
// using initial conditions
//----------------------------------------------------------------------------

Thinc::Thinc(double _xmin, double _xmax, int _IMAX, double _tend) :
    xmin(_xmin),
    xmax(_xmax),
    IMAX(_IMAX),
    tend(_tend),
    qh(extents[multi_array_types::extent_range(-N_ph,IMAX+N_ph)]),
    qhe(extents[multi_array_types::extent_range(-N_ph,IMAX+N_ph)]),
    F(extents[IMAX + 1]),
    dqh(extents[IMAX]),
    qbnd(extents[multi_array_types::extent_range(-N_ph,IMAX+N_ph)][2]), // 2 is for two faces per cell
    x(extents[IMAX]),
    qh0(extents[IMAX]),
    qh2(extents[IMAX]),
    qh3(extents[IMAX]),
    dqh1(extents[IMAX]),
    dqh2(extents[IMAX]),
    dqh3(extents[IMAX]),
    dqh4(extents[IMAX]),
    dqh5(extents[IMAX]),
    dqh6(extents[IMAX]),
    dqh7(extents[IMAX]),
    dqh8(extents[IMAX]),
    dqh9(extents[IMAX]),
    dqh10(extents[IMAX]),
    dqh11(extents[IMAX]),
    dqh12(extents[IMAX]),
    dqh13(extents[IMAX]),
    betas(extents[IMAX]),
    deltas(extents[IMAX]),
    dt(0.0),
    time(0.0),
    time_step(0),
    track(extents[IMAX+1])
    
{

    // Initialize the grid

    dx = (xmax - xmin)/static_cast<double>(IMAX);

    for (int i = 0; i < IMAX; ++i)
        x[i] = xmin + ((i+1)-0.5)*dx;

    // Loop through all the cells

    for (int i = 0; i < IMAX; i++) {
        qh[i] = 0.0;
        qhe[i] = 0.0;
        for (int q = 0; q < NGP; ++q) { 
            qh[i] += wGP[q]*initialCondition(x[i] + dx*xGP[q]); //cell average
            qhe[i] += wGP[q]*initialCondition(x[i] + dx*xGP[q]); //exact
        }

    }
    
    for (int i = 0; i < IMAX; i++) {

        betas[i] = beta_linear(qh[i-1], qh[i], qh[i+1]);
    }
    
    

    dt = CFL*dx/std::abs(adv);
}

//----------------------------------------------------------------------------
// Apply boundary conditions
//----------------------------------------------------------------------------

void Thinc::applyBoundaryConditions() {

    int i,oned_begin, oned_end, ilhs, irhs;

    oned_begin = 0; oned_end = IMAX-1;

    for (i = 0; i < N_ph; ++i) {

        // Left boundary 
        
        ilhs = oned_begin - i - 1;
        irhs = oned_end - i; // Periodic 
        qh[ilhs] = qh[irhs];

       // Right boundary

        ilhs = oned_end + i + 1;
        irhs = oned_begin + i; // Periodic

        qh[ilhs] = qh[irhs];
       
    }
}

//----------------------------------------------------------------------------
// THINC Reconstruction 
//----------------------------------------------------------------------------






void THINC(double q_im1, double q_i, double q_ip1, double beta, double& L, double& R) {

    if (  (q_ip1-q_i)*(q_i-q_im1) < 0.0 ) {
        L = q_i;
        R = q_i;   
    }

    else {
        

        double min, max, gamma; 
        
        min = std::min(q_im1,q_ip1);
        max = std::max(q_im1,q_ip1); 
        
        if (q_im1 < q_ip1)
            gamma = 1.0; 
        else
            gamma = -1.0; 

        double C = (q_i - min + 1.0e-20)/(max-min+1.0e-20); 
        double B = std::exp(gamma*beta*(2.*C-1.)); 
        double A = (B/std::cosh(beta)  - 1.)/std::tanh(beta);
        double D = (std::tanh(beta) + A)/(1. + A*std::tanh(beta));

        L = min + 0.5*(max-min)*(1. + gamma*A);
        R = min + 0.5*(max-min)*(1. + gamma*D); 
    
    }
}


void upwind(double q_im1, double q_i, double q_ip1, double& q_imh, double& q_iph) {
    double slope = 0.5*(q_ip1 - q_im1);
    q_imh = q_i - 0.5*slope;
    q_iph = q_i + 0.5*slope;        
}

// Monotized-Central limiter
void mclim(double q_im1, double q_i, double q_ip1, double& q_imh, double& q_iph) {

    double a = q_i-q_im1;
    double b = q_ip1-q_i;

    double slope;

    if (a*b < small_num)
        slope = 0.0;     
    
    else {

        double beta = 2.0;         
        slope = std::min(0.5*std::abs(a+b), std::min(beta*std::abs(a), beta*std::abs(b)));

        if (a < 0.0)
            slope *= -1.; 
    }


    q_imh = q_i - 0.5*slope;
    q_iph = q_i + 0.5*slope;  

    
}

double sigmoid(double x) {
    return 1./(1. + std::exp(-x));
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
//----------------------------------------------------------------------------
// Compute the RHS in each cell
//----------------------------------------------------------------------------

void Thinc::computeRHS() {

    const double r1_dx = 1./dx;

    int end_point = (N+1)/2;

    multi_array<double,1> stencil(extents[multi_array_types::extent_range(-end_point,end_point+1)]);
    multi_array<double,1> coeffs(extents[N+1]);
    
    applyBoundaryConditions();
    
    double beta;
    for (int i = -1; i < IMAX+1; ++i) {

        beta = beta_NN(qh[i-1], qh[i], qh[i+1]);
        
        //beta = 1.6;//std::log(3.);
        //THINC(qh[i-1], qh[i], qh[i+1], beta, qbnd[i][0], qbnd[i][1]);// choose
        mclim(qh[i-1], qh[i], qh[i+1], qbnd[i][0], qbnd[i][1]);
    }

    // Find upwind flux

    for (int i = 0; i < IMAX + 1; ++i)
        F[i] = advFlux(qbnd[i-1][1],qbnd[i][0]);


    // Find RHS

    for (int i = 0; i < IMAX; ++i)
        dqh[i] = -r1_dx*(F[i+1] - F[i]);
}



void Thinc::updateSolution() {

    int i;

    static const double Abar[] = {
    14005451.0 / 335480064.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -59238493.0 / 1068277825.0,
    181606767.0 / 758867731.0,
    561292985.0 / 797845732.0,
    -1041891430.0 / 1371343529.0,
    760417239.0 / 1151165299.0,
    118820643.0 / 751138087.0,
    -528747749.0 / 2220607170.0,
    1.0 / 4.0
    };

    static const double b21 = 1.0 / 18.0;
    static const double b3[] = { 1.0 / 48.0, 1.0 / 16.0 };
    static const double b4[] = { 1.0 / 32.0, 0.0, 3.0 / 32.0 };
    static const double b5[] = { 5.0 / 16.0, 0.0, -75.0 / 64.0, 75.0 / 64.0 };
    static const double b6[] = { 3.0 / 80.0, 0.0, 0.0, 3.0 / 16.0, 3.0 / 20.0 };
    static const double b7[] = {
    29443841.0 / 614563906.0,
    0.0,
    0.0,
    77736538.0 / 692538347.0,
    -28693883.0 / 1125000000.0,
    23124283.0 / 1800000000.0
    };
    static const double b8[] = {
    16016141.0 / 946692911.0,
    0.0,
    0.0,
    61564180.0 / 158732637.0,
    22789713.0 / 633445777.0,
    545815736.0 / 2771057229.0,
    -180193667.0 / 1043307555.0
    };
    static const double b9[] = {
    39632708.0 / 573591083.0,
    0.0,
    0.0,
    -433636366.0 / 683701615.0,
    -421739975.0 / 2616292301.0,
    100302831.0 / 723423059.0,
    790204164.0 / 839813087.0,
    800635310.0 / 3783071287.0
    };
    static const double b10[] = {
    246121993.0 / 1340847787.0,
    0.0,
    0.0,
    -37695042795.0 / 15268766246.0,
    -309121744.0 / 1061227803.0,
    -12992083.0 / 490766935.0,
    6005943493.0 / 2108947869.0,
    393006217.0 / 1396673457.0,
    123872331.0 / 1001029789.0
    };
    static const double b11[] = {
    -1028468189.0 / 846180014.0,
    0.0,
    0.0,
    8478235783.0 / 508512852.0,
    1311729495.0 / 1432422823.0,
    -10304129995.0 / 1701304382.0,
    -48777925059.0 / 3047939560.0,
    15336726248.0 / 1032824649.0,
    -45442868181.0 / 3398467696.0,
    3065993473.0 / 597172653.0
    };
    static const double b12[] = {
    185892177.0 / 718116043.0,
    0.0,
    0.0,
    -3185094517.0 / 667107341.0,
    -477755414.0 / 1098053517.0,
    -703635378.0 / 230739211.0,
    5731566787.0 / 1027545527.0,
    5232866602.0 / 850066563.0,
    -4093664535.0 / 808688257.0,
    3962137247.0 / 1805957418.0,
    65686358.0 / 487910083.0
    };
    static const double b13[] = {
    403863854.0 / 491063109.0,
    0.0,
    0.0,
    -5068492393.0 / 434740067.0,
    -411421997.0 / 543043805.0,
    652783627.0 / 914296604.0,
    11173962825.0 / 925320556.0,
    -13158990841.0 / 6184727034.0,
    3936647629.0 / 1978049680.0,
    -160528059.0 / 685178525.0,
    248638103.0 / 1413531060.0,
    0.0
    };

    // Stage 1

    computeRHS();

    for (i = 0; i < IMAX; ++i) {
        qh0[i] = qh[i];
        dqh1[i] = dqh[i];
        qh[i] = qh0[i] + b21 * dt * dqh1[i];
    }

    // Stage 2

    computeRHS();

    for (i = 0; i < IMAX; ++i) {
        dqh2[i] = dqh[i];
        qh[i] = qh0[i] + dt * (b3[0] * dqh1[i] + b3[1] * dqh2[i]);
    }

    // Stage 3

    computeRHS();

    for (i = 0; i < IMAX; ++i) {
        dqh3[i] = dqh[i];
        qh[i] = qh0[i] + dt * (b4[0] * dqh1[i] + b4[2] * dqh3[i]);
    }

    // Stage 4

    computeRHS();

    for (i = 0; i < IMAX; ++i) {
        dqh4[i] = dqh[i];
        qh[i] = qh0[i] + dt * (b5[0] * dqh1[i] + b5[2] * dqh3[i] + b5[3] * dqh4[i]);
    }

    // Stage 5

    computeRHS();

    for (i = 0; i < IMAX; ++i) {
        dqh5[i] = dqh[i];
        qh[i] = qh0[i] + dt * (b6[0] * dqh1[i] + b6[3] * dqh4[i] + b6[4] * dqh5[i]);
    }

    // Stage 6

    computeRHS();

    for (i = 0; i < IMAX; ++i) {
        dqh6[i] = dqh[i];
        qh[i] = qh0[i] + dt * (b7[0] * dqh1[i] + b7[3] * dqh4[i] + b7[4] * dqh5[i] + b7[5] * dqh6[i]);
    }

    // Stage 7

    computeRHS();

    for (i = 0; i < IMAX; ++i) {
        dqh7[i] = dqh[i];
        qh[i] =
            qh0[i] + dt * (b8[0] * dqh1[i] + b8[3] * dqh4[i] + b8[4] * dqh5[i] +
                  b8[5] * dqh6[i] + b8[6] * dqh7[i]);
    }


    // Stage 8

    computeRHS();

    for (i = 0; i < IMAX; ++i) {
        dqh8[i] = dqh[i];
        qh[i] =
            qh0[i] + dt * (b9[0] * dqh1[i] + b9[3] * dqh4[i] + b9[4] * dqh5[i] +
                  b9[5] * dqh6[i] + b9[6] * dqh7[i] + b9[7] * dqh8[i]);
    }

    // Stage 9

    computeRHS();

    for (i = 0; i < IMAX; ++i) {
        dqh9[i] = dqh[i];
        qh[i] =
            qh0[i] + dt * (b10[0] * dqh1[i] + b10[3] * dqh4[i] + b10[4] * dqh5[i] +
                  b10[5] * dqh6[i] + b10[6] * dqh7[i] + b10[7] * dqh8[i] +
                  b10[8] * dqh9[i]);
    }


    // Stage 10

    computeRHS();

    for (i = 0; i < IMAX; ++i) {
        dqh10[i] = dqh[i];
        qh[i] =
            qh0[i] + dt * (b11[0] * dqh1[i] + b11[3] * dqh4[i] + b11[4] * dqh5[i] +
                  b11[5] * dqh6[i] + b11[6] * dqh7[i] + b11[7] * dqh8[i] +
                  b11[8] * dqh9[i] + b11[9] * dqh10[i]);
    }

    // Stage 11

    computeRHS();

    for (i = 0; i < IMAX; ++i) {
        dqh11[i] = dqh[i];
        qh[i] =
        qh0[i] + dt * (b12[0] * dqh1[i] + b12[3] * dqh4[i] + b12[4] * dqh5[i] +
                  b12[5] * dqh6[i] + b12[6] * dqh7[i] + b12[7] * dqh8[i] +
                  b12[8] * dqh9[i] + b12[9] * dqh10[i] + b12[10] * dqh11[i]);
    }


    // Stage 12

    computeRHS();

    for (i = 0; i < IMAX; ++i) {
        dqh12[i] = dqh[i];
        qh[i] =
            qh0[i] + dt * (b13[0] * dqh1[i] + b13[3] * dqh4[i] + b13[4] * dqh5[i] +
                  b13[5] * dqh6[i] + b13[6] * dqh7[i] + b13[7] * dqh8[i] +
                  b13[8] * dqh9[i] + b13[9] * dqh10[i] + b13[10] * dqh11[i] +
                  b13[11] * dqh12[i]);
    }

    // Stage 13

    computeRHS();

    for (i = 0; i < IMAX; ++i) {
        dqh13[i] = dqh[i];
      const double ksum8 =
        Abar[0] * dqh1[i] + Abar[5] * dqh6[i] + Abar[6] * dqh7[i] +
        Abar[7] * dqh8[i] + Abar[8] * dqh9[i] + Abar[9] * dqh10[i] +
        Abar[10] * dqh11[i] + Abar[11] * dqh12[i] + Abar[12] * dqh13[i];
      qh[i] = qh0[i] + dt * ksum8;
    }

}

void Thinc::calcTimeStep() {

    double s, smax = 0.0;
    double smaxv = alpha;


    //double factor =10.669676460233536*std::pow(dx,1.25);
    double factor = 0.9;

    int i;

    for (i = 0; i < IMAX; ++i) {

        s = adv;

        if (s > smax)
            smax = s;

    }

    dt = factor*CFL*dx/(smax + 2.0*smaxv/dx);

    // If time step exceeds the final time, reduce it accordingly

    if((time + dt)>tend)
        dt = tend - time;
}

//----------------------------------------------------------------------------
// Print L2 and Max error norms at the end of simulation. Works only for
// periodic test cases where end of solution and initial condition are same
//----------------------------------------------------------------------------

void Thinc::errorNorms() const {

    double error, l2 = 0.0, linf = 0.0, l1 = 0.0;

    double exact;

    // Loop through all the cells

    for (int i = 0; i < IMAX; i++) {

        exact = 0.0;

        for (int q = 0; q < NGP; ++q)
            exact += wGP[q]*initialCondition(x[i] + dx*xGP[q] - adv*tend);

        error = std::abs(qh[i] - qhe[i]);

        if (error > linf)
            linf = error;

        l2 += error*error;
        l1 += error;
    }

    l2 = std::sqrt(l2/static_cast<double>(IMAX));
    l1 = l1/static_cast<double>(IMAX);

    //printf ("L2 error = %.7e; L-inf error = %.7e\n", l2, linf);
    printf ("%.7e; %.7e; %.7e\n", l1, l2, linf);
}


void Thinc::track_beta(){
    double x0 = 1.505;
    for(int i = 0; i<IMAX; i++){
    
   
        if(std::abs(x[i] - fmod(x0 +  time, 2.0) ) < small_num ) 
            std::cout<<x[i]<<", "<<betas[i]<<std::endl;
    }
}
//----------------------------------------------------------------------------
// Put everything together and run the problem
//----------------------------------------------------------------------------

/*
void Thinc::run() {

    auto start = std::chrono::system_clock::now();

    //-------------------------------------

    //updateSolution();
    
   
   
   int j = 0;
    //while (time_step < 1001) {
    while (time < tend) {

        //printf ("%d, t = %4.3e\n", time_step, time);

        // If time step exceeds the final time, reduce it accordingly

        if((time + dt)>tend)
            dt = tend - time;
        
        updateSolution();
        
        if(time_step%5 == 0){
            //track[j] = betas[152 + j];
            //std::cout<<track[j]<<std::endl;
            //j = j + 1;
           
            track_beta();
            //printf ("%d, t = %4.3e\n", time_step, time);
            
        }
            

        time += dt;
        time_step++;
    }
   
    printf ("%d, t = %4.3e\n", time_step, time);

    //-------------------------------------

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    std::cout << "Time taken = " << elapsed_seconds.count() << std::endl;
}

*/

void Thinc::run() {

    auto start = std::chrono::system_clock::now();

    //-------------------------------------

    computeRHS();


    
    while (time < tend) {

        printf ("%d, time = %4.3e, dt = %4.3e\n", time_step, time, dt);

        calcTimeStep();
        updateSolution();

        time += dt;
        time_step++;
    }
   

    printf ("%d, time = %4.3e, dt = %4.3e\n", time_step, time, dt);

    //-------------------------------------

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    std::cout << "Time taken = " << elapsed_seconds.count() << std::endl;
}




//----------------------------------------------------------------------------
// Plot solution as csv file
//----------------------------------------------------------------------------

void Thinc::plot() const {

    std::ofstream out_data;
    const std::string filename = "sol_mc.csv";
    out_data.open (filename);
    out_data.flags( std::ios::dec | std::ios::scientific );
    out_data.precision(6);

    out_data << "x,u\n";

    for (int i = 0; i < IMAX; ++i) {
        out_data << x[i] << "," << qh[i] << std::endl;
    }

    out_data.close();
    
    std::ofstream out_data1;
    const std::string filename1 = "deltas.csv";
    out_data1.open (filename1);
    out_data1.flags( std::ios::dec | std::ios::scientific );
    out_data1.precision(6);
    out_data1 << "x,delta\n";

    for (int i = 0; i < IMAX; ++i) {
        out_data1 << x[i] << "," << deltas[i] <<" beta: "<<betas[i]<<", Stencil : "<<qh[i-1]<<", "<< qh[i]<<", "<< qh[i+1]<< std::endl;
    }

    out_data1.close();
}

//----------------------------------------------------------------------------
// Main
//----------------------------------------------------------------------------

int main() {


    double xmin = -1.0;
    double xmax =  1.0;
    int IMAX = 400;
    double tend = (100.)*2.0;


    Thinc Sol(xmin, xmax, IMAX, tend);
    Sol.run();
    Sol.plot();
    Sol.errorNorms();

    return 0;

}
