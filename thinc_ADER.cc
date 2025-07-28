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
const double alpha = 0.0;
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
                                      //
                                                                
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
    multi_array<double,2> qbnd_grad; // Value of conserved variable gradients at cell faces
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




double Thinc::initialCondition(double xx) const { //icond
      
    
    double eps = 1.0e-8;    
    double x0 = 0.5;
    double smear = 2.0*dx; 
    double y;
    //y = (tanh1d(xx, x0, smear, 1.0-eps, eps) + tanh1d(xx, -x0, smear, eps, 1.-eps)) -(1.0-eps);
    //y = std::sin(M_PI*xx);
    y = jiangShuWave(xx);
    //y = semiEllipse(xx);
    //y = squareWave(xx);
    
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
    qbnd_grad(extents[multi_array_types::extent_range(-N_ph,IMAX+N_ph)][2]), // 2 is for two faces per cell
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
        for (int q = 0; q < NGP; ++q) {
            qh[i] += wGP[q]*initialCondition(x[i] + dx*xGP[q]);
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
    
    double b0[] = {2.747898101806640625e+00, 3.068449497222900391e+00, 2.299013137817382812e+00};
    double b1[] = {-6.444582939147949219e+00, 3.310102462768554688e+00};
    double b2[] = {-7.944608330726623535e-01};
    
    double W0[3][1] = {
        {-4.898131847381591797e+00},
        {-4.851841449737548828e+00},
        {-4.631204605102539062e+00}
    };
    
    double W1[2][3] = {
        {5.434078693389892578e+00, 3.051296472549438477e+00, 5.189759254455566406e+00},
        {-1.077417278289794922e+01, -1.050577068328857422e+01, -7.418824672698974609e+00}
        
    };
    
    double W2[1][2] = {
        {1.230962562561035156e+01 , -2.011163139343261719e+01}
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



void Thinc::updateSolution()  {

    int i; 
    double q_t; 
    const double r1_dx = 1./dx;
    double beta;
    
    
    applyBoundaryConditions();
    
    double beta_l = 1.6, beta_s = std::log(3.);
    
    double qiph_L_A, qiph_R_A, qiph_L_B, qiph_R_B;
    double qimh_L_A, qimh_R_A, qimh_L_B, qimh_R_B;
    double W, TBV_A = 0.0, TBV_B = 0.0;
    for (i = -1; i < IMAX+1; ++i) {

        //beta = std::log(3.);
        beta = beta_NN(qh[i-1], qh[i], qh[i+1]);

        THINC(qh[i-1], qh[i], qh[i+1], beta, qbnd[i][0], qbnd[i][1]); //choose
        //mclim(qh[i-1], qh[i], qh[i+1], qbnd[i][0], qbnd[i][1]);
        
        
        
        // TBV here
       
        /*
        THINC(qh[i-2], qh[i-1], qh[i],   beta_l, W, qimh_L_A);
        THINC(qh[i-1], qh[i],   qh[i+1], beta_l, qimh_R_A, qiph_L_A);
        THINC(qh[i],   qh[i+1], qh[i+2], beta_l, qiph_R_A, W);
        THINC(qh[i-2], qh[i-1], qh[i], beta_s, W, qimh_L_B);
        THINC(qh[i-1], qh[i], qh[i+1], beta_s, qimh_R_B, qiph_L_B);
        THINC(qh[i], qh[i+1], qh[i+2], beta_s, qiph_R_B, W);
        
        TBV_A = std::min( {std::abs(qimh_L_A - qimh_R_A) + std::abs(qiph_L_A - qiph_R_A),
        std::abs(qimh_L_A - qimh_R_A) + std::abs(qiph_L_A - qiph_R_B),
        std::abs(qimh_L_B - qimh_R_A) + std::abs(qiph_L_A - qiph_R_A),
        std::abs(qimh_L_B - qimh_R_A) + std::abs(qiph_L_A - qiph_R_B) }  );
        

        
        TBV_B = std::min( {std::abs(qimh_L_A - qimh_R_B) + std::abs(qiph_L_B - qiph_R_A),
        std::abs(qimh_L_A - qimh_R_B) + std::abs(qiph_L_B - qiph_R_B),
        std::abs(qimh_L_B - qimh_R_B) + std::abs(qiph_L_B - qiph_R_A),
        std::abs(qimh_L_B - qimh_R_B) + std::abs(qiph_L_B - qiph_R_B) }  );

        if(TBV_A < TBV_B) //choose Scheme A 
            THINC(qh[i-1], qh[i], qh[i+1], beta_l, qbnd[i][0], qbnd[i][1]);
        else
            THINC(qh[i-1], qh[i], qh[i+1], beta_s, qbnd[i][0], qbnd[i][1]);
        */
        
        // TBV here

        
        q_t = -r1_dx*(qbnd[i][1] - qbnd[i][0])*adv;

        qbnd[i][0] += 0.5*dt*q_t; 
        qbnd[i][1] += 0.5*dt*q_t; 
       
        if (i >= 0 && i < IMAX){  
            deltas[i] = calc_delta(qh[i-1], qh[i], qh[i+1]);
            betas[i] = beta;
            //if(beta <1.6)
                //std::cout << x[i] << ":" <<  beta << "delta ="<<calc_delta(qh[i-1], qh[i], qh[i+1]) << std::endl;    
        }
    }

    // Find upwind flux

    for (int i = 0; i < IMAX + 1; ++i)
        F[i] = advFlux(qbnd[i-1][1],qbnd[i][0]);


    // Find RHS

    for (int i = 0; i < IMAX; ++i)
        qh[i] -= r1_dx*dt*(F[i+1] - F[i]);
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
    
    l1 = l1/static_cast<double>(IMAX);
    l2 = std::sqrt(l2/static_cast<double>(IMAX));

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

void Thinc::run() {

    auto start = std::chrono::system_clock::now();

    //-------------------------------------

    //updateSolution();
    
   
   
   int j = 0;
    //while (time_step < 1001) {
    while (time < tend) {

        printf ("%d, t = %4.3e\n", time_step, time);

        // If time step exceeds the final time, reduce it accordingly

        if((time + dt)>tend)
            dt = tend - time;
        
        updateSolution();
        
        if(time_step%5 == 0){
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






//----------------------------------------------------------------------------
// Plot solution as csv file
//----------------------------------------------------------------------------

void Thinc::plot() const {

    std::ofstream out_data;
    const std::string filename = "sol.csv";
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
    int IMAX = 32;
    double tend = (1.)*2.0;


    Thinc Sol(xmin, xmax, IMAX, tend);
    Sol.run();
    Sol.plot();
    Sol.errorNorms();

    return 0;

}
