/*
 * pde.h
 *      Author: sunder
 */

#ifndef PDE_H_
#define PDE_H_

#include<iostream>
#include<string>
#include <cmath>
#include <string.h> 
#define BOOST_DISABLE_ASSERTS
#include "boost/multi_array.hpp"

using namespace boost; 

//-----------------------------------------------------------------------
// Base (abstract) class for hyperbolic PDEs of the form:
// dQ/dt + div(F(Q)) = 0
// All PDE templates must inherit from this class
//-----------------------------------------------------------------------

class HyperbolicPDE {
protected:
	int _nVar;
	std::string SystemName;
	std::vector<std::string> FieldName;  
    double smax = 0.0; 
public:
	HyperbolicPDE() : _nVar(0) {}
	virtual ~HyperbolicPDE() {}
	inline int nVar() const {return _nVar;}
    inline double maxEigenValue() const {return smax;}
    inline void resetMaxEigenValue() {smax = 0.0;}
	std::string getSystemName() const {return SystemName;}
	std::string getFieldName(int c) const {return FieldName[c];}

	/* Initial condition function */

	virtual void initCond(double, multi_array<double,1>&) const = 0;

	/* PDE related functions (all these functions are pure virtual functions) */

	virtual void cons2Prim(const multi_array<double,1>&, multi_array<double,1>&) const = 0;
	virtual void prim2Cons(const multi_array<double,1>&, multi_array<double,1>&) const = 0;
	virtual void flux(const multi_array<double,1>&, multi_array<double,1>&) const = 0;
    virtual void project2CharacteristicSpace(const multi_array<double,1>&, const multi_array<double,1>&, multi_array<double,1>&) const = 0; 
    virtual void project2RealSpace(const multi_array<double,1>&, const multi_array<double,1>&, multi_array<double,1>&) const = 0; 
	virtual void eigenValues(const multi_array<double,1>&, multi_array<double,1>&) const = 0;
	virtual bool checkPAD(const multi_array<double,1>&) const = 0;

	/* Boundary conditions */

	virtual void getRightState(const multi_array<double,1>&, int, multi_array<double,1>&) const = 0;

	/* LLF/Rusanov Riemann solver to find upwind flux at faces */

	void riemannSolver(const multi_array<double,1>& QL, const multi_array<double,1>& QR, multi_array<double,1>& F) {

		int c;
		double s = 0.0;

		multi_array<double,1> FL(extents[_nVar]), FR(extents[_nVar]);
		multi_array<double,1> LL(extents[_nVar]),LR(extents[_nVar]);

		flux(QL, FL);
		flux(QR, FR);

		eigenValues(QL,LL);
		eigenValues(QR,LR);

		for (c = 0; c < _nVar; ++c) {
			if ( std::abs(LL[c]) > s)
				s = std::abs(LL[c]);
			if ( std::abs(LR[c]) > s)
				s = std::abs(LR[c]);
		}

		for (c = 0; c < _nVar; ++c) 
			F[c] = 0.5*( (FL[c] + FR[c]) - s*(QR[c] - QL[c]) ); 

        if (s>smax)
            smax = s; 
    }
};

//-----------------------------------------------------------------------
// Linear Convection Equation 
//-----------------------------------------------------------------------

class LinearConvection : public HyperbolicPDE {
public:
    // Common initial conditions 

    enum IC 
    {
        sine_wave, 
        gauss_wave,
        square_wave,
        jiang_shu_wave
    }; 

    // Constructor

    LinearConvection(double _a, LinearConvection::IC _ic) : a(_a), ic(_ic)
    {
        _nVar = 1; 
        FieldName.resize(_nVar);
        FieldName[0] = "u";
        SystemName = "Linear Convection Equations";
    }

    void initCond(double, multi_array<double,1>&) const;
	void cons2Prim(const multi_array<double,1>&, multi_array<double,1>&) const;
	void prim2Cons(const multi_array<double,1>&, multi_array<double,1>&) const;
	void flux(const multi_array<double,1>&, multi_array<double,1>&) const;
	void eigenValues(const multi_array<double,1>&, multi_array<double,1>&) const;
    void project2CharacteristicSpace(const multi_array<double,1>&, const multi_array<double,1>&, multi_array<double,1>&) const; 
    void project2RealSpace(const multi_array<double,1>&, const multi_array<double,1>&, multi_array<double,1>&) const; 
	bool checkPAD(const multi_array<double,1>&) const;
    void getRightState(const multi_array<double,1>&, int, multi_array<double,1>&) const;

private:

	double a;                // Specific heat ratio of the gas
	double rho_floor, prs_floor; // Density and Pressure floor values
    LinearConvection::IC ic;

	// Common initial conditions

    void sineWave(double, multi_array<double,1>&) const;  
	void gaussWave(double, multi_array<double,1>&) const;
	void squareWave(double, multi_array<double,1>&) const; 
    double G(double,double,double) const; 
    double F(double,double,double) const; 
	void jiangShuWave(double, multi_array<double,1>&) const;  

	// Commonly used boundary conditions
	void transmissive(const multi_array<double,1>&, multi_array<double,1>&) const;
};

//-----------------------------------------------------------------------
// Inviscid compressible Euler equations for ideal gas
//-----------------------------------------------------------------------

class EulerEquations : public HyperbolicPDE  {
public:

	// Common initial conditions

	enum IC
	{
        sine_wave, 
		sod_shock_tube,
        lax_shock_tube, 
		shu_osher,
		titarev_toro,
        blast_wave 
	};

	// Constructor

	EulerEquations(double gamma, EulerEquations::IC _ic) : GAMMA(gamma), ic(_ic)
	{
		_nVar=3;
		rho_floor = 1.0e-14, prs_floor = 1.0e-12;
		FieldName.resize(_nVar);

		FieldName[0] = "Density"; 
		FieldName[1] = "Velocity"; 
		FieldName[2] = "Pressure"; 

		SystemName = "Euler Equations";
        
        if (_ic == EulerEquations::sine_wave) {
            xmin = -1.0; xmax = 1.0; tend = 20.0;  
            bL = 0; bR = 0;        
        }

        else if (_ic == EulerEquations::sod_shock_tube) {
            xmin = -5.0; xmax = 5.0; tend = 2.0;         
            bL = 1; bR = 1;         
        }

        else if (_ic == EulerEquations::lax_shock_tube) {
            xmin = -5.0; xmax = 5.0; tend = 1.3;         
            bL = 1; bR = 1;         
        }
        
        else if (_ic == EulerEquations::shu_osher) {
            xmin = -5.0; xmax = 5.0; tend = 1.8;         
            bL = 1; bR = 1;         
        }
    
        else if (_ic == EulerEquations::titarev_toro) {
            xmin = -5.0; xmax = 5.0; tend = 4.0;         
            bL = 1; bR = 1; 
       
        }

        else if (_ic == EulerEquations::blast_wave) {
            xmin = 0.0; xmax = 1.0; tend = 0.038;         
            bL = 2; bR = 2;         
        }

        else {
            xmin = 0.0; xmax = 1.0; tend = 0.0;
            bL = 1; bR = 1;         
        }
	}

    void initCond(double, multi_array<double,1>&) const;
	void cons2Prim(const multi_array<double,1>&, multi_array<double,1>&) const;
	void prim2Cons(const multi_array<double,1>&, multi_array<double,1>&) const;
	void flux(const multi_array<double,1>&, multi_array<double,1>&) const;
	void eigenValues(const multi_array<double,1>&, multi_array<double,1>&) const;
    void project2CharacteristicSpace(const multi_array<double,1>&, const multi_array<double,1>&, multi_array<double,1>&) const; 
    void project2RealSpace(const multi_array<double,1>&, const multi_array<double,1>&, multi_array<double,1>&) const; 
	bool checkPAD(const multi_array<double,1>&) const;
    void getRightState(const multi_array<double,1>&, int, multi_array<double,1>&) const;
    void getProblemParameters(double& _xmin,double& _xmax,double& _tend, int& _bL, int& _bR) {_xmin = xmin; _xmax = xmax; _tend = tend, _bL = bL, _bR = bR;} 

private:

	double GAMMA;                // Specific heat ratio of the gas
	double rho_floor, prs_floor; // Density and Pressure floor values
    double xmin, xmax, tend;     // Problem specific domain sizes and final times 
    int bL, bR;                  // Problem specific boundary conditions      
    EulerEquations::IC ic;

	// Common initial conditions

    void sineWave(double, multi_array<double,1>&) const; // Sinusoidal wave problem (for checking accuracy of reconstruction)  
	void sodShockTube(double, multi_array<double,1>&) const; // Sod Shock Tube problem 
	void laxShockTube(double, multi_array<double,1>&) const; // Lax shock Tube problem 
	void shuOsher(double, multi_array<double,1>&) const; // Shu-Osher shock wave density interaction problem 
	void titarevToro(double, multi_array<double,1>&) const; // Titarev-toro high-frequency shock wave density interaction problem 

    void blastWave(double, multi_array<double,1>&) const; // blast Wave problem 

	// Commonly used boundary conditions

	void reflective(const multi_array<double,1>&, multi_array<double,1>&) const;
	void transmissive(const multi_array<double,1>&, multi_array<double,1>&) const;

};

#endif /* PDE_H_ */
