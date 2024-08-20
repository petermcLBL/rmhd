/*    nb 2024/08/19

    RK2
x@k+1 = x@k + dt * f2
f1 = f( x@k, t@k )
f2 = f( x@k + (dt/2) * f1, t@k +(dt/2) )
works with
alf_adv(zpNew, zpOld, zpOld, zmNew, zmOld, zmOld, dt/2.0);
alf_adv(zpNew, zpOld, zpNew, zmNew, zmOld, zmNew, dt);

    alf_adv produces f2 and x@k+1 without problem

    for RK4, there is a problem
x@k+1 = x@k + (dt/6)( 1*f1 + 2*f2 + 2*f3 + 1*f4)
f1 = f(x@k, t@k)
f2 = f( x@k + (dt/2)*f1 , t@k + dt/2 )
f3 = f( x@k + (dt/2)*f2 , t@k + dt/2 )
f4 = f( x@k + (dt*f3) , t@k + dt )

alf_adv probably works fine with f2, f2, f4
f1 is problem

RK2 had no problem because it only needed f2 to produce final answer
RK4 needs standalone value of f1 without dt calculation
already tried feeding it dt=0, that disappeared to nothing

how to isolate the f(x@k, t@k) by itself?

IDEA (for after lunch)
wikipedia describes equations as
f1 = f(x@k, t@k)
f2 = f( x@k + dt*(f1/2) , t@k + dt/2 )
f3 = f( x@k + dt*(f2/2) , t@k + dt/2 )
f4 = f( x@k + (dt*f3) , t@k + dt )

that might be subtle enough to be cause of difference?
answer: no, the rate of [error reduction : step increase] remained linear
*/

// Nonlinear timestepping routine using Noah's trick
void alf_adv(cuDoubleComplex *zpNew, 
        cuDoubleComplex *zpOld, 
        cuDoubleComplex *zpstar, 
        cuDoubleComplex *zmNew, 
        cuDoubleComplex *zmOld, 
        cuDoubleComplex *zmstar, 
        double dt) {

    if (nlrun) {
        // temp1 = {zp, -kperp**2 *zm} + {zm, -kperp**2 zp}, temp2 = -kperp**2{zp,zm}
        nonlin(zpstar, zmstar, temp1, temp2, temp3);

        //ZP
        // temp3 = temp1 - temp2 for zp
        //    addsubt <<<dG,dB>>> (temp3, temp1, temp2, -1);
        ADDSUBT (temp3, temp1, temp2, -1);
        // Coeff of 0.5 in front of the nonlinear term is included in multkperpinv
        // multiply by kperp2**(-1)
        multKPerpInv <<<dG,dB>>> (temp3, temp3);

        // multiply nonlinear term by integrating factor
        linstep <<<dG,dB>>> (temp3, temp3, dt);

    }
    // zpNew = zpOld*exp(i*kz*dt)
    //  linstep <<<dG,dB>>> (zpNew, zpOld, dt);
    LINSTEP (zpNew, zpOld, dt);

    if (nlrun) {
        // Add in the nonlinear term
        fwdeuler <<<dG,dB>>> (zpNew, temp3, dt);
    }

    if (nlrun) {
        //ZM
        // temp3 = bracket1 + bracket2 for zm
        addsubt <<<dG,dB>>> (temp3, temp1, temp2, 1);

        // Coeff of .5 in front of the nonlinear term is included in multkperpinv
        // multiply by kperp2**(-1)
        multKPerpInv <<<dG,dB>>> (temp3, temp3);

        // multiply nonlinear term by integrating factor
        LINSTEP(temp3, temp3, -dt);
    }

    // zmNew = zmOld*exp(-i*kz*dt)
    LINSTEP (zmNew, zmOld, -dt);

    if (nlrun) {
        // Add in the nonlinear term
        fwdeuler <<<dG,dB>>> (zmNew, temp3, dt);
    }
}

// nb - exact same as normal alf_adv but remove anything that use
void alf_adv_rk4(cuDoubleComplex *zpNew, 
        cuDoubleComplex *zpOld, 
        cuDoubleComplex *zpstar, 
        cuDoubleComplex *zmNew, 
        cuDoubleComplex *zmOld, 
        cuDoubleComplex *zmstar, 
        double dt) {

    if (nlrun) {
        // temp1 = {zp, -kperp**2 *zm} + {zm, -kperp**2 zp}, temp2 = -kperp**2{zp,zm}
        nonlin(zpstar, zmstar, temp1, temp2, temp3);

        //ZP
        // temp3 = temp1 - temp2 for zp
        //    addsubt <<<dG,dB>>> (temp3, temp1, temp2, -1);
        ADDSUBT (temp3, temp1, temp2, -1);
        // Coeff of 0.5 in front of the nonlinear term is included in multkperpinv
        // multiply by kperp2**(-1)
        multKPerpInv <<<dG,dB>>> (temp3, temp3);

        // multiply nonlinear term by integrating factor
        linstep <<<dG,dB>>> (temp3, temp3, dt);

    }
    // zpNew = zpOld*exp(i*kz*dt)
    //  linstep <<<dG,dB>>> (zpNew, zpOld, dt);
    LINSTEP (zpNew, zpOld, dt);

    if (nlrun) {
        // Add in the nonlinear term
        fwdeuler <<<dG,dB>>> (zpNew, temp3, dt);
    }

    if (nlrun) {
        //ZM
        // temp3 = bracket1 + bracket2 for zm
        addsubt <<<dG,dB>>> (temp3, temp1, temp2, 1);

        // Coeff of .5 in front of the nonlinear term is included in multkperpinv
        // multiply by kperp2**(-1)
        multKPerpInv <<<dG,dB>>> (temp3, temp3);

        // multiply nonlinear term by integrating factor
        LINSTEP(temp3, temp3, -dt);
    }

    // zmNew = zmOld*exp(-i*kz*dt)
    LINSTEP (zmNew, zmOld, -dt);

    if (nlrun) {
        // Add in the nonlinear term
        fwdeuler <<<dG,dB>>> (zmNew, temp3, dt);
    }
}

// Timestepping routine for alfven, nonlinear, nondebug
void advance(cuDoubleComplex *zpNew, cuDoubleComplex *zpOld, cuDoubleComplex *zmNew, cuDoubleComplex *zmOld, double dt, int istep) {

    //nb changed 2024/08/02
    // first, clear out the holding areas
    zero <<<dG,dB>>> (tempZpOne, Nx, Ny/2+1, Nz);
    zero <<<dG,dB>>> (tempZmOne, Nx, Ny/2+1, Nz);
    zero <<<dG,dB>>> (tempZpTwo, Nx, Ny/2+1, Nz);
    zero <<<dG,dB>>> (tempZmTwo, Nx, Ny/2+1, Nz);
    //yes zpNew and zmNew as well, prev they were just over-written, not added to
    zero <<<dG,dB>>> (zpNew, Nx, Ny/2+1, Nz);
    zero <<<dG,dB>>> (zmNew, Nx, Ny/2+1, Nz);

    if(driven && istep%nforce==0){

        // Alfven wave forcing
        zero <<<dG,dB>>> (temp1, Nx, Ny/2+1, Nz);
        forcing(temp1, dt, kstir_x, kstir_y, kstir_z, fampl);
        fwdeuler <<<dG,dB>>> (zpOld, temp1, dt);
        fwdeuler <<<dG,dB>>> (zmOld, temp1, dt);
    }
/*
    // Half Alfven step
    alf_adv(zpNew, zpOld, zpOld, zmNew, zmOld, zmOld, dt/2.0);

    // Full Alfven step
    alf_adv(zpNew, zpOld, zpNew, zmNew, zmOld, zmNew, dt);
*/
/*
RK2
x(at k+1) = x(at k) + dt * f2
f1 = f( xk, tk )
f2 = f( xk + (dt/2) * f1, tk +(dt/2) )

RK4
x(k+1) = x(k) + (dt/6)( 1*f1 + 2*f2 + 2*f3 + 1*f4)
which is equivalent to
x(k+1) = x(k) + (dt/6)*f1 + (dt/3)*f2 + (dt/3)*f3 + (dt/6)*4
f1 = f(xk, tk)
f2 = f( xk + (dt/2*f1) , tk + dt/2 )
f3 = f( xk + (dt/2*f2) , tk + dt/2 )
f4 = f( xk + (dt*f3) , tk + dt )

^ SOMEWHERE IN HERE IS THE PROBLEM, I CAN FEEL IT
IF alf_adv REALLY IS JUST fn BY ITSELF, HOW WOULD I CHECK
OR ARE THE alf_adv CALLS JUST f2, f3, f4, and x(k+1) = [...] ?

IF xk + (dt/2*f1) IS EQUIVALENT TO
alf_adv(zpNew, zpOld, zpOld, zmNew, zmOld, zmOld, dt/2.0);
THEN THAT "dt/2 * f1" IS BAKED INTO alf_adv
SO YOU NEED TO WORK AROUND THAT
MAYBE CALL alf_adv ONE ADDITIONAL TIME BETWEEN 1st & 2nd CALLS
1st IS TO PRODUCE 
f1 = 0 + (dt/6)f1
REQUIRED FOR OVERALL EQUATION
2nd IS TO CREATE THE VALUE USED IN f2
xk + (dt/2*f1)
REQUIRED FOR f2 AND SUBSEQUENT EQUATIONS

IF TRUE THEN THIS MAY REQUIRE SOME SHUFFLING BETWEEN end-1 & end-0 alf_adv CALLS
BECAUSE, NOTE
TAKING ADVANTAGE OF STRUCTURE OF alf_adv TO RE-USE IT
f1 = f(xk, tk) IS ACTUALLY f1 = 0 + f(xk, tk)
THIS MIGHT BE JUMPING AT NOTHING, THOUGH

NO YEH IF IT HAPPENS TO ONE THEN IT HAPPENS TO THE OTHER
I NEED TO MAKE A NEW TEMP VALUE TO HOLD THINGS

for RK2
xk + (dt/2*f1) already exists as
alf_adv(zpNew, zpOld, zpOld, zmNew, zmOld, zmOld, dt/2.0);
xk + dt*f2 already exists as
alf_adv(zpNew, zpOld, zpNew, zmNew, zmOld, zmNew, dt);

so should be able to say, since tempZp/tempZm are at zero 
f1 = 0 + (dt/6)f1
which would mean
alf_adv(tempZp, zpOld, zpOld, tempZm, zmOld, zmOld, dt/6.0);

I am still unclear on the point of zpStar and zmStar (args 3 and 6)
But they are different between the the RK2 half-step and full-step
the difference between the 2 is that zpNew/zmNew are altered by the first alf_adv!
so they might need to be repalced with the tempZp/tempZm
...or I could just re-use the functions as-is, since- no wait the factor is different on all of them
i was right the first time
tempZp/tempZm will hold the 'f' 

addsubt(cuDoubleComplex* result, cuDoubleComplex* f, cuDoubleComplex* g, double a)
use addsubt() to say
zpNew = zpNew + tempZp * factor
addsubt(zpNew, zpNew, tempZp, factor) i.e.
addsubt(zpNew, zpNew, tempZp, (double)1/6)

in order: 
calc 'f1' +store in tempZp/tempZm
alf_adv(tempZp, zpOld, zpOld, tempZm, zmOld, zmOld, dt);
x(k-ish #1) = x(k) + (dt/6)*f1
addsubt(zpNew, zpNew, tempZp, (double)1/6);
addsubt(zmNew, zmNew, tempZm, (double)1/6);

calc 'f2' +store in tempZp/tempZm
alf_adv(tempZp, zpOld, tempZp, tempZm, zmOld, tempZm, dt/2);
x(k-ish #2) = x(k) + (dt/3)*f2
addsubt(zpNew, zpNew, tempZp, (double)1/3);
addsubt(zmNew, zmNew, tempZm, (double)1/3);

alf_adv(tempZp, zpOld, tempZp, tempZm, zmOld, tempZm, dt/2);
x(k-ish #3) = x(k) + (dt/3)*f3
addsubt(zpNew, zpNew, tempZp, (double)1/3);
addsubt(zmNew, zmNew, tempZm, (double)1/3);

alf_adv(tempZp, zpOld, zpOld, tempZm, zmOld, zmOld, dt);
x(k-ish #4) = x(k) + (dt/6)*f4
addsubt(zpNew, zpNew, tempZp, (double)1/6);
addsubt(zmNew, zmNew, tempZm, (double)1/6);

at this point "x(k-ish #4)" should be same value as "x(k+1)"
i think that's all?
*/


    /*nb 2024/08/16
      try #01
        replacing the scalar "(double)1/6" with "dt/6.0"
        across all "accumulator" actions
      caused things to converge faster! to zero!
      
      try #02
        f2 = f( xk + (dt/2) * f1, tk +(dt/2) ) IS EQUIVALENT TO alf_adv(zpNew, zpOld, zpOld, zmNew, zmOld, zmOld, dt/2.0);
        x(k+1) = xk + dt * f2 IS EQUIVALENT TO alf_adv(zpNew, zpOld, zpNew, zmNew, zmOld, zmNew, dt);  
        f1 IS MISSING
        TRY TO LOCATE MISSING PARTS
        x(k+1) = x(k) + (dt/6)*f1 + (dt/3)*f2 + (dt/3)*f3 + (dt/6)*4
        f1 = f(xk, tk)
        f2 = f( xk + (dt/2*f1) , tk + dt/2 ) MATCHED alf_adv(zpNew, zpOld, zpOld, zmNew, zmOld, zmOld, dt/2.0);
        f3 = f( xk + (dt/2*f2) , tk + dt/2 )
        f4 = f( xk + (dt*f3) , tk + dt )
        EXTRAPOLATE
        x(k+1) = x(k) + (dt/6)*f1 + (dt/3)*f2 + (dt/3)*f3 + (dt/6)*4
        f1 = f(xk, tk) _SHOULD_ BE BUT CANNOT CONFIRM alf_adv(tempZpOne, zpOld, zpOld, tempZmOne, zmOld, zmOld, dt);
        f2 = f( xk + (dt/2*f1) , tk + dt/2 ) MATCHED alf_adv(tempZpOne, zpOld, tempZpOne, tempZmOne, zmOld, tempZmOne, dt/2.0);
        f3 = f( xk + (dt/2*f2) , tk + dt/2 ) MATCHED alf_adv(tempZpOne, zpOld, tempZpOne, tempZmOne, zmOld, tempZmOne, dt/2.0);
        f4 = f( xk + (dt*f3) , tk + dt ) alf_adv(tempZpOne, zpOld, tempZpOne, tempZmOne, zmOld, tempZmOne, dt);  
        
        f1 HAS NO dt AT ALL - TRY alf_adv(tempZpOne, zpOld, zpOld, tempZmOne, zmOld, zmOld, 0); TO EMULATE
        OR PERHAPS IT NEEDS NO CALCULATION AT ALL
        OR IT NEEDS A NEW FUNCTION ENTIRELY
        
        AFTER ATTEMPTS
        x(k+1) = x(k) + (dt/6)*f1 + (dt/3)*f2 + (dt/3)*f3 + (dt/6)*4
        f1 = f(xk, tk) DOES NOT HAVE AN ANALOGOUS FUNCTION
        f2 = f( xk + (dt/2*f1) , tk + dt/2 ) MATCHED alf_adv(tempZpOne, zpOld, zpOld, tempZmOne, zmOld, zpOld, dt/2.0);
        f3 = f( xk + (dt/2*f2) , tk + dt/2 ) MATCHED alf_adv(tempZpOne, zpOld, tempZpOne, tempZmOne, zmOld, tempZmOne, dt/2.0);
        f4 = f( xk + (dt*f3) , tk + dt ) alf_adv(tempZpOne, zpOld, tempZpOne, tempZmOne, zmOld, tempZmOne, dt);
        
      VANISHES TO ZERO AROUND dt=0.000125 - MAYBE JUST PERFORM
      addsubt <<<dG,dB>>> (zpNew, zpNew, zpOld, (double)1.0/6.0);
      addsubt <<<dG,dB>>> (zmNew, zmNew, zmOld, (double)1.0/6.0);
      TO EMULATE dt=0
      ...or remove entirely?
        
      try #03
        the existing code has f1 'baked into' f2, it never actually produces the f1 value
        given that f1 = f(xk, tk) and these are supposed to be initial vals
        would giving func a timestep of 0 give valid output?
      NOPE - output converges down to zero, or converges linearly
      
      try #04
        what if it's an order-of-operations thing? and
        x(k+1) = x(k) + (dt/6)(1*f1 + 2*f2 + 2*f3 + 1*f4)
        turns out not equivalent to
        x(k+1) = x(k) + (dt/6)*f1 + (dt/3)*f2 + (dt/3)*f3 + (dt/6)*4
        try having zpNew/zmNew hold the (1*f1 + 2*f2 + 2*f3 + 1*f4)
        then after step 4 is done, perform
        addsubt <<<dG,dB>>> (zmNew, zmOld, zmNew, dt/6.0); etc
        this SHOULDN'T make a difference but I'm running out of ideas
    */
    
    //calc 'f1' +store in tempZp/tempZm
    //    f1 = f(xk, tk)
    //alf_adv(tempZpOne, zpOld, zpOld, tempZmOne, zmOld, zmOld, dt);
    //  nb 2024/08/20 - trying to understand the right-hand side of the equation
    //    
    LINSTEP (tempZpOne, zpOld, dt);
    LINSTEP (tempZmOne, zmOld, dt);
    //x(k-ish #1) = x(k) + (dt/6)*f1
    addsubt <<<dG,dB>>> (zpNew, zpNew, tempZpOne, (double)1.0/6.0);
    addsubt <<<dG,dB>>> (zmNew, zmNew, tempZmOne, (double)1.0/6.0);
    
    // NEWLY ADDED 2024/08/13
    //calc 'f1' TO BE USED IN f2
    //  uses old step's starting values to avoid 'contaminating' itself with results of (dt * f1)
    //    applies (dt/2 * f1) instead of (dt * f1)
    alf_adv(tempZpOne, zpOld, zpOld, tempZmOne, zmOld, zmOld, dt/2.0);
    //  and do not add the results to zpNew/zmNew

    //calc 'f2' +store in tempZp/tempZm
    //    f2 = f( xk + (dt/2*f1) , tk + dt/2 )
    /*TRYING 2024/08/15
      what if I shouldn't re-calc f1 with a "half-step"
      but should pass (f1's result values) / 2 ?
    scale <<<dG,dB>>> (tempZpTwo, tempZpOne, 0.5);
    scale <<<dG,dB>>> (tempZmTwo, tempZmOne, 0.5);
      Nope, no appreciable change. This is a record of one attempt.
    */
    alf_adv(tempZpOne, zpOld, tempZpOne, tempZmOne, zmOld, tempZmOne, dt/2.0);
    //x(k-ish #2) = x(k-ish #1) + (dt/3)*f2
    addsubt <<<dG,dB>>> (zpNew, zpNew, tempZpOne, (double)1.0/3.0);
    addsubt <<<dG,dB>>> (zmNew, zmNew, tempZmOne, (double)1.0/3.0);

    // NEWLY ADDED 2024/08/13
    //calc 'f3' TO BE USED IN f4
    //  uses old step's starting values to avoid 'contaminating' itself with results of (dt/2 * f3)
    //    applies (dt/2 * f1) instead of (dt * f1)
    alf_adv(tempZpTwo, zpOld, tempZmOne, tempZmTwo, zmOld, tempZmOne, dt);
    //and do not add the results to zpNew/zmNew

    //calc 'f3' +store in tempZp/tempZm
    //    f3 = f( xk + (dt/2*f2) , tk + dt/2 )
    alf_adv(tempZpOne, zpOld, tempZpOne, tempZmOne, zmOld, tempZmOne, dt/2.0);
    //x(k-ish #3) = x(k-ish #2) + (dt/3)*f3
    addsubt <<<dG,dB>>> (zpNew, zpNew, tempZpOne, (double)1.0/3.0);
    addsubt <<<dG,dB>>> (zmNew, zmNew, tempZmOne, (double)1.0/3.0);
    
    //calc 'f4' +store in tempZp/tempZm
    //    f4 = f( xk + (dt*f3) , tk + dt )
    alf_adv(tempZpOne, zpOld, tempZpTwo, tempZmOne, zmOld, tempZmTwo, dt);
    //x(k-ish #4) = x(k-ish #3) + (dt/6)*f4
    addsubt <<<dG,dB>>> (zpNew, zpNew, tempZpOne, (double)1.0/6.0);
    addsubt <<<dG,dB>>> (zmNew, zmNew, tempZmOne, (double)1.0/6.0);

    //at this point "x(k-ish #4)" should be same value as "x(k+1)"
    //  i.e. zpNew/zmNew should have the same value as it did before this change

    // Damping
    damp_hyper <<<dG,dB>>> (zpNew, nu_hyper, alpha_hyper, dt);
    damp_hyper <<<dG,dB>>> (zmNew, nu_hyper, alpha_hyper, dt);

    dampz <<<dG,dB>>> (zpNew, nu_kz, alpha_z, dt);
    dampz <<<dG,dB>>> (zmNew, nu_kz, alpha_z, dt);

    // Move the results, the zNew's, to the zOld's 
    CP_ON_GPU(zpOld, zpNew, Nkc);
    CP_ON_GPU(zmOld, zmNew, Nkc);
}



