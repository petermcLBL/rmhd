void fft_plan_create()
{
    if(cufftPlan3d(&plan_C2R, Nz, Nx, Ny, CUFFT_C2R) != CUFFT_SUCCESS) {
        printf("plan_C2R creation failed. Don't trust results. \n");
    };
    if(cufftPlan3d(&plan_R2C, Nz, Nx, Ny, CUFFT_R2C) != CUFFT_SUCCESS) {
        printf("plan_R2C creation failed. Don't trust results. \n");
    }
}

void fft_plan_destroy()
{
    if(cufftDestroy(plan_C2R) != CUFFT_SUCCESS) printf("plan_C2R destruction failed. \n");
    if(cufftDestroy(plan_R2C) != CUFFT_SUCCESS) printf("plan_R2C destruction failed. \n");
}

void NLPS(cuDoubleComplex *result, cuDoubleComplex *f, cuDoubleComplex *g)
{
    // nb changed 2024/07/08
    //https://spiral-software.github.io/fftx/apis.html#fftxproblem
    
    /*
    Size of transform, as a std::vector<int> of length equal to the dimension, 
    with the component in each coordinate direction representing the transform size in that direction. 
    */
    std::vector<int> zxySizes{Nz, Nx, Ny};
    
    c2r_prob.setSizes(zxySizes);
    /*
    Array of length 3 that contains the following.
    args[0]: pointer to output array.
    args[1]: pointer to input array.
    args[2]: pointer to symbol array (not used by all transforms).
    */
    //the CUDA code executes across X and Y separately, FFTX appears to do both together
    // do we need 6?
    std::vector<void*> args_C2R_fy{&fdyR, &dy}, args_C2R_fx{&fdxR, &dx},
        args_C2R_gy{&gdyR, &dy}, args_C2R_gx{&gdxR, &dx},
        args_in_R2C{&result, &fdxR};
    
    GRADIENT (f, dx, dy);
/*
    if(cufftExecC2R(plan_C2R, dy, fdyR) != CUFFT_SUCCESS) printf("fdyR calculation failed. \n");
    if(cufftExecC2R(plan_C2R, dx, fdxR) != CUFFT_SUCCESS) printf("fdxR calculation failed. \n");
*/
    // type of transform = complex-to-real
    //c2r_prob.setName("imdprdft");
    
    // set i/o location, use f on dy
    c2r_prob.setArgs(args_C2R_fy);
    // perform transform
    c2r_prob.transform();
    
    // set i/o location, use f on dx
    c2r_prob.setArgs(args_C2R_fx);
    // perform transform
    c2r_prob.transform();
    
    GRADIENT (g, dx, dy);
/*
    if(cufftExecC2R(plan_C2R, dy, gdyR) != CUFFT_SUCCESS) printf("gdyR calculation failed.  \n");
    if(cufftExecC2R(plan_C2R, dx, gdxR) != CUFFT_SUCCESS) printf("gdyR calculation failed.  \n");
*/
    // set i/o location, use g on dy
    c2r_prob.setArgs(args_C2R_gy);
    // perform transform
    c2r_prob.transform();
    
    // set i/o location, use g on dx
    c2r_prob.setArgs(args_C2R_gx);
    // perform transform
    c2r_prob.transform();
    
    // Reuse fdxR as result 
    bracket <<<dG,dB>>> (fdxR, fdxR, fdyR, gdxR, gdyR, 1.0);
/*
    if(cufftExecR2C(plan_R2C, fdxR, result) != CUFFT_SUCCESS) printf("R2C failed. \n");  
*/
    // type of transform = real-to-complex
    //r2c_prob.setName("mdprdft");
    // set i/o location, "reuse fdxR as result" sent to result
    r2c_prob.setArgs(args_in_R2C);
    // perform transform
    r2c_prob.transform();
    
    scale <<<dG,dB>>> (result,1.0f/((double) Nx*Ny*Nz));

    // Dealias
    mask <<<dG,dB>>> (result);
}


