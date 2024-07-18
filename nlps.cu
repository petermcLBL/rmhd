void fft_plan_create()
{
    if(cufftPlan3d(&plan_C2R, Nz, Nx, Ny, CUFFT_Z2D) != CUFFT_SUCCESS) {
        printf("plan_C2R creation failed. Don't trust results. \n");
    };
    if(cufftPlan3d(&plan_R2C, Nz, Nx, Ny, CUFFT_D2Z) != CUFFT_SUCCESS) {
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
    // nb changed 2024/07/17
    
    // nb changed 2024/07/08
    //https://spiral-software.github.io/fftx/apis.html#fftxproblem
        //printf("NLPS hit\n");
    /*
    Size of transform, as a std::vector<int> of length equal to the dimension, 
    with the component in each coordinate direction representing the transform size in that direction. 
    */
    std::vector<int> zxySizes{Nz, Nx, Ny};
    
    c2r_prob.setSizes(zxySizes);
        //printf("setSizes hit\n");
    /*
    Array of length 3 that contains the following.
    args[0]: pointer to output array.
    args[1]: pointer to input array.
    args[2]: pointer to symbol array (not used by all transforms).
    */
    //the CUDA code executes across X and Y separately, FFTX appears to do both together
    // do we need 6?
    std::vector<void*> args_C2R_fy{&fdyR, &dy, NULL}, 
        args_C2R_fx{&fdxR, &dx, NULL},
        args_C2R_gy{&gdyR, &dy, NULL}, 
        args_C2R_gx{&gdxR, &dx, NULL},
        args_in_R2C{&result, &fdxR, NULL};

        //printf("args vector assigned\n");
    GRADIENT (f, dx, dy);
/*
    //printf("return code: %i \n", cufftExecZ2D(plan_C2R, dy, fdyR));
    if(cufftExecZ2D(plan_C2R, dy, fdyR) != CUFFT_SUCCESS) printf("fdyR calculation failed. \n");
    if(cufftExecZ2D(plan_C2R, dx, fdxR) != CUFFT_SUCCESS) printf("fdxR calculation failed. \n");
*/
    // type of transform = complex-to-real
    //c2r_prob.setName("imdprdft");
    
    int mm = Nx, nn = Ny, kk = Nz;
    int K_adj = (int) ( kk / 2 ) + 1;
    
    fftx::box_t<3> c2r_dom ( point_t<3> ( { { 1, 1, 1 } } ),
                            point_t<3> ( { { mm, nn, K_adj } } ));

    fftx::array_t<3,double> inputHost(c2r_dom);
    
    double *sampX;
    std::complex<double> * tempX;

    DEVICE_MALLOC(&sampX, inputHost.m_domain.size() * sizeof(double));
    DEVICE_MALLOC(&tempX, mm * nn * K_adj * sizeof(std::complex<double>));
    
    std::vector<void*> args{&tempX,&sampX,NULL};
    
    printf("IMDPRDFTProblem hit\n");
    IMDPRDFTProblem imdp("imdprdft");
    imdp.setArgs(args);
    imdp.setSizes(zxySizes);
    printf("IMDPRDFTProblem created\n");
    imdp.transform();
    printf("IMDPRDFTProblem finished in %d\n",imdp.getTime());


    // set i/o location, use f on dy
    c2r_prob.setArgs(args_C2R_fy);
    printf("setArgs hit\n");
    // perform transform
    c2r_prob.transform();
    printf("transform hit\n");
    
    // set i/o location, use f on dx
    c2r_prob.setArgs(args_C2R_fx);
    // perform transform
    c2r_prob.transform();
   
    GRADIENT (g, dx, dy);
/*
    if(cufftExecZ2D(plan_C2R, dy, gdyR) != CUFFT_SUCCESS) printf("gdyR calculation failed.  \n");
    if(cufftExecZ2D(plan_C2R, dx, gdxR) != CUFFT_SUCCESS) printf("gdyR calculation failed.  \n");
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
    if(cufftExecD2Z(plan_R2C, fdxR, result) != CUFFT_SUCCESS) printf("R2C failed. \n");  
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



