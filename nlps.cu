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

    // nb changed 2024/07/08
    //https://spiral-software.github.io/fftx/apis.html#fftxproblem
        //printf("NLPS hit\n");
    //Size of transform, as a std::vector<int> of length equal to the dimension, 
    //with the component in each coordinate direction representing the transform size in that direction. 

    std::vector<int> zxySizes{Nz, Nx, Ny};
    
    c2r_prob.setSizes(zxySizes);
    r2c_prob.setSizes(zxySizes);
        //printf("setSizes hit\n");

    //Array of length 3 that contains the following.
    //args[0]: pointer to output array.
    //args[1]: pointer to input array.
    //args[2]: pointer to symbol array (not used by all transforms).

    //the CUDA code executes across X and Y separately, FFTX appears to do both together
    // do we need 6?

    CUdeviceptr symbolPtr = (CUdeviceptr) NULL; 
    
    std::vector<void*> args_C2R_fy{&fdyR, &dy, &symbolPtr}, 
        args_C2R_fx{&fdxR, &dx, &symbolPtr},
        args_C2R_gy{&gdyR, &dy, &symbolPtr}, 
        args_C2R_gx{&gdxR, &dx, &symbolPtr},
        args_in_R2C{&result, &fdxR, &symbolPtr};

        //printf("args vector assigned\n");

    GRADIENT (f, dx, dy);
/*
    //printf("return code: %i \n", cufftExecZ2D(plan_C2R, dy, fdyR));
    if(cufftExecZ2D(plan_C2R, dy, fdyR) != CUFFT_SUCCESS) printf("fdyR calculation failed. \n");
    if(cufftExecZ2D(plan_C2R, dx, fdxR) != CUFFT_SUCCESS) printf("fdxR calculation failed. \n");
*/
    // type of transform = complex-to-real
    //c2r_prob.setName("imdprdft");
    
        // nb changed 2024/07/17
/*    // TEST CODE FROM FFTX EXAMPLE, CAN BE DELETED WHENEVER
    int mm = Nx, nn = Ny, kk = Nz;
    int K_adj = (int) ( kk / 2 ) + 1;
    std::vector<int> sizes{mm,nn,kk};
    
    fftx::box_t<3> domain ( point_t<3> ( { { 1, 1, 1 } } ),
                            point_t<3> ( { { mm, nn, kk } } ));
    fftx::box_t<3> outputd ( point_t<3> ( { { 1, 1, 1 } } ),
                            point_t<3> ( { { mm, nn, K_adj } } ));

    fftx::array_t<3,double> inputHost(domain);
    fftx::array_t<3,std::complex<double>> outputHost(outputd);
    fftx::array_t<3,double> outputHost2(domain);
    fftx::array_t<3,std::complex<double>> outDevfft1(outputd);
    fftx::array_t<3,double> outDevfft2(domain);
    
    double *sampX, *sampY, *sampsym;
    std::complex<double> * tempX;

    DEVICE_MALLOC(&sampX, inputHost.m_domain.size() * sizeof(double));
    DEVICE_MALLOC(&tempX, mm * nn * K_adj * sizeof(std::complex<double>));
    
    DEVICE_MALLOC(&sampX, inputHost.m_domain.size() * sizeof(double));
    DEVICE_MALLOC(&sampY, outputHost2.m_domain.size() * sizeof(double));
    DEVICE_MALLOC(&sampsym,  outputHost.m_domain.size() * sizeof(double));
    DEVICE_MALLOC(&tempX, mm * nn * K_adj * sizeof(std::complex<double>));

    std::vector<void*> args{&tempX,&sampX,&symbolPtr};

    //printf("MDPRDFTProblem hit\n");
    MDPRDFTProblem mdp(args, sizes, "mdprdft");
    //printf("MDPRDFTProblem created\n");
    mdp.transform();
    printf("MDPRDFTProblem finished in %f\n",mdp.getTime());
    
    //printf("IMDPRDFTProblem hit\n");
    IMDPRDFTProblem imdp("imdprdft");
    imdp.setArgs(args);
    imdp.setSizes(sizes);
    //printf("IMDPRDFTProblem created\n");
    imdp.transform();
    printf("IMDPRDFTProblem finished in %f\n",imdp.getTime());
*/

    // set i/o location, use f on dy
    c2r_prob.setArgs(args_C2R_fy);
    //printf("setArgs 1 hit\n");
    // perform transform
    c2r_prob.transform();
    //printf("c2r_prob 1 finished in %f\n",c2r_prob.getTime());
    
    // set i/o location, use f on dx
    c2r_prob.setArgs(args_C2R_fx);
    //printf("setArgs 2 hit\n");
    // perform transform
    c2r_prob.transform();
    //printf("c2r_prob 2 finished in %f\n",c2r_prob.getTime());
 
    GRADIENT (g, dx, dy);
/*
    if(cufftExecZ2D(plan_C2R, dy, gdyR) != CUFFT_SUCCESS) printf("gdyR calculation failed.  \n");
    if(cufftExecZ2D(plan_C2R, dx, gdxR) != CUFFT_SUCCESS) printf("gdyR calculation failed.  \n");
*/
    // set i/o location, use g on dy
    c2r_prob.setArgs(args_C2R_gy);
    //printf("setArgs 3 hit\n");
    // perform transform
    c2r_prob.transform();
    //printf("c2r_prob 3 finished in %f\n",c2r_prob.getTime());
    
    // set i/o location, use g on dx
    c2r_prob.setArgs(args_C2R_gx);
    //printf("setArgs 4 hit\n");
    // perform transform
    c2r_prob.transform();
    //printf("c2r_prob 4 finished in %f\n",c2r_prob.getTime());

    // Reuse fdxR as result 
    bracket <<<dG,dB>>> (fdxR, fdxR, fdyR, gdxR, gdyR, 1.0);
/*
    if(cufftExecD2Z(plan_R2C, fdxR, result) != CUFFT_SUCCESS) printf("R2C failed. \n");  
*/
    // type of transform = real-to-complex
    //r2c_prob.setName("mdprdft");
    // set i/o location, "reuse fdxR as result" sent to result
    r2c_prob.setArgs(args_in_R2C);
    //printf("setArgs 5 hit\n");
    // perform transform
    r2c_prob.transform();
    //printf("r2c_prob 1 finished in %f\n",r2c_prob.getTime());

    scale <<<dG,dB>>> (result, 1.0/((double) Nx*Ny*Nz));
    //printf("scale hit\n");
    // Dealias
    mask <<<dG,dB>>> (result);
    //printf("mask hit\n");
}



