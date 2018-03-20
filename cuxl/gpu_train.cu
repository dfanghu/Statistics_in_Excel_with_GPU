#include "gpu_.cuh"
#include "gpu_train.cuh"
#include "transfer_and_loss.cuh"

//#define DEBUG
#define SHM
//#define DEBUG
#define LMAX 100


__global__ void set_W_first_row(int L, const int* dpp, double* W) {
  for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < L - 1; tid += blockDim.x * gridDim.x) {
    //each tid takes care of the first row in W[tid + 1]
    int start = 0;
    for (int i = 1; i <= tid; ++i) {
      start += dpp[i - 1] * dpp[i];
    }
    W[start] = 999.;
    for (int i = 1; i < dpp[tid]; ++i) {  //require W[layer] to have size pp[layer] * pp[layer-1] such that the first row has length pp[layer-1]
      W[start + i] = 0.;
    }
  }
}
__global__ void init_W_random(int lW, double* W) {
  for (unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < lW; tid += blockDim.x * gridDim.x) {
    W[tid] = runif(tid) / 100.;
    //W[tid] = 0.01;
  }
}

__global__ void updt_D_yhat_B(int L, const int* pp, const int* s, int* sp, int* ss, double* D, double* yhat, double* yobs, double* B, const double* W, int sample_size, int fnid, int fintrid, double* ddd, int* idTransFns = nullptr, int idLossFn = 0) {
  /*
  pp[0..L]

  D is stored as passed in from Excel, ie., traversing observation by observation (row-major), NOT variable by variable
  D = D0 D1 D2 D3 .. DL    Each Di is not transposed, unlike those appear on the study sheet
  D0 = D[0..(n x s[0] - 1)]   s is the partial sum sequence of pp, starting from pp[0].    s[0] = pp[0]
  D1 = D[(n x s[0]) .. (n x s[1] - 1)]       s[1] = s[0] + pp[1]
  D2 = D[(n x s[1]) .. (n x s[2] - 1)]
  D3 = D[(n x s[2]) .. (n x s[3] - 1)]
  ......
  DL = D[(n x s[L-1])..(n x s[L] - 1)]

  yhat[0..(n x L)]
  yobs[0..(n x L)]

  B is also stored row-major as D
  B = B1 B2 .. BL    Each Bi is not transposed, unlike those appear on the study sheet
  B1 = B[0 .. (n x s'[1] - 1)]    s' is another partial sum sequence of pp, but starting from pp[1].  s'[1] = pp[1]
  B2 = B[(n x s'[1]) .. (n x s'[2] - 1)]     s'[2] = s'[1] + pp[2]
  B3 = B[(n x s'[2]) .. (n x s'[3] - 1)]
  ......
  BL = B[(n x s'[L-1]) .. (n x s'[L] - 1)]

  W = W1 W2 W3 .. WL, but each Wi is in transposed position of what we see on the study sheet
  W1 = W[0 .. (ss[1] - 1)]        ss is the partial sum sequence of pp[0..L-1] x pp[1..L].  ss[1] = pp[0] x pp[1]
  W2 = W[ss[1] .. (ss[2] - 1)]    ss[2] = ss[1] + pp[1] x pp[2]
  W3 = W[ss[2] .. (ss[3] - 1)]
  ......
  WL = W[ss[L-1]..(ss[L] - 1)]

  */

  //body
lbl_declare_variables:
  //indexing letters
  int i; //index of a sample (row in D) range: 0..(sample_size - 1)
  int j; //index of a variable (column in D) range: 0..(p[ell]-1)
  int ell; //index of a layer range: 0..L or 1..L
  int p, p0, pnxt, stD, stDpre, stW, stB, stBnxt;
  //pointers to transfer function and loss function
  double(*fn)(double); //transfer function
  double(*dfn)(double); //derivativle function of the transfer function
  double(*loss)(int, const double*, const double*); //lossfun
  double*(*dloss)(int, double*, const double*, const double*); //derivative function of the loss function
  getLossFn(idLossFn, loss, dloss);
  //The current thread manages a particular variable in a particular observation row of the data.
  //The row is managed by a block, potentially looping when too many variables. 
  //The block uses shared memory
lbl_declare_sharedmem:
  //limited by total shared memory => require careful formula for num blks to launch, say total is 96KB, a 1024-node layer requires 8KB per block => maximum 12 blocks.
  //since we row the thread idx by block size, there is no limit on the layer width and shared memory needed.
  //with gtx1080 the max shared mem per block is 96KB hence max nodes per layers is 12*1024, practically enough.
#ifdef SHM
  extern __shared__ double D0[];
#endif
lbl_the_block_rolls_on_an_observation_row:
  for (int bid = blockIdx.x; bid < sample_size; bid += gridDim.x) {
    for (ell = 1; ell <= L; ++ell) {  // [1,L]
                                      //env: (bid, ell)
      getTransFn(idTransFns[ell - 1], fn, dfn); //idTransFns was entered starting position 0 for layer1's tranfer function, so proper indexing should be 1 less.
      p0 = pp[ell - 1];
      p = pp[ell];
      stDpre = (ell == 1 ? 0 : (s[ell - 2])) + bid * p0;
      stD = s[ell - 1] + bid * p;
      stW = ss[ell - 1];
#ifdef SHM
      //load D(ell-1) = D[s[ell-2] to s[ell-1]] to shared memory before computing D(ell)
      for (int tid = threadIdx.x; tid < p0; tid += blockDim.x) {
        //env: (bid,ell-1,tid)
        D0[tid] = D[stDpre + tid];   // starts 0 for the D[0]
      }
      __syncthreads();  //need to syncthreads
#endif
                        //compute D[ell]
      for (int tid = threadIdx.x; tid < p; tid += blockDim.x) {
        //env:(bid,ell,tid)
        //inner product of the bid-th observation vector in D[ell-1] and the tid-th row vector of the weight matrix W[ell] to produce the tid-th element of the bid-th observation vector in D[ell]
        double inpd = 0.;
        for (int i = 0; i < p0; ++i) {
          //env:(bid,ell,tid,i)
#ifdef SHM
          inpd += D0[i] * W[stW + tid * p0 + i];
#else
          inpd += D[st + i] * W[stW + tid * p0 + i];
#endif
        }
        //inner product computed, now feed it to the transfer function
        D[stD + tid] = fn(inpd);
      }
      __syncthreads();
    }

    //next update yhat and the last layer for B
    //env: bid
    for (int tid = threadIdx.x; tid < pp[L]; tid += blockDim.x) {
      //env: (bid, tid)
      int k = bid * pp[L] + tid;
      double DL = D[s[L - 1] + k];
      if (idLossFn == 1) {
        //xen loss, requires softmax final transform
        //todo
        double den = 0.;
        for (int i = 0; i < pp[L]; ++i) {
          den += exp(D[s[L - 1] + bid * pp[L] + i]);
        }
        if (bid < 5 && tid == 0) ddd[bid] = den;
        yhat[k] = exp(DL) / den;
        //yhat[ k ] = 0;
      }
      else {
        //identity final transform
        yhat[k] = DL;
      }

      //update the head of the backward recurrence
      B[sp[L - 1] + k] = dfn(DL) * (yhat[k] - yobs[k]);
    }

    //next update the rest of the backward recurrence
  lbl_update_B:
    for (int ell = L - 1; ell >= 1; --ell) {
      pnxt = pp[ell + 1];
      p = pp[ell];
      stBnxt = sp[ell] + bid * pnxt;
      stB = sp[ell - 1] + bid * p;
      stD = s[ell - 1] + bid * p;
      stW = ss[ell];
      //env: (bid, ell)
      getTransFn(idTransFns[ell - 1], fn, dfn);
#ifdef SHM
      //load B(ell+1) to shared memory
      //reuse D0 for B, they are the same size
      for (int tid = threadIdx.x; tid < pnxt; tid += blockDim.x) {
        D0[tid] = B[stBnxt + tid]; //load B(ell+1)'s (bid, tid) element to shared mem of block bid
      }
      __syncthreads();
#endif
      for (int tid = threadIdx.x; tid < p; tid += blockDim.x) {
        //env: (bid, ell, tid)
        //B(bid,ell,tid) = innerproduct(B(bid,ell+1), W(ell+1,tid))
        double inpd = 0.;
        for (int i = 0; i < pnxt; ++i) {
#ifdef SHM
          inpd += D0[i] * W[stW + i * p + tid];
#else
          inpd += B[stBnxt + i] * W[stW + i * p + tid];
#endif
        }
        B[stB + tid] = inpd * dfn(D[stD + tid]);
      }
      __syncthreads();
    lbl_next_B:
    }
  lbl_next_bid:
  } //D[ell] filled by now
}

__global__ void init_F_n_U(int leng, double* F, double* U) {
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < leng; tid += gridDim.x * blockDim.x) {
    U[tid] = F[tid] = 0.;
  }
}
__global__ void updt_G_n_F(int L, double* G, const double* D, const double* B, double* F, const int* pp, int sample_size) {
  int stG(0), stD(0), stB(0);
  for (int layer = 1; layer <= L; ++layer) {
    int p0 = pp[layer - 1];
    int p = pp[layer];
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < p0 * p; tid += blockDim.x * gridDim.x) {
      int r = tid / p0;
      int c = tid % p0;
      double s = 0.;
      for (int i = 0; i < sample_size; ++i) {
        s += B[stB + i * p + r] * D[stD + i * p0 + c];
      }
      F[stG + tid] = G[stG + tid]; //F is previous G
      G[stG + tid] = s;
    }
    stG += p0 * p;
    stD += p0 * sample_size;
    stB += p * sample_size;
  }
}
__global__ void rprop(int leng, const double* G, double* W, double* U, const double* F, double propup, double propdn, double cap, double floor, double* ddd) {
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < leng; tid += gridDim.x * blockDim.x) {
    double u = U[tid], g = G[tid], f = F[tid];
    int sgnG = SGN(g);
    int sgnChg = sgnG * SGN(f);
    u *= (sgnChg > 0 ? propup : ((sgnChg < 0) ? propdn : 0));
    u = (u > cap ? cap : (u < floor ? floor : u));
    U[tid] = u;
    W[tid] -= sgnG * u;
  }
}
void init_W_host(int lW, double* hW, LPXLOPER12 W0, int* pp, int L) {
  int offset = 0;
  int offset_W0 = 0;
  int maxwid = -1;
  int r0, c0;
  for (int i = 1; i < L; ++i) {
    if (pp[i]>maxwid) {
      maxwid = pp[i];
    }
  }
  for (int i = 0; i < L; ++i) {
    int szWcurr = pp[i] * pp[i + 1];
    for (int k = 0; k < szWcurr; ++k) {
      int r = k / pp[i + 1];
      int c = k % pp[i + 1];
      hW[offset + c * pp[i] + r] = W0->val.array.lparray[offset_W0 + r * maxwid + (maxwid - pp[i + 1] + c)].val.num;
    }
    offset += szWcurr;
    offset_W0 += pp[i] * maxwid;
  }
}
LPXLOPER12 WINAPI gpu_train_numeric(FP12 D0, FP12 Yobs, FP12 layers, FP12 idTransFnInput, int nIter = -1, int retMode = 0, int useSuppliedInitialW0 = 0, LPXLOPER12 W0 = nullptr) {
  //Documentation: (MacBook) /Users/jd/Documents/On_Model_Selection.tex
  std::ofstream fs("c:/temp/debug.txt");
  fs << "in gpu_train_numeric"; fs.flush();
  OutputDebugString(L"in gpu_train_numeric");
  bool ok = false;
  int L; //legal within [2..LMAX]
  cudaErrorCode = cudaSuccess;
  XCHAR* errmsg = L"";
  if (D0.rows != Yobs.rows) {
    errmsg = L"#X and Y have different sample sizes";
    goto lbl_return;
  }

  if (layers.rows == 1) {
    L = layers.columns - 1;
  }
  else if (layers.columns == 1) {
    L = layers.rows - 1;
  }
  else {
    errmsg = L"#Illegal layers input";
    goto lbl_return;
  }
  if (L > LMAX) {
    errmsg = L"#maximum layers exceeded";
    goto lbl_return;
  }
  if (L < 2) {
    errmsg = L"#at least 1 hidden layers";
    goto lbl_return;
  }
  if (idTransFnInput.rows * idTransFnInput.columns != L + 1) {
    errmsg = L"#Number of transfer functions should equal the number of layers";
    goto lbl_return;
  }
  if (D0.columns != (INT32)layers.array[0] || Yobs.columns != (INT32)layers.array[L]) {
    errmsg = L"#0th layer width should equal ncol X, last layer width should equal ncol Yobs";
    goto lbl_return;
  }

  int sample_size = D0.rows;
  cudaDeviceReset();

  //set layer sizes:
  //the input layer and all-except-the-last hidden layers all have a bias term
  //the bias term of layer ell is not included in that layer's width, p[ell]
  //the last layer must have the same width as yhat, which doesn't have a bias term
  //the array pp is used to store full width, including the bias for the input and all-except-the-last hidden layers.
  int p[LMAX + 2], pp[LMAX + 2], ppmax(0), idTransFns[LMAX + 2], idLossFn;
  for (int ell = 0; ell < L; ++ell) { //[0,L)
    int currLayerWidth = (int)layers.array[ell];
    if (currLayerWidth < 1) {
      errmsg = L"#Nonpositive layer width";
      goto lbl_return;
    }
    else {
      pp[ell] = currLayerWidth;
      p[ell] = pp[ell] - 1;
      if (ppmax < currLayerWidth && ell > 0) ppmax = currLayerWidth;
    }
    int currTransFnId = (int)idTransFnInput.array[ell];
    if (currTransFnId < 0 || currTransFnId > 3) {
      errmsg = L"#Transfer Function Id should be among 0(logit), 1(tanh), 2(relu), and 3(identity)";
      goto lbl_return;
    }
    else {
      //index starts 0 ends L-1, but is used for layer starts 1 ends L
      idTransFns[ell] = currTransFnId;
    }
  }
  p[L] = pp[L] = (int)layers.array[L];   //{L}
  if (ppmax < p[L]) ppmax = p[L];
  int s[LMAX + 2], sp[LMAX + 2], ss[LMAX + 2];
  s[0] = pp[0] * sample_size; //{0}
  sp[0] = 0;  //sp meaning starts from index 1, set 0th position to 0 as extension  {0}
  ss[0] = 0;  //ss meaning starts from index 1, set 0th position to 0 as extension  {0}
  for (int ell = 1; ell <= L; ++ell) { //[1, L]
    s[ell] = s[ell - 1] + pp[ell] * sample_size;
    sp[ell] = sp[ell - 1] + pp[ell] * sample_size;
    ss[ell] = ss[ell - 1] + pp[ell - 1] * pp[ell];
  }


lbl_set_loss_function:  OutputDebugString(L"@@lbl_set_loss_function");
  idLossFn = idTransFns[L] = (int)idTransFnInput.array[L]; //loss function  {L}

  double estGpuMemRequired_B;
  if (nIter < 0) {
    //estimate required gpu memory in MB
    estGpuMemRequired_B = sample_size * 1. * (pp[0] + pp[L] * 2.) * sizeof(double);  //X, Yobs, and Yhat
    for (int ell = 1; ell <= L; ++ell) {
      estGpuMemRequired_B += 2. * pp[ell] * 1. * sample_size * 1. * sizeof(double);  //for D[ell] and B[ell], each of pp[ell] * sample_size * sizeof(double)
      estGpuMemRequired_B += 3. * pp[ell - 1] * 1. * pp[ell] * 1. * sizeof(double);  //for W[ell] and G[ell], their previous states, and additional scratch pads
    }
    return TempNum12(estGpuMemRequired_B / 1024. / 1024.);
  }

  //allocate gpu arrays
  int lD(0);  double* D; //Data,    0,1,..,L
  int lB(0);  double* B; //BackRec,   1,..,L
  int lW(0);  double* W; //Weights,   1,..,L
  int lG(0);  double* G; //Grads,     1,..,L  
  int lU(0);  double* U; //Update state parameter, 1..L
  int lF(0);  double* F; //next Grads,1,..,L
  for (int ell = 1; ell <= L; ++ell) {
    lD += pp[ell];
    lW += pp[ell] * pp[ell - 1];
  }
  lB = lD * sample_size;
  lD = (pp[0] + lD) * sample_size;
  lF = lU = lG = lW;

lbl_allodate_debug_array:  OutputDebugString(L"@@lbl_allodate_debug_array");
  double* ddd;  cuAsrt(lbl_return, cudaMalloc((void**)&ddd, 100 * sizeof(double)));
lbl_allocate_dpp_ds_dsp_dss:  OutputDebugString(L"@@lbl_allocate_dpp_ds_dsp_dss");
  int* dpp; //device copy of the pp array: width of all layers from 0..L
  cuAsrt(lbl_return, cudaMalloc((void**)&dpp, (1 + L) * sizeof(int)));
  cuAsrt(lbl_cudaFree, cudaMemcpy(dpp, pp, (1 + L) * sizeof(int), cudaMemcpyHostToDevice));
  int* ds;
  cuAsrt(lbl_return, cudaMalloc((void**)&ds, (1 + L) * sizeof(int)));
  cuAsrt(lbl_cudaFree, cudaMemcpy(ds, s, (1 + L) * sizeof(int), cudaMemcpyHostToDevice));
  int* dsp;
  cuAsrt(lbl_return, cudaMalloc((void**)&dsp, (1 + L) * sizeof(int)));
  cuAsrt(lbl_cudaFree, cudaMemcpy(dsp, sp, (1 + L) * sizeof(int), cudaMemcpyHostToDevice));
  int* dss;
  cuAsrt(lbl_return, cudaMalloc((void**)&dss, (1 + L) * sizeof(int)));
  cuAsrt(lbl_cudaFree, cudaMemcpy(dss, ss, (1 + L) * sizeof(int), cudaMemcpyHostToDevice));
lbl_allocate_idTransFns:  OutputDebugString(L"@@lbl_allocate_idTransFn");
  int* d_idTransFns;
  cuAsrt(lbl_return, cudaMalloc((void**)&d_idTransFns, (1 + L) * sizeof(int)));
  cuAsrt(lbl_cudaFree, cudaMemcpy(d_idTransFns, idTransFns, (1 + L) * sizeof(int), cudaMemcpyHostToDevice));
lbl_allocate_D_and_copy_only_the_input_layer:  OutputDebugString(L"@@lbl_allocate_D_and_copy_only_the_input_layer");
  cuAsrt(lbl_cudaFree, cudaMalloc((void**)&D, lD * sizeof(double)));
  cuAsrt(lbl_cudaFree, cudaMemcpy(D, D0.array, pp[0] * sample_size * sizeof(double), cudaMemcpyHostToDevice));
lbl_allocate_and_copy_output_layers:  OutputDebugString(L"@@lbl_allocate_and_copy_output_layers");
  double* yobs;
  cuAsrt(lbl_cudaFree, cudaMalloc((void**)&yobs, p[L] * sample_size * sizeof(double)));
  cuAsrt(lbl_cudaFree, cudaMemcpy(yobs, Yobs.array, p[L] * sample_size * sizeof(double), cudaMemcpyHostToDevice));

lbl_allocate_B_W_G_U_F_layers:  OutputDebugString(L"@@lbl_allocate_B_W_G_U_F_layers");
  cuAsrt(lbl_cudaFree, cudaMalloc((void**)&B, lB * sizeof(double)));
  cuAsrt(lbl_cudaFree, cudaMalloc((void**)&W, lW * sizeof(double)));
  cuAsrt(lbl_cudaFree, cudaMalloc((void**)&G, lG * sizeof(double)));  //one grad per weight, same size
  cuAsrt(lbl_cudaFree, cudaMalloc((void**)&U, lU * sizeof(double)));
  cuAsrt(lbl_cudaFree, cudaMalloc((void**)&F, lF * sizeof(double)));

  double* yhat;
  cuAsrt(lbl_cudaFree, cudaMalloc((void**)&yhat, p[L] * sample_size * sizeof(double))); //y layers can use p[L] (==pp[L])
                                                                                        //initializations
  dim3 tpb, nb; size_t smem_B;

lbl_initialize_W:  OutputDebugString(L"@@lbl_initialize_W");
  fs << "@@lbl_initialize_W"; fs.flush();
  double* hW = (double*)malloc(lW * sizeof(double));
  OutputDebugString(L"@@a"); fs << "@@a"; fs.flush();

  setLaunchParams(tpb, nb, lW);

  if (useSuppliedInitialW0 == 0) {
    OutputDebugString(L"@@b"); fs << "@@b"; fs.flush();
    init_W_random << <nb, tpb >> >(lW, W);
    OutputDebugString(L"@@c"); fs << "@@c"; fs.flush();
    set_W_first_row << <L, 1 >> >(L, dpp, W);
    OutputDebugString(L"@@d"); fs << "@@d"; fs.flush();
    cuAsrt(lbl_cudaFree, cudaGetLastError());
  }
  else {
    OutputDebugString(L"@@e"); fs << "@@e"; fs.flush();
    init_W_host(lW, hW, W0, pp, L);
    OutputDebugString(L"@@f"); fs << "@@f"; fs.flush();
    cuAsrt(lbl_cudaFree, cudaMemcpy(W, hW, lW * sizeof(double), cudaMemcpyHostToDevice));
  }
  OutputDebugString(L"@@g"); fs << "@@g"; fs.flush();
  cuAsrt(lbl_cudaFree, cudaGetLastError());


lbl_initialize_D_yhat_B:  OutputDebugString(L"@@lbl_initialize_D_yhat_B");
  smem_B = ppmax * sizeof(double);
  //TODO: if smem_B exceeds maximum shared mem per block then error out
  int iDev;
  cuAsrt(lbl_cudaFree, cudaGetDevice(&iDev));
  cudaDeviceProp prop;
  cuAsrt(lbl_cudaFree, cudaGetDeviceProperties(&prop, iDev));
  if (smem_B > prop.sharedMemPerBlock) {
    errmsg = L"#shared mem per block exceeds allowed maximum, reduce the widest layer!";
    goto lbl_cudaFree;
  }
  tpb.x = ppmax > prop.maxThreadsPerBlock / 2 ? prop.maxThreadsPerBlock / 2 : ppmax;  //the 1/2 factor avoids "too many resources requested at launch" error
  nb.x = (sample_size * smem_B > prop.sharedMemPerBlock) ? prop.sharedMemPerBlock / smem_B : sample_size; //ensure total shared mem within limit
  updt_D_yhat_B << <nb, tpb, smem_B >> >(L, dpp, ds, dsp, dss, D, yhat, yobs, B, W, sample_size, 0, 0, ddd, d_idTransFns, idLossFn); //<<<nb, tpb, smem_B >>>
  cuAsrt(lbl_cudaFree, cudaGetLastError());

lbl_initialize_G:  OutputDebugString(L"@@lbl_initialize_G");
  setLaunchParams(tpb, nb, ppmax * ppmax);  //ppmax * ppmax is an upper bound of the largest weight matrix
  updt_G_n_F << <nb, tpb >> >(L, G, D, B, F, dpp, sample_size);  //G is updated, F will be reinitialized to 0 in the next
  cuAsrt(lbl_cudaFree, cudaGetLastError());

lbl_initilaize_F:  OutputDebugString(L"@@lbl_initilaize_F");
  setLaunchParams(tpb, nb, lF);
  init_F_n_U << <nb, tpb >> >(lF, F, U);  //all initialized to 0.0
  cuAsrt(lbl_cudaFree, cudaGetLastError());

lbl_main_loop:  OutputDebugString(L"@@lbl_main_loop");
  double propup(1.2), propdn(0.5), cap(50), floor(0.000001);
  for (int epoch = 0; epoch < nIter; ++epoch) {
    setLaunchParams(tpb, nb, lW);
    rprop << <nb, tpb >> >(lW, G, W, U, F, propup, propdn, cap, floor, ddd);
    cuAsrt(lbl_cudaFree, cudaGetLastError());
    set_W_first_row << <L, 1 >> >(L, dpp, W);
    cuAsrt(lbl_cudaFree, cudaGetLastError());

    tpb.x = ppmax > prop.maxThreadsPerBlock / 2 ? prop.maxThreadsPerBlock / 2 : ppmax;  //the 1/2 factor avoids "too many resources requested at launch" error
    nb.x = (sample_size * smem_B > prop.sharedMemPerBlock) ? prop.sharedMemPerBlock / smem_B : sample_size; //ensure total shared mem within limit
    updt_D_yhat_B << <nb, tpb, smem_B >> >(L, dpp, ds, dsp, dss, D, yhat, yobs, B, W, sample_size, 0, 0, ddd, d_idTransFns, idLossFn);
    cuAsrt(lbl_cudaFree, cudaGetLastError());

    setLaunchParams(tpb, nb, ppmax * ppmax);  //ppmax * ppmax is an upper bound of the largest weight matrix
    updt_G_n_F << <nb, tpb >> >(L, G, D, B, F, dpp, sample_size);
    cuAsrt(lbl_cudaFree, cudaGetLastError());
  }
lbl_copy_result_weights_to_CPU:  OutputDebugString(L"@@lbl_copy_result_weights_to_CPU");
  cudaMemcpy(hW, W, lW * sizeof(double), cudaMemcpyDeviceToHost);

lbl_prepare_results:  OutputDebugString(L"@@lbl_prepare_results");
  int nrowsWeights(0);
  for (int i = 1; i <= L; ++i) nrowsWeights += pp[i - 1];
  int nrowsout = (retMode > 0) ? (1 + sample_size) + (nrowsWeights + 1) * retMode : nrowsWeights;
  int ncolsout = ppmax;
  static XLOPER12 xMulti;
  xMulti.xltype = xltypeMulti | xlbitDLLFree;
  xMulti.val.array.rows = nrowsout;
  xMulti.val.array.columns = ncolsout;
  xMulti.val.array.lparray = (LPXLOPER12)GlobalLock(hArray = GlobalAlloc(GMEM_ZEROINIT, nrowsout * ncolsout * sizeof(XLOPER12)));
  for (int i = 0; i < nrowsout * ncolsout; ++i) {
    xMulti.val.array.lparray[i].xltype = xltypeStr;
    xMulti.val.array.lparray[i].val.str = L"";
  }

  int offset(0);
  if (retMode == 1) {
    double* hyhat = (double*)malloc(p[L] * sample_size * sizeof(double));
    cudaMemcpy(hyhat, yhat, p[L] * sample_size * sizeof(double), cudaMemcpyDeviceToHost);
    double error_train = 0.;
    for (int i = 0; i < sample_size; ++i) {
      double e = 0.;
      for (int j = 0; j < p[L]; ++j) {
        int k = i * p[L] + j;
        switch (idLossFn) {
        case 1: //cross-entropy
          e += -log(hyhat[k]) * Yobs.array[k];
          break;
        default:
          e += (hyhat[k] - Yobs.array[k]) * (hyhat[k] - Yobs.array[k]);
        }
        xMulti.val.array.lparray[(1 + i) * ncolsout + j].val.num = hyhat[k];
        xMulti.val.array.lparray[(1 + i) * ncolsout + j].xltype = xltypeNum;
      }
      for (int j = p[L]; j < ncolsout; ++j) {
        xMulti.val.array.lparray[(1 + i) * ncolsout + j].val.str = L"";
        xMulti.val.array.lparray[(1 + i) * ncolsout + j].xltype = xltypeStr;
      }
      error_train += e;
    }
    error_train /= sample_size;
    xMulti.val.array.lparray[0].val.num = error_train;
    xMulti.val.array.lparray[0].xltype = xltypeNum;
    for (int i = 1; i < ncolsout; ++i) {
      xMulti.val.array.lparray[i].val.str = L"";
      xMulti.val.array.lparray[i].xltype = xltypeStr;
    }
    free(hyhat);
    offset += ncolsout * (1 + sample_size + 1);
  }
  int woffset = 0;
  for (int i = 1; i <= L; ++i) {
    for (int irow = 0; irow < pp[i]; ++irow) {
      for (int icol = 0; icol < pp[i - 1]; ++icol) {
        xMulti.val.array.lparray[offset + icol * ncolsout + ncolsout - pp[i] + irow].val.num = hW[woffset + irow * pp[i - 1] + icol];
        xMulti.val.array.lparray[offset + icol * ncolsout + ncolsout - pp[i] + irow].xltype = xltypeNum;
      }
    }
    offset += pp[i - 1] * ncolsout;
    woffset += pp[i - 1] * pp[i];
  }
#ifdef DEBUG
  lbl_debug_cpu_watch :
                      double* hG = (double*)malloc(lG * sizeof(double));
                      cudaMemcpy(hG, G, lG * sizeof(double), cudaMemcpyDeviceToHost);

                      double* hD = (double*)malloc(lD * sizeof(double));
                      cudaMemcpy(hD, D, lD * sizeof(double), cudaMemcpyDeviceToHost);

                      double* hB = (double*)malloc(lB * sizeof(double));
                      cudaMemcpy(hB, B, lB * sizeof(double), cudaMemcpyDeviceToHost);

                      double* hU = (double*)malloc(lU * sizeof(double));
                      cudaMemcpy(hU, U, lU * sizeof(double), cudaMemcpyDeviceToHost);

                      double* hF = (double*)malloc(lF * sizeof(double));
                      cudaMemcpy(hF, F, lF * sizeof(double), cudaMemcpyDeviceToHost);

                      double* hdd = (double*)malloc(100 * sizeof(double));
                      cudaMemcpy(hdd, ddd, 100 * sizeof(double), cudaMemcpyDeviceToHost);

                      free(hdd);
                      free(hF);
                      free(hU);
                      free(hB);
                      free(hD);
                      free(hG);
#endif
                      free(hW);
                      ok = true;
                    lbl_cudaFree:
                    lbl_return:
                      cudaDeviceSynchronize();
                      cudaDeviceReset();
                      //fs.close();
                      if (ok) {
                        if (nIter < 0) {
                          return TempNum12(estGpuMemRequired_B / 1024. / 1024.);
                        }
                        else {

                          return (LPXLOPER12)&xMulti;
                          //return TempNum12(error_train);

                        }
                      }
                      else {
                        if (cudaSuccess != cudaErrorCode) {
                          errmsg = nullptr;
                          errmsg = pc2pwc(errmsg, cudaGetErrorString(cudaErrorCode));
                        }
                        return TempStr12(errmsg);
                      }
}