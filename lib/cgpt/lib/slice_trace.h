// sliceSum from Grid but with vector of lattices as input
template<class vobj>
inline void cgpt_slice_trace_sums(const PVector<Lattice<vobj>> &Data,
                            //std::vector<typename vobj::scalar_object> &result,
                            std::vector<ComplexD> &result,
                            int orthogdim)
{
  ///////////////////////////////////////////////////////
  // FIXME precision promoted summation
  // may be important for correlation functions
  // But easily avoided by using double precision fields
  ///////////////////////////////////////////////////////
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;

  GridBase *grid = Data[0].Grid();
  assert(grid!=NULL);

  const int     Nd = grid->_ndimension;
  const int  Nsimd = grid->Nsimd();
  const int Nbasis = Data.size();

  assert(orthogdim >= 0);
  assert(orthogdim < Nd);

  int fd = grid->_fdimensions[orthogdim];
  int ld = grid->_ldimensions[orthogdim];
  int rd = grid->_rdimensions[orthogdim];

  Vector<vobj> lvSum(rd * Nbasis);         // will locally sum vectors first
  Vector<sobj> lsSum(ld * Nbasis, Zero()); // sum across these down to scalars
  result.resize(fd * Nbasis);              // And then global sum to return the same vector to every node

  int      e1 = grid->_slice_nblock[orthogdim];
  int      e2 = grid->_slice_block [orthogdim];
  int  stride = grid->_slice_stride[orthogdim];
  int ostride = grid->_ostride[orthogdim];

  // sum over reduced dimension planes, breaking out orthog dir
  // Parallel over orthog direction
  VECTOR_VIEW_OPEN(Data, Data_v, AcceleratorRead);
  auto lvSum_p = &lvSum[0];
  typedef decltype(coalescedRead(Data_v[0][0])) CalcElem;

  accelerator_for(r, rd * Nbasis, grid->Nsimd(), {
    CalcElem elem = Zero();

    int n_base = r / rd;
    int so = (r % rd) * ostride; // base offset for start of plane
    for(int n = 0; n < e1; n++){
      for(int b = 0; b < e2; b++){
        int ss = so + n * stride + b;
        elem += coalescedRead(Data_v[n_base][ss]);
      }
    }
    coalescedWrite(lvSum_p[r], elem);
  });
  VECTOR_VIEW_CLOSE(Data_v);

  thread_for(n_base, Nbasis, {
    // Sum across simd lanes in the plane, breaking out orthog dir.
    ExtractBuffer<sobj> extracted(Nsimd); // splitting the SIMD
    Coordinate icoor(Nd);

    for(int rt = 0; rt < rd; rt++){
      extract(lvSum[n_base * rd + rt], extracted);
      for(int idx = 0; idx < Nsimd; idx++){
        grid->iCoorFromIindex(icoor, idx);
        int ldx = rt + icoor[orthogdim] * rd;
        lsSum[n_base * ld + ldx] = lsSum[n_base * ld + ldx] + extracted[idx];
      }
    }

    for(int t = 0; t < fd; t++){
      int pt = t / ld; // processor plane
      int lt = t % ld;
      if ( pt == grid->_processor_coor[orthogdim] ) {
        result[n_base * fd + t] = TensorRemove(trace(lsSum[n_base * ld + lt]));
      } else {
        result[n_base * fd + t] = scalar_type(0.0); //Zero();
      }
    }
  });
  scalar_type* ptr = (scalar_type *) &result[0];
  int words = fd * sizeof(sobj) / sizeof(scalar_type) * Nbasis;
  grid->GlobalSumVector(ptr, words);
}

template<typename T>
PyObject* cgpt_slice_trace(const PVector<Lattice<T>>& basis, int dim) {
  typedef typename Lattice<T>::vector_object vobj;
  //typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;

  //std::vector<sobj> result;
  std::vector<scalar_type> result;
  cgpt_slice_trace_sums(basis, result, dim);

  int Nbasis = basis.size();
  int Nsobj  = result.size() / basis.size();

  PyObject* ret = PyList_New(Nbasis);
  for (size_t ii = 0; ii < Nbasis; ii++) {

    PyObject* corr = PyList_New(Nsobj);
    for (size_t jj = 0; jj < Nsobj; jj++) {
      int nn = ii * Nsobj + jj;
      //PyList_SET_ITEM(corr, jj, cgpt_numpy_export(result[nn]));
      PyList_SET_ITEM(corr, jj, PyComplex_FromDoubles(result[nn].real(),result[nn].imag()));
    }

    PyList_SET_ITEM(ret, ii, corr);
  }

  return ret;
}
