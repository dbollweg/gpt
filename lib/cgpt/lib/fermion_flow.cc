#include "lib.h"


EXPORT(Fermionflow_fixedstepsize,{
    
    PyObject* _U;
    PyObject* _chi;
    PyObject* _epsilon;
    PyObject* _Nstep;
    PyObject* _meas_interval;
    PyObject* _Nckpoints;
    // bool symanzik_improved;


    if (!PyArg_ParseTuple(args, "OOOOOO", &_U, &_chi, &_epsilon, &_Nstep, &_meas_interval, &_Nckpoints)) {
      std::cout << "Error reading arguments" << std::endl;
      return NULL;
    }

    auto grid = get_pointer<GridCartesian>(_U,"U_grid");

    LatticeGaugeFieldD U_flow(grid);
    LatticeGaugeFieldD U(grid);

    PVector<LatticeFermionD> chi_in;
    LatticeFermionD chi_out(grid);

    std::vector<cgpt_Lattice_base*> ferm_basis;
    std::cout << "fill ferm_basis from _chi" << std::endl;
    cgpt_basis_fill(ferm_basis,_chi);
    std::cout << "fill chi_in from ferm_basis" <<std::endl;
    cgpt_basis_fill(chi_in,ferm_basis);
    std::cout << "basis fill successful" <<std::endl;

    for (int mu=0;mu<Nd;mu++) {
        auto l = get_pointer<cgpt_Lattice_base>(_U,"U",mu);
        auto& Umu = compatible<iColourMatrix<vComplexD>>(l)->l;
        PokeIndex<LorentzIndex>(U,Umu,mu);
    }

    // Now do zeuthen flow 
    Real epsilon;
    int Nstep;
    int meas_interval;
    int Nckpoints;
    cgpt_convert(_epsilon, epsilon);
    cgpt_convert(_Nstep, Nstep);
    cgpt_convert(_meas_interval, meas_interval);
    cgpt_convert(_Nckpoints,Nckpoints);

    LatticeColourMatrixD xform1(grid);

    typedef typename PeriodicGimplR::GaugeField GaugeLorentz;

    FermionFlow<PeriodicGimplR,WilsonGaugeAction<PeriodicGimplR>,LatticeFermionD> WF(epsilon,Nstep,U,meas_interval,Nckpoints);

    // ZeuthenFlow<PeriodicGimplR> ZF(epsilon, Nstep, meas_interval);
    // if (meas_interval == 0) {
    //   ZF.resetActions();
    // }
    WF.smear(U_flow, chi_out, U, chi_in[0]);

    // Transfrom back to stuff that gpt can deal with
    std::vector< cgpt_Lattice_base* > U_prime(4);
    for (int mu=0;mu<4;mu++) {
      auto lat = new cgpt_Lattice< iColourMatrix< vComplexD > >(grid);
      lat->l = PeekIndex<LorentzIndex>(U_flow,mu);
      U_prime[mu] = lat;
    }

    vComplexD vScalar = 0; // TODO: grid->to_decl()
    auto U_flow_return = Py_BuildValue("(l,[i,i,i,i],s,s,[N,N,N,N])", grid, grid->_gdimensions[0],
      grid->_gdimensions[1], grid->_gdimensions[2], grid->_gdimensions[3],
      get_prec(vScalar).c_str(), "full", U_prime[0]->to_decl(), U_prime[1]->to_decl(),
      U_prime[2]->to_decl(), U_prime[3]->to_decl());

    // auto phi_flow_return = Py_BuildValue("(l,[i,i,i,i],s,s,N)")
    return U_flow_return;
});
