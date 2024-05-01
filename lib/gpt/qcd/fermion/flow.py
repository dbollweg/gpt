#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import cgpt, gpt, numpy

def Fermionflow_fixedstepsize(U, chi, epsilon=0.1, Nstep=10, meas_interval=1, Ncheckpoints=10, improvement=False):
    field = {
            "U_grid": U[0].grid.obj,
            "U": [u.v_obj[0] for u in U],
        }
  #  gpt.message("otype of chi field:", chi.otype.v_otype[0])
    if chi.otype.v_otype[0] == 'ot_vspin4color3':
        
        r,c = cgpt.Fermionflow_fixedstepsize_vspincolor(field, [chi], epsilon, Nstep, meas_interval, Ncheckpoints, improvement)
        #return r
        result=[]
        otype = gpt.ot_matrix_su_n_fundamental_group(3)
        for t_obj, s_ot, s_pr in r[4]:
            assert s_pr == r[2]
            assert s_ot == "ot_mcolor3"
            l = gpt.lattice(U[0].grid, otype, [t_obj])
            result.append(l)
        
        
        otype_chi = gpt.ot_vector_spin_color(4,3)
        t_obj, s_ot, s_pr = c[4]
        assert s_pr == c[2]
        assert s_ot == "ot_vspin4color3"
        result_chi = gpt.lattice(U[0].grid, otype_chi, [t_obj])
        
        return result,result_chi
        
    elif chi.otype.v_otype[0] == "ot_mspin4color3":
        r,c = cgpt.Fermionflow_fixedstepsize_mspincolor(field, [chi], epsilon, Nstep, meas_interval, Ncheckpoints, improvement)
        #return r
        result=[]
        otype = gpt.ot_matrix_su_n_fundamental_group(3)
        for t_obj, s_ot, s_pr in r[4]:
            assert s_pr == r[2]
            assert s_ot == "ot_mcolor3"
            l = gpt.lattice(U[0].grid, otype, [t_obj])
            result.append(l)
        
        
        otype_chi = gpt.ot_matrix_spin_color(4,3)
        t_obj, s_ot, s_pr = c[4]
        assert s_pr == c[2]
        assert s_ot == "ot_mspin4color3"
        result_chi = gpt.lattice(U[0].grid, otype_chi, [t_obj])
        
    
        return result,result_chi
    else:
        gpt.message("Fermionflow_fixedstepsize error: flow for detected otype of chi field is not implemented")
        assert 0

def Fermionflow_fixedstepsize_adjoint(U, chi, epsilon=0.1, Nstep=10, meas_interval=1, Ncheckpoints=10, improvement=False):
    field = {
            "U_grid": U[0].grid.obj,
            "U": [u.v_obj[0] for u in U],
        }
    if chi.otype.v_otype[0] == "ot_vspin4color3":
        
        c = cgpt.Fermionflow_fixedstepsize_adjoint_vspincolor(field, [chi], epsilon, Nstep, meas_interval, Ncheckpoints, improvement)
        
        otype_chi = gpt.ot_vector_spin_color(4,3)
        t_obj, s_ot, s_pr = c[4]
        assert s_pr == c[2]
        assert s_ot == "ot_vspin4color3"
        result_chi = gpt.lattice(U[0].grid, otype_chi, [t_obj])
            
        return result_chi
    elif chi.otype.v_otype[0] == "ot_mspin4color3":
        c = cgpt.Fermionflow_fixedstepsize_adjoint_mspincolor(field, [chi], epsilon, Nstep, meas_interval, Ncheckpoints, improvement)
        
        otype_chi = gpt.ot_vector_spin_color(4,3)
        t_obj, s_ot, s_pr = c[4]
        assert s_pr == c[2]
        assert s_ot == "ot_vspin4color3"
        result_chi = gpt.lattice(U[0].grid, otype_chi, [t_obj])
            
        return result_chi
    else:
        gpt.message("Fermionflow_fixedstepsize_adjoint error: flow for detected otype of chi field is not implemented")
        assert 0