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
import cgpt
import gpt
import numpy as np


def conj(l):
    if type(l) == gpt.expr:
        return gpt.expr(
            [
                (
                    complex(a[0]).conjugate(),
                    [(x[0] ^ (gpt.factor_unary.BIT_CONJ), x[1]) for x in a[1]],
                )
                for a in l.val
            ]
        )
    elif type(l) == gpt.tensor:
        return l.conj()
    else:
        return conj(gpt.expr(l))


def transpose(l):
    if type(l) == gpt.expr:
        return gpt.expr(
            [
                (
                    a[0],
                    [(x[0] ^ (gpt.factor_unary.BIT_TRANS), x[1]) for x in reversed(a[1])],
                )
                for a in l.val
            ]
        )
    elif type(l) == gpt.tensor and l.transposable():
        return l.transpose()
    else:
        return transpose(gpt.expr(l))


def adj(l):
    if type(l) == gpt.expr:
        return gpt.expr(
            [
                (
                    complex(a[0]).conjugate(),
                    [
                        (
                            x[0] ^ (gpt.factor_unary.BIT_TRANS | gpt.factor_unary.BIT_CONJ),
                            x[1],
                        )
                        for x in reversed(a[1])
                    ],
                )
                for a in l.val
            ]
        )
    elif (type(l) == gpt.tensor and l.transposable()) or type(l) == gpt.matrix_operator:
        return l.adj()
    else:
        return adj(gpt.expr(l))


def inv(l):
    if type(l) == gpt.matrix_operator:
        return l.inv()
    else:
        assert 0


def apply_expr_unary(l):
    if l.unary == gpt.expr_unary.NONE:
        return l
    return gpt.expr(gpt.eval(l))


def trace(l, t=None):
    if t is None:
        t = gpt.expr_unary.BIT_SPINTRACE | gpt.expr_unary.BIT_COLORTRACE
    if type(l) == gpt.tensor:
        return l.trace(t)
    return gpt.expr(l, t)


def spin_trace(l):
    return trace(l, gpt.expr_unary.BIT_SPINTRACE)


def color_trace(l):
    return trace(l, gpt.expr_unary.BIT_COLORTRACE)


def sum(e):
    l = gpt.eval(e)
    val = cgpt.lattice_sum(l.v_obj[0])
    for i in l.otype.v_idx[1:]:
        val = np.append(val, cgpt.lattice_sum(l.v_obj[i]))
    return gpt.util.value_to_tensor(val, l.otype)
