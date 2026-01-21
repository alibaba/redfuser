# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Analysis on TIR blocks, loops and functions."""

from typing import List, Optional, Set, Union, Dict, Any

from typing_extensions import Literal

from tvm import ir, tir
from tvm import get_global_func

_is_reduction_block = get_global_func("tir.schedule.IsReductionBlock")
_get_blockrealize = get_global_func("tir.schedule.GetBlockRealize")
is_gemm_stmt = get_global_func("tir.schedule.IsGemmStmt")

def collect_block_iter_vars_used_in_indices(
    block: tir.Block, indices: List[tir.PrimExpr]
) -> Set[tir.Var]:
    """Collect the block iter variables used in the access region of a buffer region."""
    tir_vars = set()
    for expr in indices:
        tir_vars |= collect_vars_used_in_prim_expr(expr)
    tir_vars &= set(iter_var.var for iter_var in block.iter_vars)
    return tir_vars


def collect_vars_used_in_prim_expr(expr: tir.PrimExpr) -> Set[tir.Var]:
    """Collect the variables used in the PrimExpr."""
    tir_vars = set()

    def _collect_tir_var(expr):
        if isinstance(expr, tir.Var):
            tir_vars.add(expr)

    tir.stmt_functor.post_order_visit(expr, _collect_tir_var)
    return tir_vars


def get_buffer_load_from_prim_expr(expr: tir.PrimExpr) -> Set[tir.BufferLoad]:
    buffer_load_set = set()

    def _get_buffer_load(expr: tir.PrimExpr):
        if isinstance(expr, tir.BufferLoad):
            buffer_load_set.add(expr)

    tir.stmt_functor.post_order_visit(expr, _get_buffer_load)
    return buffer_load_set


class IterInfo:
    """Information about a loop/iter var."""

    kind: Literal["S", "R", "O"]
    var: tir.Var
    _dom: tir.PrimExpr
    loop_rv: List[tir.schedule.LoopRV]
    annotations: List[Dict[str, Any]]

    def __init__(
        self,
        kind: Literal["S", "R", "O"],
        var: tir.Var,
        dom: tir.PrimExpr,
        loop_rv: List[tir.schedule.LoopRV],
        annotations: List[Dict[str, Any]],
    ):
        """Construct an IterInfo object."""
        self.kind = kind
        self.var = var
        self._dom = dom
        self.loop_rv = loop_rv
        self.annotations = annotations

    @property
    def dom(self) -> Union[int, tir.PrimExpr]:
        """The iteration domain of the loop."""
        return int(self._dom) if isinstance(self._dom, tir.IntImm) else self._dom

    def __str__(self) -> str:
        return f'Iter("{self.kind}", {self.dom}, annotations={self.annotations})'

    def __repr__(self) -> str:
        return str(self)


class BlockInfo:
    """Information about a TIR block."""

    name: str
    iters: List[IterInfo]
    block_rv: tir.schedule.BlockRV
    _reduction_block: bool

    def __init__(
        self,
        name: str,
        iters: List[IterInfo],
        block_rv: tir.schedule.BlockRV,
        sch: tir.Schedule,
        reduction_block: bool = False,
    ):
        """Construct a BlockInfo object."""
        self.name = name
        self.block_rv = block_rv
        self.iters = iters
        self.sch = sch
        self._reduction_block = reduction_block

    def dom(self) -> List[Union[int, tir.PrimExpr]]:
        """The iteration domain of the block."""
        return [i.dom for i in self.iters]

    def dom_kind(self) -> str:
        """The iteration domain kind of the block, for example, SSSS, SSSR."""
        return "".join(i.kind for i in self.iters)

    def is_injective(self) -> bool:
        """Whether the block is injective, i.e. all its iteration domains are injective."""
        return all(k == "S" for k in self.dom_kind())

    def is_elementwise(self) -> bool:
        """Whether the block is elementwise, i.e. trivial mapping between read/write region"""

        def _check_unit_var_range(dom: ir.Range, var: tir.Var) -> bool:
            return dom.min.same_as(var) and dom.extent == 1

        if not self.is_injective():
            return False
        block = self.sch.get(self.block_rv)
        if len(block.reads) != 1 or len(block.writes) != 1:
            return False
        r_region = block.reads[0].region
        w_region = block.writes[0].region
        if len(r_region) != len(w_region):
            return False
        for var, r_dom, w_dom in zip(block.iter_vars, r_region, w_region):
            if not _check_unit_var_range(var, r_dom) or not _check_unit_var_range(
                var, w_dom
            ):
                return False
        return True

    def is_reduction(self) -> bool:
        """Whether the block is a reduction workload."""
        return self._reduction_block

    def is_gemm(self) -> bool:
        """Whether the block is a GEMM workload."""
        block = self.sch.get(self.block_rv)

        # 1. Must be a reduction operation
        if not self.is_reduction():
            return False
        # 2. Must be C[..., M, N] = C[..., M, N] + A[..., M/K, K/M] * B[..., K/N, N/K] pattern
        gemm_stmts = []
        if isinstance(block.body, tir.SeqStmt):
            for stmt in block.body:
                if is_gemm_stmt(stmt):
                    gemm_stmts.append(stmt)
        elif isinstance(block.body, tir.BufferStore):
            if is_gemm_stmt(block.body):
                gemm_stmts.append(block.body)
        if len(gemm_stmts) != 1:
            return False

        return True

    def __str__(self) -> str:
        return f'BlockInfo("{self.name}", "{self.dom_kind()}", {self.dom()})'

    def __repr__(self) -> str:
        return str(self)


def normalize_prim_func(sch: tir.Schedule) -> Optional[List[BlockInfo]]:
    def _iter_kind(i: tir.IterVar) -> str:
        return {
            tir.IterVar.DataPar: "S",
            tir.IterVar.CommReduce: "R",
        }.get(i.iter_type, "O")

    blocks_info = []
    root_block_rv = sch.get_block(sch.mod[sch.func_working_on].body.block.name_hint)
    block_rvs = sch.get_child_blocks(root_block_rv)
    for block_rv in block_rvs:
        block_realize: tir.BlockRealize = _get_blockrealize(sch, block_rv)
        block = sch.get(block_rv)
        iter_values = block_realize.iter_values
        loop_rvs = list(sch.get_loops(block_rv))
        iter_vars = list(block.iter_vars)
        ordered_loop_rvs = []
        for iter_value in iter_values:
            vars = collect_vars_used_in_prim_expr(iter_value)
            var_loop_rvs = [loop_rv for loop_rv in loop_rvs if sch.get(loop_rv).loop_var in vars]
            ordered_loop_rvs.append(var_loop_rvs)
        blocks_info.append(
            BlockInfo(
                name=block.name_hint,
                iters=[
                    IterInfo(
                        kind=_iter_kind(iter_var),
                        var=iter_var.var,
                        dom=iter_var.dom.extent,
                        loop_rv=loop_rv,
                        annotations=[dict(sch.get(rv).annotations) for rv in loop_rv],
                    )
                    for loop_rv, iter_var in zip(ordered_loop_rvs, iter_vars)
                ],
                block_rv=block_rv,
                sch=sch,
                reduction_block=_is_reduction_block(sch, block_rv, root_block_rv),
            )
        )
    return blocks_info
