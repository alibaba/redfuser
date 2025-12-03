/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file convert_to_tilelang_builtins.cc
 * \brief Convert TIR statements to TileLang builtin function calls.
 *
 * This pass converts standard TIR constructs into TileLang-specific builtin calls:
 * - Parallel loops -> tl_parallel
 * - Reduction operations -> tl_reduce
 * - GEMM operations -> tl_gemm
 * - Simple copy operations -> tl_copy
 * - Fill operations -> tl_fill
 * - Region operations -> tl_region
 */

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../schedule/analysis.h"
#include "../schedule/utils.h"

namespace tvm {

namespace tir {

namespace {
using namespace ffi;

/*!
 * \brief Main converter class that transforms TIR statements to TileLang builtins.
 */
class TileLangBuiltinConverter : public StmtExprMutator {
 public:
  explicit TileLangBuiltinConverter() : analyzer_(), inside_tiling_(false) {}

  PrimFunc Convert(PrimFunc func);

 private:
  // Statement visitors
  Stmt VisitStmt_(const ForNode* op) override;
  Stmt VisitStmt_(const BlockRealizeNode* op) override;
  Stmt VisitStmt_(const BufferStoreNode* op) override;

  bool IsCopyStmt(const BufferStoreNode* stmt);
  bool IsFillStmt(const BufferStoreNode* stmt);

  Stmt Convert2TL_GEMM(const BufferStoreNode* stmt);
  Stmt Convert2TL_REDUCE(const BufferStoreNode* stmt);
  Stmt Convert2TL_COPY(const BufferStoreNode* stmt);
  Stmt Convert2TL_FILL(const BufferStoreNode* stmt);
  Stmt Convert2TL_PARALLEL(const BufferStoreNode* stmt);
  PrimExpr Convert2TL_Region(const Buffer& buffer, const Array<PrimExpr>& indices);

  // Analyzer for expression simplification
  ffi::Map<Var, ObjectRef> tiles_;
  arith::Analyzer analyzer_;
  ffi::Map<Var, ObjectRef> binding_;
  bool inside_tiling_;
};

PrimFunc TileLangBuiltinConverter::Convert(PrimFunc func) {
  // TODO: Implement function-level conversion
  auto new_body = this->VisitStmt(func->body);
  if (new_body.same_as(func->body)) {
    return func;
  }
  auto write_ptr = func.CopyOnWrite();
  write_ptr->body = new_body;
  return func;
}

Stmt TileLangBuiltinConverter::VisitStmt_(const ForNode* op) {
  if (inside_tiling_) {
    tiles_.Set(op->loop_var, GetRef<For>(op));
  }
  auto stmt = StmtExprMutator::VisitStmt_(op);
  // 如果是inside_tiling_, 则只需要返回内部的语句，不需要For节点
  if (inside_tiling_) {
    tiles_.erase(op->loop_var);
    return stmt.as<ForNode>()->body;
  }
  return stmt;
}

Stmt TileLangBuiltinConverter::VisitStmt_(const BlockRealizeNode* op) {
  auto ends_with = [](const std::string& str, const std::string& suffix) {
    size_t suffix_len = suffix.length();
    if (str.length() < suffix_len) return false;
    return str.compare(str.length() - suffix_len, suffix_len, suffix) == 0;
  };
  bool already_inside_tiling = inside_tiling_;
  // Step 0. enter the tiling block, set inside_tiling_ to true
  if (ends_with(std::string(op->block->name_hint), "_outer")) {
    inside_tiling_ = true;
  }
  auto cur_binding = GetBindings(GetRef<BlockRealize>(op));
  for (const auto& [var, expr] : cur_binding) {
    if (expr.as<VarNode>()) {
      auto value = tiles_.Get(Downcast<Var>(expr));
      if (value.has_value()) {
        binding_.Set(var, value.value());
      }
    }
  }

  auto stmt = StmtExprMutator::VisitStmt_(op);

  if (ends_with(std::string(op->block->name_hint), "_outer")) {
    inside_tiling_ = false;
  }
  for (const auto& [var, expr] : cur_binding) {
    binding_.erase(var);
  }
  if (already_inside_tiling) {
    return stmt.as<BlockRealizeNode>()->block->body;
  }
  return stmt;
}

Stmt TileLangBuiltinConverter::VisitStmt_(const BufferStoreNode* op) {
  ICHECK(inside_tiling_);
  if (IsGemmStmt(GetRef<Stmt>(op))) {
    return Convert2TL_GEMM(op);
  } else if (IsReduceStmt(GetRef<Stmt>(op))) {
    return Convert2TL_REDUCE(op);
  } else if (IsCopyStmt(op)) {
    return Convert2TL_COPY(op);
  } else if (IsFillStmt(op)) {
    return Convert2TL_FILL(op);
  } else {
    return Convert2TL_PARALLEL(op);
  }
}

bool TileLangBuiltinConverter::IsCopyStmt(const BufferStoreNode* stmt) {
  return stmt->value.as<BufferLoadNode>() != nullptr;
}

bool TileLangBuiltinConverter::IsFillStmt(const BufferStoreNode* stmt) {
  return stmt->value.as<IntImmNode>() != nullptr || stmt->value.as<FloatImmNode>() != nullptr;
}

Stmt TileLangBuiltinConverter::Convert2TL_GEMM(const BufferStoreNode* stmt) {
  size_t n_dims = stmt->indices.size();
  auto mul_node = stmt->value.as<AddNode>()->b.as<MulNode>();

  BufferLoad A_load = CollectExprs<BufferLoad>(mul_node->a)[0];
  BufferLoad B_load = CollectExprs<BufferLoad>(mul_node->b)[0];
  BufferLoad C_load = BufferLoad(stmt->buffer, stmt->indices);

  auto A_region = Convert2TL_Region(A_load->buffer, A_load->indices);
  auto B_region = Convert2TL_Region(B_load->buffer, B_load->indices);
  auto C_region = Convert2TL_Region(stmt->buffer, stmt->indices);

  PrimExpr M = stmt->indices[n_dims - 2];
  PrimExpr N = stmt->indices[n_dims - 1];
  PrimExpr K = A_load->indices[n_dims - 1];
  bool is_transB = tvm::StructuralEqual()(B_load->indices[n_dims - 1], K) ? true : false;

  ffi::Array<PrimExpr> args;
  args.push_back(A_region);   // A
  args.push_back(B_region);   // B
  args.push_back(C_region);   // C
  args.push_back(false);      // is_transA
  args.push_back(is_transB);  // is_transB
  args.push_back(1);          // GemmWarpPolicy
  args.push_back(false);      // clear_accum
  args.push_back(1);          // k_pack
  args.push_back(0);          // wg_wait
  args.push_back(0);          // mbar

  auto call = Call(DataType::Void(), builtin::tl_gemm(), args);
  return Evaluate(call);
}

Stmt TileLangBuiltinConverter::Convert2TL_REDUCE(const BufferStoreNode* stmt) {
  auto [a_expr, b_expr] = GetBinaryOpOperands(stmt->value);
  auto a = Downcast<BufferLoad>(a_expr);
  auto b = Downcast<BufferLoad>(b_expr);
  auto a_region = Convert2TL_Region(a->buffer, a->indices);
  auto b_region = Convert2TL_Region(b->buffer, b->indices);
  ffi::String reduce_type;
  // TODO: min, absmax, abssum
  if (stmt->value.as<AddNode>()) {
    reduce_type = "sum";
  } else if (stmt->value.as<MaxNode>()) {
    reduce_type = "max";
  } else {
    ICHECK(false) << "Unsupported reduction operation: " << stmt->value;
  }
  if (tvm::StructuralEqual()(stmt->buffer, a->buffer)) {
    int reduce_dim = GetReduceDim(stmt->indices, b->indices);
    return Evaluate(Call(DataType::Void(), builtin::tl_reduce(),
                         {b_region, a_region, StringImm(reduce_type), reduce_dim, false}));
  } else {
    int reduce_dim = GetReduceDim(stmt->indices, a->indices);
    return Evaluate(Call(DataType::Void(), builtin::tl_reduce(),
                         {a_region, b_region, StringImm(reduce_type), reduce_dim, false}));
  }
}

Stmt TileLangBuiltinConverter::Convert2TL_COPY(const BufferStoreNode* stmt) {
  auto src = Downcast<BufferLoad>(stmt->value);
  PrimExpr src_region = Convert2TL_Region(src->buffer, src->indices);
  PrimExpr dst_region = Convert2TL_Region(stmt->buffer, stmt->indices);
  int coalesced_width = -1;
  bool disable_tma = false;
  int eviction_policy = 0;
  auto call = Call(DataType::Void(), builtin::tl_copy(),
                   {src_region, dst_region, coalesced_width, disable_tma, eviction_policy});
  return Evaluate(call);
}

Stmt TileLangBuiltinConverter::Convert2TL_FILL(const BufferStoreNode* stmt) {
  PrimExpr dst_region = Convert2TL_Region(stmt->buffer, stmt->indices);
  if (auto expr = stmt->value.as<IntImmNode>()) {
    auto value = GetRef<IntImm>(expr);
    auto call = Call(DataType::Void(), builtin::tl_fill(), {dst_region, value});
    return Evaluate(call);
  } else if (auto expr = stmt->value.as<FloatImmNode>()) {
    auto value = GetRef<FloatImm>(expr);
    auto call = Call(DataType::Void(), builtin::tl_fill(), {dst_region, value});
    return Evaluate(call);
  } else {
    ICHECK(false) << "Unsupported fill operation: " << stmt->value;
  }
}

Stmt TileLangBuiltinConverter::Convert2TL_PARALLEL(const BufferStoreNode* stmt) {
  auto dst = BufferLoad(stmt->buffer, stmt->indices);

  ffi::Map<Var, Var> index_map;
  ffi::Map<Var, PrimExpr> dom_map;

  auto operands = CollectExprs<BufferLoad>(stmt->value);
  operands.push_back(dst);
  //   collect all iter vars in operands
  for (const auto& operand : operands) {
    for (const auto& index : operand->indices) {
      auto tile_vars = CollectExprs<Var>(index);
      for (const auto& tile_var : tile_vars) {
        if (binding_.Get(tile_var).has_value()) {
          if (auto tile_for = binding_.Get(tile_var)->as<ForNode>()) {
            index_map.Set(tile_var, tile_for->loop_var);
            dom_map.Set(tile_for->loop_var, tile_for->extent);
          }
        }
      }
    }
  }
  //   FIXME: this will create a new Var in TIR automatically, because tile_for->loop_var is
  //   eliminated in VisitStmt_(const ForNode* op)
  auto new_dst = Substitute(dst, index_map);
  auto new_value = Substitute(stmt->value, index_map);

  ffi::Array<PrimExpr> args;
  args.push_back(new_dst);
  args.push_back(new_value);
  for (const auto& [var, expr] : dom_map) {
    args.push_back(var);
    args.push_back(expr);
  }
  auto call = Call(DataType::Void(), builtin::tl_parallel(), args);
  return Evaluate(call);
}

PrimExpr TileLangBuiltinConverter::Convert2TL_Region(const Buffer& buffer,
                                                     const Array<PrimExpr>& indices) {
  ffi::Array<PrimExpr> new_indices;
  new_indices.reserve(indices.size());
  ffi::Array<PrimExpr> extents;
  extents.reserve(indices.size());

  for (const auto& index : indices) {
    ffi::Optional<Var> tile_var;
    ffi::Optional<For> tile_for;
    // FIXME: only support single tile var now
    PostOrderVisit(index, [&](const ObjectRef& node) {
      if (const auto* var_node = node.as<VarNode>()) {
        auto var = GetRef<Var>(var_node);
        if (const auto* for_node = binding_.Get(var)->as<ForNode>()) {
          tile_var = var;
          tile_for = GetRef<For>(for_node);
        }
      }
    });

    if (tile_var.has_value()) {
      PrimExpr tile_min = analyzer_.Simplify(tile_for.value()->min);
      PrimExpr tile_extent = analyzer_.Simplify(tile_for.value()->extent);

      Map<Var, PrimExpr> vmap;
      vmap.Set(tile_var.value(), tile_min);
      new_indices.push_back(analyzer_.Simplify(Substitute(index, vmap)));
      extents.push_back(tile_extent);
    } else {
      new_indices.push_back(analyzer_.Simplify(index));
      extents.push_back(make_const(index.dtype(), 1));
    }
  }
  BufferLoad load = BufferLoad(buffer, new_indices);
  ffi::Array<PrimExpr> args;
  args.push_back(load);
  for (auto extent : extents) {
    args.push_back(extent);
  }
  return Call(DataType::Void(), builtin::tl_region(), args);
}

}  // namespace

/* ==================== Pass Registration ==================== */

namespace transform {

Pass ConvertToTileLangBuiltins() {
  auto pass_func = [](PrimFunc func, IRModule m, PassContext ctx) {
    TileLangBuiltinConverter converter;
    return converter.Convert(func);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ConvertToTileLangBuiltins", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.transform.ConvertToTileLangBuiltins", ConvertToTileLangBuiltins);
}

}  // namespace transform
}  // namespace tir
}  // namespace tvm
