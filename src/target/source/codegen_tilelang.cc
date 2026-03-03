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

#include <../../src/tir/schedule/analysis.h>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/name_supply.h>
#include <tvm/node/script_printer.h>
#include <tvm/script/printer/doc.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../../runtime/thread_storage_scope.h"

namespace tvm::codegen {

namespace {
using namespace tir;
using namespace script::printer;

ExprDoc TileLangPrefix(const ffi::String& attr) { return IdDoc("T")->Attr(attr); }

ExprDoc CallTileLang(const ffi::String& name, const ffi::Array<ExprDoc>& args,
                     const std::vector<std::pair<ffi::String, ExprDoc>>& kwargs) {
  ffi::Array<ffi::String> kw_keys;
  ffi::Array<ExprDoc> kw_values;
  for (const auto& [key, value] : kwargs) {
    kw_keys.push_back(key);
    kw_values.push_back(value);
  }
  return TileLangPrefix(name)->Call(args, kw_keys, kw_values);
}

ExprDoc CallTileLang(const ffi::String& name, const ffi::Array<ExprDoc>& args) {
  return TileLangPrefix(name)->Call(args);
}

LiteralDoc TileLangDataType(const DataType& dtype) {
  ICHECK(dtype.is_scalar());
  if (dtype.is_int()) {
    return LiteralDoc::Str("int" + std::to_string(dtype.bits()), std::nullopt);
  } else if (dtype.is_uint()) {
    return LiteralDoc::Str("uint" + std::to_string(dtype.bits()), std::nullopt);
  } else if (dtype.is_float()) {
    return LiteralDoc::Str("float" + std::to_string(dtype.bits()), std::nullopt);
  } else if (dtype.is_bfloat16()) {
    return LiteralDoc::Str("bfloat16", std::nullopt);
  } else if (dtype.is_float8()) {
    if (dtype.code() == DataType::kFloat8_e4m3fn) {
      return LiteralDoc::Str("float8_e4m3fn", std::nullopt);
    } else if (dtype.code() == DataType::kFloat8_e4m3fnuz) {
      return LiteralDoc::Str("float8_e4m3fnuz", std::nullopt);
    }
  }
  LOG(FATAL) << "Unsupported data type: " << dtype;
}

template <typename Iter>
ExprDoc IntoListDoc(const Iter& begin, const Iter& end) {
  return ListDoc({begin, end});
}

class CodeGenTileLang : protected StmtFunctor<Doc(const Stmt&)>,
                        protected ExprFunctor<ExprDoc(const PrimExpr&)> {
 public:
  FunctionDoc VisitFunc(const std::string& func_name, const PrimFuncNode* func) {
    ffi::Array<AssignDoc> params;
    for (auto param : func->params) {
      if (func->buffer_map.find(param) != func->buffer_map.end()) {
        const Buffer& io_buffer = func->buffer_map[param];
        ListDoc shape = ListDoc(VisitExprArray(io_buffer->shape));
        LiteralDoc dtype = TileLangDataType(io_buffer->dtype);
        ExprDoc anno = CallTileLang("Tensor", {shape, dtype});
        AssignDoc assign_doc = AssignDoc(VisitExpr(io_buffer->data), std::nullopt, anno);
        params.push_back(assign_doc);
      }
    }
    ffi::Array<StmtDoc> body = Flatten({VisitStmt(func->body)});

    ffi::Array<ExprDoc> kernel_decorators, func_decorators;
    // Add decorator: @T.prim_func
    kernel_decorators.push_back(TileLangPrefix("prim_func"));
    auto kernel_doc = FunctionDoc(IdDoc("kernel"), params, kernel_decorators, std::nullopt, body);

    // Add decorator: @tilelang.jit(out_idx=[...])
    if (auto out_idx = func->attrs.GetAttr<ffi::Array<IntImm>>("out_idx")) {
      ffi::Array<ExprDoc> out_idx_elements;
      for (const auto& idx : out_idx.value()) {
        out_idx_elements.push_back(LiteralDoc::Int(idx->value, std::nullopt));
      }
      auto out_idx_doc = ListDoc(out_idx_elements);
      func_decorators.push_back(
          IdDoc("tilelang")->Attr("jit")->Call({}, {ffi::String("out_idx")}, {out_idx_doc}));
    }
    auto return_doc = ReturnDoc(IdDoc("kernel"));

    return FunctionDoc(IdDoc(func_name), {}, func_decorators, std::nullopt,
                       {kernel_doc, return_doc});
  }

 protected:
  using ExprSelf = ExprFunctor<ExprDoc(const PrimExpr&)>;
  using StmtSelf = StmtFunctor<Doc(const Stmt&)>;

  using ExprSelf::VisitExpr;
  using StmtSelf::VisitStmt;

  /*!
   * \brief Collect nested AttrStmt nodes with thread_extent (blockIdx bindings).
   *
   * This function traverses nested AttrStmt nodes and collects all blockIdx bindings
   * in order (x, y, z), then returns the innermost body.
   *
   * \param op The AttrStmt node to start from.
   * \param block_dims Output: extents for blockIdx.x, y, z (in that order).
   * \param block_vars Output: variable names for blockIdx.x, y, z (in that order).
   * \return The innermost body after all blockIdx AttrStmt nodes.
   */
  Stmt CollectBlockIdxBindings(const AttrStmtNode* op,
                               ffi::Array<ffi::Optional<PrimExpr>>& block_dims,
                               ffi::Array<ffi::Optional<Var>>& block_vars) {
    const AttrStmtNode* current = op;
    while (current != nullptr) {
      if (current->attr_key == tir::attr::thread_extent) {
        auto iv = Downcast<IterVar>(current->node);
        const std::string& thread_tag = iv->thread_tag;
        int dim_index = thread_tag[thread_tag.size() - 1] - '0';
        if (dim_index >= 0) {
          block_dims.Set(dim_index, current->value);
          block_vars.Set(dim_index, iv->var);
        }
      }
      current = current->body.as<AttrStmtNode>();
    }

    // Find the innermost body
    const Stmt* body = &op->body;
    while (const auto* nested = body->as<AttrStmtNode>()) {
      body = &nested->body;
    }
    return *body;
  }

  Doc VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == tir::attr::thread_extent) {
      auto iv = Downcast<IterVar>(op->node);
      const std::string& thread_tag = iv->thread_tag;

      // Only handle vblockIdx (virtual blockIdx)
      if (thread_tag.compare(0, 10, "vblockIdx.") == 0) {
        // Collect all nested blockIdx bindings
        ffi::Array<ffi::Optional<PrimExpr>> block_dims(10, std::nullopt);
        ffi::Array<ffi::Optional<Var>> block_vars(10, std::nullopt);

        Stmt innermost_body = CollectBlockIdxBindings(op, block_dims, block_vars);

        // Visit the innermost body
        auto body_doc = VisitStmt(innermost_body);

        // Collect all valid dims and vars in order
        ffi::Array<PrimExpr> all_dims;
        ffi::Array<Var> all_vars;
        for (int i = 0; i < 10; ++i) {
          if (block_dims[i].has_value()) {
            all_dims.push_back(block_dims[i].value());
            all_vars.push_back(block_vars[i].value());
          }
        }

        // Build T.Kernel(...) call
        ffi::Array<ExprDoc> kernel_args;
        ffi::Array<ExprDoc> var_docs;
        ffi::Array<StmtDoc> final_body = Flatten({body_doc});

        if (all_dims.size() <= 3) {
          // Use dims and vars directly
          for (size_t i = 0; i < all_dims.size(); ++i) {
            kernel_args.push_back(VisitExpr(all_dims[i]));
            var_docs.push_back(VisitExpr(all_vars[i]));
          }
        } else {
          // First 2 dims stay as-is, merge dims from index 2 onwards
          kernel_args.push_back(VisitExpr(all_dims[0]));
          kernel_args.push_back(VisitExpr(all_dims[1]));
          var_docs.push_back(VisitExpr(all_vars[0]));
          var_docs.push_back(VisitExpr(all_vars[1]));

          // Merge dims[2], dims[3], ... into a single expression
          PrimExpr merged_dim = all_dims[2];
          for (size_t i = 3; i < all_dims.size(); ++i) {
            merged_dim = merged_dim * all_dims[i];
          }
          kernel_args.push_back(VisitExpr(merged_dim));

          // Create a new fused var for lhs[2]
          std::string fused_var_name = name_supply_->FreshName("fused_" + all_vars[2]->name_hint);
          ExprDoc fused_var_doc = IdDoc(fused_var_name);
          var_docs.push_back(fused_var_doc);

          // Build index_to_coordinates call and prepend to body
          ffi::Array<ExprDoc> original_vars_from_2;
          ffi::Array<ExprDoc> dims_from_2;
          for (size_t i = 2; i < all_vars.size(); ++i) {
            original_vars_from_2.push_back(VisitExpr(all_vars[i]));
            dims_from_2.push_back(VisitExpr(all_dims[i]));
          }
          ExprDoc coord_lhs = TupleDoc(original_vars_from_2);
          ExprDoc coord_rhs =
              CallTileLang("index_to_coordinates", {fused_var_doc, ListDoc(dims_from_2)});
          AssignDoc coord_assign = AssignDoc(coord_lhs, coord_rhs, std::nullopt);

          ffi::Array<StmtDoc> new_body;
          new_body.push_back(coord_assign);
          for (auto stmt : final_body) {
            new_body.push_back(stmt);
          }
          final_body = new_body;
        }

        // Create lhs: single var or tuple
        ffi::Optional<ExprDoc> lhs;
        if (var_docs.size() == 1) {
          lhs = var_docs[0];
        } else if (var_docs.size() > 1) {
          lhs = TupleDoc(var_docs);
        }

        // Create rhs: T.Kernel(...)
        ExprDoc rhs = TileLangPrefix("Kernel")->Call(kernel_args);

        // Create ScopeDoc: with T.Kernel(...) as (...):
        return ScopeDoc(lhs, rhs, final_body);
      }
    }

    // For non-blockIdx AttrStmt, just visit the body
    return VisitStmt(op->body);
  }

  Doc VisitStmt_(const BlockRealizeNode* op) override {
    auto cur_binding = GetBindings(ffi::GetRef<BlockRealize>(op));
    for (const auto& [var, expr] : cur_binding) {
      binding_.Set(var, expr);
    }
    const Block block = op->block;
    auto body_doc = VisitStmt(block->body);
    for (const auto& [var, expr] : cur_binding) {
      binding_.erase(var);
    }
    ffi::Array<Doc> stmts;
    for (auto buffer : block->alloc_buffers) {
      auto buffer_name = VisitExpr(buffer->data);
      auto buffer_shape = ListDoc(VisitExprArray(buffer->shape));
      auto buffer_dtype = TileLangDataType(buffer->dtype);
      auto scope = buffer.scope();
      if (scope == "shared") {
        stmts.push_back(AssignDoc(
            buffer_name, CallTileLang("alloc_shared", {buffer_shape, buffer_dtype}), std::nullopt));
      } else if (scope == "local.fragment") {
        stmts.push_back(AssignDoc(buffer_name,
                                  CallTileLang("alloc_fragment", {buffer_shape, buffer_dtype}),
                                  std::nullopt));
      } else {
        LOG_FATAL << "Unsupported scope: " << scope;
      }
    }
    stmts.push_back(body_doc);
    return StmtBlockDoc(Flatten(stmts));
  }

  Doc VisitStmt_(const ForNode* op) override {
    auto body_doc = VisitStmt(op->body);

    auto max = analyzer_.Simplify(op->min + op->extent);
    std::vector<std::pair<ffi::String, ExprDoc>> kwargs;
    if (auto num_stages = op->annotations.Get("num_stages")) {
      kwargs.push_back(
          {"num_stages", LiteralDoc::Int(num_stages.value().cast<int64_t>(), std::nullopt)});
    } else {
      kwargs.push_back({"num_stages", LiteralDoc::Int(1, std::nullopt)});
    }
    auto range = CallTileLang("Pipelined", {VisitExpr(op->min), VisitExpr(max)}, kwargs);
    return ForDoc(VisitExpr(op->loop_var), range, Flatten({body_doc}));
  }

  Doc VisitStmt_(const SeqStmtNode* op) override {
    ffi::Array<Doc> stmts;
    for (auto stmt : op->seq) {
      stmts.push_back(VisitStmt(stmt));
    }
    return StmtBlockDoc(Flatten(stmts));
  }

  Doc VisitStmt_(const EvaluateNode* op) override {
    // Special handling for tl_parallel - it generates a ForDoc (statement), not an expression
    if (const auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::tl_parallel())) {
        return VisitTLParallel(call);
      }
    }
    return ExprStmtDoc(VisitExpr(op->value));
  }

  Doc VisitTLParallel(const CallNode* op) {
    auto stmt_lhs = VisitExpr(op->args[0]);
    auto stmt_rhs = VisitExpr(op->args[1]);
    auto stmt = AssignDoc(stmt_lhs, stmt_rhs, std::nullopt);

    // Collect loop variables and extents
    ffi::Array<ExprDoc> loop_vars;     // i, j, ...
    ffi::Array<ExprDoc> loop_extents;  // 128, 128, ...
    for (size_t i = 2; i < op->args.size(); i += 2) {
      loop_vars.push_back(VisitExpr(op->args[i]));         // var
      loop_extents.push_back(VisitExpr(op->args[i + 1]));  // extent
    }

    // Create lhs: single var or tuple (i, j, ...)
    // Create rhs: T.Parallel(128, 128, ...)
    ExprDoc for_rhs = CallTileLang("Parallel", loop_extents);

    if (loop_vars.size() == 1) {
      return ForDoc(loop_vars[0], for_rhs, {stmt});
    } else {
      return ForDoc(TupleDoc(loop_vars), for_rhs, {stmt});
    }
  }

  ExprDoc VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::tl_region())) {
      ffi::Array<Doc> regions;
      auto buffer_load = Downcast<BufferLoad>(op->args[0]);
      auto buffer = VisitExpr(buffer_load->buffer->data);
      for (size_t i = 0; i < buffer_load->indices.size(); i++) {
        auto start_doc = VisitExpr(buffer_load->indices[i]);
        auto stop_doc = VisitExpr(buffer_load->indices[i] + op->args[i + 1]);
        regions.push_back(SliceDoc(start_doc, stop_doc, std::nullopt));
      }
      return buffer[regions];
    } else if (op->op.same_as(builtin::tl_gemm())) {
      std::vector<std::pair<ffi::String, ExprDoc>> kwargs;
      // We only need the name of tl_region for gemm and reduce
      auto A = Downcast<IndexDoc>(VisitExpr(op->args[0]))->value;
      auto B = Downcast<IndexDoc>(VisitExpr(op->args[1]))->value;
      auto C = Downcast<IndexDoc>(VisitExpr(op->args[2]))->value;
      auto [is_transA, is_transA_doc] = ToBoolean(VisitExpr(op->args[3]));
      if (is_transA) {
        kwargs.push_back({"transpose_A", is_transA_doc});
      }
      auto [is_transB, is_transB_doc] = ToBoolean(VisitExpr(op->args[4]));
      if (is_transB) {
        kwargs.push_back({"transpose_B", is_transB_doc});
      }
      auto policy = VisitExpr(op->args[5]);
      if (Downcast<IntImm>(Downcast<LiteralDoc>(policy)->value)->value != 0) {
        kwargs.push_back({"policy", policy});
      }
      auto [clear_accum, clear_accum_doc] = ToBoolean(VisitExpr(op->args[6]));
      if (clear_accum) {
        kwargs.push_back({"clear_accum", clear_accum_doc});
      }
      auto k_pack_doc = VisitExpr(op->args[7]);
      if (Downcast<IntImm>(Downcast<LiteralDoc>(k_pack_doc)->value)->value != 1) {
        kwargs.push_back({"k_pack", k_pack_doc});
      }
      auto wg_wait_doc = VisitExpr(op->args[8]);
      if (Downcast<IntImm>(Downcast<LiteralDoc>(wg_wait_doc)->value)->value != 0) {
        kwargs.push_back({"wg_wait", wg_wait_doc});
      }
      // FIXME: mbar is not supported yet
      return CallTileLang("gemm", {A, B, C}, kwargs);
    } else if (op->op.same_as(builtin::tl_reduce())) {
      auto src = Downcast<IndexDoc>(VisitExpr(op->args[0]))->value;
      auto dst = Downcast<IndexDoc>(VisitExpr(op->args[1]))->value;
      auto reduce_type = VisitExpr(op->args[2]);
      auto reduce_dim = VisitExpr(op->args[3]);
      auto [clear, clear_doc] = ToBoolean(VisitExpr(op->args[4]));
      return CallTileLang("reduce", {src, dst, reduce_type, reduce_dim, clear_doc});
    } else if (op->op.same_as(builtin::tl_reduce_topk())) {
      auto input = Downcast<IndexDoc>(VisitExpr(op->args[0]))->value;
      auto topk_values = Downcast<IndexDoc>(VisitExpr(op->args[1]))->value;
      auto topk_indices = Downcast<IndexDoc>(VisitExpr(op->args[2]))->value;
      auto topk = VisitExpr(op->args[3]);
      auto axis = VisitExpr(op->args[4]);
      auto start_offset = VisitExpr(op->args[5]);
      return IdDoc("reduce_topk")
          ->Call({input, topk_values, topk_indices, topk, axis, start_offset});
    } else if (op->op.same_as(builtin::tl_copy())) {
      std::vector<std::pair<ffi::String, ExprDoc>> kwargs;
      // we need a real region in T.copy
      auto src = VisitExpr(op->args[0]);
      auto dst = VisitExpr(op->args[1]);
      auto coalesced_width = VisitExpr(op->args[2]);
      if (Downcast<IntImm>(Downcast<LiteralDoc>(coalesced_width)->value)->value != -1) {
        kwargs.push_back({"coalesced_width", coalesced_width});
      }
      auto [disable_tma, disable_tma_doc] = ToBoolean(VisitExpr(op->args[3]));
      if (disable_tma) {
        kwargs.push_back({"disable_tma", disable_tma_doc});
      }
      auto eviction_policy = VisitExpr(op->args[4]);
      ffi::Map<int64_t, LiteralDoc> eviction_policy_map = {
          // {0, LiteralDoc::Str("evict_normal", std::nullopt)}, // default value
          {1, LiteralDoc::Str("evict_first", std::nullopt)},
          {2, LiteralDoc::Str("evict_last", std::nullopt)},
      };
      auto eviction_policy_doc = eviction_policy_map.Get(
          Downcast<IntImm>(Downcast<LiteralDoc>(eviction_policy)->value)->value);
      if (eviction_policy_doc.has_value()) {
        kwargs.push_back({"eviction_policy", eviction_policy_doc.value()});
      }
      return CallTileLang("copy", {src, dst}, kwargs);
    } else if (op->op.same_as(builtin::tl_fill())) {
      return CallTileLang("fill", VisitExprArray(op->args));
    } else {
      // Math functions: tir.xxx -> T.xxx
      if (op->op.same_as(Op::Get("tir.exp"))) {
        return CallTileLang("exp", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.exp2"))) {
        return CallTileLang("exp2", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.exp10"))) {
        return CallTileLang("exp10", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.log"))) {
        return CallTileLang("log", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.log2"))) {
        return CallTileLang("log2", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.log10"))) {
        return CallTileLang("log10", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.sin"))) {
        return CallTileLang("sin", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.cos"))) {
        return CallTileLang("cos", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.tan"))) {
        return CallTileLang("tan", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.sinh"))) {
        return CallTileLang("sinh", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.cosh"))) {
        return CallTileLang("cosh", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.tanh"))) {
        return CallTileLang("tanh", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.atan"))) {
        return CallTileLang("atan", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.sqrt"))) {
        return CallTileLang("sqrt", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.rsqrt"))) {
        return CallTileLang("rsqrt", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.erf"))) {
        return CallTileLang("erf", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.floor"))) {
        return CallTileLang("floor", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.ceil"))) {
        return CallTileLang("ceil", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.trunc"))) {
        return CallTileLang("trunc", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.round"))) {
        return CallTileLang("round", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.nearbyint"))) {
        return CallTileLang("nearbyint", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.fabs"))) {
        return CallTileLang("abs", {VisitExpr(op->args[0])});
      } else if (op->op.same_as(Op::Get("tir.pow"))) {
        return CallTileLang("pow", {VisitExpr(op->args[0]), VisitExpr(op->args[1])});
      } else if (op->op.same_as(Op::Get("tir.sigmoid"))) {
        return CallTileLang("sigmoid", {VisitExpr(op->args[0])});
      }
      LOG_FATAL << "Unsupported call: " << op->op;
    }
  }

  ExprDoc VisitExpr_(const BufferLoadNode* op) override {
    ExprDoc buffer = VisitExpr(op->buffer->data);
    ffi::Array<Doc> indices;
    for (const auto& idx : op->indices) {
      indices.push_back(VisitExpr(idx));
    }
    return buffer[indices];
  }

  ExprDoc VisitExpr_(const CastNode* op) override {
    return CallTileLang("Cast", {TileLangDataType(op->dtype), VisitExpr(op->value)});
  }

  ExprDoc VisitExpr_(const MaxNode* op) override {
    return CallTileLang("max", {VisitExpr(op->a), VisitExpr(op->b)});
  }

  ExprDoc VisitExpr_(const AddNode* op) override {
    return OperationDoc(OperationDocNode::Kind::kAdd, {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const SubNode* op) override {
    return OperationDoc(OperationDocNode::Kind::kSub, {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const MulNode* op) override {
    return OperationDoc(OperationDocNode::Kind::kMult, {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const DivNode* op) override {
    return OperationDoc(OperationDocNode::Kind::kDiv, {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const FloorDivNode* op) override {
    return OperationDoc(OperationDocNode::Kind::kFloorDiv, {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const FloorModNode* op) override {
    return OperationDoc(OperationDocNode::Kind::kMod, {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const LTNode* op) override {
    return OperationDoc(OperationDocNode::Kind::kLt, {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const LENode* op) override {
    return OperationDoc(OperationDocNode::Kind::kLtE, {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const EQNode* op) override {
    return OperationDoc(OperationDocNode::Kind::kEq, {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const NENode* op) override {
    return OperationDoc(OperationDocNode::Kind::kNotEq, {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const GTNode* op) override {
    return OperationDoc(OperationDocNode::Kind::kGt, {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const GENode* op) override {
    return OperationDoc(OperationDocNode::Kind::kGtE, {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const AndNode* op) override {
    return OperationDoc(OperationDocNode::Kind::kAnd, {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const OrNode* op) override {
    return OperationDoc(OperationDocNode::Kind::kOr, {VisitExpr(op->a), VisitExpr(op->b)});
  }

  ExprDoc VisitExpr_(const NotNode* op) override {
    return OperationDoc(OperationDocNode::Kind::kNot, {VisitExpr(op->a)});
  }

  ExprDoc VisitExpr_(const IntImmNode* op) override {
    return LiteralDoc::Int(op->value, std::nullopt);
  }

  ExprDoc VisitExpr_(const FloatImmNode* op) override {
    if (op->value == std::numeric_limits<float>::lowest()) {
      return OperationDoc(OperationDocNode::Kind::kUSub,
                          {CallTileLang("infinity", {TileLangDataType(op->dtype)})});
    }
    if (op->value == std::numeric_limits<float>::max()) {
      return CallTileLang("infinity", {TileLangDataType(op->dtype)});
    }
    if (std::isinf(op->value)) {
      if (std::signbit(op->value)) {
        return OperationDoc(OperationDocNode::Kind::kUSub,
                            {CallTileLang("infinity", {TileLangDataType(op->dtype)})});
      } else {
        return CallTileLang("infinity", {TileLangDataType(op->dtype)});
      }
    }
    return LiteralDoc::Float(op->value, std::nullopt);
  }

  ExprDoc VisitExpr_(const StringImmNode* op) override {
    return LiteralDoc::Str(op->value, std::nullopt);
  }

  ExprDoc VisitExpr_(const VarNode* op) override {
    Var var = binding_.Get(ffi::GetRef<Var>(op)).has_value()
                  ? Downcast<Var>(binding_.Get(ffi::GetRef<Var>(op)).value())
                  : ffi::GetRef<Var>(op);
    auto it = var_name_map_.find(var);
    std::string name;
    if (it != var_name_map_.end()) {
      name = (*it).second;
    } else {
      name = name_supply_->FreshName(SimplifyName(op->name_hint));
      var_name_map_.Set(var, name);
    }
    return IdDoc(name);
  }

  std::pair<bool, ExprDoc> ToBoolean(const ExprDoc& expr) {
    if (Downcast<IntImm>(Downcast<LiteralDoc>(expr)->value)->value == 1) {
      return {true, LiteralDoc::Boolean(true, std::nullopt)};
    } else {
      return {false, LiteralDoc::Boolean(false, std::nullopt)};
    }
  }

  ffi::Array<ExprDoc> VisitExprArray(const ffi::Array<PrimExpr>& op) {
    ffi::Array<ExprDoc> args;
    for (auto expr : op) {
      args.push_back(VisitExpr(expr));
    }
    return args;
  }

  ffi::Array<StmtDoc> Flatten(const ffi::Array<Doc>& docs) {
    ffi::Array<StmtDoc> stmts;
    for (auto doc : docs) {
      if (auto block = doc.as<StmtBlockDocNode>()) {
        stmts.insert(stmts.end(), block->stmts.begin(), block->stmts.end());
      } else {
        stmts.push_back(Downcast<StmtDoc>(doc));
      }
    }
    return stmts;
  }

  std::string SimplifyName(std::string name) {
    std::replace(name.begin(), name.end(), '.', '_');
    // 1. find _local_fragment and remove it
    auto idx = name.find("_local_fragment");
    if (idx != std::string::npos) {
      name.replace(idx, 15, "");
    }
    // 2. find _fragment and remove it
    idx = name.find("_fragment");
    if (idx != std::string::npos) {
      name.replace(idx, 9, "");
    }
    // 3. find _shared and remove it
    idx = name.find("_shared");
    if (idx != std::string::npos) {
      name.replace(idx, 7, "");
    }
    // 4. find T_ and remove it
    idx = name.find("T_");
    if (idx != std::string::npos) {
      name.replace(idx, 2, "");
    }
    return name;
  }

  arith::Analyzer analyzer_;

  // The name supply for generating unique names for variables.
  NameSupply name_supply_;
  // A map from TVM variables to their names in the TileLang script.
  ffi::Map<Var, ffi::String> var_name_map_;

  ffi::Map<Var, PrimExpr> binding_;
};

}  // namespace

ffi::String FunctionToTileLangScript(const std::string& func_name, const PrimFunc& func) {
  // Generate the function doc
  FunctionDoc func_doc = CodeGenTileLang().VisitFunc(func_name, func.get());

  // Convert to Python script
  std::string script = DocToPythonScript(func_doc, PrinterConfig());

  return script + "\n\n";
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.FunctionToTileLangScript", FunctionToTileLangScript);
}

}  // namespace tvm::codegen
