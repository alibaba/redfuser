import ast
from dataclasses import dataclass
import sympy as sp
from sympy.matrices.expressions import MatrixExpr
from sympy.printing.str import StrPrinter


@dataclass
class ReductionConfig:
    node: ast.AST
    is_reduction: bool
    reduce_target: str
    iter: str
    reduce_op: str
    reduce_func: str
    x_map: dict
    y_map: dict
    c_map: dict


D_PREFIX = "d_"
BMat_M = 2
BMat_N = 2


def _patch_sympy_functions():
    """让sympy函数自动支持BMat对象"""
    original_functions = {}
    function_names = [
        "exp",
        "log",
        "sqrt",
        "Abs",
        "Max",
        "Mul"
    ]

    for func_name in function_names:
        original_functions[func_name] = getattr(sp, func_name)

    def create_patched_function(original_func, func_name):
        def patched_func(*args, **kwargs):
            d_vars = []
            free_vars = []
            bmat_args = []
            i = 0
            for arg in args:
                if isinstance(arg, BMat):
                    d = sp.Dummy(f"d{i}")
                    d_vars.append(d)
                    free_vars.append(d)
                    bmat_args.append(arg)
                    i += 1
                else:
                    d_vars.append(arg)
            if i > 0:
                func_call = original_func(*d_vars, **kwargs)
                return BMat.ew(sp.Lambda(tuple(free_vars), func_call), *bmat_args)
            else:
                return original_func(*args, **kwargs)

        return patched_func

    # 应用patch
    for func_name, original_func in original_functions.items():
        setattr(sp, func_name, create_patched_function(original_func, func_name))


_patch_sympy_functions()


class TopK(sp.Function):
    @classmethod
    def eval(cls, x, y):
        return None

    def __str__(self):
        return f"reduce_topk({self.args[0]}, {self.args[1]})"


def _eval_expr_with_bmat(expr, subs_map):
    """
    递归求值表达式，用 Python 运算符替代 sympy 内部构造，
    从而触发 BMat 的 __add__, __mul__ 等重载方法
    """
    # 变量替换
    if expr in subs_map:
        return subs_map[expr]
    
    # 基本类型直接返回
    if expr.is_Number:
        return expr
    if expr.is_Symbol and expr not in subs_map:
        return expr
    
    # 递归处理子表达式
    args = [_eval_expr_with_bmat(arg, subs_map) for arg in expr.args]
    
    # 根据表达式类型，用 Python 运算符重建
    if expr.is_Add:
        result = args[0]
        for arg in args[1:]:
            result = result + arg
        return result
    elif expr.is_Mul:
        result = args[0]
        for arg in args[1:]:
            result = result * arg
        return result
    elif expr.is_Pow:
        return args[0] ** args[1]
    elif isinstance(expr, sp.exp):
        return sp.exp(args[0])
    elif isinstance(expr, sp.log):
        return sp.log(args[0])
    elif isinstance(expr, sp.Abs):
        return sp.Abs(args[0])
    elif isinstance(expr, sp.Max):
        return sp.Max(*args)
    else:
        # 其他类型用原始类重建
        return type(expr)(*args)


class ElementwiseApplyNAry(MatrixExpr):
    def __new__(cls, func, *ops):
        if not all(isinstance(op, BMat) for op in ops):
            subs_map = dict(zip(func.variables, ops))
            return _eval_expr_with_bmat(func.expr, subs_map)
        mat_shapes = [op.shape for op in ops]
        bshape = BMat._get_bshape(mat_shapes)
        ops_b = [BMat._broadcast_to(op, bshape) for op in ops]
        if not isinstance(func, sp.core.function.Lambda):
            tmps = tuple(sp.Dummy(f"d{i}") for i in range(len(ops_b)))
            func = sp.core.function.Lambda(tmps, func(*tmps))
        obj = MatrixExpr.__new__(cls, func, *ops_b)
        obj._shape = bshape
        return obj

    @property
    def shape(self):
        return self._shape

    @property
    def function(self):
        return self.args[0]

    @property
    def operands(self):
        return self.args[1:]

    @property
    def all_operands(self):
        ops = []
        for op in self.args[1:]:
            ops.extend(list(op.all_operands))
        return ops

    def _entry(self, i, j, **kwargs):
        vals = [op._entry(i, j, **kwargs) for op in self.operands]
        return self.function(*vals)

    def __str__(self):
        func_str = str(self.function)
        ops_str = []
        for op in self.operands:
            if isinstance(op.expr, sp.MatMul):
                ops_str.append("@".join(str(arg) for arg in op.operands))
            else:
                ops_str.append(str(op))
        return f"ew({func_str}, {', '.join(ops_str)})"

    def subs(self, *args, **kwargs):
        subs_dict = args[0]
        if not subs_dict:
            return self
        new_operands = [op.subs(subs_dict, **kwargs) for op in self.operands]
        return ElementwiseApplyNAry(self.function, *new_operands)


class BMat(MatrixExpr):
    _op_priority = 10000

    def __new__(cls, x, bshape=None):
        x = x.expr if isinstance(x, BMat) else x
        if not isinstance(x, (sp.MatrixSymbol, ElementwiseApplyNAry, sp.MatMul, sp.ZeroMatrix, sp.Identity, sp.OneMatrix)):
            raise TypeError(
                f"BMat need MatMul/ElementwiseApplyNAry/MatrixSymbol/ZeroMatrix/Identity/OneMatrix, got: {type(x)}"
            )
        obj = MatrixExpr.__new__(cls, x)
        obj._expr = x
        obj._bshape = bshape if bshape else x.shape
        return obj

    @property
    def expr(self):
        return self._expr

    @property
    def shape(self):
        return self._bshape

    @property
    def function(self):
        if isinstance(self.expr, ElementwiseApplyNAry):
            return self.expr.function
        else:
            return None

    @property
    def operands(self):
        if isinstance(self.expr, ElementwiseApplyNAry):
            return list(self.expr.operands)
        elif isinstance(self.expr, sp.MatMul):
            return list(self.expr.args)
        return [self]
    
    @property
    def all_operands(self):
        ops = []
        if isinstance(self.expr, ElementwiseApplyNAry):
            for op in self.expr.operands:
                ops.extend(list(op.all_operands))
        elif isinstance(self.expr, sp.MatMul):
            for op in self.expr.args:
                ops.extend(list(op.all_operands))
        else:
            ops.append(self)
        return ops

    def __str__(self):
        if isinstance(self.expr, sp.MatMul):
            return "@".join(str(arg) for arg in self.expr.args)

        return f"{str(self.expr)}"

    @classmethod
    def _get_bshape(cls, shape_list):
        if len(shape_list) == 1:
            return shape_list[0]

        result = shape_list[0]
        for shape in shape_list[1:]:
            M, N = result
            MB, NB = shape

            # 行维度广播
            if M == 1 or MB == 1 or M == MB:
                result_M = max(M, MB)
            else:
                raise ValueError(f"行维度不兼容: {M} vs {MB}")

            # 列维度广播
            if N == 1 or NB == 1 or N == NB:
                result_N = max(N, NB)
            else:
                raise ValueError(f"列维度不兼容: {N} vs {NB}")

            result = (result_M, result_N)

        return result

    @classmethod
    def _broadcast_to(cls, x, shape):
        M, N = shape
        m, n = x.shape
        if m in (1, M) and n in (1, N):
            return cls(x, (M, N))

        raise ValueError(f"无法广播：目标形状={shape}，给定={x.shape}")

    @classmethod
    def ew(cls, func, *args):
        result = cls(ElementwiseApplyNAry(func, *args))

        return result.flatten_elementwise()

    def _check_type(self, other):
        if not isinstance(other, BMat) and not other.is_real:
            raise ValueError(f"BMat need BMat/Symbol/Number, got: {type(other)}")

    def __add__(self, other):
        self._check_type(other)
        d1 = sp.Dummy("d1")
        d2 = sp.Dummy("d2")
        if isinstance(other, BMat):
            return self.ew(sp.Lambda((d1, d2), d1 + d2), self, other)
        else:
            return self.ew(sp.Lambda((d1), d1 + other), self)

    def __radd__(self, other):
        self._check_type(other)
        d1 = sp.Dummy("d1")
        d2 = sp.Dummy("d2")
        if isinstance(other, BMat):
            return self.ew(sp.Lambda((d1, d2), d1 + d2), other, self)
        else:
            return self.ew(sp.Lambda((d1), other + d1), self)

    def __sub__(self, other):
        self._check_type(other)
        d1 = sp.Dummy("d1")
        d2 = sp.Dummy("d2")
        if isinstance(other, BMat):
            return self.ew(sp.Lambda((d1, d2), d1 - d2), self, other)
        else:
            return self.ew(sp.Lambda((d1), d1 - other), self)

    def __rsub__(self, other):
        self._check_type(other)
        d1 = sp.Dummy("d1")
        d2 = sp.Dummy("d2")
        if isinstance(other, BMat):
            return self.ew(sp.Lambda((d1, d2), d1 - d2), other, self)
        else:
            return self.ew(sp.Lambda((d1), other - d1), self)

    def __truediv__(self, other):
        self._check_type(other)
        d1 = sp.Dummy("d1")
        d2 = sp.Dummy("d2")
        if isinstance(other, BMat):
            return self.ew(sp.Lambda((d1, d2), d1 / d2), self, other)
        else:
            return self.ew(sp.Lambda((d1), d1 / other), self)

    def __rtruediv__(self, other):
        self._check_type(other)
        d1 = sp.Dummy("d1")
        d2 = sp.Dummy("d2")
        if isinstance(other, BMat):
            return self.ew(sp.Lambda((d1, d2), d1 / d2), other, self)
        else:
            return self.ew(sp.Lambda((d1), other / d1), self)

    def __mul__(self, other):
        self._check_type(other)
        d1 = sp.Dummy("d1")
        d2 = sp.Dummy("d2")
        if isinstance(other, BMat):
            return self.ew(sp.Lambda((d1, d2), d1 * d2), self, other)
        else:
            return self.ew(sp.Lambda((d1), d1 * other), self)

    def __rmul__(self, other):
        self._check_type(other)
        d1 = sp.Dummy("d1")
        d2 = sp.Dummy("d2")
        if isinstance(other, BMat):
            return self.ew(sp.Lambda((d1, d2), d1 * d2), other, self)
        else:
            return self.ew(sp.Lambda((d1), other * d1), self)

    def __matmul__(self, other):
        if not isinstance(other, BMat):
            raise ValueError(f"BMatMul need BMat, got: {type(other)}")
        return BMat(sp.MatMul(self, other))

    def __rmatmul__(self, other):
        if not isinstance(other, BMat):
            raise ValueError(f"BMatMul need BMat, got: {type(other)}")
        return BMat(sp.MatMul(other, self))

    def __neg__(self):
        d = sp.Dummy("d")
        return self.ew(sp.Lambda((d), -d), self)

    def __pow__(self, other):
        if not isinstance(other, (sp.Number, sp.Symbol)):
            raise ValueError(f"BMatPow need Symbol/Number, got: {type(other)}")
        d = sp.Dummy("d")
        return self.ew(sp.Lambda(d, sp.Pow(d, other)), self)

    def __abs__(self):
        d = sp.Dummy("d")
        return self.ew(sp.Lambda((d), sp.Abs(d)), self)

    def flatten_elementwise(self):
        """
        展开嵌套的Elementwise函数为单个Elementwise函数
        将 f(g(A, B), C) 转换为 h(A, B, C)，其中 h = lambda a,b,c: f(g(a,b), c)
        """

        def _collect_base_matrices_and_build_expr(
            expr, matrix_to_var={}, unique_matrices=[]
        ):
            """
            递归收集基础矩阵并构建组合表达式，处理重复矩阵
            返回: (表达式)
            """
            expr = expr.expr if isinstance(expr, BMat) else expr
            if isinstance(expr, (ElementwiseApplyNAry)):
                func = expr.function  # Lambda函数
                matrices = list(expr.operands)  # 矩阵参数

                arg_exprs = []

                for matrix in matrices:
                    if isinstance(matrix.expr, (ElementwiseApplyNAry)):
                        # 嵌套的Elementwise，递归处理
                        sub_expr = _collect_base_matrices_and_build_expr(
                            matrix, matrix_to_var, unique_matrices
                        )
                        arg_exprs.append(sub_expr)
                    else:
                        # 基础矩阵，检查是否已经存在
                        matrix_key = str(matrix)
                        if matrix_key in matrix_to_var:
                            arg_exprs.append(matrix_to_var[matrix_key])
                        else:
                            var_name = sp.Dummy(f"d{len(unique_matrices)}")
                            matrix_to_var[matrix_key] = var_name
                            unique_matrices.append(matrix)
                            arg_exprs.append(var_name)

                # 构建组合表达式：将func应用到arg_exprs
                if len(func.variables) == len(arg_exprs):
                    substitutions = dict(zip(func.variables, arg_exprs))
                    combined_expr = func.expr.subs(substitutions)
                else:
                    combined_expr = func.expr

                return combined_expr
            else:
                # 不是Elementwise表达式，作为基础矩阵
                matrix_key = str(expr)
                if matrix_key in matrix_to_var:
                    # 复用已有的变量
                    return matrix_to_var[matrix_key]
                else:
                    # 创建新的变量
                    var_name = sp.Dummy(f"d{len(unique_matrices)}")
                    matrix_to_var[matrix_key] = var_name
                    unique_matrices.append(expr)
                    return var_name

        def _update_shape(expr: BMat):
            new_ops = [BMat(op.expr) for op in expr.operands]
            if isinstance(expr.expr, ElementwiseApplyNAry):
                return BMat(ElementwiseApplyNAry(expr.function, *new_ops))
            elif isinstance(expr.expr, sp.MatMul):
                return BMat(sp.MatMul(*new_ops))
            else:
                return BMat(expr.expr)

        try:
            matrix_to_var = {}
            unique_matrices = []
            combined_expr = sp.simplify(
                _collect_base_matrices_and_build_expr(
                    self, matrix_to_var, unique_matrices
                ),
                force=True,
            )

            if not combined_expr.free_symbols:
                # 表达式中没有变量（是常数）
                return BMat(sp.OneMatrix(*self.shape) * combined_expr)

            # 找出化简后表达式中实际使用的变量
            used_symbols = combined_expr.free_symbols

            # 过滤出实际使用的矩阵和变量
            used_matrices = []
            used_vars = []
            for var in matrix_to_var.values():
                if var in used_symbols:
                    # 找到对应的矩阵
                    for matrix_key, mapped_var in matrix_to_var.items():
                        if mapped_var == var:
                            # 找到原始矩阵对象
                            for orig_matrix in unique_matrices:
                                if str(orig_matrix) == matrix_key:
                                    used_matrices.append(orig_matrix)
                                    used_vars.append(var)
                                    break
                            break

            new_func = sp.Lambda(tuple(used_vars), combined_expr)
            flattened = ElementwiseApplyNAry(new_func, *used_matrices)

            return _update_shape(BMat(flattened))

        except Exception as e:
            print(f"展开失败: {e}")
            return self

    def _entry(self, i, j, **kwargs):
        expr_rows, expr_cols = self.expr.shape
        i_actual = sp.Integer(0) if expr_rows == 1 else i
        j_actual = sp.Integer(0) if expr_cols == 1 else j
        return self.expr._entry(i_actual, j_actual, **kwargs)

    def subs(self, *args, **kwargs):
        subs_dict = args[0]
        for bmat_key, replacement in subs_dict.items():
            if self.expr == bmat_key.expr:
                return BMat(replacement)

        new_expr = self.expr.subs(subs_dict, **kwargs)
        return BMat(new_expr)

    def eliminate_fixed_suffix(self, elim_const: bool):
        def _is_fixed_suffix_symbol(e):
            e = e.expr if isinstance(e, BMat) else e
            return isinstance(e, sp.MatrixSymbol) and str(e).endswith("_fixed")

        def _replace_symbol(elim_flag, e, ctx):
            if elim_flag or _is_fixed_suffix_symbol(e):
                if ctx == "eye":
                    return BMat(sp.Identity(e.shape[0]))
                else:
                    # TODO:应该返回0矩阵吗?
                    return BMat(sp.ZeroMatrix(*e.shape))
            return None

        def _elim_fixed_args(e, vars_to_ops: dict, ctx):
            if e.is_Dummy or e.is_Symbol:
                if _is_fixed_suffix_symbol(vars_to_ops.get(e, None)):
                    vars_to_ops.pop(e)
                    return sp.sympify(ctx)
                else:
                    return e
            if e.is_number:
                return e
            cur_ctx = None
            if e.is_Add or isinstance(e, (sp.functions.elementary.complexes.Abs)):
                cur_ctx = "0"
            elif e.is_Mul or e.is_Pow:
                cur_ctx = "1"
            elif isinstance(
                e,
                (
                    sp.functions.elementary.exponential.exp,
                    sp.functions.elementary.miscellaneous.Max,
                ),
            ):
                cur_ctx = "-oo"
            parts = [_elim_fixed_args(arg, vars_to_ops, cur_ctx) for arg in e.args]
            if elim_const:
                parts = [part for part in parts if not str(part).startswith("c")]
            if (not parts) or all([part.is_number for part in parts]):
                return sp.sympify(ctx)
            return type(e)(*parts)

        def _elim_rec(elim_flag, e, ctx=None):
            e = e.expr if isinstance(e, BMat) else e

            # 若是fixed MatrixSymbol，按当前上下文替换
            repl = _replace_symbol(elim_flag, e, ctx)
            if repl:
                return repl

            if isinstance(e, sp.MatMul):
                parts = [_elim_rec(elim_flag, a, ctx="eye") for a in e.args]
                # 消除所有单位阵
                parts = [part for part in parts if not isinstance(part.expr if isinstance(part, BMat) else part, sp.Identity)]
                # 全都被消除了，则根据当前ctx进行替换
                if not parts:
                    return _replace_symbol(True, e, ctx)
                if len(parts) == 1:
                    return parts[0]
                return BMat(sp.MatMul(*parts))

            if isinstance(e, ElementwiseApplyNAry):
                parts = []
                vars = e.function.args[0]
                func = e.function.expr
                operands = e.operands
                for op in operands:
                    op_expr = op.expr if isinstance(op, BMat) else op
                    # fixed MatrixSymbol 在后续进行处理
                    if isinstance(op_expr, sp.MatrixSymbol):
                        parts.append(op)
                    else:
                        part = _elim_rec(elim_flag, op, ctx)
                        parts.append(
                            sp.MatrixSymbol(f"{op}_fixed", *op.shape)
                            if isinstance(part, BMat) and part.expr.is_ZeroMatrix
                            else part
                        )
                vars_to_ops = dict(zip(vars, parts))

                new_func = _elim_fixed_args(func, vars_to_ops, None)
                if new_func is None or new_func.is_number:
                    return _replace_symbol(True, e, ctx)
                # f(x) = x
                if new_func.is_Dummy:
                    return BMat(list(vars_to_ops.values())[0])
                return BMat(
                    ElementwiseApplyNAry(
                        sp.Lambda(tuple(vars_to_ops.keys()), new_func),
                        *list(vars_to_ops.values()),
                    )
                ) if vars_to_ops else new_func

            return BMat(e)

        return BMat(_elim_rec(False, self.expr))

def strip_bmat(obj):
    return obj.expr if isinstance(obj, BMat) else obj


class TileLangPrinter(StrPrinter):
    """将 SymPy 表达式转换为 TileLang 代码"""
    
    def _print_MatrixElement(self, expr):
        """skip printing the zero index"""
        parent = self._print(expr.parent)
        indices = []
        for idx in expr.args[1:]:
            # 如果索引不是 0，则添加到索引列表
            if idx != 0:
                indices.append(self._print(idx))
        
        return f"{parent}[{', '.join(indices)}]"
    
    def _print_exp(self, expr):
        arg = self._print(expr.args[0])
        return f"T.exp2(({arg}) * 1.44269504)"
    
    def _print_log(self, expr):
        arg = self._print(expr.args[0])
        return f"T.log({arg})"
    
    def _print_sqrt(self, expr):
        arg = self._print(expr.args[0])
        return f"T.sqrt({arg})"
    
    def _print_Abs(self, expr):
        arg = self._print(expr.args[0])
        return f"T.abs({arg})"
    
    def _print_Max(self, expr):
        args = ', '.join([self._print(arg) for arg in expr.args])
        return f"T.max({args})"
    
    def _print_Pow(self, expr):
        base = self._print(expr.args[0])
        exp = self._print(expr.args[1])
        # 如果指数是 0.5, 使用 sqrt
        if expr.args[1] == sp.Rational(1, 2):
            return f"T.sqrt({base})"
        # 如果指数是 2, 使用乘法
        elif expr.args[1] == 2:
            return f"({base}) * ({base})"
        else:
            return f"T.pow({base}, {exp})"
    
    def _print_floor(self, expr):
        arg = self._print(expr.args[0])
        return f"T.floor({arg})"
    
    def _print_ceiling(self, expr):
        arg = self._print(expr.args[0])
        return f"T.ceil({arg})"

def to_tilelang(expr):
    """将 SymPy 表达式转换为 TileLang 代码字符串"""
    return TileLangPrinter().doprint(expr)

