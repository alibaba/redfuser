from tvm import tir

from typing import List

from .common_analysis_v2 import BlockInfo
from .utils import ReductionConfig


class CascadedGroupInfo:
    """记录一个Cascaded Group的详细信息"""
    
    def __init__(self, cascaded_group: List[BlockInfo], sch: tir.Schedule):
        self.cascaded_group = cascaded_group
        self.sch = sch

        self.x_counter = 0
        self.y_counter = 0
        self.c_counter = 0
        
        self.reduction_configs: List[ReductionConfig] = []
        self.var_map: dict = {}
        self.x_map: dict = {}
        self.y_map: dict = {}
        self.c_map: dict = {}

        # e.g. 'x0' -> some BufferLoad, 'c0' -> some FloatImm(1.0)
        self.x_buffer_load_map: dict = {}
        self.y_buffer_load_map: dict = {}
        self.c_buffer_load_map: dict = {}
        self.prev_buffer_load_map: dict = {}

        self.reduce_funcs: list = []
        self.op_params: dict = {} # 存储可能的额外参数 {reduce_target: {...}}

        self._analyze()


    def _analyze_buffer_store(self, stmt: tir.BufferStore) -> tuple:
        """
            分析BufferStore类型的stmt,返回(reduce_op, reduce_func, reduce_target)

            e.g.
                - a = a + expr  -> reduce_op="+"
                - a = max(a, expr) -> reduce_op="max"
        """
        stmt_buffer = stmt.buffer
        stmt_value = stmt.value

        def _is_buffer_load(expr, buffer):
            if isinstance(expr, tir.BufferLoad):
                return expr.buffer.same_as(buffer)
            return False

        if isinstance(stmt_value, tir.Add):
            reduce_op = "+"
            if _is_buffer_load(stmt_value.a, stmt_buffer):
                reduce_func = stmt_value.b
                reduce_target = stmt_value.a
            else:
                reduce_func = stmt_value.a
                reduce_target = stmt_value.b
        elif isinstance(stmt_value, tir.Max):
            reduce_op = "max"
            if _is_buffer_load(stmt_value.a, stmt_buffer):
                reduce_func = stmt_value.b
                reduce_target = stmt_value.a
            else:
                reduce_func = stmt_value.a
                reduce_target = stmt_value.b
        else:
            raise NotImplementedError(f"Unsupported stmt_value type: {type(stmt_value)}")

        return reduce_op, reduce_func, reduce_target


    def _analyze_evaluate(self, stmt: tir.Evaluate, block: tir.Block) -> tuple:
        """
            分析Evaluate类型的stmt,返回(reduce_op, reduce_func, reduce_target, reduce_target_buffers, op_extra_params)

            支持：
                - vec_reduce("topk", ...) -> reduce_op="topk"
        """
        call = stmt.value

        # 检查是否是Call
        if not isinstance(call, tir.Call):
            raise NotImplementedError(f"Unsupported Evaluate content: {type(call)}")

        if call.op.name == "tir.vec_reduce":
            op_type = call.args[0] # "topk"

            if op_type == "topk":
                return self._analyze_vec_reduce_topk(call, block)
            else:
                raise NotImplementedError(f"Unsupported vec_reduce op_type: {op_type}")
        else:
            raise NotImplementedError(f"Unsupported Call op: {call.op.name}")


    def _analyze_vec_reduce_topk(self, call: tir.Call, block: tir.Block) -> tuple:
        """
            分析T.vec_reduce("topk", ...)

            参数结构:
            - args[0]: op_type ("topk")
            - args[1]: num_topk (8)
            - args[2]: axis (-1)
            - args[3]: input_load (BufferLoad)
            - args[4:-1]: output_loads [values, indices]
            - args[-1]: reduce_loop_var

            返回: (reduce_op, reduce_func, reduce_target, reduce_target_buffers, op_extra_params)
                - reduce_func: input_load (BufferLoad)，统一类型
                - op_extra_params: {"num_topk": 8, "axis": -1}
        """
        num_topk = int(call.args[1])
        axis = int(call.args[2])
        input_load = call.args[3]
        output_loads = list(call.args[4:-1])  # [values, indices]

        reduce_op = "topk"
        # 直接取了x0做reduce_func,在decompose中暂时没影响
        reduce_func = input_load
        # reduce_target取第一个输出(values),暂时没影响
        reduce_target = output_loads[0]
        # 所有输出buffer(用于变量收集时识别y)
        reduce_target_buffers = [load.buffer for load in output_loads]
        # 额外参数
        op_extra_params = {"num_topk": num_topk, "axis": axis}

        return reduce_op, reduce_func, reduce_target, reduce_target_buffers, op_extra_params


    def _analyze(self):
        for block_info in self.cascaded_group:
            # 得到block_info对应的TIR block
            block = self.sch.get(block_info.block_rv)

            if isinstance(block.body, tir.SeqStmt):
                # 如果block的body是SeqStmt且长度大于1的话,通常第一个Stmt是init,第二个是表达式
                stmts = block.body
                if len(stmts) >= 2:
                    stmt = stmts[-1]
                else:
                    stmt = stmts[0]
            else:
                stmt = block.body

            # stmt -> BufferStore or Evaluate(Call)
            if isinstance(stmt, tir.BufferStore):
                reduce_op, reduce_func, reduce_target = self._analyze_buffer_store(stmt)
                reduce_target_buffers = [reduce_target.buffer]
                op_extra_params = None
            elif isinstance(stmt, tir.Evaluate):
                reduce_op, reduce_func, reduce_target, reduce_target_buffers, op_extra_params = self._analyze_evaluate(stmt, block)
            else:
                raise NotImplementedError(f"Unsupported stmt type: {type(stmt)}")

            self.reduce_funcs.append(reduce_func)

            # 开始构造var_map
            # 收集该block中的所有规约轴,这个数组的长度有可能大于1吗?
            reduction_vars = [loop_info.var for loop_info in block_info.iters 
                              if loop_info.kind == "R"]

            # 一个Block中拥有多个规约轴的情况暂时不考虑
            assert len(reduction_vars) == 1

            # 收集表达式中所有的BufferLoad和常量变量
            # 对于topk,不收集IntImm/FloatImm(workaround了这里)
            stmt_buffer_loads = []
 
            def _collect_loads(e):
                if reduce_op == "topk":
                    # 偷懒
                    if isinstance(e, tir.BufferLoad):
                        stmt_buffer_loads.append(e)
                else:
                    if isinstance(e, (tir.BufferLoad, tir.FloatImm, tir.IntImm)):
                        stmt_buffer_loads.append(e)

            tir.stmt_functor.post_order_visit(stmt, _collect_loads)

            def _depends_on_reduction(indices, vars):
                for idx in indices:
                    if idx in vars:
                        return True
                return False

            # 为每个BufferLoad和常量变量创建符号变量
            for load in stmt_buffer_loads:
                load_str = str(load)

                if load_str in self.var_map:
                    continue

                # 常量可以直接赋值
                if isinstance(load, (tir.FloatImm, tir.IntImm)):
                    var_name = f"c{self.c_counter}"
                    self.c_map[var_name] = load_str
                    self.c_buffer_load_map[var_name] = load
                    self.c_counter += 1                
                else:
                    # BufferLoad分成两种情况:
                    # 1. 如果包含有规约轴,则表明是被规约的数据,属于输入
                    # 2. 如果不包含规约轴,则可能是被规约的数据(reduce target);还有一种可能是和本次规约操作无关的常量,如果该常量没有在先前的Block中被划为reduce target,则应该归属于常量
                    if _depends_on_reduction(load.indices, reduction_vars):
                        var_name = f"x{self.x_counter}"
                        self.x_map[var_name] = load_str
                        self.x_buffer_load_map[var_name] = load
                        self.x_counter += 1                    
                    elif any(load.buffer.same_as(buf) for buf in reduce_target_buffers):
                        # 检查是否是任意一个输出buffer
                        var_name = f"y{self.y_counter}"
                        self.y_map[var_name] = load_str
                        self.y_buffer_load_map[var_name] = load
                        self.y_counter += 1
                    else:
                        var_name = f"c{self.c_counter}"
                        self.c_map[var_name] = load_str
                        self.c_buffer_load_map[var_name] = load
                        self.c_counter += 1

                self.var_map[load_str] = var_name

            # 根据var_map,构造替换变量后的reduce_func字符串
            # 需要重构,这里是纯字符串的构造
            def _remove_cast_extract_mapped_func(reduce_func, is_gemm):
                def _extract(e, gemm_counter):
                    if isinstance(e, (tir.Add, tir.Sub, tir.Mul, tir.Div, tir.Max, tir.Min)):
                        if isinstance(e, tir.Add):
                            return f"({_extract(e.a, gemm_counter)}) + ({_extract(e.b, gemm_counter)})"
                        if isinstance(e, tir.Sub):
                            return f"({_extract(e.a, gemm_counter)}) - ({_extract(e.b, gemm_counter)})"
                        if isinstance(e, tir.Mul):
                            if gemm_counter:
                                gemm_counter -= 1
                                return f"({_extract(e.a, gemm_counter)}) @ ({_extract(e.b, gemm_counter)})"
                            else:
                                return f"({_extract(e.a, gemm_counter)}) * ({_extract(e.b, gemm_counter)})"
                        if isinstance(e, tir.Div):
                            return f"({_extract(e.a, gemm_counter)}) / ({_extract(e.b, gemm_counter)})"
                        if isinstance(e, tir.Max):
                            return f"max(({_extract(e.a, gemm_counter)}), ({_extract(e.b, gemm_counter)}))"
                        if isinstance(e, tir.Min):
                            return f"min(({_extract(e.a, gemm_counter)}), ({_extract(e.b, gemm_counter)}))"
                    elif isinstance(e, (tir.Call)):
                        return f"{e.op.name.removeprefix('tir.')}({', '.join([(_extract(arg, gemm_counter)) for arg in e.args])})"
                    elif isinstance(e, tir.Cast):
                        return f"{(_extract(e.value, gemm_counter))}"
                    elif isinstance(e, (tir.BufferLoad, tir.FloatImm, tir.IntImm)):
                        return f"{self.var_map[str(e)]}"
                    else:
                        print("extract mapped func: should be here?")

                return _extract(reduce_func, 1 if is_gemm else 0)

            # 对于topk来说,因为取了x0作为reduce_func,所以这里不会出错了
            mapped_func = _remove_cast_extract_mapped_func(reduce_func, block_info.is_gemm())

            # 获取 reduce_target 的变量名
            reduce_target_name = self.var_map[str(reduce_target)]

            # 存储额外参数(如果有)
            if op_extra_params is not None:
                self.op_params[reduce_target_name] = op_extra_params

            config = ReductionConfig(
                node=None,
                is_reduction=block_info.is_reduction(),
                reduce_target=reduce_target_name,
                iter=reduction_vars[0].name.removeprefix('v_'),
                reduce_op=reduce_op,
                reduce_func=mapped_func,
                x_map=self.x_map,
                y_map=self.y_map,
                c_map=self.c_map
            )

            self.reduction_configs.append(config)


def analyze_cascaded_group(cascaded_group: List[BlockInfo], sch: tir.Schedule) -> CascadedGroupInfo:
    return CascadedGroupInfo(cascaded_group, sch)
