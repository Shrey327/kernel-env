"""AST checks that run before any candidate code executes.

Goal: reject submissions that compute with PyTorch instead of custom kernels, or that try to
tamper with the evaluator (imports, monkeypatching, introspection escapes).

Tensor arithmetic operators are checked at runtime in worker.kernel_only_mode, where tensors can
be distinguished from legitimate scalar/shape arithmetic.
"""

import ast
from typing import Dict, List, Optional, Tuple

ALLOWED_IMPORTS = {"torch", "triton", "math", "typing", "functools"}
BANNED_CALLS = {
    "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "open", "breakpoint", "input", "memoryview",
}
# torch.* calls allowed in host code (allocation only)
TORCH_ALLOWED = {
    f"torch.{n}"
    for n in (
        "empty", "empty_like", "empty_strided", "zeros", "zeros_like", "ones", "ones_like",
        "full", "full_like", "arange", "Size", "device", "is_tensor",
        # random generation: harmless, and models often copy the reference's get_inputs()
        "rand", "randn", "randint", "rand_like", "randn_like", "manual_seed",
    )
}
# tensor methods that do real compute
TENSOR_COMPUTE = {
    "matmul", "mm", "bmm", "addmm", "baddbmm", "einsum", "sum", "mean", "max", "min", "amax",
    "amin", "prod", "cumsum", "cumprod", "softmax", "log_softmax", "relu", "relu_", "sigmoid",
    "tanh", "exp", "log", "sqrt", "rsqrt", "pow", "abs", "clamp", "clamp_", "norm", "var", "std",
    "argmax", "argmin", "topk", "sort", "mul", "mul_", "add", "add_", "sub", "sub_", "div",
    "div_", "neg", "gelu", "silu", "conv1d", "conv2d", "conv3d", "matrix_power", "tril", "triu",
    "masked_fill", "where", "gather", "scatter", "index_select", "logsumexp", "outer", "dot",
}
KERNEL_DECORATORS = {"triton.jit", "triton.autotune", "triton.heuristics"}


def _path(node: ast.AST, aliases: Dict[str, str]) -> Tuple[Optional[str], bool]:
    """Dotted path of a Name/Attribute chain with import aliases expanded; bool = root is an import."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None, False
    root = aliases.get(node.id)
    return ".".join([root or node.id] + parts[::-1]), root is not None


def _is_self_attr(node: ast.AST) -> bool:
    """self.<name>: a user-chosen attribute name (e.g. self.conv3d), not a tensor method."""
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"


def _is_kernel(fn: ast.FunctionDef, aliases: Dict[str, str]) -> bool:
    for d in fn.decorator_list:
        p, _ = _path(d.func if isinstance(d, ast.Call) else d, aliases)
        if p in KERNEL_DECORATORS:
            return True
    return False


def check_source(code: str, language: str = "triton") -> List[str]:
    """Return a list of violations; empty list means the source may be executed."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} (line {e.lineno})"]

    errors: List[str] = []
    aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    errors.append(f"import of '{a.name}' not allowed")
                aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or mod.split(".")[0] not in ALLOWED_IMPORTS:
                errors.append(f"import from '{mod}' not allowed")
            for a in node.names:
                aliases[a.asname or a.name] = f"{mod}.{a.name}"
    # module-level `ext = load_inline(...)`: ext.relu(x) is a custom kernel, not a tensor op
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if _path(node.value.func, aliases)[0] == "torch.utils.cpp_extension.load_inline":
                aliases.update({t.id: f"cuda_ext.{t.id}" for t in node.targets if isinstance(t, ast.Name)})

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            errors.append(f"call to '{node.func.id}' not allowed (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__":
            errors.append(f"dunder attribute '{node.attr}' not allowed (line {node.lineno})")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Attribute):
                        root = sub
                        while isinstance(root, ast.Attribute):
                            root = root.value
                        if not (isinstance(root, ast.Name) and root.id == "self"):
                            errors.append(f"assigning attributes outside self not allowed (line {node.lineno})")

    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    if "ModelNew" not in classes:
        errors.append("no top-level class ModelNew")

    # self attributes holding nn layers (calling them = PyTorch compute)
    nn_attrs = set()
    for cls in classes.values():
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                for n in ast.walk(fn):
                    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                        p, _ = _path(n.value.func, aliases)
                        if p and p.startswith("torch.nn.") and p != "torch.nn.Parameter":
                            nn_attrs.update(
                                t.attr for t in n.targets
                                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"
                            )

    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    kernels = [f for f in funcs if _is_kernel(f, aliases)]
    kernel_ids = {id(n) for k in kernels for n in ast.walk(k)}
    init_ids = {id(n) for f in funcs if f.name == "__init__" for n in ast.walk(f)}
    call_func_ids = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}

    for n in ast.walk(tree):
        if id(n) in kernel_ids:
            continue
        # references that smuggle compute through an alias, e.g. `mm = torch.matmul; mm(a, b)`
        if isinstance(n, (ast.Name, ast.Attribute)) and id(n) not in call_func_ids:
            p, imp = _path(n, aliases)
            if p and imp and p.startswith("torch.") and (
                p.rsplit(".", 1)[-1] in TENSOR_COMPUTE
                or p.startswith(("torch.nn.functional.", "torch.ops", "torch.compile", "torch._"))
            ):
                errors.append(f"reference to PyTorch op '{p}' (line {n.lineno})")
            elif not imp and isinstance(n, ast.Attribute) and n.attr in TENSOR_COMPUTE and not _is_self_attr(n):
                errors.append(f"reference to tensor op '.{n.attr}' (line {n.lineno})")
        if id(n) in init_ids:
            continue
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
            errors.append(f"'@' matmul in host code (line {n.lineno})")
        if not isinstance(n, ast.Call):
            continue
        p, root_is_import = _path(n.func, aliases)
        if p is None:
            continue
        if p.startswith("torch.") and p not in TORCH_ALLOWED and not p.startswith(
            ("torch.cuda.", "torch.utils.cpp_extension.")
        ):
            errors.append(f"PyTorch op '{p}' in host code (line {n.lineno}); use a custom kernel")
        elif not root_is_import and isinstance(n.func, ast.Attribute):
            if _is_self_attr(n.func):
                if n.func.attr in nn_attrs:
                    errors.append(f"calling nn layer '{p}' in host code (line {n.lineno})")
            elif n.func.attr in TENSOR_COMPUTE:
                errors.append(f"tensor op '.{n.func.attr}()' in host code (line {n.lineno})")

    if language == "triton" and not kernels:
        errors.append("no @triton.jit kernel defined")
    if language == "cuda":
        loads = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and _path(n.func, aliases)[0] == "torch.utils.cpp_extension.load_inline"
        ]
        if not loads:
            errors.append("no torch.utils.cpp_extension.load_inline call")

    return list(dict.fromkeys(errors))  # dedupe, keep order
