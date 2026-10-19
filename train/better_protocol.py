"""Fixed training-only curriculum and shape variants; frozen evaluation remains original-sized."""
import ast
from pathlib import Path

ACTIVATIONS = {
 '19_ReLU': 'tl.maximum(x, 0.)',
 '20_LeakyReLU': 'tl.where(x >= 0, x, P*x)',
 '21_Sigmoid': '1. / (1. + tl.exp(-x))',
 '22_Tanh': '2. / (1. + tl.exp(-2.*x)) - 1.',
 '26_GELU_': '.5*x*(1. + tl.erf(x*0.7071067811865476))',
 '27_SELU_': '1.0507009873554805*tl.where(x>0,x,1.6732632423543772*(tl.exp(x)-1.))',
 '29_Softplus': 'tl.where(x>20.,x,tl.log(1.+tl.exp(x)))',
 '30_Softsign': 'x / (1. + tl.abs(x))',
 '31_ELU': 'tl.where(x>0,x,P*(tl.exp(x)-1.))',
}
CURRICULUM = ['level1/'+name for name in [*ACTIVATIONS,
 '47_Sum_reduction_over_a_dimension','48_Mean_reduction_over_a_dimension',
 '94_MSELoss','96_HuberLoss','1_Square_matrix_multiplication_']]

# Level-2 fusion tasks, original shapes, hybrid contract. Fixed by rule before any measurement: train split,
# head Gemm/Matmul/Conv2d, every later op elementwise (no reductions, norms, pooling, softmax, min/max, dropout).
FUSION_CURRICULUM = ['level2/'+name for name in [
 '12_Gemm_Multiply_LeakyReLU','1_Conv2D_ReLU_BiasAdd','29_Matmul_Mish_Mish','40_Matmul_Scaling_ResidualAdd',
 '53_Gemm_Scaling_Hardtanh_GELU','54_Conv2d_Multiply_LeakyReLU_GELU','57_Conv2d_ReLU_HardSwish',
 '59_Matmul_Swish_Scaling','63_Gemm_ReLU_Divide','69_Conv2d_HardSwish_ReLU','70_Gemm_Sigmoid_Scaling_ResidualAdd',
 '71_Conv2d_Divide_LeakyReLU','76_Gemm_Add_ReLU','81_Gemm_Swish_Divide_Clamp_Tanh_Clamp','86_Matmul_Divide_GELU',
 '87_Conv2d_Subtract_Subtract_Mish','95_Matmul_Add_Swish_Tanh_GELU_Hardtanh','9_Matmul_Subtract_Multiply_ReLU']]
FUSION_ELEMENTWISE = {'multiply','leakyrelu','divide','scaling','scale','relu','add','swish','sigmoid','tanh','gelu',
 'mish','hardtanh','clamp','subtract','residualadd','biasadd','hardswish'}


def fusion_selection(tasks):
    selected=[]
    for task_id,task in sorted(tasks.items()):
        if task.level!=2 or task.split!='train':continue
        head,*chain=task_id.split('/')[1].split('_',1)[1].split('_')
        if head.lower() in ('gemm','matmul','conv2d') and chain and all(c.lower() in FUSION_ELEMENTWISE for c in chain):
            selected.append(task_id)
    return selected


# Freeze these before GPU calibration; evaluation always keeps the original task source.
TRAINING_VARIANT = 0
TRAINING_SHAPES = {
    'activations': [{'batch_size':32,'dim':262144}, {'batch_size':64,'dim':262144}],
    'reductions': [{'batch_size':32,'dim1':128,'dim2':2048}, {'batch_size':64,'dim1':128,'dim2':2048}],
    'losses': [{'batch_size':32,'input_shape':(131072,)}, {'batch_size':64,'input_shape':(131072,)}],
    'matmul': [{'N':1024}, {'N':1536}],
}


def variant_source(task, variant):
    if task.task_id not in CURRICULUM or task.split != 'train':
        raise ValueError('shape changes are restricted to preregistered training tasks')
    if variant not in (0,1):raise ValueError('unknown shape variant')
    name=task.task_id.split('/')[1]
    family=('activations' if name in ACTIVATIONS else 'reductions' if name.startswith(('47_','48_'))
            else 'losses' if name.startswith(('94_','96_')) else 'matmul')
    overrides=TRAINING_SHAPES[family][variant]
    tree=ast.parse(task.src)
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            key=node.targets[0].id
            if key in overrides:node.value=ast.parse(repr(overrides[key]),mode='eval').body
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def training_copy(tasks, destination, variant=1):
    root=Path(destination);root.mkdir(parents=True,exist_ok=False)
    for task in tasks.values():
        path=root/(task.task_id+'.py');path.parent.mkdir(exist_ok=True)
        path.write_text(variant_source(task,variant) if task.task_id in CURRICULUM else task.src)
    return root


HEADER='import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n'


def model_source(task, forward):
    tree=ast.parse(task.src)
    model=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Model')
    init=next(node for node in model.body if isinstance(node,ast.FunctionDef) and node.name=='__init__')
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='Model':node.id='ModelNew'
            return node
    init=Rename().visit(init)
    init.body=[n for n in init.body if not isinstance(n,ast.Expr) or not isinstance(n.value,ast.Constant)]
    body='\n'.join('    '+line for line in ast.unparse(init).splitlines())
    return '\nclass ModelNew(nn.Module):\n'+body+'\n\n'+forward


def teacher_source(task):
    name=task.task_id.split('/')[1]
    if name in ACTIVATIONS:
        kernel='''
@triton.jit
def activation_kernel(X,Y,N,P:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,other=0.)
    y=EXPRESSION
    tl.store(Y+i,y,i<N)
'''.replace('EXPRESSION',ACTIVATIONS[name])
        param='self.negative_slope' if name=='20_LeakyReLU' else 'self.alpha' if name=='31_ELU' else '1.'
        forward=f'''    def forward(self,x):
        x=x.contiguous()
        y=torch.empty_like(x)
        activation_kernel[(triton.cdiv(x.numel(),1024),)](x,y,x.numel(),{param},1024)
        return y
'''
    elif name.startswith(('47_','48_')):
        mean=name.startswith('48_')
        kernel='''
@triton.jit
def reduction_kernel(X,Y,R:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
    batch=tl.program_id(0)
    c=tl.program_id(1)*32+tl.arange(0,32)
    r=tl.arange(0,B)
    x=tl.load(X+batch*R*C+r[:,None]*C+c[None,:],(r[:,None]<R)&(c[None,:]<C),other=0.)
    y=tl.sum(x,axis=0)DIVISOR
    tl.store(Y+batch*C+c,y,c<C)
'''.replace('DIVISOR',' / R' if mean else '')
        shape='(x.shape[0],x.shape[2])' if mean else '(x.shape[0],1,x.shape[2])'
        forward=f'''    def forward(self,x):
        assert self.dim==1
        x=x.contiguous()
        y=torch.empty({shape},device=x.device,dtype=x.dtype)
        reduction_kernel[(x.shape[0],triton.cdiv(x.shape[2],32))](x,y,x.shape[1],x.shape[2],triton.next_power_of_2(x.shape[1]))
        return y
'''
    elif name.startswith(('94_','96_')):
        loss='d*d' if name.startswith('94_') else 'tl.where(tl.abs(d)<1.,.5*d*d,tl.abs(d)-.5)'
        kernel='''
@triton.jit
def loss_partial_kernel(X,T,P,N,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    d=tl.load(X+i,i<N,other=0.)-tl.load(T+i,i<N,other=0.)
    value=EXPRESSION
    value=tl.where(i<N,value,0.)
    tl.store(P+tl.program_id(0),tl.sum(value,axis=0))
@triton.jit
def loss_final_kernel(P,Y,M:tl.constexpr,N,B:tl.constexpr):
    i=tl.arange(0,B)
    value=tl.load(P+i,i<M,other=0.)
    tl.store(Y,tl.sum(value,axis=0)/N)
'''.replace('EXPRESSION',loss)
        forward='''    def forward(self,predictions,targets):
        x=predictions.contiguous()
        t=targets.contiguous()
        n=x.numel()
        m=triton.cdiv(n,1024)
        partial=torch.empty((m,),device=x.device,dtype=torch.float32)
        y=torch.empty((),device=x.device,dtype=x.dtype)
        loss_partial_kernel[(m,)](x,t,partial,n,1024)
        loss_final_kernel[(1,)](partial,y,m,n,triton.next_power_of_2(m))
        return y
'''
    elif name.startswith('1_'):
        # Existing handwritten test kernel, verified again on both training shapes before SFT.
        from kernel_env.tests.kernels import MATMUL
        return MATMUL
    else:raise ValueError('no preregistered teacher for '+task.task_id)
    return HEADER+kernel+model_source(task,forward)
