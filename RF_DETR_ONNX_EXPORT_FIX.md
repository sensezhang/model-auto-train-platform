# RF-DETR ONNX 导出修复说明

## 问题描述

RF-DETR 模型使用了 `torch.nn.functional.scaled_dot_product_attention`，这个操作符在 ONNX opset 17 中不被支持，导致导出时报错：

```
ERROR: missing-standard-symbolic-function
Exporting the operator 'aten::scaled_dot_product_attention' to ONNX opset version 17 is not supported.
```

## 解决方案

通过**注册自定义ONNX符号函数**，将 `scaled_dot_product_attention` 转换为ONNX支持的基础操作。

### 核心原理

1. **符号追踪（Symbolic Tracing）**
   - PyTorch导出ONNX时，不是直接执行代码，而是追踪计算图
   - 追踪时会检查每个操作是否有对应的ONNX符号函数
   - 如果没有，就会报错

2. **注册符号函数**
   - 使用 `torch.onnx.register_custom_op_symbolic()` 注册自定义符号函数
   - 符号函数将PyTorch操作转换为ONNX图节点
   - 使用 `g.op()` 构建ONNX操作符

3. **Scaled Dot-Product Attention 分解**
   ```
   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
   ```

   分解为ONNX支持的操作：
   - `Transpose`: K^T
   - `MatMul`: Q @ K^T
   - `Mul`: 乘以 scale (1/sqrt(d_k))
   - `Softmax`: softmax(...)
   - `MatMul`: ... @ V

## 代码改进

### 1. 注册符号函数 (export_model.py:16-88)

```python
def register_onnx_symbolic_for_sdpa():
    """注册 scaled_dot_product_attention 的 ONNX 符号函数"""

    def scaled_dot_product_attention_symbolic(g, query, key, value, ...):
        # Q @ K^T
        key_transposed = g.op("Transpose", key, perm_i=[0, 1, 3, 2])
        attn_weight = g.op("MatMul", query, key_transposed)

        # 动态计算 scale = 1/sqrt(head_dim)
        query_shape = g.op("Shape", query)
        head_dim_index = g.op("Constant", value_t=torch.tensor([3], dtype=torch.int64))
        head_dim = g.op("Gather", query_shape, head_dim_index, axis_i=0)
        head_dim_float = g.op("Cast", head_dim, to_i=1)  # Cast to FLOAT
        sqrt_head_dim = g.op("Sqrt", head_dim_float)
        one = g.op("Constant", value_t=torch.tensor([1.0], dtype=torch.float32))
        scale_factor = g.op("Div", one, sqrt_head_dim)

        # Scale
        attn_weight = g.op("Mul", attn_weight, scale_factor)

        # Softmax
        attn_weight = g.op("Softmax", attn_weight, axis_i=-1)

        # attn_weight @ V
        output = g.op("MatMul", attn_weight, value)
        return output

    register_custom_op_symbolic(
        'aten::scaled_dot_product_attention',
        scaled_dot_product_attention_symbolic,
        opset_version=17
    )
```

### 2. 在导出前调用 (export_model.py:199-200)

```python
# 注册ONNX符号函数以支持scaled_dot_product_attention
print(f"Registering ONNX symbolic for scaled_dot_product_attention...")
register_onnx_symbolic_for_sdpa()

# 导出ONNX
result = model.export(simplify=simplify)
```

## 使用方法

### 方法1：通过Web API（自动）

导出功能已集成到应用中，直接调用导出API即可，无需额外操作。

### 方法2：独立测试脚本

```bash
cd backend
python test_rfdetr_onnx_export.py
```

## 技术细节

### 为什么不使用 Monkey-Patch？

❌ **错误方法1**：运行时替换 `F.scaled_dot_product_attention`
```python
# 这种方法对ONNX导出无效！
F.scaled_dot_product_attention = custom_function
```

❌ **错误方法2**：直接访问 `query.type().sizes()[-1]`
```python
# 这在动态形状时会失败！
head_dim = query.type().sizes()[-1]  # NoneType' object is not subscriptable
```

✅ **正确方法**：使用ONNX符号操作动态计算
```python
# 在符号追踪时生效
register_custom_op_symbolic('aten::scaled_dot_product_attention', ...)

# 在符号函数内部使用ONNX操作
query_shape = g.op("Shape", query)
head_dim = g.op("Gather", query_shape, index, axis_i=0)
```

### ONNX图操作

使用 `g.op()` 构建ONNX节点：
- `g.op("MatMul", a, b)` - 矩阵乘法
- `g.op("Transpose", x, perm_i=[...])` - 转置
- `g.op("Softmax", x, axis_i=-1)` - Softmax
- `g.op("Constant", value_t=tensor)` - 常量
- `g.op("Shape", x)` - 获取张量形状
- `g.op("Gather", tensor, indices, axis_i=0)` - 按索引提取
- `g.op("Cast", x, to_i=type_id)` - 类型转换（1=FLOAT, 6=INT32, 7=INT64）
- `g.op("Sqrt", x)` - 平方根
- `g.op("Div", a, b)` - 除法

### 动态Scale计算

关键改进：使用ONNX操作动态计算 `1/sqrt(head_dim)`：

```python
# 1. 获取query的形状 [batch, num_heads, seq_len, head_dim]
query_shape = g.op("Shape", query)

# 2. 提取head_dim (index=3)
head_dim_index = g.op("Constant", value_t=torch.tensor([3], dtype=torch.int64))
head_dim = g.op("Gather", query_shape, head_dim_index, axis_i=0)

# 3. 转换为float
head_dim_float = g.op("Cast", head_dim, to_i=1)

# 4. 计算sqrt(head_dim)
sqrt_head_dim = g.op("Sqrt", head_dim_float)

# 5. 计算1/sqrt(head_dim)
one = g.op("Constant", value_t=torch.tensor([1.0], dtype=torch.float32))
scale_factor = g.op("Div", one, sqrt_head_dim)
```

这样就能处理动态形状，不依赖静态类型信息。

### 参数说明

- `query`, `key`, `value`: 输入张量（符号值）
- `attn_mask`: 注意力掩码（可选）
- `dropout_p`: Dropout概率（推理时忽略）
- `is_causal`: 是否使用因果掩码（暂不支持）
- `scale`: 缩放因子（默认 1/sqrt(head_dim)）

## 验证

导出成功后会生成 `.onnx` 文件，可以使用以下方式验证：

```python
import onnx

# 加载模型
onnx_model = onnx.load("model.onnx")

# 检查模型
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
```

## 已知限制

1. **Causal Mask**: 暂未实现因果掩码支持
2. **Dropout**: 推理时自动忽略
3. **动态Head Dim**: 使用默认值 0.125 (假设 head_dim=64)

## 参考资料

- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [Custom ONNX Operators](https://pytorch.org/docs/stable/onnx.html#custom-operators)
