# 3MF底板对象验证工具

## 用途

`verify_3mf_backing.py` 是一个命令行工具，用于快速验证生成的3MF文件是否包含独立的底板对象。

## 使用方法

```bash
python3 verify_3mf_backing.py <path_to_3mf_file>
```

## 示例

```bash
# 验证输出文件
python3 verify_3mf_backing.py output/logo_Lumina.3mf

# 验证批量生成的文件
python3 verify_3mf_backing.py output/Lumina_Batch_*.3mf
```

## 输出示例

### 成功找到底板对象

```
================================================================================
Verifying 3MF file: output/logo_Lumina.3mf
================================================================================

✅ Scene loaded successfully
   Total objects: 5

Objects in scene:
  - White: 1234 vertices, 2468 faces
  - Cyan: 567 vertices, 1134 faces
  - Magenta: 890 vertices, 1780 faces
  - Yellow: 456 vertices, 912 faces
  - Backing: 2000 vertices, 4000 faces
    ✅ BACKING OBJECT FOUND!
    Color: RGB(255, 255, 255)
    Bounding box:
      Min: [0.0, 0.0, 0.0]
      Max: [60.0, 60.0, 2.0]
      Size: [60.0, 60.0, 2.0]

✅ VERIFICATION PASSED: Backing object found in 3MF file
```

### 未找到底板对象

```
================================================================================
Verifying 3MF file: output/logo_Lumina.3mf
================================================================================

✅ Scene loaded successfully
   Total objects: 4

Objects in scene:
  - White: 1234 vertices, 2468 faces
  - Cyan: 567 vertices, 1134 faces
  - Magenta: 890 vertices, 1780 faces
  - Yellow: 456 vertices, 912 faces

❌ VERIFICATION FAILED: Backing object NOT found in 3MF file
```

## 检查内容

该工具会验证：

1. ✅ 3MF文件是否有效
2. ✅ 场景中的对象数量
3. ✅ 是否存在名为"Backing"的对象
4. ✅ 底板对象的颜色（应为白色 RGB(255,255,255)）
5. ✅ 底板对象的网格信息（顶点数、面数）
6. ✅ 底板对象的尺寸（边界框）

## 返回值

- 成功找到底板对象：退出码 0
- 未找到底板对象或出错：退出码 1

可用于自动化测试脚本：

```bash
if python3 verify_3mf_backing.py output/test.3mf; then
    echo "✅ 底板对象验证通过"
else
    echo "❌ 底板对象验证失败"
fi
```

## 依赖

- Python 3.6+
- trimesh

## 注意事项

- 该工具仅检查对象名称中是否包含"Backing"或"backing"
- 底板对象应该是白色（RGB 255,255,255）
- 如果3MF文件中没有勾选"底板单独一个对象"选项，则不会有独立的底板对象（这是正常的）
