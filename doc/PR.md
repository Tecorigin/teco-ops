## 算子提交规范

### Commit 消息格式

```
[type](algo<num>)：<subject>
```

- `type` 必须为以下之一：
  - `feat` — 新增算子或功能
  - `fix` — 修复 bug
  - `perf` — 性能优化
  - `refactor` — 代码重构
  - `ci` — 持续集成配置变更
  - `tool` — 工具脚本变更
  - `docs` — 文档变更
  - `test` — 测试用例变更
- `algo<num>`：算法编号，如 `algo0` 表示默认算法实现，`algo1` 表示优化算法实现等
- `subject`：简短描述本次提交的内容（英文）

示例：
```bash
git commit -m "[feat](algo0)：add sigmoid operator"
git commit -m "[perf](algo1)：optimize memory access pattern for flatten_rays"
git commit -m "[fix](algo0)：correct boundary check in flatten_rays"
```

### PR 规范

- **1 PR = 1 个算子**：每个 PR 对应一个完整的算子实现
- 每个算子开发使用独立的 git 分支，分支名建议使用 `op/<op_name>` 格式
- PR 描述需包含：算子功能说明、接口设计、性能数据（如有）
- PR 提交后，后续 commit 会自动同步到已有 PR 中，无需新建 PR

**PR 自查清单：**
- [ ] 所有新增文件已添加 BSD License 头部
- [ ] Commit 消息符合规范
- [ ] 新功能不破坏原有功能
- [ ] SPM 内存申请未超过 235KB
- [ ] 测例精度和性能测试通过
- [ ] 已更新或新增算子设计文档（`doc/op_docs/`）

### 代码风格

- 遵循 [Google C++ Style Guide](https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/contents.html)
- 所有新增文件需添加 BSD License 头部
- SPM 内存申请上限 235KB

### 代码格式化

贡献者可以使用 `tools/format2google` 脚本将代码规范化为 Google C++ 风格：

```bash
# 格式化单个文件
./tools/format2google path/to/file.cpp

# 格式化整个目录
./tools/format2google path/to/directory
```

### 代码风格检查

项目使用 cpplint 进行代码风格检查。在 `source env.sh` 后，git hooks 会自动安装，提交时会自动检查暂存区的 C++ 文件：

```bash
# 安装依赖
pip install cpplint

# 首次使用需要 source env.sh 来安装 git hooks
source env.sh

# 提交时自动检查
git add <files>
git commit -m "message"

# 跳过检查（不推荐）
git commit -n -m "message"
```

检查配置文件位于 `tools/CPPLINT.cfg`。