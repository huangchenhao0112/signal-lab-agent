# SignalLab Agent

SignalLab Agent 是一个面向通信信号处理与频谱分析场景的
大模型智能助手。

## 当前版本

v0.1.0

当前已经完成：

- FastAPI基础后端
- 环境配置管理
- 大模型统一调用客户端
- 健康检查接口
- 基础对话接口
- Pydantic请求与响应校验
- 基础自动化测试

## 项目结构

- `app/main.py`：FastAPI入口
- `app/schemas.py`：接口数据结构
- `app/core/config.py`：环境配置
- `app/core/llm_client.py`：大模型调用
- `tests/`：自动化测试

## 本地运行

安装依赖：

```bash
python -m pip install -r requirements.txt