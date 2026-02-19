"""
SFBT策略生成器

负责调用模型进行SFBT策略推理，生成下一轮的干预指令包。

作者：Claude
日期：2025-01-12
版本：v1.2（支持可配置API）
"""

import json
import yaml
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from openai import OpenAI

# 导入配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import APIConfig


@dataclass
class StrategyPackage:
    """SFBT策略包"""
    selected_method: str      # 选中的干预方法
    strategic_intend: str     # 战略意图（包含当前子模块）
    current_module: str       # 当前子模块
    todo: List[str]          # 必须做的事
    notdo: List[str]         # 禁止的事


class SFBTStrategyGenerator:
    """SFBT策略生成器"""

    # 可用方法白名单
    AVAILABLE_METHODS = [
        "例外情境",
        "奇迹问题",
        "应对问题",
        "关系问题",
        "量尺问题",
        "赞美"
    ]

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        初始化策略生成器

        Args:
            api_key: API密钥（默认使用config.BRAIN_API_KEY）
            base_url: API地址（默认使用config.BRAIN_API_BASE_URL）
            model: 模型名称（默认使用config.BRAIN_API_MODEL）
        """
        # 使用config中的默认值，如果参数未提供
        api_key = api_key or APIConfig.BRAIN_API_KEY
        base_url = base_url or APIConfig.BRAIN_API_BASE_URL
        model = model or APIConfig.BRAIN_API_MODEL

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.model = model

        # 加载Prompt模板
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> dict:
        """加载decider_prompt.yaml"""
        prompt_path = Path(__file__).parent / "decider_prompt.yaml"

        if not prompt_path.exists():
            print(f"[错误] decider_prompt.yaml不存在: {prompt_path}")
            raise FileNotFoundError(f"找不到decider_prompt.yaml: {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # 检查是否为空
        if not data or data is None:
            print("[错误] decider_prompt.yaml内容为空，请先填写Prompt内容")
            raise ValueError("decider_prompt.yaml内容为空，无法加载")

        return data

    def generate_strategy(
        self,
        user_msg: str,
        full_history: List[Dict],
        previous_method: str,
        sfbt_turn: int,
        remaining_minutes: int
    ) -> Tuple[StrategyPackage, str]:
        """
        生成SFBT策略包

        Args:
            user_msg: 用户当前回复
            full_history: 完整10轮对话历史（每轮包含role和content）
            previous_method: 上一轮使用的干预方法
            sfbt_turn: SFBT阶段内的第几轮
            remaining_minutes: 剩余时间（分钟）

        Returns:
            (StrategyPackage, reasoning): 策略包和完整推理过程
        """
        # 1. 构建输入数据
        t_input_start = time.time()
        input_data = self._format_input(
            user_msg, full_history, previous_method,
            sfbt_turn, remaining_minutes
        )
        t_input = time.time() - t_input_start

        # 2. 构建完整Prompt
        t_build_prompt_start = time.time()
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(input_data)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        t_build_prompt = time.time() - t_build_prompt_start

        t_api_start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,
                stream=True
            )

            # 4. 处理流式响应，收集全部内容
            full_content = ""
            reasoning_content = ""

            t_stream_start = time.time()
            for chunk in response:
                try:
                    # 调试：打印chunk结构
                    # print(f"\n[DEBUG] Chunk: {chunk}")

                    if len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content is not None:
                            full_content += delta.content
                            # 实时打印进度
                            print(".", end="", flush=True)
                        elif hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                            # GPT-4o-mini可能返回reasoning_content
                            reasoning_content += delta.reasoning_content
                    else:
                        # chunk.choices为空，可能已经结束
                        pass
                except Exception as e:
                    print(f"\n⚠️ Chunk解析错误: {e}")
                    print(f"Chunk类型: {type(chunk)}")
                    print(f"Chunk内容: {chunk}")
                    continue

            t_stream = time.time() - t_stream_start
            print()  # 换行
            t_api_total = time.time() - t_api_start

            # 5. 分离决策过程(CoT)和JSON
            t_parse_start = time.time()
            reasoning_content, json_str = self._split_cot_and_json(full_content)
            t_parse = time.time() - t_parse_start

            # 6. 解析JSON
            try:
                package_dict = json.loads(json_str)

                # 7. 验证方法名
                method = package_dict.get("selected_method", "")
                package_dict["selected_method"] = self._normalize_method(method)

                # 8. 构建策略包对象
                package = StrategyPackage(
                    selected_method=package_dict["selected_method"],
                    strategic_intend=package_dict["strategic_intend"],
                    current_module=package_dict["current_module"],
                    todo=package_dict["todo"],
                    notdo=package_dict["notdo"]
                )

                # 9. 输出时间分解
                print(f"\n⏱️ 时间分解:")
                print(f"  - 输入格式化: {t_input:.3f}秒")
                print(f"  - Prompt构建: {t_build_prompt:.3f}秒")
                print(f"  - API调用总计: {t_api_total:.3f}秒")
                print(f"    - 流式接收: {t_stream:.3f}秒")
                print(f"    - API往返延迟: {t_api_total - t_stream:.3f}秒")
                print(f"  - 解析分离: {t_parse:.3f}秒")
                print(f"  - 总耗时: {t_input + t_build_prompt + t_api_total + t_parse:.3f}秒")

                # 10. 返回策略包和推理过程
                # reasoning_content 是分离出的决策过程(CoT)
                return package, reasoning_content

            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
                print(f"JSON字符串: {json_str[:500]}...")
                # 返回默认策略包（解包后）
                fallback_package, fallback_reasoning = self._get_fallback_package()
                return fallback_package, fallback_reasoning

        except Exception as e:
            print(f"❌ API调用失败: {e}")
            # 返回默认策略包（解包后）
            fallback_package, fallback_reasoning = self._get_fallback_package()
            return fallback_package, fallback_reasoning

    def _format_input(
        self,
        user_msg: str,
        full_history: List[Dict],
        previous_method: str,
        sfbt_turn: int,
        remaining_minutes: int
    ) -> str:
        """格式化输入数据"""

        # 格式化历史对话
        history_text = self._format_history(full_history)

        input_data = f"""
## 当前输入数据

用户回复：
{user_msg}

上一轮方法：{previous_method}

SFBT阶段：第{sfbt_turn}轮
剩余时间：{remaining_minutes}分钟

完整对话历史（10轮）：
{history_text}
"""
        return input_data

    def _format_history(self, history: List[Dict]) -> str:
        """格式化历史对话"""
        lines = []
        for msg in history[-10:]:  # 确保只取10轮
            role = msg.get("role", "")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        """构建系统Prompt"""
        template = self.prompt_template.get("system_role", "")
        return template.strip()

    def _build_user_prompt(self, input_data: str) -> str:
        """构建用户Prompt"""
        template = self.prompt_template.get("feature_extraction", "")
        modules = self.prompt_template.get("module_definitions", "")
        methods = self.prompt_template.get("intervention_methods", "")
        chains = self.prompt_template.get("chain_rules", "")
        priority = self.prompt_template.get("priority_system", "")
        output = self.prompt_template.get("output_format", "")

        prompt = f"{template}\n\n{modules}\n\n{methods}\n\n{chains}\n\n{priority}\n\n{output}\n\n{input_data}"
        return prompt

    def _normalize_method(self, raw_method: str) -> str:
        """
        规范化方法名（白名单+模糊匹配）

        Args:
            raw_method: LLM输出的方法名

        Returns:
            规范化后的方法名
        """
        # 1. 完全匹配
        if raw_method in self.AVAILABLE_METHODS:
            return raw_method

        # 2. 模糊匹配（包含关键词）
        method_keywords = {
            "例外情境": ["例外", "寻找例外", "例外情境"],
            "奇迹问题": ["奇迹", "奇迹问题", "奇迹"],
            "应对问题": ["应对", "应对问题", "肯定", "支持"],
            "关系问题": ["关系", "他人", "重要他人"],
            "量尺问题": ["量尺", "分数", "打分", "量化"],
            "赞美": ["赞美", "肯定", "鼓励"]
        }

        for standard_name, keywords in method_keywords.items():
            if any(kw in raw_method for kw in keywords):
                print(f"⚠️ 方法名模糊匹配: '{raw_method}' -> '{standard_name}'")
                return standard_name

        # 3. 无法匹配，返回默认
        print(f"❌ 无法识别方法: '{raw_method}'，使用默认方法'应对问题'")
        return "应对问题"

    def _split_cot_and_json(self, full_content: str) -> tuple[str, str]:
        """
        分离决策过程(CoT)和JSON

        Args:
            full_content: 完整输出（决策过程 + JSON）

        Returns:
            (reasoning_content, json_string)
        """
        # 查找JSON代码块标记
        json_start = full_content.find("```json")

        if json_start == -1:
            # 没有找到```json，尝试查找单独的```标记
            json_start = full_content.find("```")

        if json_start == -1:
            # 都找不到，假设全部是JSON
            print("⚠️ 未检测到JSON代码块标记，尝试解析全部内容")
            return "", full_content.strip()

        # 分离CoT和JSON
        reasoning_content = full_content[:json_start].strip()
        json_part = full_content[json_start:]

        # 提取JSON字符串（移除```json和```标记）
        lines = json_part.split('\n')
        json_lines = []
        in_json = False

        for line in lines:
            if line.strip().startswith("```"):
                in_json = not in_json
                continue
            if in_json or line.strip().startswith('{'):
                json_lines.append(line)

        json_str = '\n'.join(json_lines).strip()

        return reasoning_content, json_str

    def _get_fallback_package(self) -> Tuple[StrategyPackage, str]:
        """获取默认策略包（兜底）"""
        package = StrategyPackage(
            selected_method="应对问题",
            strategic_intend="提供支持和陪伴，当前处于S1_合作构建阶段",
            current_module="S1_合作构建",
            todo=["保持温和支持的态度", "跟随用户的流向"],
            notdo=["不要急于推动改变", "不要提供建议"]
        )
        reasoning = "使用兜底策略包"
        return package, reasoning

    def save_test_result(self, test_name: str, package: StrategyPackage, reasoning: str):
        """保存测试结果到文件"""
        result = {
            "test_name": test_name,
            "package": {
                "selected_method": package.selected_method,
                "strategic_intend": package.strategic_intend,
                "current_module": package.current_module,
                "todo": package.todo,
                "notdo": package.notdo
            },
            "reasoning": reasoning
        }

        output_path = Path(__file__).parent / "test_results.json"

        # 追加模式写入
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except:
                    data = []
                data.append(result)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([result], f, ensure_ascii=False, indent=2)

        print(f"✅ 测试结果已保存到: {output_path}")


# =============================================================================
# 测试代码
# =============================================================================
def test_strategy_generator():
    """测试策略生成器"""

    # 初始化（使用config中的配置）
    generator = SFBTStrategyGenerator()

    # 加载测试数据
    test_data_path = Path(__file__).parent / "test_data.yaml"
    if not test_data_path.exists():
        print(f"❌ 测试数据文件不存在: {test_data_path}")
        return

    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_cases = yaml.safe_load(f)

    # 运行测试
    for case in test_cases.get("test_cases", []):
        print(f"\n{'='*60}")
        print(f"测试案例: {case.get('name', 'Unknown')}")
        print(f"{'='*60}")

        # 构建输入
        full_history = case.get("context", "").split('\n')
        history_list = []
        for line in full_history:
            if ': ' in line:
                role, content = line.split(':', 1)
                history_list.append({"role": role.strip(), "content": content.strip()})

        # 记录开始时间
        start_time = time.time()

        # 生成策略
        package, reasoning = generator.generate_strategy(
            user_msg=case.get("user_msg", ""),
            full_history=history_list,
            previous_method=case.get("current_package", {}).get("selected_method", ""),
            sfbt_turn=case.get("sfbt_turn", 1),
            remaining_minutes=case.get("remaining_minutes", 10)
        )

        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 输出结果
        print(f"\n✅ 生成结果 (总耗时: {elapsed_time:.2f}秒):")
        print(f"  方法: {package.selected_method}")
        print(f"  意图: {package.strategic_intend}")
        print(f"  模块: {package.current_module}")
        print(f"  Todo: {package.todo}")
        print(f"  Notdo: {package.notdo}")

        # 保存结果
        generator.save_test_result(case.get('name', 'unknown'), package, reasoning)

        print(f"\n推理过程前500字:")
        print(reasoning[:500] + "..." if len(reasoning) > 500 else reasoning)


if __name__ == "__main__":
    test_strategy_generator()
