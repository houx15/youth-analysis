"""
判断给定名字是否是娱乐圈名人或作品的脚本

使用GPT API判断wordlists/entertainment_nouns_2020.txt中的每个名字是否是：
- 娱乐圈名人（歌手、明星、演员、艺人）
- 作品（电影、电视剧）

返回1（是）或0（不是），结果输出为CSV文件。

使用方法:
    python check_entertainment_names.py
    python check_entertainment_names.py --input_file wordlists/entertainment_nouns_2020.txt --output_file results.csv
    python check_entertainment_names.py --delay 0.5 --model gpt-4
"""

import csv
import logging
import time
from pathlib import Path
from typing import Optional
import fire
from openai import OpenAI

from configs.configs import OPENAI_API_KEY, OPENAI_BASE_URL

# 默认配置
DEFAULT_INPUT_FILE = "wordlists/entertainment_nouns_2020.txt"
DEFAULT_OUTPUT_FILE = "entertainment_names_classification.csv"
DEFAULT_MODEL = "gpt-5.2"  # 可以根据需要修改
DEFAULT_DELAY = 0.1  # API调用间隔（秒），避免请求过快

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EntertainmentNameClassifier:
    """使用GPT API判断名字是否是娱乐圈名人或作品"""

    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        初始化分类器

        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL（可选，默认从configs读取）
            model: 使用的模型名称
        """
        if not api_key:
            raise ValueError(
                "请设置OPENAI_API_KEY环境变量，或在configs/configs.py中配置API密钥"
            )

        # if base_url is None:
        #     base_url = OPENAI_BASE_URL
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model
        logger.info(f"初始化分类器，使用模型: {model}, base_url: {base_url}")

    def _build_prompt(self, name: str) -> str:
        """
        构建判断提示词

        Args:
            name: 要判断的名字

        Returns:
            提示词字符串
        """
        prompt = f"""请判断以下名字是否是娱乐圈相关的内容：

名字：{name}

返回标准：
1: 娱乐圈名人：包括歌手、明星、演员、艺人等
2: 作品：包括电影、电视剧等
0: 既不是娱乐圈名人也不是作品

请只回答"1"、"2"或"0"，不要添加任何其他文字或解释。"""
        return prompt

    def classify(self, name: str) -> int:
        """
        判断一个名字是否是娱乐圈名人或作品

        Args:
            name: 要判断的名字

        Returns:
            1表示是，0表示不是
        """
        prompt = self._build_prompt(name)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "我正在分析微博中人们讨论的内容，请帮我判断下列词条是否为娱乐圈的名人（如明星、歌手、演员、艺人、脱口秀表演者等）或娱乐作品（电视剧、电影）。",
                    },
                    {"role": "user", "content": prompt},
                ],
                # temperature=0.1,  # 降低随机性，提高一致性
                # max_tokens=10,  # 只需要返回1或0
            )

            result_text = response.choices[0].message.content.strip()

            # 提取数字结果
            if "1" in result_text:
                return 1
            elif "0" in result_text:
                return 0
            else:
                # 如果返回的不是1或0，记录警告并默认返回0
                logger.warning(
                    f"名字 '{name}' 的返回结果异常: {result_text}，默认返回0"
                )
                return 0

        except Exception as e:
            logger.error(f"判断名字 '{name}' 时出错: {e}")
            return 0


def classify_names(
    input_file: str = DEFAULT_INPUT_FILE,
    output_file: str = DEFAULT_OUTPUT_FILE,
    delay: float = DEFAULT_DELAY,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    批量判断名字列表

    Args:
        input_file: 输入文件路径（每行一个名字）
        output_file: 输出CSV文件路径
        delay: API调用间隔（秒）
        model: 使用的模型名称
        api_key: OpenAI API密钥（默认从configs.configs读取）
        base_url: API基础URL（默认从configs.configs读取）
    """
    logger.info("=== 开始判断娱乐圈名字 ===")

    # 读取输入文件
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    logger.info(f"从 {input_path} 读取名字列表")
    with open(input_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    logger.info(f"共读取 {len(names)} 个名字")

    # 初始化分类器
    api_key_to_use = api_key if api_key is not None else OPENAI_API_KEY
    base_url_to_use = base_url if base_url is not None else OPENAI_BASE_URL
    classifier = EntertainmentNameClassifier(
        api_key=api_key_to_use,
        # base_url=base_url_to_use,
        model=model,
    )

    # 判断每个名字
    results = []
    for i, name in enumerate(names, 1):
        logger.info(f"正在判断 [{i}/{len(names)}]: {name}")
        result = classifier.classify(name)
        results.append({"name": name, "is_entertainment": result})
        logger.info(f"  -> 结果: {result}")

        # 延迟以避免请求过快
        if i < len(names) and delay > 0:
            time.sleep(delay)

    # 保存结果到CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"保存结果到 {output_path}")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "is_entertainment"])
        writer.writeheader()
        writer.writerows(results)

    # 统计结果
    total = len(results)
    positive = sum(1 for r in results if r["is_entertainment"] == 1)
    negative = total - positive

    logger.info("=== 判断完成 ===")
    logger.info(f"总计: {total}")
    logger.info(f"是娱乐圈相关: {positive} ({positive/total*100:.1f}%)")
    logger.info(f"不是娱乐圈相关: {negative} ({negative/total*100:.1f}%)")
    logger.info(f"结果已保存到: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    fire.Fire(classify_names)
