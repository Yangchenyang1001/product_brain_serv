"""
图片处理节点

处理 Markdown 文档中的图片，包括图片提取、存储到 MinIO、以及替换引用路径。
"""
import base64
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Set, List, Dict, Deque

from langchain_openai import OpenAI

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.config import get_config, ImportConfig
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError, ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class ImageContext:
    """md中图片上下文"""
    heading: str  # 标题
    pre_text: str  # 上文
    post_text: str  # 下文


@dataclass
class ImageInfo:
    """完整图片信息    """
    name: str  # 图片文件名，如 "abc123.jpg"
    path: str  # 图片完整路径
    context: ImageContext  # 图片上下文


class MdFileHandler:
    """负责MD文件的读取/路径校验/图片目录构建以及处理后续备份"""

    def __init__(self, logger):
        self.logger = logger

    def read_md(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """
        读取 Markdown 文件内容，并构建对应的 images 目录路径。
        :param state:
        :return: md_context md正文, md_path_obj md文件路径对象, images_dir images目录
        """
        self.logger.info("【step_1】 (MdFileHandler.read_md)：该函数负责读取 Markdown 文件")
        md_path = state.get("md_path", "")
        # 首先校验输入状态中的 md_path 字段是否合法（若非法抛出 StateFieldError）
        if not md_path:
            raise StateFieldError(node_name="md_img_node", field_name="md_path", expected_type=str)

        md_path_obj = Path(md_path)
        # 接着检查文件是否存在（若不存在抛出 FileProcessingError）
        if not md_path_obj.exists():
            raise FileProcessingError(
                f"md文件路径无效: {md_path}", node_name="md_img_node"
            )
        # 最后读取 MD 文件的原始内容，并构建对应的 images 目录路径。
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_context = f.read()

        images_dir = md_path_obj.parent / "images"
        return md_context, md_path_obj, images_dir

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        """
        备份新文件，将原文件重命名为原文件名_new，然后写入新内容。
        :param md_path_obj: md旧路径对象
        :param new_md_content: md新路径对象
        :return: 新路径字符串
        """
        self.logger.info("【step_5】备份新文件")
        # 修改原文件名，在末尾添加 _new，保留原文件扩展名(当前方法仅仅只是生成对象, 并不会实际重命名文件)
        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        # 将更新后的文件对象写入磁盘(真正实现写入)
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)


class ImageScanner:
    """扫描图片目录，提取每张图片在 MD 中的上下文信息。"""

    def __init__(self, logger):
        self.logger = logger

    def scan_img_dir(self, images_dir: Path, md_content: str, image_extensions: Set[str], context_length: int) -> list[
        ImageInfo]:
        image_list: list[ImageInfo] = []
        # 通过 pathlib.iterdir 遍历 images 目录
        for image_path in images_dir.iterdir():
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in image_extensions:
                continue
            # 子功能调用 Step 2.1_find_context()：对每张图片调用该方法定位其在 Markdown 中的上下文。
            ctx = self._find_context(md_content, image_path.name, context_length)

            if ctx is None:
                continue
            # 组装图片信息，包含图片名、路径和上下文ctx
            image_list.append(ImageInfo(name=image_path.name, path=str(image_path), context=ctx))

        return image_list

    def _find_context(self, md_content: str, img_name: str, max_chars: int = 200) -> ImageContext | None:
        """
        找到图片在md中第一次出现的文职的上下文, 找不到返回None
        :param md_content: md全部内容
        :param img_name: 图片名
        :param max_chars: 上下文最大字符数
        :return: 图片的上下文对象ImageContext, 包含标题、前文和后文
        """

        self.logger.info(f"【step_2.1】定位{img_name}其在 Markdown 中的上下文")

        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)"
        )
        # 所有行包含图片 文本 空行
        md_lines = md_content.split("\n")
        for line_idx, line in enumerate(md_lines):
            if not pattern.search(line):
                continue
            #    Step 2.2_find_heading_above()：向上寻找最近的标题。
            prev_title, prev_boundary = self._find_heading_above(
                md_lines, line_idx
            )
            # 取上文片段
            pre_content = md_lines[prev_boundary + 1: line_idx]

            # 取图片的前文
            img_pre: str = self._extract_limited_context(
                pre_content, max_chars, direction="front"
            )
            # Step 2.3_find_heading_below()：向下寻找最近的标题。
            next_boundary = self._find_heading_below(
                md_lines, line_idx
            )
            # 取下文片段
            post_content = md_lines[line_idx + 1: next_boundary]
            # 取图片的后文
            img_post: str = self._extract_limited_context(
                post_content, max_chars, direction="end"
            )
            # 返回图片的上下文, 包含标题、前文和后文
            return ImageContext(
                heading=prev_title,
                pre_text=img_pre,
                post_text=img_post,
            )
        return None

    @staticmethod
    def _extract_limited_context(lines: List[str], max_chars: int, direction: str) -> str:
        """
        按段落分割, 按direction方向贪心填装文本, 保持段落完整性
        :param lines: 符合要求的文本行列表
        :param max_chars: 上下文最大字符数
        :param direction: 贪心方向，"front" 或 "end"
        :return: 符合要求的文本,组合成一个字符

        front 表示从前往后贪心  反转两次
        如果max_chars=2 正文如下
          pre_content1      pre_content3
          pre_content2 =>   pre_content2  =>  pre_content3/pre_content2
          pre_content3      pre_content1
          这样做可以获取更靠近图片的文本, 提供vlm更精确的语义

        end 无需反转 和语义顺序一致仅仅截取即可

        """
        current_paragraph: List[str] = []  # 装填段落的行列表 当前段落
        paragraphs: List[str] = []  # 所有的段落列表 总段落
        # 遍历每一行数据
        for line in lines:
            is_blank_line = not line.strip()
            is_other_image = re.match(
                r"^!\[.*?\]\(.*?\)$", line.strip()
            )
            # 如果是空行或图片行，则将当前段落添加到 paragraphs 总段落列表中，并开始新的段落
            if is_blank_line or is_other_image:
                # 仅仅在当前段落有值时操作
                if current_paragraph:
                    # 将当前段落添加到 paragraphs 列表中
                    paragraphs.append("\n".join(current_paragraph))
                    # 清空行列表, 标志重新组装新段落
                    current_paragraph = []
                continue
            # 否则将当前行添加到当前段落中
            current_paragraph.append(line)
        # 添加最后一个当前段落 到总段落列表中
        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        # 如果是front，则反转 paragraphs 列表，以便从前往后选择
        if direction == "front":
            paragraphs.reverse()

        # 初始化总字符数和选中的段落列表
        total: int = 0
        selected: List[str] = []
        # 遍历 paragraphs 列表
        for para in paragraphs:

            """
            如果当前段落添加到已选段落后超过 max_chars，则停止添加
            如果当前段落添加后会超重，并且‘我们已经至少选了前面的一些段落了’，则停止添加
            如果当前一个都没选（selected 为空），哪怕这个段落重得像头大象，我也得先把它硬塞进去，不然最后返回的就是空字符串了。
            """

            if (total + len(para) > max_chars) and selected:
                break
            selected.append(para)
            total += len(para)

        # 如果是front，则反转选中的段落列表，以便保持正确的顺序
        if direction == "front":
            selected.reverse()
        return "\n\n".join(selected)

    @staticmethod
    def _find_heading_above(md_lines: List[str], from_idx: int) -> tuple[str, int]:
        """
        从 from_idx 向上查找最近的标题。
        :param md_lines: md文件内容列表
        :param from_idx: 开始查找的索引
        :return: 标题字符串和索引，未找到则返回空字符串和-1
        """
        for i in range(from_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return md_lines[i], i
        return "", -1

    @staticmethod
    def _find_heading_below(md_lines: List[str], from_idx: int) -> int:
        """
        从 from_idx 向下查找下一个标题。
        :param md_lines: md文件内容列表
        :param from_idx: 开始查找的索引
        :return: 索引，未找到则返回段落长度
        """
        for i in range(from_idx + 1, len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return i
        return len(md_lines)


class ImageUploader:
    """将本地图片上传至minio, 并在MD内容中替换为远程url+摘要"""

    def __init__(self, logger):
        self.logger = logger

    def upload_and_replace(self,
                           document_name: str,
                           md_content: str,
                           images_summaries: Dict[str, str],
                           image_list: List[ImageInfo],
                           minio_bucket: str, minio_base_url: str) -> str:
        """
        上传图片至 MinIO 并替换 MD 内容中的图片引用为远程 URL + 摘要。
        :param document_name: 文档标题
        :param md_content: Markdown 内容
        :param images_summaries: 图片摘要列表
        :param image_list: 图片列表
        :param minio_bucket: MinIO 桶名
        :param minio_base_url: MinIO 基础 URL
        :return: 替换后的 Markdown 全部内容
        """
        self.logger.info("【step_4】 (ImageUploader.upload_and_replace)：上传与正则替换。")

        # 上传所有图片至 MinIO
        remote_urls = self._upload_all(document_name, image_list, minio_bucket, minio_base_url)
        # 替换 MD 内容中的图片引用为远程 URL + 摘要
        return self._replace_in_md(md_content, images_summaries, remote_urls)

    def _upload_all(self, document_name: str, image_list: List[ImageInfo], minio_bucket: str, minio_base_url: str) -> \
            Dict[str, str]:
        """
        上传所有图片至 MinIO
        :param document_name: 文档标题
        :param image_list: 图片列表
        :param minio_bucket: MinIO 桶名
        :param minio_base_url: MinIO 基础 URL
        :return: 远程 URL 字典   key image_name
        """
        self.logger.info("【step_4.1】 (ImageUploader.upload_and_replace)：上传与正则替换")
        remote_urls: Dict[str, str] = {}

        try:
            # 获取 MinIO 客户端
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            # MinIO 不可用，所有图片保留本地路径
            self.logger.warning(
                f"MinIO 不可用，所有图片保留本地路径: {e}"
            )
            for img in image_list:
                remote_urls[img.name] = img.path
            return remote_urls
        # 遍历所有图片信息
        for img in image_list:
            # 构建图片对象名
            object_name = f"{document_name}/{img.name}"
            try:
                # minio上传文件
                minio_client.fput_object(
                    minio_bucket, object_name, img.path
                )
                # 构建url地址
                remote_url = (
                    f"{minio_base_url}/{minio_bucket}/{object_name}"
                )
                self.logger.info(
                    f"【step_4.1】 (ImageUploader.upload_and_replace)：{img.name} 上传成功, 远程地址: {remote_url}")
                remote_urls[img.name] = remote_url
            except Exception:
                # 上传失败，当前图片保留本地路径
                self.logger.warning(
                    f"{img.name} 上传失败，保留本地路径"
                )
                remote_urls[img.name] = img.path
        return remote_urls

    @staticmethod
    def _replace_in_md(md_content: str,
                       summaries: Dict[str, str],
                       remote_urls: Dict[str, str], ) -> str:
        """
        替换 MD 中的图片引用为远程 URL + 摘要。
        :param md_content: Markdown 内容
        :param summaries: 图片摘要列表
        :param remote_urls: 远程 URL 列表
        :return: 替换后的 Markdown 内容
        """

        # 正则表达式匹配图片引用
        pattern: re.Pattern[str] = re.compile(r"!\[(.*?)\]\((.*?)\)")

        # 替换函数
        def replacer(match: re.Match[str]) -> str:
            #
            """
            获取图片原始路径  match.group 括号 () 是“分组”的标志，它们把需要提取的数据框了起来
            match.group(0) ➡️ ![系统界面](./assets/interface.png) （完整的原始引用）
            match.group(1) ➡️ 系统界面 （方括号里的原始描述）
            match.group(2) ➡️ ./assets/interface.png （圆括号里的原始路径）
            :param match:
            :return:
            """
            original_path = match.group(2).strip()
            file_name_in_md = Path(original_path).name

            # for img_name, summary in summaries.items():
            #     if img_name == file_name_in_md:
            #         return f"![{summary}]({remote_urls[img_name]})"

            # 直接用文件名在两个字典里查，速度最快，不需要遍历
            if file_name_in_md in summaries:
                summary = summaries[file_name_in_md]
                remote_url = remote_urls[file_name_in_md]
                return f"![{summary}]({remote_url})"

            return match.group(0)

        return pattern.sub(replacer, md_content)


class VLMSummarizer:
    """通过视觉语言模型为每张图片生成中文标题/摘要。"""

    def __init__(self, logger):
        self.logger = logger

    def summarize_all(self,
                      document_title: str,
                      image_list: list[ImageInfo],
                      vlm: str,
                      requests_per_minute) -> Dict[str, str]:
        """
        通过视觉语言模型为每张图片生成中文标题/摘要。
        :param document_title: 文档标题
        :param image_list: 图片列表
        :param vlm: 视觉语言模型
        :param requests_per_minute: 每分钟请求限制
        :return:
        """
        summaries: Dict[str, str] = {}
        request_timestamps: Deque[float] = deque()
        self.logger.info("【step_3.1】(VLMSummarizer.summarize_all)：VLM 智能摘要")
        try:
            # 获取 OpenAI 客户端 用于vlm生成摘要
            client = AIClients.get_openai()
        except Exception as e:
            # 如果无法获取客户端就默认所有图片的摘要部分都是"图片描述"
            for img in image_list:
                summaries[img.name] = "图片描述"
            return summaries
        # 遍历所有图片信息
        for img in image_list:
            # 滑动窗口控制请求频率
            self._enforce_rate_limit(request_timestamps, requests_per_minute)
            # 调用 _summarize_one() 方法生成图片摘要
            summaries[img.name] = self._summarize_one(client, vlm, document_title, img)

        return summaries

    def _summarize_one(self, client: OpenAI, vlm: str, document_title: str, img: ImageInfo) -> str:
        """
        为单张图片生成中文标题/摘要。
        :param client: OpenAI 客户端
        :param vlm: 视觉语言模型
        :param document_title: 文档标题
        :param img: 图片信息
        :return:
        """
        self.logger.info(f"【step_3.2】_summarize_one()生成单张图片{img.name}摘要")
        # 拼接上下文（标题+上文+下文）
        """
        列表推导式 [p for p in (img.context.heading, img.context.pre_text, img.context.post_text) if p]
        遍历元组(img.context.heading, img.context.pre_text, img.context.post_text)
        if p  判断p状态是否异常  仅支持非空字符  
        ("标题","前文",None ) => ("标题","前文) 
        "\n".join(parts)
        ("标题","前文",None ) => ["标题","前文] =>"标题\n前文"
        """
        parts = [p for p in (img.context.heading, img.context.pre_text, img.context.post_text) if p]
        final_context = "\n".join(parts) if parts else "暂时无可用上下文"

        try:
            # 读取图片文件并进行 base64 编码
            with open(img.path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except:
            return "暂无图片"
        # 调用官方SDK调用方式大模型生成摘要
        """
        # 也可以使用langchain库调用
        from langchain import OpenAI, PromptTemplate
        prompt = PromptTemplate(
            template="任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                     "背景信息：\n"
                     "  1. 所属文档标题：\"{document_title}\"\n"
                     "  2. 图片上下文：{final_context}\n"
                     "请结合图片内容和上述上下文信息，"
                     "用中文简要总结这张图片的内容，"
                     "生成一个精准的中文标题（不要包含图片二字）。",
            input_variables=["document_title", "final_context"],
        )
        formatted_text = prompt.format(
            document_title=document_title, 
            final_context=final_context
        )
        #构建 LangChain 标准的多模态消息结构 (文本 + 图片Base64)
        langchain_messages = [
            HumanMessage(
                content=[
                    # 传入文本
                    {"type": "text", "text": formatted_text},
                    # 传入图片
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    }
                ]
            )
        ]
        llm = OpenAI(model=vlm)
        response = llm.invoke(langchain_messages).content  
        """

        try:
            resp = client.chat.completions.create(
                model=vlm,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_title}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img.path}: {e}")
            return "图片描述"

    def _enforce_rate_limit(
            self, timestamps: Deque[float],
            max_requests: int, window: int = 60,
    ):
        """
        滑动窗口限流
        :param timestamps: 窗口双端队列
        :param max_requests: 最大请求数
        :param window: 窗口大小（秒）
        :return:
        """
        # 当前时间
        now = time.time()
        # 移除窗口外的请求时间  最早时间 <= 当前时间 - 窗口大小
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()
        # 当前请求数量大于等于最大请求数也就是窗口大小
        if len(timestamps) >= max_requests:
            # 计算需要睡眠的时间
            sleep_dur = window - (now - timestamps[0])
            # 如果需要睡眠的时间大于0
            if sleep_dur > 0:
                self.logger.info(
                    f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                )
                time.sleep(sleep_dur)
                self.logger.info("睡眠结束，继续处理请求...")
                self.logger.info(f"当前请求数量：{len(timestamps)}")
            # 当前时间
            now = time.time()
            # 再次移除窗口外的请求时间
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()
        # 记录当前请求时间
        timestamps.append(now)


class MarkDownImageNode(BaseNode):
    """处理 MarkDown 图片的管道节点 —— 仅负责编排，不含业务细节。点"""
    name: str = "md_img_node"

    def __init__(self):
        super().__init__()
        self.file_handler = MdFileHandler(self.logger)
        self.scanner = ImageScanner(self.logger)
        self.summarizer = VLMSummarizer(self.logger)
        self.uploader = ImageUploader(self.logger)

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """
        MD 节点处理流程
        :param state: 状态
        :return: 状态
        """

        # 环境初始化与前置检查并 获取系统配置（config）。
        config = get_config()
        # Step 1 (MdFileHandler.read_md)：该函数负责读取 Markdown 文件。
        md_content, md_path_obj, image_dir = self.file_handler.read_md(state)

        # 关键分支判断：检查 images 目录是否存在。若不存在，则跳过所有处理，直接返回原始文档内容；若存在，则进入后续处理流程。
        if not image_dir.exists():
            state["md_content"] = md_content
            return state
        # Step 2 (ImageScanner.scan_img_dir)：扫描与上下文提取
        image_list: List[ImageInfo] = self.scanner.scan_img_dir(
            images_dir=image_dir,
            md_content=md_content,
            image_extensions=config.image_extensions,
            context_length=config.img_content_length
        )
        print(image_list)
        # for image in image_list:
        #     print(image)
        #     # print(f"图片名称****{image.name}****, 图片路径****{image.path}**** ")
        #     # print(f"图片标题：****{repr(image.context.heading)}****, 图片前文：****{repr(image.context.pre_text)}****, 图片后文：****{repr(image.context.post_text)}****")
        #

        # Step 3 (VLMSummarizer.summarize_all)：VLM 智能摘要
        summaries: dict[str, str] = self.summarizer.summarize_all(document_title=md_path_obj.stem,
                                                                  image_list=image_list,
                                                                  vlm=config.vl_model,
                                                                  requests_per_minute=config.requests_per_minute)

        for i, content in enumerate(summaries):
            print(f"Image {i + 1}: {content}{summaries[content]}")

        # Step 4 (ImageUploader.upload_and_replace)：上传与正则替换
        new_md_content: str = self.uploader.upload_and_replace(document_name=md_path_obj.stem,
                                                               md_content=md_content,
                                                               images_summaries=summaries,
                                                               image_list=image_list,
                                                               minio_bucket=config.minio_bucket,
                                                               minio_base_url=config.get_minio_base_url())
        print(new_md_content)

        # 5. Step 5 (MdFileHandler.backup)：文档备份写入。将替换完成后的新内容写入磁盘，生成名为 {原文件名}_new.md 的备份文件。
        self.file_handler.backup(md_path_obj, new_md_content)
        # 6. 状态更新与返回更新。处理管道中的状态（state['md_content']）。返回更新后的 state，供后续节点使用。
        state["md_content"] = new_md_content
        return state


if __name__ == '__main__':
    setup_logging()

    node = MarkDownImageNode()

    state = node(
        {
            # "file_title":"万用表RS-12的使用",
            "md_path": r"C:\Users\43779\Desktop\AIlearnrecord\shopkeeper_brain\knowledge\test\output\华为擎云 W525 用户指南-(PGUW-WBX0,KOS&UOS_02,zh-cn)\auto\华为擎云 W525 用户指南-(PGUW-WBX0,KOS&UOS_02,zh-cn).md"})
