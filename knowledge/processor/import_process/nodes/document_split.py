import json
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.utils.markdown_util import MarkdownTableLinearizer

"""
文档切分节点

将 Markdown 文档按配置的块大小和重叠长度进行切分，生成适合向量嵌入的文本块。
"""
from typing import Dict, Any, List

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class DocumentSplitNode(BaseNode):
    name: str = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """
        1. step1 获取输入 _get_input()
        2. step2 按标题一级切分
        3. step3 切分与合并 split_and_merge()
          a. step3.1 超长章节二次切分
            ⅰ. 使用 RecursiveCharacterTextSplitter 切分
          b. step3.2 合并过短章节
            ⅰ. 合并 body 长度小于 min_content_length 的相邻片段，避免信息碎片化。 merge_short_section()
        4. step4 组装最终chunk
          a. _assemble_chunk()
        5. step5 日志统计
          a. 输出切分统计信息，便于调试和监控。 _log_summary()
        6. step6 备份切片
          a. 将切分结果备份到 JSON 文件，便于调试和追溯。_backup_chunks()
        :param state:
        :return:
        """

        # config = get_config()
        #
        # md_content       = state.get("md_content", "")
        #
        # if md_content:
        #     md_content.replace("\r\n","\n").replace("\r","\n")
        #
        # file_title = state.get("file_title")

        # step1 获取输入 _get_input()
        md_content, file_title, max_content_length, min_content_length = self._get_inputs(state)
        # step2 按标题一级切分
        sections: List[Dict[str, Any]] = self._split_by_headings(md_content, file_title)
        print(sections)
        # step3 切分与合并 split_and_merge()
        final_chunks = self.split_and_merge(sections, max_content_length, min_content_length)
        print(final_chunks)
        # 组装最终chunk
        chunks = self._assemble_chunk(final_chunks)
        #
        state['chunks'] = chunks
        # # 6. 日志统计
        self._log_summary(md_content, chunks, max_content_length)
        #  7. 备份
        self._backup_chunks(state, chunks)

        return state

    def _get_inputs(self, state: ImportGraphState) -> tuple[str, str, int, int]:
        """
        获取输入参数
        :param state:
        :return:
        """
        self.log_step("step1", "切分文档的参数校验以及获取...")
        # 获取全文
        md_content = state.get("md_content")
        if md_content:
            # 替换回车符
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        else:
            raise StateFieldError(self.name, "md_content", str)
        # 获取文件标题
        file_title = state.get("file_title", "")
        # 校验文件标题
        if not file_title:
            raise StateFieldError(self.name, "file_title", str)
        max_content_length = self.config.max_content_length
        min_content_length = self.config.min_content_length
        if max_content_length < 0 or min_content_length < 0 or max_content_length < min_content_length:
            raise ValidationError("切分最大值和最小值参数错误", self.name)

        return md_content, file_title, max_content_length, min_content_length

    def _split_by_headings(self, md_content, file_title) -> List[Dict[str, Any]]:
        """
        识别标题切分文档
        :param md_content: 整个文档内容
        :param file_title: 文件的标题  无扩展名
        :return: 字典集合  包含标题title/内容body文件标题/file_title/父标题parent_title
        """

        sections: List[Dict[str, Any]] = []  # 输出结果集合
        in_fence = False  # 代码围挡判断

        heading_pattern = re.compile(r"^\s*(#{1,6})\s+(.+)")  # 匹配标题的正则

        body_lines: List[str] = []  # 临时存储正文内容 , 仅支持存放两个标题间的内容

        content_lines = md_content.split("\n")  # 按换行分割的全部正文数据

        current_title = ""  # 当前标题
        current_level = 0  # 当前等级
        hierarchy = [""] * 7  # 7个长度，第一个（0）不用  等级、层级  存放各级标题信息

        def _flush():
            # 去除空格
            body = "\n".join(body_lines).strip()
            parent_title = ""
            if not (current_title or body):
                return None
                # 遍历标题数组获取父标题
            for i in range(current_level - 1, 0, -1):
                if hierarchy[i]:
                    parent_title = hierarchy[i]
                    break

            if not parent_title:
                parent_title = current_title if current_title else file_title

            return sections.append({
                "title": current_title if current_title else file_title,
                "body": body,
                "file_title": file_title,
                "parent_title": parent_title
            })

        for index, line in enumerate(content_lines):
            # 代码围栏跳过
            if line.strip().startswith("```") or line.strip().startswith("~~~"):
                in_fence = not in_fence

            # 匹配标题
            match = heading_pattern.match(line) if not in_fence else None

            if match:
                _flush()  # 帮我把当前标题之前的内容封装成一个section

                level = len(match.group(1))
                current_title = line
                current_level = level
                # 复制当前数据最新的标题对应等级
                hierarchy[level] = current_title
                # 每次更新标题数组都需要清空后续标题, 反之重复使用  清空后续标题
                for i in range(level + 1, 7):
                    hierarchy[i] = ""
                # 重置正文信息
                body_lines = []

            else:
                # 标题以外的信息全部收集起来 做正文
                body_lines.append(line)
        _flush()

        return sections

    def split_and_merge(self, sections: List[Dict[str, Any]], max_content_length, min_content_length):
        """
        长切短合操作
        :param sections: 正文片段集合
        :param max_content_length: 最大长度
        :param min_content_length: 最小长度
        :return:
        """
        current_sections = []

        # 拆分
        for section in sections:
            # append 添加单个元素
            # extend 添加多个元素
            current_sections.extend(self.split_long_section(section, max_content_length))
        # 合并
        final_sections = self.merge_short_section(current_sections, min_content_length)

        return final_sections

    def _assemble_chunk(self, final_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """最终组合 chunk"""
        self.log_step("step4", "组装最终的切片信息...")
        chunks = []

        for chunk in final_chunks:
            # 1. 获取 chunk 的信息
            title = chunk.get('title')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')
            body = chunk.get('body')
            content = f"{title}\n\n{body}"

            # 2. 构建最终 chunk 对象
            assemble_chunk = {
                "title": title,
                "file_title": file_title,
                "parent_title": parent_title,
                "content": content,
            }

            # 3. 判断 part 是否存在
            if "part" in chunk:
                assemble_chunk['part'] = chunk.get('part')

            chunks.append(assemble_chunk)

        return chunks

    def split_long_section(self, section: Dict[str, Any], max_content_length) -> List[Dict[str, Any]]:
        """
        拆分过长的章节
        :param section:  文本片段
        :param max_content_length:  最大文本长度
        :return:
        """
        #  获取 section 对象属性
        title = section.get('title')
        body = section.get('body')
        file_title = section.get('file_title')
        parent_title = section.get('parent_title')

        # 表格处理
        if "<table>" in body:
            self.logger.info("检测到了表格数据...")
            # 使用降维度转移法  处理表格数据
            # 手动处理表格内容, 也可以结合实际使用vlm大模型处理复杂表格
            body = MarkdownTableLinearizer.process(body)
            section['body'] = body

        # 标题处理 大于50则自动切断
        TITLE_MAX_LENGTH = 50
        if len(title) > TITLE_MAX_LENGTH:
            self.logger.warning(f"检测文件{file_title}对应的{title}长度过长...")
            title = title[:TITLE_MAX_LENGTH]

        # 标题+正文的长度 如果未大于临界值就不拆分
        title_prefix = f"{title}\n\n"
        total_length = len(title_prefix) + len(body)
        if total_length <= max_content_length:
            return [section]

        # 计算 body 的可用长度
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]

        # 使用递归字符文本拆分器进行拆分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,  # 不重叠，默认值为0
            # 最后的""必须保留, 用于取保所有切片长度都是不大于body_length
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            keep_separator=False  # 不包含分隔符， 默认值为True
        )
        texts = text_splitter.split_text(body)
        # 如果仅仅切割出一个元素,说明当前拆分未达到临界值所以直接返回
        if len(texts) <= 1:
            return [section]

        final_sections = []
        for index, textStr in enumerate(texts):
            final_sections.append({
                "title": title,
                "body": textStr,
                "file_title": file_title,
                "parent_title": parent_title,
                "part": index + 1  # 标记序号
            })

        return final_sections

    def merge_short_section(self, current_sections: List[Dict], min_content_length):
        """
        合并过短的章节
        :param current_sections: 当前章节
        :param min_content_length: 最小内容长度
        :return:
        """

        final_sections = []  # 最终的箱子

        # 空列表直接返回，避免后续引用未定义变量
        if len(current_sections) == 0:
            return final_sections

        current_section = current_sections[0]

        # 2. 遍历以及合并：仅合并「同源」且「当前段过短」的相邻片段
        for next_section in current_sections[1:]:
            # 同源检查
            same_parent = (current_section['parent_title'] == next_section['parent_title'])
            # 当前段过短（< min_content_length）且与下一段同源时，才向后合并
            if same_parent and len(current_section.get('body')) < min_content_length:
                # body的合并
                current_section['body'] = (
                        current_section.get('body').rstrip() + "\n\n" + next_section.get('body').lstrip()
                )
                # 合并后重新回到「父标题」语义，part 由后续统一编号
                current_section['title'] = current_section['parent_title']
                current_section['part'] = 0
            else:
                # 将原来 current_section 进行封箱
                final_sections.append(current_section)
                # 移动窗口
                current_section = next_section

        # 最后一个（封装起来）
        final_sections.append(current_section)
        # 3. 过滤空正文切片
        final_sections = [s for s in final_sections if s.get('body', '').strip()]

        # 4. 对同源被拆分的 section 统一重编号并拼接到 title
        part_counter = {}
        result = []
        for final_section in final_sections:
            parent_title = final_section.get('parent_title')
            if "part" in final_section:
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                new_part = part_counter[parent_title]
                final_section['part'] = new_part
                final_section['title'] = final_section['title'] + f"- {new_part}"

            result.append(final_section)

        return result

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        """
        输出切分统计信息
        :param raw_content: 原始内容
        :param chunks: 切分后的章节
        :param max_length: 最大长度
        :return:
        """
        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    setup_logging()
    document_node = DocumentSplitNode()
    # 构造状态字典
    file_path = r"C:\Users\43779\Desktop\AIlearnrecord\shopkeeper_brain\knowledge\test\output\万用表RS-12的使用\auto\万用表RS-12的使用_new.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    state = {
        "file_title": "hak180产品安全千册",
        "md_content": content,
        "file_dir": r"C:\Users\43779\Desktop\AIlearnrecord\shopkeeper_brain\knowledge\test\output\万用表RS-12的使用\auto"}
    document_node.process(state)
