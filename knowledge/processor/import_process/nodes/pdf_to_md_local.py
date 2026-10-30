"""
  @Author:桌角是小黑
  @Time:2026/10/9
  @Desc:
"""
"""
对接本地 mineru-api 服务
启动方式：mineru-api --host 127.0.0.1 --port 8000
API文档：http://127.0.0.1:8000/docs

本地API接口：
    POST /tasks          — 异步任务提交（上传文件，立即返回task_id）
    GET  /tasks/{id}     — 轮询任务状态
    GET  /tasks/{id}/result — 获取解析结果（ZIP包）
    GET  /health         — 健康检查
"""
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import requests

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state


class PdfToMdLocalNode(BaseNode):
    """
    节点: PDF转Markdown (本地mineru-api版)
    将 PDF 非结构化数据转换为 Markdown 结构化数据。
    通过HTTP调用本地 mineru-api 服务完成解析。
    """

    name = "pdf_to_md_local_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        必要参数：pdf_path、local_dir
        更新参数：md_path、md_content
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：校验PDF路径和输出目录
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 步骤2：调用本地MinerU API同步解析，获取ZIP包
        zip_content = self._step_2_parse(pdf_path_obj)

        # 步骤3：解压ZIP包并提取MD文件
        md_path = self._step_3_extract(zip_content, output_dir_obj, pdf_path_obj.stem)

        # 步骤4：读取md的内容
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 步骤5：更新state状态
        state["md_path"] = str(md_path)
        state["md_content"] = md_content

        return state

    def _step_1_validate_paths(self, state: ImportGraphState):
        """
        步骤1：校验PDF文件路径和输出目录
        核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
        返回：合法的PDF文件Path对象、输出目录Path对象
        异常：ValueError(参数缺失)、FileNotFoundError(文件无效)
        """

        # 1、参数非空校验
        pdf_path = state.get("pdf_path", "").strip()
        local_dir = state.get("local_dir", "").strip()
        if not pdf_path:
            raise ValueError("缺失参数：pdf_path")
        if not local_dir:
            raise ValueError("缺失参数：local_dir")

        # 2、转换为Path对象统一处理路径
        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(local_dir)

        # 3、PDF文件有效性校验
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF文件不存在，绝对路径：{pdf_path_obj.absolute()}")

        # 4、确保输出目录存在，不存在则递归创建
        if not output_dir_obj.exists():
            self.logger.info(f"输出目录不存在，自动创建：{output_dir_obj.absolute()}")
            output_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, output_dir_obj

    def _step_2_parse(self, pdf_path_obj: Path) -> bytes:
        """
        步骤2：调用本地MinerU API同步解析接口 POST /file_parse
        核心流程：配置校验 → 提交同步解析请求 → 等待完成并返回ZIP字节
        参数：pdf_path_obj-已校验的PDF Path对象
        返回：ZIP结果包的字节内容
        异常：ValueError(配置缺失)、RuntimeError(请求失败)
        """
        # 1、配置校验
        if not self.config.mineru_base_url:
            raise ValueError("MinerU配置缺失：请在 .env 文件中配置 MINERU_BASE_URL，例如：http://127.0.0.1:8000")
        self.logger.info(f"【配置校验】MinerU本地API配置校验成功，开始处理文件：{pdf_path_obj.name}")

        # 2、调用同步解析接口：POST /file_parse
        url = f"{self.config.mineru_base_url}/file_parse"
        self.logger.info(f"【同步解析】调用接口：{url}")

        start_time = time.time()

        with open(pdf_path_obj, "rb") as f:
            files = {"files": (pdf_path_obj.name, f, "application/pdf")}
            data = {
                "return_md": "true",      # 返回 Markdown
                "return_images": "true",  # ← 新增！返回图片
                "return_content_list": "true",  # ← 新增！返回结构化数据
                "response_format_zip": "true",  # 以 ZIP 格式返回
                "return_original_file": "true",  # 保留原始 PDF
                "backend": "pipeline", #基于CPU模型
            }
            response = requests.post(url, files=files, data=data, timeout=600)  #文件上传请求。post请求.

        elapsed_time = time.time() - start_time

        if response.status_code != 200:
            raise RuntimeError(
                f"【同步解析】请求失败：状态码：{response.status_code}，响应内容：{response.text}"
            )

        content_type = response.headers.get("content-type", "")
        self.logger.info(f"【同步解析】解析完成，耗时{int(elapsed_time)}s，响应类型：{content_type}，大小：{len(response.content) / 1024:.1f}KB")

        return response.content

    def _step_3_extract(self, zip_content: bytes, output_dir_obj: Path, pdf_stem: str) -> str:
        """
        步骤3：解压MinerU解析结果ZIP包，提取目标MD文件
        核心流程：保存ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
        参数：zip_content-ZIP字节内容；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
        返回：最终MD文件的字符串格式绝对路径
        异常：RuntimeError(解压失败)、FileNotFoundError(无MD文件)
        """

        # 1、保存ZIP包
        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        with open(zip_save_path, "wb") as f:
            f.write(zip_content)
        self.logger.info(f"【ZIP保存】保存路径：{zip_save_path}，大小：{len(zip_content) / 1024:.1f}KB")

        # 2、清空解压目录（ZIP内部已包含{pdf_stem}/前缀，直接解压到output_dir_obj即可）
        extract_target_dir = output_dir_obj / pdf_stem
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir, ignore_errors=True)
            self.logger.info(f"【ZIP解压】已清空旧的解压目录：{extract_target_dir}")

        # 3、直接解压到output_dir_obj（ZIP内自带{pdf_stem}/前缀，解压后自动形成 output_dir_obj/{pdf_stem}/auto/...）
        self.logger.info(f"【ZIP解压】开始解压ZIP包到：{output_dir_obj} ...")
        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(output_dir_obj)  #解压
            all_files = zip_file_obj.namelist()
        self.logger.info(f"【ZIP解压】解压完成")
        self.logger.info(f"【ZIP解压】ZIP内文件列表（共{len(all_files)}个）：")
        for f_name in all_files:
            self.logger.info(f"  - {f_name}")

        # 4、查找MD文件（ZIP内结构：{pdf_stem}/auto/{pdf_stem}.md）
        md_auto_dir = extract_target_dir / "auto"
        target_md_file = md_auto_dir / f"{pdf_stem}.md"

        if not target_md_file.exists():
            # 兜底：优先找full.md，再递归搜索任意.md
            full_md = md_auto_dir / "full.md"
            if full_md.exists():
                target_md_file = full_md
            else:
                md_files = list(extract_target_dir.rglob("*.md")) #Recursive Glob — 递归搜索所有匹配 *.md 的文件
                if not md_files:
                    raise FileNotFoundError(f"【MD查找】解压目录中未找到任何MD文件：{extract_target_dir}")
                target_md_file = md_files[0]
            self.logger.info(f"【MD查找】未找到预期路径，使用：{target_md_file}")

        self.logger.info(f"【MD查找】找到MD文件：{target_md_file}")

        # 5、检查images目录
        images_dir = target_md_file.parent / "images"
        if images_dir.exists():
            img_files = list(images_dir.iterdir())
            self.logger.info(f"【资源检查】images目录存在，共{len(img_files)}个图片文件 ✅")
            for img_file in img_files:
                self.logger.info(f"  - {img_file.name} ({img_file.stat().st_size / 1024:.1f}KB)")
        else:
            self.logger.warning(f"【资源检查】images目录不存在：{images_dir}，MD中的图片引用将无法显示 ")

        return str(target_md_file.absolute())
# ================================================================== #
#                        测试                                        #
# ================================================================== #
if __name__ == "__main__":
    from knowledge.processor.import_process.base import setup_logging

    setup_logging()

    pdf_path = os.path.join(r"E:\doc\hak180产品安全手册.pdf")
    local_dir = os.path.join(r"E:\doc\output2")

    init_state = create_default_state(
        task_id="task_001",
        pdf_path=pdf_path,
        local_dir=local_dir
    )

    node_pdf_to_md = PdfToMdLocalNode()
    final_state = node_pdf_to_md(init_state)
    print(json.dumps(final_state, indent=4, ensure_ascii=False))
