"""
PDF 转 MD 节点（网络版）

通过 MinerU 开放 API（默认 https://mineru.net/api/v4）在服务端解析 PDF，
不依赖本地 mineru 命令行 / 本地模型，也不占用本机 GPU。

接口流程（MinerU v4 开放接口）：
    1. POST {base_url}/file-urls/batch                      申请上传链接，返回 batch_id + file_urls
    2. PUT  file_urls[i]                                    把 PDF 二进制上传到该链接（不需要鉴权头）
    3. GET  {base_url}/extract-results/batch/{batch_id}     轮询解析状态（waiting-file/pending/running/converting/done/failed）
    4. 下载 full_zip_url 结果包，解压后定位 {stem}/auto/{stem}.md

配置（.env）：
    MINERU_BASE_URL=https://mineru.net/api/v4
    MINERU_API_TOKEN=sk-xxxxxxxx
    MINERU_MODEL_VERSION=pipeline   # 可选 pipeline / vlm
    MINERU_LANGUAGE=ch              # 可选，OCR 语言
"""
import json
import shutil
import time
import zipfile
from pathlib import Path

import requests

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import (
    ConfigurationError,
    FileProcessingError,
    PdfConversionError,
    ValidationError,
)
from knowledge.processor.import_process.state import ImportGraphState

# MinerU 批处理任务中代表“已结束”的状态
_TERMINAL_STATES = {"done", "failed"}


class PdfToMdNetNode(BaseNode):
    """
    节点: PDF 转 Markdown（网络版）

    调用 MinerU 开放 API 完成解析：
        申请上传链接 -> 上传 PDF -> 轮询任务 -> 下载 ZIP -> 解压取 MD
    """

    name: str = "pdf_to_md_net_node"

    # ==================== 主流程 ====================

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        必要参数：import_file_path、file_dir
        更新参数：md_path、md_content

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # 步骤1：校验输入路径
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 步骤2：申请上传链接（拿到 batch_id 与上传地址）
        batch_id, upload_url = self._step_2_apply_upload_url(pdf_path_obj)

        # 步骤3：上传 PDF 到对象存储
        self._step_3_upload_file(pdf_path_obj, upload_url)

        # 步骤4：轮询解析任务，拿到结果包下载地址
        zip_url = self._step_4_wait_for_result(batch_id, pdf_path_obj.name)

        # 步骤5：下载结果包并解压，定位 Markdown
        md_path = self._step_5_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        # 步骤6：回填状态
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        state["md_path"] = str(md_path)
        state["md_content"] = md_content
        self.logger.info(f"PDF 解析完成：{pdf_path_obj.name} -> {md_path}（{len(md_content)} 字符）")

        return state

    # ==================== 步骤1：校验路径 ====================

    def _step_1_validate_paths(self, state: ImportGraphState) -> tuple[Path, Path]:
        """
        校验 PDF 文件路径与输出目录

        返回：PDF 的 Path 对象、输出目录的 Path 对象
        """
        self.log_step("step1", "校验输入路径")

        import_file_path = state.get("import_file_path", "") or ""
        file_dir = state.get("file_dir", "") or ""

        if not import_file_path:
            raise ValidationError("解析的文件不存在", self.name)

        pdf_path_obj = Path(import_file_path)
        if not pdf_path_obj.exists():
            raise FileProcessingError(f"解析的文件路径不存在：{pdf_path_obj.absolute()}", self.name)

        # 输出目录为空时降级到 PDF 所在目录
        output_dir_obj = Path(file_dir) if file_dir else pdf_path_obj.parent
        output_dir_obj.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"上传文件的路径: {pdf_path_obj}")
        self.logger.info(f"输出的目录: {output_dir_obj}")

        return pdf_path_obj, output_dir_obj

    # ==================== 步骤2：申请上传链接 ====================

    def _step_2_apply_upload_url(self, pdf_path_obj: Path) -> tuple[str, str]:
        """
        POST {base_url}/file-urls/batch 申请文件上传链接

        返回：batch_id、第一个文件的上传地址
        """
        self.log_step("step2", "向 MinerU 申请上传链接")

        base_url, token = self._get_api_settings()

        url = f"{base_url}/file-urls/batch"
        payload = {
            "enable_formula": self.config.mineru_enable_formula,
            "enable_table": self.config.mineru_enable_table,
            "language": self.config.mineru_language,
            "model_version": self.config.mineru_model_version,
            "files": [
                {
                    "name": pdf_path_obj.name,
                    # data_id 用于在结果里关联业务数据，这里用文件名即可
                    "data_id": pdf_path_obj.stem[:128],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }

        self.logger.info(f"【申请上传链接】调用接口：{url}")
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=self.config.mineru_request_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise PdfConversionError(f"申请上传链接请求失败：{exc}", self.name) from exc

        if response.status_code != 200:
            raise PdfConversionError(
                f"申请上传链接失败：状态码 {response.status_code}，响应 {response.text[:500]}",
                self.name,
            )

        data = self._parse_api_response(response, "申请上传链接")

        batch_id = data.get("batch_id", "")
        file_urls = data.get("file_urls") or []
        if not batch_id or not file_urls:
            raise PdfConversionError(f"申请上传链接返回内容不完整：{data}", self.name)

        self.logger.info(f"【申请上传链接】batch_id={batch_id}")
        return str(batch_id), str(file_urls[0])

    # ==================== 步骤3：上传文件 ====================

    def _step_3_upload_file(self, pdf_path_obj: Path, upload_url: str) -> None:
        """
        PUT 上传 PDF 二进制到申请到的上传链接（该链接自带签名，不需要再加鉴权头）
        """
        self.log_step("step3", "上传 PDF 到 MinerU 对象存储")

        file_size = pdf_path_obj.stat().st_size
        self.logger.info(f"【上传文件】{pdf_path_obj.name}，大小 {file_size / 1024 / 1024:.2f}MB")

        start_time = time.time()
        try:
            with open(pdf_path_obj, "rb") as f:
                response = requests.put(
                    upload_url,
                    data=f,
                    timeout=self.config.mineru_request_timeout_seconds,
                )
        except requests.RequestException as exc:
            raise PdfConversionError(f"上传 PDF 失败：{exc}", self.name) from exc

        if response.status_code not in (200, 201):
            raise PdfConversionError(
                f"上传 PDF 失败：状态码 {response.status_code}，响应 {response.text[:500]}",
                self.name,
            )

        self.logger.info(f"【上传文件】完成，耗时 {time.time() - start_time:.2f}s")

    # ==================== 步骤4：轮询解析结果 ====================

    def _step_4_wait_for_result(self, batch_id: str, file_name: str) -> str:
        """
        GET {base_url}/extract-results/batch/{batch_id} 轮询任务状态

        返回：结果 ZIP 包的下载地址
        """
        self.log_step("step4", "等待 MinerU 解析完成")

        base_url, token = self._get_api_settings()
        url = f"{base_url}/extract-results/batch/{batch_id}"
        headers = {"Authorization": f"Bearer {token}", "Accept": "*/*"}

        poll_interval = max(1, int(self.config.mineru_poll_interval_seconds))
        deadline = time.time() + max(1, int(self.config.mineru_max_wait_seconds))
        last_state = ""
        last_log_time = 0.0

        self.logger.info(f"【解析进度】开始轮询：{url}（每 {poll_interval}s 一次）")

        while True:
            if time.time() > deadline:
                raise PdfConversionError(
                    f"等待 MinerU 解析超时（>{self.config.mineru_max_wait_seconds}s），"
                    f"batch_id={batch_id}，最后状态={last_state or '未知'}",
                    self.name,
                )

            try:
                response = requests.get(url, headers=headers, timeout=self.config.mineru_request_timeout_seconds)
            except requests.RequestException as exc:
                # 轮询期间的网络抖动不直接失败，记录后继续重试
                self.logger.warning(f"【解析进度】轮询请求异常，将重试：{exc}")
                response = None

            if response is not None:
                if response.status_code != 200:
                    raise PdfConversionError(
                        f"查询解析结果失败：状态码 {response.status_code}，响应 {response.text[:500]}",
                        self.name,
                    )

                data = self._parse_api_response(response, "查询解析结果")
                results = data.get("extract_result") or []
                # 批处理里只有一个文件，取同名项（找不到就取第一条）
                item = next((r for r in results if str(r.get("file_name", "")) == file_name), None)
                if item is None and results:
                    item = results[0]

                if item is not None:
                    current_state = str(item.get("state", "")).lower()
                    # 状态变化或每 30 秒打印一次进度
                    now = time.time()
                    if current_state != last_state or now - last_log_time >= 30:
                        self.logger.info(
                            f"【解析进度】state={current_state or '未知'}"
                            + (f"，progress={item.get('extract_progress')}" if item.get("extract_progress") else "")
                        )
                        last_state = current_state
                        last_log_time = now

                    if current_state in _TERMINAL_STATES:
                        if current_state == "failed":
                            raise PdfConversionError(
                                f"MinerU 解析失败：{item.get('err_msg') or '未返回错误信息'}",
                                self.name,
                            )
                        zip_url = item.get("full_zip_url") or ""
                        if not zip_url:
                            raise PdfConversionError(f"MinerU 解析完成但没有返回结果包地址：{item}", self.name)
                        self.logger.info(f"【解析进度】解析完成，结果包：{zip_url}")
                        return str(zip_url)

            time.sleep(poll_interval)

    # ==================== 步骤5：下载并解压 ====================

    def _step_5_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> Path:
        """
        下载结果 ZIP 包，整理成本地 mineru 命令行的同款目录结构，定位 Markdown 文件

        最终结构与本地解析完全一致（保证下游节点、存储路径、MinIO 对象前缀不变）：
            {file_dir}/{pdf_stem}/auto/{pdf_stem}.md
            {file_dir}/{pdf_stem}/auto/images/*.jpg
            {file_dir}/{pdf_stem}/auto/*_content_list.json ...

        返回：MD 文件的 Path 对象
        """
        self.log_step("step5", "下载并解压解析结果")

        self.logger.info(f"【下载结果】{zip_url}")
        try:
            response = requests.get(zip_url, stream=True, timeout=self.config.mineru_request_timeout_seconds)
        except requests.RequestException as exc:
            raise PdfConversionError(f"下载解析结果失败：{exc}", self.name) from exc

        if response.status_code != 200:
            raise PdfConversionError(
                f"下载解析结果失败：状态码 {response.status_code}", self.name
            )

        # 先在临时目录解压，整理成与本地 mineru 命令行完全一致的目录结构后再落到文件目录
        staging_dir = output_dir_obj / f".{pdf_stem}.mineru_tmp"
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        zip_save_path = staging_dir / f"{pdf_stem}_result.zip"
        with open(zip_save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        self.logger.info(f"【下载结果】保存到临时目录 {zip_save_path}，大小 {zip_save_path.stat().st_size / 1024:.1f}KB")

        try:
            with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
                # 云端接口返回扁平结构（full.md、images/、xxx_content_list.json），
                # 自建 mineru-api 返回 {pdf_stem}/auto/{pdf_stem}.md 结构，统一先解压到临时目录
                zip_file_obj.extractall(staging_dir)
                all_files = zip_file_obj.namelist()
        except zipfile.BadZipFile as exc:
            raise PdfConversionError(f"解析结果不是合法的 ZIP 包：{zip_save_path}", self.name) from exc

        self.logger.info(f"【解压】完成，压缩包内 {len(all_files)} 个文件")

        # 目标结构（与本地 mineru 命令行一致）：{file_dir}/{pdf_stem}/auto/{pdf_stem}.md + auto/images/
        document_dir = output_dir_obj / pdf_stem
        auto_dir = document_dir / "auto"
        if document_dir.exists():
            shutil.rmtree(document_dir, ignore_errors=True)
            self.logger.info(f"【解压】已清空旧的文档目录：{document_dir}")
        auto_dir.mkdir(parents=True, exist_ok=True)

        # 找到临时目录里的 md 及其同级资源目录（images、content_list 等），整体搬进 auto/
        source_md_file = self._find_md_file(staging_dir, pdf_stem)
        source_md_dir = source_md_file.parent
        for entry in sorted(source_md_dir.iterdir()):
            # 结果 ZIP 包只是中间产物，不落到文件目录里
            if entry.resolve() == zip_save_path.resolve():
                continue
            shutil.move(str(entry), str(auto_dir / entry.name))
        self.logger.info(f"【解压】迁移 {source_md_dir} -> {auto_dir}")

        # 清理临时目录（含结果 ZIP 包），不污染文件目录
        shutil.rmtree(staging_dir, ignore_errors=True)

        target_md_file = auto_dir / source_md_file.name
        self.logger.info(f"【MD 查找】解析结果 Markdown：{target_md_file}")

        images_dir = target_md_file.parent / "images"
        if images_dir.exists():
            self.logger.info(f"【资源检查】images 目录存在，共 {len(list(images_dir.iterdir()))} 个图片文件")
        else:
            self.logger.warning(f"【资源检查】images 目录不存在：{images_dir}")

        return target_md_file.absolute()

    def _find_md_file(self, extract_target_dir: Path, pdf_stem: str) -> Path:
        """
        在解压目录中定位 Markdown，并统一重命名为 {pdf_stem}.md

        兼容两种目录结构：
            {pdf_stem}/auto/{pdf_stem}.md   （自建 mineru-api）
            full.md                          （MinerU 云端接口）
        """
        candidates = [
            extract_target_dir / pdf_stem / "auto" / f"{pdf_stem}.md",
            extract_target_dir / pdf_stem / "auto" / "full.md",
            extract_target_dir / "auto" / f"{pdf_stem}.md",
            extract_target_dir / "auto" / "full.md",
            extract_target_dir / pdf_stem / f"{pdf_stem}.md",
            extract_target_dir / pdf_stem / "full.md",
            extract_target_dir / f"{pdf_stem}.md",
            extract_target_dir / "full.md",
        ]
        target_md_file = next((path for path in candidates if path.exists()), None)

        if target_md_file is None:
            md_files = sorted(extract_target_dir.rglob("*.md"))
            if not md_files:
                raise FileProcessingError(f"解压目录中未找到任何 MD 文件：{extract_target_dir}", self.name)
            target_md_file = md_files[0]
            self.logger.info(f"【MD 查找】未命中预期路径，使用兜底路径：{target_md_file}")

        # 统一成 {pdf_stem}.md，保证下游 md_img_node 用文件名作为文档名/图片上传前缀
        if target_md_file.name != f"{pdf_stem}.md":
            normalized_path = target_md_file.with_name(f"{pdf_stem}.md")
            if not normalized_path.exists():
                target_md_file.rename(normalized_path)
                self.logger.info(f"【MD 查找】重命名 {target_md_file.name} -> {normalized_path.name}")
                target_md_file = normalized_path

        return target_md_file

    # ==================== 通用工具 ====================

    def _get_api_settings(self) -> tuple[str, str]:
        """读取并校验 MinerU API 配置，返回 (base_url, token)"""
        base_url = (self.config.mineru_base_url or "").strip().rstrip("/")
        token = (self.config.mineru_api_token or "").strip()

        if not base_url:
            raise ConfigurationError(
                "MinerU 配置缺失：请在 .env 中配置 MINERU_BASE_URL，例如 https://mineru.net/api/v4",
                self.name,
            )
        if not token:
            raise ConfigurationError(
                "MinerU 配置缺失：请在 .env 中配置 MINERU_API_TOKEN",
                self.name,
            )
        return base_url, token

    def _parse_api_response(self, response: requests.Response, action: str) -> dict:
        """解析 MinerU 接口响应，code != 0 时抛出可读错误"""
        try:
            body = response.json()
        except ValueError as exc:
            raise PdfConversionError(f"{action}返回的不是 JSON：{response.text[:300]}", self.name) from exc

        if not isinstance(body, dict):
            raise PdfConversionError(f"{action}返回格式异常：{body}", self.name)

        if body.get("code", 0) != 0:
            raise PdfConversionError(
                f"{action}失败：code={body.get('code')}，msg={body.get('msg')}",
                self.name,
            )

        data = body.get("data")
        return data if isinstance(data, dict) else {}


# ================================================================== #
#                        测试                                        #
# ================================================================== #
if __name__ == "__main__":
    import os

    setup_logging()

    init_state = {
        "import_file_path": r"C:\Users\43779\Desktop\AIlearnrecord\shopkeeper_brain_docs\doc\hak180产品安全手册.pdf",
        "file_dir": os.path.join(os.path.dirname(__file__), "..", "..", "..", "test", "output_net"),
    }

    node_pdf_to_md_net = PdfToMdNetNode()
    print(json.dumps(node_pdf_to_md_net(init_state), indent=4, ensure_ascii=False))
