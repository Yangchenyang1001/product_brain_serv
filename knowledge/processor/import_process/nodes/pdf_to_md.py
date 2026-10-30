"""
PDF 转 MD 节点

将上传的 PDF 文件转换为 Markdown 格式，提取文本内容并保留基本结构信息。
"""
import json
import subprocess
import time
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    """
    PDF 转 MD 节点
    调用MinerU命令行工具将pdf转换为Markdown格式
    支持实时日志输出，方便用户查看转换进度

    """
    name: str = "pdf_to_md_node"

    def _validate_state_inputs_path(self, state: ImportGraphState) -> tuple[Path, Path]:
        """
        验证输入路径
        :param state:
        :return:
        """
        self.log_step("step1", "对状态的路径输入参数做校验")
        # 1. 获取输入pdf文件路径 与输出后路径
        import_file_path = state.get('import_file_path', '')
        file_dir = state.get('file_dir', '')

        # 2. 校验文件是否存在(非空判断)
        if not import_file_path:
            raise ValidationError("解析的文件不存在", self.name)

        # 3. 用Path标准化
        import_file_path_obj = Path(import_file_path)
        # 4. 校验文件是否存在
        if not import_file_path_obj.exists():
            raise FileProcessingError("解析的文件路径不存在", self.name)
        # 5. 判断输出目录是否为空
        if not file_dir:
            # 默认目录做兜底 降级处理
            file_dir = import_file_path_obj.parent
        # 6. 标准输出目录
        file_dir_path_obj = Path(file_dir)
        self.logger.info(f"上传文件的路径:{import_file_path}")
        self.logger.info(f"输出的目录:{file_dir}")

        # 7. 返回 输出文件以及输出目录的标准path
        return import_file_path_obj, file_dir_path_obj

    def _get_md_paths(self, import_file_path, file_dir_path) -> str:
        file_name = import_file_path.stem
        md_path = file_dir_path / file_name / "auto" / f"{file_name}.md"
        return str(md_path)

    def _execute_mineru(self, import_file_path, file_dir_path) -> int:
        """
        执行 MinerU 命令

        Args:
            import_file_path: 解析的文件路径
            file_dir_path: 解析后的文件目录

        Returns:
            命令返回码（0 表示成功）
        """
        self.log_step("step2", "执行MinerU解析PDF")

        # 1. 构建命令行
        # pipeline 使用原始推理模型, 完成pdf解析,无需使用GPU
        cmd = [
            "mineru",
            "-p",
            str(import_file_path),
            "-o",
            str(file_dir_path),
            "--source",
            "local",
            "--backend",
            "pipeline"
        ]

        self.logger.info(f"执行命令: {' '.join(cmd)}")
        process_start_time = time.time()

        # 2. 执行命令行(子进程执行命令行)
        proc = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",  # 遇到乱码时替换
            text=True,  # 输出的内容是字符串 不是字节
            encoding="utf-8",  # 用指定的中文字符集进行编解码
            bufsize=1  # 按行缓冲，只要缓冲区一行满了就输出
        )
        # 3. 获取日志信息
        for line in proc.stdout:
            self.logger.info(f"执行MinerU产生的日志：{line}")

        # 4. 等待子进程做完
        processed_code = proc.wait()
        process_end_time = time.time()

        if processed_code == 0:
            self.logger.info(
                f"MinerU成功解析PDF文件：{import_file_path.name} "
                f"耗时:{process_end_time - process_start_time:.2f}s"
            )
        else:
            self.logger.error(f"MinerU解析PDF文件：{import_file_path.name}失败")

        # 5. 返回状态码
        return processed_code

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """
        执行pdf转换
        :param state:  状态图
        :return:  更新后的状态图
        """
        # 1. 验证路径
        import_file_path_obj, file_dir_path_obj = self._validate_state_inputs_path(state)

        # 2. 利用MinerU工具解析pdf成为md
        processed_code = self._execute_mineru(import_file_path_obj, file_dir_path_obj)
        if processed_code != 0:
            raise PdfConversionError("MinerU解析PDF失败", self.name)
        # 3. 获取md的path
        md_path = self._get_md_paths(import_file_path_obj, file_dir_path_obj)
        # 4. 更新state 字典的md_path
        state['md_path'] = md_path

        # 5. 返回state
        return state

if __name__ == '__main__':
    setup_logging()
    current_node = PdfToMdNode()

    input_state = {
        "import_file_path": r"C:/Users/43779/Desktop/AIlearnrecord/shopkeeper_brain_docs/doc/万用表RS-12的使用.pdf",
        "file_dir": r"output/"
    }
    processed_result = current_node(input_state)
	#indent=4 缩进
    #ensure_ascii=False确保不出现乱码
    print(json.dumps(processed_result, indent=4, ensure_ascii=False))