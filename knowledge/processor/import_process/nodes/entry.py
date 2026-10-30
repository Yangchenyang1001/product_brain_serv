"""
入口节点

导入流程的第一个节点，负责接收上传的文件，初始化状态并进行基本校验（文件格式、大小等）。
"""
import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    """
    入口节点
    根据输入文件的扩展名设置相应的处理标志
    决定后续流程走PDF转换分支还是直接处理MD分支
    """
    name: str = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """
        处理文件类型的检测
        :param state:  ImportGraphState
        :return: ImportGraphState
        """
        # 1. 获取导入的文件路径以及文件名所在目录
        self.log_step("step1","[获取文件路径]")
        file_dir = state.get('file_dir')
        import_file_path = state.get('import_file_path')

        # 2. 简单校验一下文件  文件路径以及所在目录
        self.log_step("step2","[检测文件路径]")
        if not file_dir or not import_file_path:
            raise ValidationError("文件路径或所在目录不存在",self.name)

        #3. 使用标准的path对象操作文件逻辑 并获取文件后缀
        self.log_step("step3","[文件格式解析并判断]")
        path = Path(import_file_path)
        #4. 获取文件后缀 并转换成小写
        suffix_lower = path.suffix.lower()


        #5. 判断文件的后缀
        if suffix_lower == ".pdf":
            state["is_pdf_read_enabled"]=True
            state["is_md_read_enabled"] = False
            state["pdf_path"]=import_file_path
        elif suffix_lower == ".md":
            state["is_md_read_enabled"] = True
            state["is_pdf_read_enabled"] = False
            state["md_path"] = import_file_path
        else:
            self.logger.debug(f"文件类型{suffix_lower}不支持")
            raise ValidationError("文件格式不支持",self.name)

        file_title = path.stem
        state["file_title"] = file_title
        self.logger.debug(f"文件标题{file_title}")

        return state

if __name__ == '__main__':
    setup_logging()

    # 方式：直接实例该节点对象，调用 process 方法
    # 1. 构建该节点需要的 state
    test_entry_state = {
        "file_dir": r"D:\test_data",
        "import_file_path": r"D:\test_data\万用表的使用.pdf"
    }
    # 2. 实例 EntryNode 节点
    entry_node = EntryNode()
    # 3. 调用 process 方法
    processed_state = entry_node(test_entry_state)

    # 序列化打印
    print(json.dumps(processed_state, ensure_ascii=False, indent=4))