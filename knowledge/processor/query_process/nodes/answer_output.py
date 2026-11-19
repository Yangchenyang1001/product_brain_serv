"""
答案生成节点

将重排后的文档作为上下文，调用 LLM 生成最终答案，并通过 SSE 流式推送给前端。
"""
import re
from typing import List, Dict, Tuple

from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.answer_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import set_task_result

# 匹配 markdown 图片语法：![描述](http://...)
# 注意：MinIO 对象名可能含中文/空格（如 "H3C LA2608室内无线网关 用户手册-6W100-整本手册"），
# 所以括号内必须允许空格，只在遇到右括号时结束，否则会一条都匹配不到
MD_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(\s*(https?://[^)]+?)\s*\)")
# 匹配答案末尾可能存在的【图片】区块（模型自己输出的）
IMAGE_BLOCK_PATTERN = re.compile(r"\n*【\s*图片\s*】[\s\S]*$")


class AnswerOutputNode(BaseNode):
    name: str = "answer_output_node"

    def _push_existing_answer(self, state: QueryGraphState):
        if not state.get("answer"):
            set_task_result(state["task_id"], "answer", state["answer"])

    @staticmethod
    def _normalize_image_url(url: str) -> str:
        """规范化图片 URL：对象名里可能带空格（如中文文档名），不编码会截断链接。"""
        return (url or "").strip().replace(" ", "%20")

    def _collect_image_urls(self, reranked_docs: List[Dict]) -> List[str]:
        """从重排文档的 content 中确定性提取 markdown 图片 URL。

        不再依赖大模型把长 URL 原样抄进答案，避免模型直接省略图片区块。
        """
        self.log_step("_collect_image_urls", "提取切片图片")
        image_urls: List[str] = []
        for doc in reranked_docs or []:
            content = doc.get("content", "") or ""
            for raw_url in MD_IMAGE_PATTERN.findall(content):
                url = self._normalize_image_url(raw_url)
                if url and url not in image_urls:
                    image_urls.append(url)
        max_images = self.config.max_answer_images
        if max_images and max_images > 0:
            image_urls = image_urls[:max_images]
        self.logger.info(f"从检索结果中提取到 {len(image_urls)} 张图片")
        return image_urls

    def _attach_images(self, state: QueryGraphState):
        """把图片 URL 以【图片】区块确定性追加到答案末尾。

        模型自己输出的图片区块/markdown 图片会先被清掉，统一由后端追加，
        保证 URL 合法（空格已编码）且格式与前端解析逻辑一致。
        """
        self.log_step("_attach_images", "追加图片区块")
        image_urls = state.get("image_urls") or []
        answer = (state.get("answer") or "").strip()

        if not image_urls:
            return
        if not answer:
            # 生成失败也给出图片，避免用户什么都看不到
            answer = "暂无文字答案，可参考以下图片："

        # 去掉模型自己写的【图片】区块和 markdown 图片，避免重复或 URL 抄错
        answer = IMAGE_BLOCK_PATTERN.sub("", answer)
        answer = MD_IMAGE_PATTERN.sub("", answer).strip()

        image_block = "【图片】\n" + "\n".join(image_urls)
        state["answer"] = f"{answer}\n\n{image_block}"

        # 流式场景：图片区块作为一次 delta 补发，前端会立即渲染
        if state.get("is_stream") and state.get("task_id"):
            push_sse_event(
                state["task_id"],
                SSEEvent.DELTA,
                {"delta": f"\n\n{image_block}"}
            )

    def _build_prompt(self, state: QueryGraphState) -> str:
        self.log_step("_build_prompt", "构建提示词")
        char_budget = self.config.max_context_chars
        # 获取问题和商品名
        question =state.get("rewritten_query") or state.get("original_query","")
        item_names = state["item_names"]

        # step3 格式化检索文档
        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )
        # step4 格式化历史对话
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )
        # 组装提示词
        return ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=", ".join(item_names),
            question=question,
        )

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        self.log_step("_format_reranked_docs", "格式化重排序文档")
        """格式化重排序文档，带字符预算控制"""
        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, 1):
            content = doc.get("content", "").strip()
            self.log_step("_format_reranked_docs", f"topN ： {content}")
            if not content:
                continue

            # 构建元信息标签
            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field, "")).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            # 添加相关性得分
            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            # 组合元信息和内容
            doc_entry = " ".join(meta_tags) + "\n" + content

            # 检查预算
            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        self.log_step("_format_chat_history", "格式化历史对话")
        """格式化历史对话"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"

            # 检查预算
            if used_chars + len(formatted_line) > char_budget:
                  break

            formatted_lines.append(formatted_line)
            used_chars += len(formatted_line) + 1
        return "\n".join(formatted_lines), char_budget - used_chars

    def _generate_answer(self, state, prompt):

        self.log_step("_generate_answer","生成答案")
        llm_client = AIClients.get_llm_openai(response_format=False)
        if llm_client is None:
            raise ValueError("LLM 客户端初始化失败")
        task_id = state["task_id"]

        if state.get("is_stream"):
            state["answer"] = self._stream_generate(llm_client, prompt, task_id)
        else:
            state["answer"] = self._invoke_generate(llm_client,prompt)
            set_task_result(task_id, "answer", state["answer"])

    def _invoke_generate(self,llm_client, prompt: str) -> str:
        """非流式生成"""
        self.log_step("_invoke_generate", "非流式生成答案")
        try:
            response = llm_client.invoke(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"生成回答出错: {e}")
            return "抱歉，生成回答时出现错误。"

    def _stream_generate(self, llm_client, prompt, task_id):
        self.log_step("_stream_generate", "流式生成答案")
        """流式生成，逐 chunk 推送 delta 事件"""
        accumulated_answer = ""
        try:
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk, "content", "") or ""
                if delta_text:
                    accumulated_answer += delta_text
                    push_sse_event(task_id, "delta", {"delta": delta_text})
        except Exception as e:
            self.logger.error(f"流式生成出错: {e}")
        return accumulated_answer

    def _write_history(self, state: QueryGraphState):
        self.log_step("_write_history", "写入历史记录")
        """将用户问题和助手回答写入 MongoDB 历史记录"""
        session_id = state["session_id"]
        rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
        item_names = state.get("item_names") or []

        try:
            # 1. 写用户问题
            save_chat_message(
                session_id=session_id,
                role="user",
                text=state["original_query"],
                rewritten_query=rewritten_query,
                item_names=item_names,
            )

            # 2. 写助手回复
            if state.get("answer"):
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=state["answer"],
                    rewritten_query=rewritten_query,
                    item_names=item_names,
                    image_urls=state.get("image_urls") or [],
                )
        except Exception as e:
            self.logger.warning(f"写入历史记录失败: {e}")
    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        1. step1 检查已有答案
        2. step2 获取查询文本和商品名
        3. step3 格式化检索文档
          a. 预算控制
        4. step4 格式化历史对话
          a. 预算控制
        5. step5 组装完整提示词
        6. step6 调用LLM大模型生成答案
        7. step7 写入历史记录
        8. step8 发送结束事件
        :param state:
        :return:
        """
        # step1 检查已有答案
        if state.get("answer"):
            self._push_existing_answer(state)

        else:
            # step5 组装完整提示词
            prompt = self._build_prompt(state)
            state["prompt"] = prompt
            # step6 调用LLM大模型生成答案
            self._generate_answer(state,prompt)
            # step6.1 从检索切片里提取图片并确定性追加到答案
            state["image_urls"] = self._collect_image_urls(state.get("reranked_docs") or [])
            self._attach_images(state)
            set_task_result(state["task_id"], "image_urls", state.get("image_urls") or [])

        # step7 写入历史记录
        self._write_history(state)

        # step8 发送结束事件
        # if state.get("is_stream"):
        #     push_sse_event(
        #         state["task_id"],
        #         SSEEvent.FINAL,
        #         {"answer": state.get("answer", "")}
        #     )

        return state
if __name__ == "__main__":
    from dotenv import load_dotenv
    import json

    load_dotenv()

    from knowledge.processor.query_process.base import setup_logging
    setup_logging()

    print("=" * 60)
    print("开始测试: 答案生成节点 (AnswerOutputNode)")
    print("=" * 60)

    # 构造模拟状态
    mock_state = {
        "task_id": "test_task_001",
        "session_id": "test_session_001",
        "is_stream": False,
        "original_query": "万用表怎么测电压？",
        "rewritten_query": "RS-12数字万用表如何测量电压？",
        "item_names": ["RS-12数字万用表"],
        "reranked_docs": [
            {
                "content": "数字万用表测量电压步骤：1. 将旋钮转到V档位；2. 黑表笔插COM孔，红表笔插V孔；3. 将表笔并联到被测点两端。",
                "source": "local",
                "chunk_id": "chunk_001",
                "title": "万用表使用手册",
                "score": 0.9234
            },
            {
                "content": "测量直流电压时需注意正负极性，红表笔接正极，黑表笔接负极。",
                "source": "web",
                "url": "https://example.com/guide",
                "title": "电压测量指南",
                "score": 0.8756
            }
        ],
        "history": [
            {"role": "user", "text": "万用表是什么？"},
            {"role": "assistant", "text": "万用表是一种多功能电子测量仪器..."}
        ],
    }

    print("【输入状态】:")
    print(f"  query: {mock_state['rewritten_query']}")
    print(f"  item_names: {mock_state['item_names']}")
    print(f"  reranked_docs: {len(mock_state['reranked_docs'])} 篇")
    print("-" * 60)

    # 执行答案生成
    node = AnswerOutputNode()
    result = node.process(mock_state)

    # 打印结果
    print("\n【生成结果】:")
    print("-" * 60)
    print(result.get("answer", "无答案"))
    print("-" * 60)

    print("\n测试完成")

