"""
Web 搜索节点

通过 MCP 协议调用外部 Web 搜索服务，获取互联网上的补充信息，弥补知识库中可能缺失的内容。
"""
import asyncio
import json
from typing import Tuple, List, Dict, Any

from agents.mcp import MCPServerSse, MCPServerStreamableHttp

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState


class McpSearchNode(BaseNode):
    name: str = "web_search_mcp_node"

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验输入参数"""

        # 1. 获取 state 的 rewritten_query
        rewritten_query = state.get('rewritten_query', "")

        # 2. 获取 state 的 item_names
        item_names = state.get('item_names', "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(
                node_name=self.name,
                field_name="rewritten_query",
                expected_type=str
            )

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(
                node_name=self.name,
                field_name="item_names",
                expected_type=list
            )

        # 4. 返回
        return rewritten_query, item_names

    async def _create_execute_web_search(self, validated_rewritten_query: str) -> List[Dict[str, Any]]:
        """创建并执行Web搜索"""
        # 2. step2 创建MCP客户端
        async  with MCPServerStreamableHttp(
                name="联网搜索",
                params={
                    "url": self.config.mcp_dashscope_base_url,  # MCP 服务端点
                    "headers": {"Authorization": f"Bearer {self.config.openai_api_key}"},  # 认证头需带 Bearer 前缀
                    "timeout": 300,  # 请求超时时间（秒）
                    "terminate_on_close": True,  # 关闭时终止连接
                },
                max_retry_attempts=2,  # 最大重试次数
                cache_tools_list=True,  # 缓存工具列表，避免重复请求
        ) as mcp_client:
            # try:
            # 3. step3 建立MCP连接
            self.logger.info(f"正在连接 MCP SSE 端点: {self.config.mcp_dashscope_base_url}")
            await mcp_client.connect()
            # 4. step4 调用搜索工具
            execute_tool_result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": validated_rewritten_query, "count": 3}
            )
            # 5. step5 解析工具执行完的结果
            if not execute_tool_result or not execute_tool_result.content[0]:
                return []
            text_content_text: str = execute_tool_result.content[0].text
            if not text_content_text:
                return []

            try:
                text_content_text: Dict[str, Any] = json.loads(text_content_text)

                pages = text_content_text.get('pages', "")
                if not pages:
                    return []

                search_result = []
                # b) 遍历得到每一个结果
                for page in pages:
                    snippet = page.get('snippet', "").strip()
                    title = page.get('title', "").strip()
                    url = page.get('url', "").strip()
                    search_result.append({"snippet": snippet, "title": title, "url": url})

                return search_result
            except Exception as e:
                self.logger.error("反序列MCP结果失败")
                return []
        # finally:
        #     await mcp_client.cleanup()  # 关闭连接

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        业务流程
        1. step1 参数校验
        2. step2 创建MCP客户端
        3. step3 建立MCP连接
        4. step4 调用搜索工具
        5. step5 解析搜索结果
        6. step6 更新并返回结果
        :param state:
        :return:
        """
        # 1. step1 参数校验
        rewritten_query, item_names = self._validate_query_inputs(state)

        mcp_result = asyncio.run(self._create_execute_web_search(rewritten_query))

        # 6. step6 更新并返回结果
        if not mcp_result:
            return {"web_search_docs": []}

        # 更新 state 的 web_search_docs
        return {"web_search_docs": mcp_result}


if __name__ == '__main__':
    state = {
        "rewritten_query": "今天的小米汽车的股价是多少",
        "item_names": ["RS-12 数字万用表"]
    }

    mcp_search = McpSearchNode()
    result = mcp_search.process(state)
    print(result)

    for r in result.get('web_search_docs') or []:
        print(json.dumps(r, ensure_ascii=False, indent=2))
