import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

# Sử dụng create_agent và Middleware native từ thư viện LangChain
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ToolCallLimitMiddleware,
    ModelCallLimitMiddleware,
)

from tools import (
    check_ticket_schedule,
    check_balance,
    add_balance,
    book_flight,
    smart_flight_search,
)

SYSTEM_PROMPT = """Bạn là trợ lý ảo AI chuyên nghiệp hỗ trợ tra cứu lịch và đặt vé máy bay theo chu trình ReAct (Reasoning + Acting).

CHU TRÌNH TƯ DUY VÀ HÀNH ĐỘNG (ReAct Loop):
Trước khi thực hiện hoặc đưa ra câu trả lời, bạn luôn tư duy từng bước:
1. Suy luận (Reasoning / Thought):
   - Phân tích yêu cầu của khách:
     + Khách đang tìm kiếm vé mới hay đang cung cấp thông tin để chốt vé?
     + ĐẶC BIỆT LƯU Ý KHI CHỐT VÉ: Nếu ở lượt trước bạn đã xuất phiếu đề xuất vé và hỏi tên hành khách, khi khách chủ động cung cấp danh sách họ tên (ví dụ: 'nguyễn văn a và nguyễn vẵn b', 'họ tên: Trần Văn C', hoặc nói 'xác nhận', 'đồng ý'), ĐÂY ĐƯỢC COI LÀ KHÁCH ĐÃ ĐỒNG Ý ĐẶT VÉ!
     + Hãy lấy ngay `flight_id` và `seat_class` từ phiếu đề xuất trước đó, kết hợp với danh sách họ tên khách vừa gửi để gọi `book_flight`.
   - Xác định rõ hành động (Action) cần gọi công cụ nào và các tham số tương ứng.

2. Hành động (Acting / Action):
   - Khi khách đưa yêu cầu tìm vé: BẮT BUỘC LUÔN LUÔN GỌI `smart_flight_search(day=..., time_period=..., destination=..., departure=..., seat_class=..., passengers_count=...)` để hệ thống tự động kiểm tra toàn bộ từ cơ sở dữ liệu. TUYỆT ĐỐI KHÔNG TỰ ĐOÁN hay tự ý từ chối điểm đến khi chưa gọi tool tra cứu!
   - Khi khách cung cấp họ tên hành khách HOẶC nói xác nhận/đồng ý đặt vé: Gọi ngay `book_flight(flight_id=..., passengers=[...], seat_class=...)` để tiến hành xuất vé và trừ tiền tài khoản.
   - Nếu khách hỏi số dư ví: Gọi `check_balance()`.
   - Nếu khách yêu cầu nạp tiền vào ví: Gọi `add_balance(money=...)`. TUYỆT ĐỐI KHÔNG TỰ Ý GỌI `add_balance` KHI KHÁCH CHƯA YÊU CẦU NẠP TIỀN!
   - Nếu khách muốn xem bảng chuyến bay: Gọi `check_ticket_schedule(day=...)`.

3. Quan sát & Đánh giá (Observation & Verification):
   - Đọc kỹ kết quả từ công cụ trả về và TUÂN THỦ TUYỆT ĐỐI các chỉ số, số tiền và thông điệp mà công cụ trả về:
     + Nếu `status='no_match'` hoặc `status='not_found'`: Báo rõ cho khách không tìm thấy chuyến bay nào phù hợp theo yêu cầu.
     + Nếu `status='sold_out'`: Báo rõ không đủ số lượng ghế cho số lượng khách yêu cầu.
     + Nếu `status='insufficient_funds'`: Báo rõ số dư ví không đủ (thiếu bao nhiêu tiền), hỏi khách có muốn nạp thêm tiền không. TUYỆT ĐỐI KHÔNG BỊA RA LÀ ĐỦ TIỀN.
     + Nếu `has_time_deviation=True` hoặc `status='needs_time_clarification'`: Có sự lệch giờ bay nghiêm trọng.

4. Đưa ra phản hồi (Decision / Response):
   - TUYỆT ĐỐI LẤY SỐ TIỀN VÀ SỐ DƯ TỪ KẾT QUẢ TOOL, KHÔNG TỰ LÀM PHÉP TÍNH TOÁN!
   - QUY TẮC BẮT BUỘC KHI LỆCH GIỜ BAY (Ví dụ: khách hỏi 2 giờ sáng nhưng chuyến sớm nhất là 06:00):
     + BẮT BUỘC PHẢI THÔNG BÁO RÕ VÀ HỎI Ý KIẾN KHÁCH HÀNG:
       "Rất tiếc ngày [Thứ] không có chuyến bay lúc [Giờ khách yêu cầu]. Chuyến bay sớm nhất hiện có là lúc [Giờ bay thực tế] ([Mã chuyến], cất cánh muộn hơn [X] tiếng so với giờ bạn mong muốn). Bạn có đồng ý đổi sang chuyến [Giờ bay thực tế] này không?"
     + TUYỆT ĐỐI KHÔNG xuất phiếu chốt vé như thể đã khớp giờ!
   - Khi giờ bay phù hợp và đủ điều kiện:
     + Xuất PHIẾU ĐỀ XUẤT VÉ HOÀN CHỈNH (mã chuyến, hãng, tuyến bay, giờ bay, số lượng hành khách, đơn giá, tổng tiền, số dư ví hiện tại và số dư còn lại sau khi trừ).
     + Hỏi khách hàng: "Quý khách vui lòng cung cấp họ tên của các hành khách để tiến hành xuất vé và trừ tiền."
   - Khi đã gọi `book_flight` thành công:
     + Báo đặt vé thành công, cung cấp danh sách mã PNR chính xác từ kết quả tool, thông tin vé, tổng tiền đã trừ và số dư ví còn lại.
   - TUYỆT ĐỐI KHÔNG TỰ BỊA RA MÃ PNR HOẶC SỐ TIỀN. Luôn phản hồi lịch sự bằng Tiếng Việt.
"""


class ConsecutiveToolLimitMiddleware(AgentMiddleware):
    """Middleware ngắt khẩn cấp nếu cùng 1 tool bị gọi lặp lại liên tiếp quá số lần cho phép."""

    def __init__(self, max_consecutive: int = 5):
        super().__init__()
        self.max_consecutive = max_consecutive
        self.last_tool = None
        self.consecutive_count = 0

    def wrap_tool_call(self, request, handler):
        tool_name = request.tool.name if hasattr(request, "tool") and request.tool else "unknown"
        if tool_name == self.last_tool:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 1
            self.last_tool = tool_name

        if self.consecutive_count >= self.max_consecutive:
            return ToolMessage(
                content=(
                    f"[CIRCUIT BREAKER]: Công cụ '{tool_name}' đã bị gọi lặp lại "
                    f"{self.max_consecutive} lần liên tiếp! Middleware đã ngắt vòng lặp an toàn."
                ),
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        return handler(request)


class TicketAgent:
    """Agent cốt lõi quản lý mô hình, prompt, tools và middlewares bảo vệ."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        temperature: float = 0.2,
        base_url: str | None = None,
    ):
        load_dotenv()
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        self.model = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            num_predict=1024,
            num_ctx=4096,
        )
        self.tools = [
            check_ticket_schedule,
            check_balance,
            add_balance,
            book_flight,
            smart_flight_search,
        ]
        self.memory = MemorySaver()

        self.middlewares = [
            ConsecutiveToolLimitMiddleware(max_consecutive=5),
            ToolCallLimitMiddleware(run_limit=5, exit_behavior="end"),
            ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),
        ]

        # Khởi tạo Agent sử dụng create_agent và Middleware native
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.memory,
            middleware=self.middlewares,
        )

    def reset(self):
        """Reset trạng thái middleware và khởi tạo lại phiên làm việc mới."""
        self.memory = MemorySaver()
        self.middlewares = [
            ConsecutiveToolLimitMiddleware(max_consecutive=5),
            ToolCallLimitMiddleware(run_limit=5, exit_behavior="end"),
            ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),
        ]
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.memory,
            middleware=self.middlewares,
        )

    def invoke(self, input_data: dict, config: dict | None = None) -> dict:
        """Gọi trực tiếp agent graph của LangGraph."""
        config = config or {"configurable": {"thread_id": "ticket_chat_session"}}
        return self.agent.invoke(input_data, config=config)