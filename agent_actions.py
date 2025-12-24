# agent_actions.py
from typing import Dict, List
from rag import rag_search_context  # เรียกฟังก์ชันจาก rag.py
from ddgs import DDGS

def search_context(query: str, top_k: int = 1) -> List[Dict[str, str]]:
    """
    Action: ค้นหาข้อมูลจาก Mock RAG Knowledge Base
    - query: คำถามหรือ keyword ของผู้ใช้
    - top_k: จำนวน document ที่ต้องการคืนค่า
    Return:
        list ของ document ที่เกี่ยวข้อง [{'title': ..., 'content': ...}]
    Note:
        ฟังก์ชันนี้ใช้ RAG system (mock) เพื่อจำลอง retrieval
    """
    # เรียกใช้ฟังก์ชันค้นหาจาก RAG system และส่งคืนผลลัพธ์
    return rag_search_context(query, top_k=top_k)


def call_web_search(query: str, max_results: int = 2) -> str:
    """
    Action: Web search using DuckDuckGo
    - query: search query string
    - max_results: number of search results to return
    Return: plain text summary combining top results
    """
    try:
        # ใช้ DuckDuckGo API เพื่อค้นหาข้อมูลจากอินเทอร์เน็ต
        with DDGS() as ddgs:
            # ค้นหาและแปลงผลลัพธ์เป็น list
            results = list(ddgs.text(query, max_results=max_results))
        
        # ตรวจสอบว่าพบข้อมูลหรือไม่
        if not results:
            return "No relevant information found on the web."
        
        # รวมผลการค้นหาหลายๆ รายการเป็นข้อความเดียว
        combined = " | ".join([r['body'] for r in results if 'body' in r])
        return combined
    except Exception as e:
        # จัดการข้อผิดพลาดที่อาจเกิดขึ้นระหว่างการค้นหา
        return f"Error during web search: {e}"