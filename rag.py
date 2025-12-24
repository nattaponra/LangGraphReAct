from typing import List, Dict
import re
def rag_load_context(file_path: str) -> List[Dict[str, str]]:
    """
    อ่านไฟล์ Markdown และแยกเป็น sections
    แต่ละ section จะคืนค่าเป็น dict {'title': ..., 'content': ...}
    Assumption: แต่ละ section เริ่มด้วย ## และข้อมูลแยกด้วย ---
    """
    documents = []
    current_doc = None

    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            
            # ข้าม header หลัก "# Knowledge Base"
            if line.startswith("# "):
                continue
                
            # เริ่ม section ใหม่ที่ขึ้นต้นด้วย ##
            if line.startswith("## "):
                if current_doc:
                    # ล้าง --- ออกจากท้าย content
                    current_doc["content"] = current_doc["content"].strip()
                    documents.append(current_doc)
                
                title = line[3:].strip()  # เอา "## " ออก
                current_doc = {"title": title, "content": ""}
            
            # ข้ามเส้น separator ---
            elif line == "---":
                continue
            
            # เพิ่มเนื้อหาถ้ามี section ปัจจุบัน และไม่ใช่บรรทัดว่าง
            elif current_doc is not None and line:
                current_doc["content"] += line + "\n"

    # append last section
    if current_doc:
        current_doc["content"] = current_doc["content"].strip()
        documents.append(current_doc)

    return documents

def rag_search_context(query: str, top_k: int = 1) -> List[Dict[str, str]]:
    """
    ค้นหา context จาก Mock RAG
    - query: คำค้นหา (string)
    - top_k: จำนวน document ที่ต้องการคืนค่า
    คืนค่าเป็น list ของ document ที่เกี่ยวข้องที่สุด
    """
    
    # โหลดเอกสารทั้งหมดจาก Mock RAG database
    rag_docs = rag_load_context("data/mock_rag_document.md")
    
    # แปลงคำค้นหาเป็นตัวพิมพ์เล็กเพื่อการเปรียบเทียบที่ไม่สนใจตัวพิมพ์ใหญ่เล็ก    
    query_lower = query.lower()
    scored_docs = []

    # วนลูปผ่านเอกสารทั้งหมดเพื่อหาความเกี่ยวข้อง
    for doc in rag_docs:
        # นับจำนวนครั้งที่คำค้นหาปรากฏใน title และ content (ไม่สนใจตัวพิมพ์ใหญ่เล็ก)
        title_score = len(re.findall(re.escape(query_lower), doc["title"].lower()))
        content_score = len(re.findall(re.escape(query_lower), doc["content"].lower()))
        # คำนวณคะแนนรวม โดยให้น้ำหนัก title มากกว่า content เป็น 2 เท่า
        score = title_score * 2 + content_score
        
        # เก็บเฉพาะเอกสารที่มีคะแนนมากกว่า 0 (มีความเกี่ยวข้อง)
        if score > 0:
            scored_docs.append((score, doc))

    # เรียงลำดับเอกสารตามคะแนนจากมากไปน้อย
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # คืนค่าเอกสาร top_k อันดับแรกที่เกี่ยวข้องที่สุด
    return [doc for _, doc in scored_docs[:top_k]]


# ------------------------
# ตัวอย่างการใช้งาน
# ------------------------
if __name__ == "__main__":
    # ค้นหา context
    query = "Amazon"
    results = rag_search_context(query, top_k=2)

    for r in results:
        print(f"Title: {r['title']}")
        print(f"Content preview: {r['content'][:200]}...\n")

