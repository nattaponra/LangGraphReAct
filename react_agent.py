from typing import List, Dict
from agent_actions import search_context, call_web_search
from constant import GOOGLE_GEMINI_API_KEY, GOOGLE_GEMINI_MODEL_NAME

import google.generativeai as genai
import json
import re
import os
from datetime import datetime

# ตั้งค่า Google Gemini API
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
gemini_client = genai.GenerativeModel(GOOGLE_GEMINI_MODEL_NAME)


class ReActAgent:
    def __init__(self, max_steps: int = 5, enable_logging: bool = True):
        self.observations: List[str] = []
        self.max_steps = max_steps
        self.enable_logging = enable_logging
        self.log_lines: List[str] = []

    # -------------------------
    # Logging helpers
    # -------------------------
    def log(self, message: str):
        if self.enable_logging:
            self.log_lines.append(message)
            print(message)

    # -------------------------
    # Save log to file (output data/debug/<file>.md)
    # -------------------------
    def save_log(self):
        output_dir = "data/debug"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{output_dir}/react_agent_log_{timestamp}.md"
        
        # สร้างโฟลเดอร์ถ้ายังไม่มี
        os.makedirs(output_dir, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(self.log_lines))

    # -------------------------
    # Reasoning step
    # -------------------------
    def reason(self, user_input: str, observations: List[str], step: int) -> Dict[str, str]:
        """
        LLM reasoning - let AI decide what to do based on the query type and current observations
        """

        # -------------------------
        # Clean ข้อมูล observations สำหรับ input ของ LLM
        # -------------------------
        # กรองเอา HTML tags ออกและไม่รวม "FINAL_STEP" 
        obs_texts = [re.sub(r"<[^>]+>", "", o) for o in observations if o != "FINAL_STEP"]
        # รวมข้อความทั้งหมดเป็นข้อความเดียว
        obs_text = " ".join(obs_texts)
        # ลบช่องว่างเกินและทำให้เป็นรูปแบบมาตรฐาน
        obs_text = re.sub(r"\s+", " ", obs_text).strip()

        # สร้าง prompt สำหรับให้ LLM ตัดสินใจ
        prompt = f"""
        You are a ReAct Agent AI assistant. Analyze the current situation and decide the next action.
        
        Current observations: {obs_text if obs_text else "No observations yet"}
        User question: {user_input}
        Current step: {step}

        Available actions:
        - 'search_context' if you need to search the internal knowledge base/RAG system for relevant information
        - 'web_search' if you need additional current/external information from the internet
        - 'final_answer' if you have sufficient information to provide a complete answer

        Decision criteria:
        - If no observations yet, consider what type of information is needed first
        - If observations exist, evaluate if they provide sufficient information
        - Consider whether the question requires real-time/current information vs historical/factual information
        - Think about the most logical sequence of actions to answer the question effectively
        
        Respond in JSON format: 
        
        Query field should contain:
        - For 'search_context': search keywords or phrases for the knowledge base
        - For 'web_search': search query for the internet
        - For 'final_answer': the original user question being answered
        """
        try:
            # เรียกใช้ Gemini AI เพื่อตัดสินใจการกระทำต่อไป
            response = gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # ใช้ความสร้างสรรค์ปานกลาง
                    max_output_tokens=500  # จำกัด token สำหรับการตอบกลับ
                )
            )

            # ดึงข้อความจาก response
            text = getattr(response, "text", None)
            if not text:
                raise ValueError("No text returned from LLM")
            # ลบ code block markers ออกจาก JSON response
            text = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()
            # แปลง JSON string เป็น dictionary
            decision = json.loads(text)
        except Exception as e:
            # หากเกิดข้อผิดพลาดให้บันทึกและใช้การตัดสินใจ default
            self.log(f"Error parsing LLM response: {e}")
            decision = {"action": "final_answer", "query": user_input}

        return decision

    # -------------------------
    # Action step
    # -------------------------
    def act(self, action_type: str, query: str, user_input: str = None) -> str:
        """
        Execute action and return observation
        """
        # ตรวจสอบประเภทของ action_type ที่ต้องการทำ
        if action_type == "search_context":
            # ค้นหาข้อมูลในฐานความรู้ภายใน (RAG system) โดยใช้คำค้นหา
            docs = search_context(query, top_k=2) 
            if docs:
                # สร้างสรุปเอกสารที่พบจากการค้นหา
                doc_summaries = [f"Title: {doc['title']}, Content: {doc['content']}" for doc in docs]
                obs = f"Found {len(docs)} relevant document(s) in knowledge base: " + "; ".join(doc_summaries)
            else:
                # ไม่พบข้อมูลที่เกี่ยวข้องในฐานความรู้ภายใน (RAG system)
                obs = "No relevant information found in the internal knowledge base"
        elif action_type == "web_search":
            # ค้นหาข้อมูลจากอินเทอร์เน็ต โดยใช้คำค้นหา
            obs = call_web_search(query)
        elif action_type == "final_answer":
            # สร้างคำตอบสุดท้ายโดยใช้ข้อมูลทั้งหมดที่รวบรวมได้
            obs = self.generate_final_answer(user_input or query)
        else:
            # ประเภทการกระทำที่ไม่รู้จัก
            obs = f"Unknown action: {action_type}"

        # บันทึกผลการสังเกตลงในรายการ observations
        self.observations.append(obs)
        return obs

    # -------------------------
    # Final answer generation
    # -------------------------
    def generate_final_answer(self, user_input: str) -> str:
        """
        Generate final answer based on all current observations (true ReAct pattern)
        """
        # Clean ข้อมูล observations โดยลบ HTML tags ออก
        obs_texts = [re.sub(r"<[^>]+>", "", o) for o in self.observations]
        # รวมข้อความทั้งหมดเป็นข้อความเดียว
        obs_text = " ".join(obs_texts)
        # ลบช่องว่างที่ไม่จำเป็นและ normalize ข้อความ
        obs_text = re.sub(r"\s+", " ", obs_text).strip()

        # ตรวจสอบว่ามีข้อมูลจาก observations หรือไม่
        if not obs_text or obs_text.strip() == "":
            # กรณีไม่มีข้อมูลจาก observations ให้ตอบคำถามโดยตรง
            prompt = f"""
            Please provide a helpful, direct answer to this user question:

            {user_input}

            Provide a clear, informative response in 2-3 sentences.
            """
        else:
            # กรณีมีข้อมูลจาก observations ให้ใช้ข้อมูลนั้นในการตอบ
            prompt = f"""
            Based on the information gathered, provide a clear final answer:

            Question: {user_input}
            Information gathered: {obs_text}

            Provide a complete answer in 2-3 clear sentences.
            """

        try:
            # เรียกใช้ Gemini AI เพื่อสร้างคำตอบสุดท้าย
            response = gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # ใช้ temperature ต่ำเพื่อความแม่นยำ
                    max_output_tokens=2000  # จำกัดจำนวน token สูงสุด
                )
            )
            
            # ดึงข้อความคำตอบจาก response
            final_answer = getattr(response, "text", None)
            if not final_answer:
                raise ValueError("No text returned from LLM")
            
            # ส่งคืนคำตอบพร้อมกับ prefix "FINAL_ANSWER: "
            return f"FINAL_ANSWER: {final_answer}"
                
        except Exception as e:
            # จัดการข้อผิดพลาดและบันทึก log
            self.log(f"Error generating final answer: {type(e).__name__}: {e}")
            return f"FINAL_ANSWER: Unable to generate final answer due to error: {e}"



    # -------------------------
    # Main agent flow
    # -------------------------
    def run(self, user_input: str) -> str:
        # เริ่มต้นการทำงานใหม่โดยเคลียร์ข้อมูลเก่า
        self.observations = []
        self.log_lines = []
        # สร้าง log header สำหรับการทำงานครั้งนี้
        self.log(f"# ReAct Agent Log")
        self.log(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"**User Query:** {user_input}\n")
        self.log(f"## 🚀 Starting ReAct Agent Process\nMaximum Steps: {self.max_steps}\n")
        
        # วนลูปทำงานตามจำนวนขั้นตอนสูงสุดที่กำหนด
        for step in range(1, self.max_steps + 1):
            # ให้ LLM ตัดสินใจการกระทำต่อไปตามสถานการณ์ปัจจุบัน
            decision = self.reason(user_input, self.observations, step)
            action_type = decision.get("action", "final_answer")
            query = decision.get("query", user_input)
            
            # บันทึก log สำหรับขั้นตอนนี้
            self.log(f"# Step {step}/{self.max_steps}")
            self.log(f"**Thought:** LLM decided to perform '{action_type}' action")
            self.log(f"**Action:** {action_type}")
            self.log(f"**Query:** {query}")

            # ดำเนินการตามที่ LLM ตัดสินใจและรับผลการสังเกต
            obs = self.act(action_type, query, user_input)
            self.log(f"**Observation:** {obs}\n")

            # ตรวจสอบว่าได้คำตอบสุดท้ายแล้วหรือไม่
            if action_type == "final_answer":
                # ดึงคำตอบสุดท้ายจากผลการสังเกต (รูปแบบ ReAct แท้จริง)
                if obs.startswith("FINAL_ANSWER: "):
                    final_answer = obs[14:]  # ลบ prefix "FINAL_ANSWER: " ออก
                    self.log(f"=== Final Answer ===\n{final_answer}\n")
                    self.save_log()
                    return final_answer
                break

        # กรณีที่จบลูปโดยไม่มีการทำ final_answer action (fallback)
        self.log("⚠️  Agent reached maximum steps without final_answer action")
        self.log("🔄  Forcing final answer generation...\n")
        
        # บังคับสร้างคำตอบสุดท้าย
        fallback_obs = self.generate_final_answer(user_input)
        if fallback_obs.startswith("FINAL_ANSWER: "):
            fallback_answer = fallback_obs[14:]  # ลบ prefix "FINAL_ANSWER: " ออก
            self.log(f"=== Fallback Answer ===\n{fallback_answer}\n")
            self.save_log()
            return fallback_answer
        else:
            # กรณีพิเศษ: แม้แต่ generate_final_answer ก็ล้มเหลว
            error_msg = f"Unable to generate answer after {self.max_steps} steps"
            self.log(f"❌ {error_msg}")
            self.save_log()
            return error_msg


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    agent = ReActAgent()
    
    # Test with a question that might benefit from internal knowledge first
    question = "Employee Benefits"
    print("Testing with question:", question)
    answer = agent.run(question)