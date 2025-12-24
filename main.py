from react_agent import ReActAgent

def main():
    """
    ตัวรันหลักของ ReAct Agent
    - สร้าง Agent instance
    - รับ input จากผู้ใช้
    - เรียก Agent.run() และแสดง Final Answer
    """
    # 1️⃣ สร้าง ReAct Agent
    agent = ReActAgent()

    # 2️⃣ รับ input จากผู้ใช้ (ตัวอย่าง)
    user_query = input("Query: ")

    # 3️⃣ เรียก Agent.run() → ได้ Final Answer
    final_answer = agent.run(user_query)

    # 4️⃣ แสดงผล
    print("\n=== Final Answer ===")
    print(final_answer)

if __name__ == "__main__":
    main()