import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import random
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import datetime
import json
import os
import PyPDF2
import glob
import tiktoken
import re

# 設置日誌系統
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 從環境變量獲取 OpenAI API 金鑰
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("未提供有效的 API 金鑰。請設置 OPENAI_API_KEY 環境變量。")
    st.stop()

# 初始化 OpenAI 客戶端
client = OpenAI(api_key=api_key)

# 初始化 tiktoken 編碼器
encoding = tiktoken.encoding_for_model("gpt-4o")

# 初始化 SentenceTransformer 模型（使用多語言模型）
try:
    sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    logging.info("成功加載 SentenceTransformer 模型")
except Exception as e:
    logging.error(f"加載 SentenceTransformer 模型時出錯: {e}")
    st.error("加載 SentenceTransformer 模型失敗。請檢查您的網絡連接並重試。")
    st.stop()

# 改進：從PDF檔案讀取內容的函數
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# 改進：文本預處理函數
def preprocess_text(text):
    # 移除多餘的空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    # 這裡可以添加更多的預處理步驟，如移除特殊字符、標準化等
    return text

# 改進：將文本分割成更小的段落
def split_into_chunks(text, max_chunk_size=1000):
    chunks = []
    current_chunk = ""
    for sentence in re.split(r'(?<=[。！？.])', text):
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# 改進：建立知識庫的函數
def build_knowledge_base(pdf_directory):
    knowledge_base = []
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        text = preprocess_text(text)
        chunks = split_into_chunks(text)
        knowledge_base.extend(chunks)
    return knowledge_base

# 修改：定義知識庫
pdf_directory = "data"  # 替換為實際的PDF檔案目錄
knowledge_base = build_knowledge_base(pdf_directory)

# 將知識庫轉換為嵌入向量
try:
    knowledge_embeddings = sentence_model.encode(knowledge_base)
    logging.info("成功編碼知識庫")
except Exception as e:
    logging.error(f"編碼知識庫時出錯: {e}")
    st.error("編碼知識庫失敗。請稍後再試。")
    st.stop()

# 主動學習和知識庫更新系統
class ActiveLearningSystem:
    def __init__(self, knowledge_base, confidence_threshold=0.7):
        self.knowledge_base = knowledge_base
        self.confidence_threshold = confidence_threshold
        self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        self.uncertain_queries = []

    def evaluate_certainty(self, query, response):
        return len(response) > 50

    def add_uncertain_query(self, query, response):
        self.uncertain_queries.append((query, response))

    def cluster_uncertain_queries(self, n_clusters=5):
        if len(self.uncertain_queries) < n_clusters:
            return self.uncertain_queries

        queries = [q for q, _ in self.uncertain_queries]
        embeddings = self.sentence_model.encode(queries)

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(embeddings)

        centers = kmeans.cluster_centers_
        representative_queries = []

        for i in range(n_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            center_embedding = centers[i]
            distances = np.linalg.norm(embeddings[cluster_indices] - center_embedding, axis=1)
            representative_index = cluster_indices[np.argmin(distances)]
            representative_queries.append(self.uncertain_queries[representative_index])

        return representative_queries

    def update_knowledge_base(self, new_entries):
        self.knowledge_base.extend(new_entries)
        global knowledge_embeddings
        knowledge_embeddings = sentence_model.encode(self.knowledge_base)

# 初始化主動學習系統
active_learning_system = ActiveLearningSystem(knowledge_base)

# 報告和分析系統
class AnalyticsSystem:
    def __init__(self):
        self.data_file = 'analytics_data.json'
        self.load_data()

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.queries = data.get('queries', [])
                self.response_times = data.get('response_times', [])
                self.topics = data.get('topics', [])
        else:
            self.queries = []
            self.response_times = []
            self.topics = []

    def save_data(self):
        data = {
            'queries': self.queries,
            'response_times': self.response_times,
            'topics': self.topics
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f)

    def log_interaction(self, query, response_time, topic):
        self.queries.append(query)
        self.response_times.append(response_time)
        self.topics.append(topic)
        self.save_data()

    def generate_report(self, start_date, end_date):
        if not self.queries:  # 檢查是否有數據
            return "沒有可用於報告的數據。", None

        df = pd.DataFrame({
            'query': self.queries,
            'response_time': self.response_times,
            'topic': self.topics,
            'timestamp': pd.date_range(start=start_date, periods=len(self.queries), freq='H')
        })

        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        df = df.loc[mask]

        if df.empty:  # 檢查過濾後的數據框是否為空
            return "選定日期範圍內沒有可用的數據。", None

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # 每日查詢量
        df.groupby(df['timestamp'].dt.date).size().plot(kind='line', ax=axs[0, 0])
        axs[0, 0].set_title('每日查詢量')
        axs[0, 0].set_xlabel('日期')
        axs[0, 0].set_ylabel('查詢次數')

        # 回應時間分佈
        df['response_time'].hist(bins=20, ax=axs[0, 1])
        axs[0, 1].set_title('回應時間分佈')
        axs[0, 1].set_xlabel('回應時間（秒）')
        axs[0, 1].set_ylabel('頻率')

        # 熱門主題
        topic_counts = Counter(df['topic']).most_common(5)
        if topic_counts:  # 確保有主題數據
            topics, counts = zip(*topic_counts)
            axs[1, 0].bar(topics, counts)
            axs[1, 0].set_title('前5大熱門主題')
            axs[1, 0].set_xlabel('主題')
            axs[1, 0].set_ylabel('次數')
            plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha='right')
        else:
            axs[1, 0].text(0.5, 0.5, '無主題數據', ha='center', va='center')

        # 清空未使用的子圖
        axs[1, 1].axis('off')

        plt.tight_layout()

        # 生成報告文字
        total_queries = len(df)
        avg_response_time = df['response_time'].mean() if not df['response_time'].empty else 0
        report = f"""
        分析報告（{start_date} 至 {end_date}）

        1. 總查詢次數：{total_queries}
        2. 平均回應時間：{avg_response_time:.2f} 秒
        """

        if topic_counts:
            report += "3. 前5大熱門主題：\n"
            report += ", ".join([f"{topic} ({count}次)" for topic, count in topic_counts])

        return report, fig

# 初始化分析系統
analytics_system = AnalyticsSystem()

# 從知識庫中檢索相關內容
def retrieve_relevant_context(query, top_k=5, max_tokens=3000):
    try:
        query_embedding = sentence_model.encode([query])
        similarities = cosine_similarity(query_embedding, knowledge_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_contexts = []
        total_tokens = 0
        for idx in top_indices:
            context = knowledge_base[idx]
            context_tokens = len(encoding.encode(context))
            if total_tokens + context_tokens > max_tokens:
                break
            relevant_contexts.append(context)
            total_tokens += context_tokens
        
        return relevant_contexts
    except Exception as e:
        logging.error(f"檢索相關內容時出錯: {e}")
        return []

# 生成回應的主要函數
def generate_response(query):
    start_time = datetime.datetime.now()
    try:
        relevant_context = retrieve_relevant_context(query)
        context_text = "\n".join(relevant_context)

        messages = [
            {"role": "system", "content": "你是一位專業的客戶服務助理，專門回答關於食品營養成分的問題。請仔細閱讀提供的上下文，並使用這些信息來回答用戶的問題。如果上下文中沒有直接相關的信息，請基於你的專業知識提供最相關的回答，但要明確指出這可能不是來自官方資料。"},
            {"role": "user", "content": f"基於以下信息回答問題：\n\n上下文：{context_text}\n\n問題：{query}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",  # 使用 GPT-4 模型
            messages=messages,
            max_tokens=500,  # 限制回應的長度
            n=1,
            stop=None,
            temperature=0.7,
        )

        raw_response = response.choices[0].message.content.strip()

        end_time = datetime.datetime.now()
        response_time = (end_time - start_time).total_seconds()

        topic = "食品營養"  # 根據實際情況調整

        analytics_system.log_interaction(query, response_time, topic)

        return raw_response
    except Exception as e:
        logging.error(f"生成回應時出錯: {e}")
        return f"抱歉，生成回應時發生錯誤: {str(e)}"

# Streamlit 應用
def main():
    st.title("進階 RAG 增強客戶服務助理")
    st.write("請詢問任何與我們服務相關的問題，或輸入「生成報告」以查看分析結果！")

    # 初始化聊天歷史
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # 顯示聊天歷史
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 獲取用戶輸入
    user_input = st.chat_input("在這裡輸入您的訊息...")

    if user_input:
        # 添加用戶消息到歷史
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # 在聊天界面顯示用戶消息
        with st.chat_message("user"):
            st.markdown(user_input)

        # 生成回應
        if user_input.lower() == "生成報告":
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=30)
            report, fig = analytics_system.generate_report(start_date, end_date)
            
            with st.chat_message("assistant"):
                if report == "沒有可用於報告的數據。":
                    st.markdown("目前還沒有可用於生成報告的數據。先與我聊天來生成一些數據吧！")
                else:
                    st.markdown(f"以下是最新的分析報告：\n\n{report}")
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.write("沒有可用於生成圖表的數據。")
            
            # 添加助手回應到歷史
            st.session_state.chat_history.append({"role": "assistant", "content": f"以下是最新的分析報告：\n\n{report}"})
        else:
            response = generate_response(user_input)
            
            # 顯示助手回應
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # 添加助手回應到歷史
            st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()