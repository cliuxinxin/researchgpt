from flask import Flask, request, render_template
from io import BytesIO
from PyPDF2 import PdfReader
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import requests
from flask_cors import CORS
from apikey import OPENAI_API_KEY

app = Flask(__name__)
CORS(app)


class Chatbot():
    
    def parse_paper(self, pdf):
        print("解析论文")
        number_of_pages = len(pdf.pages)
        print(f"总页数: {number_of_pages}")
        paper_text = []
        for i in range(number_of_pages):
            page = pdf.pages[i]
            page_text = []

            def visitor_body(text, cm, tm, fontDict, fontSize):
                x = tm[4]
                y = tm[5]
                # 忽略页眉页脚
                if (y > 50 and y < 720) and (len(text.strip()) > 1):
                    page_text.append({
                    'fontsize': fontSize,
                    'text': text.strip().replace('\x03', ''),
                    'x': x,
                    'y': y
                    })

            _ = page.extract_text(visitor_text=visitor_body)

            blob_font_size = None
            blob_text = ''
            processed_text = []

            for t in page_text:
                if t['fontsize'] == blob_font_size:
                    blob_text += f" {t['text']}"
                    if len(blob_text) >= 2000:
                        processed_text.append({
                            'fontsize': blob_font_size,
                            'text': blob_text,
                            'page': i
                        })
                        blob_font_size = None
                        blob_text = ''
                else:
                    if blob_font_size is not None and len(blob_text) >= 1:
                        processed_text.append({
                            'fontsize': blob_font_size,
                            'text': blob_text,
                            'page': i
                        })
                    blob_font_size = t['fontsize']
                    blob_text = t['text']
                paper_text += processed_text
        print("完成解析论文")
        return paper_text

    def paper_df(self, pdf):
        print('创建数据框')
        filtered_pdf= []
        for row in pdf:
            if len(row['text']) < 30:
                continue
            filtered_pdf.append(row)
        df = pd.DataFrame(filtered_pdf)
        df = df.drop_duplicates(subset=['text', 'page'], keep='first')
        df['length'] = df['text'].apply(lambda x: len(x))
        print('完成创建数据框')
        return df

    def calculate_embeddings(self, df):
        print('计算嵌入')
        openai.api_key = OPENAI_API_KEY
        embedding_model = "text-embedding-ada-002"
        embeddings = df.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
        df["embeddings"] = embeddings
        print('完成计算嵌入')
        return df

    def search_embeddings(self, df, query, n=3, pprint=True):
        query_embedding = get_embedding(
            query,
            engine="text-embedding-ada-002"
        )
        df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))
        
        results = df.sort_values("similarity", ascending=False, ignore_index=True)
        results = results.head(n)
        global sources 
        sources = []
        for i in range(n):
            sources.append({'第'+str(results.iloc[i]['page'])+'页': results.iloc[i]['text'][:150]+'...'})
        print(sources)
        return results.head(n)


    def create_prompt(self, df, user_input):
        result = self.search_embeddings(df, user_input, n=3)
        print(result)
        prompt = """你是一个专门阅读和总结科学论文的大型语言模型。 
        你需要根据提出的问题和一系列与问题相似的论文中的文本嵌入来回答问题。
        请根据给定的嵌入，返回一篇详细的论文摘要来回答问题。
            
            问题：""" + user_input + """
            
            以下是数据中的文本嵌入：
            
            1.""" + str(result.iloc[0]['text']) + """
            2.""" + str(result.iloc[1]['text']) + """
            3.""" + str(result.iloc[2]['text']) + """

            请根据论文返回一个详细的回答:"""

        print('完成创建提示')
        return prompt

    def gpt(self, prompt):
        print('发送请求到 GPT-3')
        openai.api_key = OPENAI_API_KEY
    
        messages = [{"role": "system", "content": prompt}]
    
        r = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.4,
            max_tokens=1500
        )
    
        answer = r.choices[0].message.content.strip()
        print('完成发送请求到 GPT-3')
        response = {'answer': answer, 'sources': sources}
        return response

    def reply(self, prompt):
        print(prompt)
        prompt = self.create_prompt(df, prompt)
        return self.gpt(prompt)
    
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/process_pdf", methods=['POST'])
def process_pdf():
    print("处理 PDF")
    file = request.data
    pdf = PdfReader(BytesIO(file))
    chatbot = Chatbot()
    paper_text = chatbot.parse_paper(pdf)
    global df
    df = chatbot.paper_df(paper_text)
    df = chatbot.calculate_embeddings(df)
    print("完成处理 PDF")
    return {'key': ''}

@app.route("/download_pdf", methods=['POST'])
def download_pdf():
    chatbot = Chatbot()
    url = request.json['url']
    r = requests.get(str(url))
    print(r.headers)
    pdf = PdfReader(BytesIO(r.content))
    paper_text = chatbot.parse_paper(pdf)
    global df
    df = chatbot.paper_df(paper_text)
    df = chatbot.calculate_embeddings(df)
    print("完成处理 PDF")
    return {'key': ''}

@app.route("/reply", methods=['POST'])
def reply():
    chatbot = Chatbot()
    query = request.json['query']
    query = str(query)
    prompt = chatbot.create_prompt(df, query)
    response = chatbot.gpt(prompt)
    print(response)
    return response, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)