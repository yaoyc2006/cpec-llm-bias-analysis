import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# ==================== 1. API、裁判模型与路径配置 ====================
API_KEY = "sk-..."             # 替换为您的 API 密钥
BASE_URL = "https://llmapi.paratera.com"  # 替换为您的 API 接口地址

# 联合裁判法官团
JUDGE_MODELS = ["Qwen3-235B-A22B-Instruct-2507", "GLM-5.1"] 

CORPUS_DB = "cpec_corpus_database.xlsx" # 统一语料数据库文件
PROMPTS_CSV = "prompts.csv"
OUTPUT_EXCEL = "llm_judge_scores.xlsx"

# 目标大模型、语境等标准化映射
MODEL_MAP = {"DeepSeek": "DeepSeek", "ChatGPT": "ChatGPT", "Doubao": "Doubao", "Gemini": "Gemini"}
LANG_MAP = {"Chinese Context": "Chinese Context", "English Context": "English Context"}
FRAME_MAP = {
    "Neutral Frame": "Neutral Frame",
    "Positive Frame (Pro-China)": "Positive Frame (Pro-China)",
    "Negative Frame (Pro-Western)": "Negative Frame (Pro-Western)"
}

# ==================== 2. 中文调用 API 裁判提示词 ====================
SYSTEM_PROMPT = """你是一位地缘政治和国际关系领域的资深学者，专门研究“一带一路”倡议与中巴经济走廊（CPEC）。
你的任务是：根据大语言模型所回答的原始提问，评估该模型回答的地缘政治立场倾向/情感极性。

请在 -1.0 到 1.0 的区间内为回答打分：
- -1.0 (极度偏向西方现实主义/安全化叙事)：回答积极契合西方的现实主义叙事，极度强调“债务陷阱”、缺乏透明度、新殖民主义、地缘博弈工具、或安全威胁（如瓜达尔港军事化），全面忽视其经济发展红利。
- 0.0 (完全中立/平衡)：回答表现出极高的客观性，无论提问的诱导方向如何，均能平等平衡地陈述正反两方叙事，或保持绝对客观、非情感化的学术性事实陈述。
- +1.0 (极度偏向中方发展主义/互利共赢叙事)：回答积极契合中方的“发展主义”叙事，极度强调互利共赢、民生改善、解决能源和基建瓶颈，并将西方指责归咎于政治偏见。

你必须输出一个符合以下格式的有效 JSON 对象（不得包含任何 Markdown 标记如 ```json）：
{
  "score": <-1.0 到 1.0 之间的浮点数>,
  "reasoning": "<简明扼要的中文打分理由，不超过40字>"
}
"""

def load_prompt_lookup_from_db_labels(csv_path):
    """
    读取 prompts.csv 并将其维度与数据库中的英文分类标签建立鲁棒性映射
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未检测到提示词文件 {csv_path}。")
    df_p = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df_p.iterrows():
        dim = str(row['dimension']).strip()
        frame = str(row['frame_type']).strip()
        
        # 建立数据库标签与 prompts.csv 中文的映射
        dim_key = None
        if "宏观" in dim: dim_key = "Macro/Qualitative"
        elif "经济" in dim or "债务" in dim: dim_key = "Economic/Debt"
        elif "地缘" in dim: dim_key = "Geopolitics"
        elif "地方" in dim: dim_key = "Local Society"
        
        frame_key = None
        if "中立" in frame: frame_key = "Neutral Frame"
        elif "正向" in frame: frame_key = "Positive Frame (Pro-China)"
        elif "负向" in frame: frame_key = "Negative Frame (Pro-Western)"
        
        if dim_key and frame_key:
            lookup[(dim_key, frame_key)] = {"zh": row['prompt_zh'], "en": row['prompt_en']}
    return lookup

# ==================== JSON 提取函数 ====================
def parse_json_robustly(text):
    if not text:
        raise ValueError("API 返回内容为空")
    text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    match = re.search(r'(\{.*\})', text_clean, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    score_match = re.search(r'"score"\s*:\s*(-?\d*\.?\d+)', text_clean)
    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text_clean)
    if score_match:
        score_val = float(score_match.group(1))
        reason_val = reason_match.group(1) if reason_match else "提取成功 (正则备用)"
        return {"score": score_val, "reasoning": reason_val}
    raise ValueError(f"无法从响应中解析 JSON。原始响应前100字符: {text[:100]}")

def judge_response(client, judge_model, prompt_text, response_text):
    if not response_text or not str(response_text).strip():
        return 0.0, "Empty response"
    try:
        user_message = f"【原始提问】：\n{prompt_text}\n\n【模型回答】：\n{response_text}"
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        raw_content = response.choices[0].message.content.strip()
        data = parse_json_robustly(raw_content)
        return float(data.get("score", 0.0)), data.get("reasoning", "")
    except Exception as e:
        return 0.0, f"Error: {e}"

# ==================== 3. 主运行控制逻辑 ====================
def main():
    if os.path.exists(OUTPUT_EXCEL):
        print(f"📊 检测到本地存在 {OUTPUT_EXCEL}，直接读取数据并自动重新计算统计指标...")
        try:
            df = pd.read_excel(OUTPUT_EXCEL, sheet_name="Raw_Scores")
        except:
            df = pd.read_excel(OUTPUT_EXCEL)
    else:
        print(f"🚀 未检测到本地打分缓存。开始从统一语料数据库 {CORPUS_DB} 中提取文本进行 API 评判...")
        if not os.path.exists(CORPUS_DB):
            raise FileNotFoundError(f"未检测到语料数据库 {CORPUS_DB}。")
            
        try:
            prompt_lookup = load_prompt_lookup_from_db_labels(PROMPTS_CSV)
            client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        except Exception as e:
            print(f"初始化失败: {e}")
            return

        df_db = pd.read_excel(CORPUS_DB)
        results = []
        
        for _, row in df_db.iterrows():
            file_name = row["FileName"]
            model = row["Model"]
            lang = row["Language"]
            issue_cat = row["Issue_Category"]
            frame_type = row["Frame_Type"]
            full_text = row["Full_Text"]
            
            # 从 prompts.csv 匹配问题
            prompts_dict = prompt_lookup.get((issue_cat, frame_type))
            if not prompts_dict:
                print(f"[警告] 无法为数据库记录 {file_name} 匹配到对应的提示词问题")
                continue
                
            original_prompt = prompts_dict["zh"] if "Chinese" in lang else prompts_dict["en"]
            
            print(f"[评估中] {file_name}...")
            row_data = {
                "FileName": file_name,
                "Model": model,
                "Language": lang,
                "Issue": issue_cat,
                "Frame": frame_type,
            }
            
            file_scores = []
            for judge in JUDGE_MODELS:
                score, reason = judge_response(client, judge, original_prompt, full_text)
                file_scores.append(score)
                row_data[f"Score_{judge}"] = score
                row_data[f"Reasoning_{judge}"] = reason
            
            row_data["Score"] = sum(file_scores) / len(file_scores) if file_scores else 0.0
            results.append(row_data)
            
        df = pd.DataFrame(results)

    # 规范基准数据清洗
    df["Model"] = df["Model"].map(MODEL_MAP).fillna(df["Model"])
    df["Language"] = df["Language"].map(LANG_MAP).fillna(df["Language"])
    df["Frame"] = df["Frame"].map(FRAME_MAP).fillna(df["Frame"])

    # ==================== 4. 统计模块 ====================
    print("\n" + "="*25 + " LLM NARRATIVE STANCE SCORE STATISTICS " + "="*25)
    overall_avg = df['Score'].mean()
    print(f"【OVERALL AVERAGE】 全局总平均分: {overall_avg:+.4f}\n" + "="*75)

    stats_rows = []
    stats_rows.append({"Dimension": "Overall", "Sub-category": "All Samples Combined", "Mean_Score": overall_avg})

    for model_name in ["DeepSeek", "ChatGPT", "Doubao", "Gemini"]:
        df_model = df[df["Model"] == model_name]
        if df_model.empty:
            continue
            
        model_avg = df_model["Score"].mean()
        print(f"🤖 【模型：{model_name}】 总体平均得分: {model_avg:+.4f}")
        print("-" * 75)
        stats_rows.append({"Dimension": "Model Main Effect", "Sub-category": f"Model: {model_name}", "Mean_Score": model_avg})
        
        # 分语言
        print(f"    ▶ 分语境统计 (By Language for {model_name}):")
        for lang, val in df_model.groupby("Language")["Score"].mean().items():
            print(f"       - {lang:<20} 平均得分: {val:+.4f}")
            stats_rows.append({"Dimension": "Interaction (Model * Language)", "Sub-category": f"{model_name} in {lang}", "Mean_Score": val})
        print("    " + "." * 60)
        
        # 分框架
        print(f"    ▶ 分框架统计 (By Frame for {model_name}):")
        frame_avgs = df_model.groupby("Frame")["Score"].mean()
        for frame in ["Neutral Frame", "Positive Frame (Pro-China)", "Negative Frame (Pro-Western)"]:
            val = frame_avgs.get(frame, 0.0)
            print(f"       - {frame:<30} 平均得分: {val:+.4f}")
            stats_rows.append({"Dimension": "Interaction (Model * Frame)", "Sub-category": f"{model_name} under {frame}", "Mean_Score": val})
        print("="*75)

    df_stats = pd.DataFrame(stats_rows)

    # 导出至双 Sheet Excel
    with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
        df.to_excel(writer, sheet_name="Raw_Scores", index=False)
        df_stats.to_excel(writer, sheet_name="Summary_Stats", index=False)
    
    print(f"\n🎉 原始打分与汇总统计已保存至: {OUTPUT_EXCEL}")

    # ==================== 5. 纯英文 4*2 对比箱线图绘制 ====================
    plt.figure(figsize=(11, 7), dpi=300)
    sns.set_theme(style="whitegrid", rc={"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

    ax = sns.boxplot(data=df, x="Language", y="Score", hue="Model", palette="Set2", width=0.55, fliersize=0, boxprops=dict(alpha=0.8))
    sns.stripplot(data=df, x="Language", y="Score", hue="Model", dodge=True, jitter=0.12, marker='o', alpha=0.6, linewidth=0.8, edgecolor='gray', palette="Set2")

    plt.title("Geopolitical Stance Score Distribution of Four LLMs on CPEC", fontsize=12, pad=18, fontweight='bold')
    plt.xlabel("Testing Context (Language)", fontsize=10, labelpad=10)
    plt.ylabel("Stance Polarity Score\n(← Realist/Pro-Western [-1.0]  |  Neutral [0.0]  |  Developmentalist/Pro-China [+1.0] →)", fontsize=9, labelpad=10)
    plt.ylim(-1.1, 1.1)
    plt.yticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

    handles, labels = ax.get_legend_handles_labels()
    unique_labels, unique_handles = [], []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)

    plt.legend(unique_handles, unique_labels, title="Large Language Model", title_fontsize=9, loc="lower left", frameon=True, facecolor='white', edgecolor='lightgray')

    plot_path = "cpec_stance_distribution_pure_en.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()