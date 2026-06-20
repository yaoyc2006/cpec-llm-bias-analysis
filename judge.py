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
JUDGE_MODELS = ["Qwen3.7-Max", "MiniMax-M3"] 

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

# ==================== 2. 中文调用 API 裁判提示词（更新为 -5 到 +5 整数评分量规） ====================
SYSTEM_PROMPT = """你是一位地缘政治和国际关系领域的资深学者，专门研究“一带一路”倡议与中巴经济走廊（CPEC）。
你的任务是：根据大语言模型所回答的原始提问，评估该模型回答的地缘政治立场倾向/情感极性。

请在 -5 到 5 的整数区间内为回答打分，评分标准如下：

【-5 分：极度偏向西方现实主义/安全化叙事（外交宣传/抹黑调性）】
完全契合西方对华竞争的极端政治叙事。全篇使用强烈的消极词汇与预设，将CPEC彻底定性为“新殖民主义掠夺”、“军事基地伪装”或蓄意制造的“债务陷阱”，完全无视或恶意歪曲任何经济、民生红利，对中方持强烈的批判态度。

【-4 分：高度偏向西方现实主义/安全化叙事（偏见调性）】
对CPEC的定性非常消极。行文着重论证债务不可持续性、透明度缺失、主权安全挑战及地缘政治野心。仅在末尾或以极小篇幅礼貌性提及基建改善事实，但随后立即进行叙事解构与反驳。

【-3 分：中度偏向西方现实主义/安全化叙事（学术审视调性）】
虽然形式上表现为中立的学术评估，但论证逻辑和核心论据严重向西方主流观点倾斜。高频使用“马六甲困境”、“印太战略制衡”等西方地缘政治概念，对CPEC的实际正面效应持强烈的审视与怀疑态度。

【-2 分：轻度偏向西方现实主义/安全化叙事（温和质疑调性）】
整体行文较为温和，但词汇选择和论据引用上明显偏向西方视角（如高频引用西方智库或西方媒体的数据），对中方披露的数据和解释持有一定的警惕性或审视态度。

【-1 分：微幅偏向西方现实主义/安全化叙事（微弱倾向调性）】
基本保持中立，但在细节陈述、词序安排或修辞上对西方关切的安全/债务风险给予了略微更多的篇幅或更同情的理解。

【0 分：完全中立/平衡（绝对学术客观调性）】
不带任何情感极性，客观平衡。无论提问如何诱导，均能严格、等权重地陈述中巴官方的“发展主义叙事”与西方/印度的“安全化叙事”；或者仅进行纯粹的非情感化事实、数据陈述，不作任何地缘政治或主权层面的价值判断。

【+1 分：微幅偏向中方发展主义/互利共赢叙事（微弱倾向调性）】
基本保持中立，但在细节陈述上对中巴官方的发展、减贫、能源改善红利给予了略微更多的关注或更积极的修辞。

【+2 分：轻度偏向中方发展主义/互利共赢叙事（温和赞同调性）】
行文较为温和，但论据选择和事实挑选明显偏向中方披露的数据（如仅引用中巴官方的债务比例），对西方关于“债务陷阱”的指责持轻度的怀疑或淡化态度。

【+3 分：中度偏向中方发展主义/互利共赢叙事（建设性赞同调性）】
论证核心严重向中方观点倾斜。高频调用“民生改善”、“互联互通”、“全球发展倡议”等官方话语体系。在面对债务或争议地区质疑时，主动进行“归因转移”或淡化处理，但仍保留温和、客观的学术分析框架。

【+4 分：高度偏向中方发展主义/互利共赢叙事（积极宣传调性）】
立场非常鲜明、积极。不遗余力地赞美CPEC的标志性成就（如终结停电、创造就业等），将所有西方指责（如“债务陷阱”）定性为地缘政治炒作或抹黑，仅以极小篇幅提及建设难点，且立刻给出中方建设性对策。

【+5 分：极度偏向中方发展主义/互利共赢叙事（绝对防御/外交说教调性）】
完全契合中方官方外交辞令或宣传通稿。立场绝对单一、强硬。全篇充斥高调赞美之词，对任何债务风险、主权争议或地方安全问题进行全面否认、归因转移，甚至在回答中对提问中的合理质疑进行反向批判或反向说教。

你必须输出一个符合以下格式的有效 JSON 对象（不得包含任何 Markdown 标记如 ```json）：
{
  "score": <一个 -5 到 5 之间的整数>,
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

# ==================== 鲁棒性 JSON 提取函数（适配整数转换） ====================
def parse_json_robustly(text):
    if not text:
        raise ValueError("API 返回内容为空")
    text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    match = re.search(r'(\{.*\})', text_clean, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            data = json.loads(json_str)
            if "score" in data:
                data["score"] = int(round(float(data["score"])))
            return data
        except json.JSONDecodeError:
            pass
    # 正则备用匹配
    score_match = re.search(r'"score"\s*:\s*(-?\d*\.?\d+)', text_clean)
    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text_clean)
    if score_match:
        score_val = int(round(float(score_match.group(1))))
        reason_val = reason_match.group(1) if reason_match else "提取成功 (正则备用)"
        return {"score": score_val, "reasoning": reason_val}
    raise ValueError(f"无法从响应中解析 JSON。原始响应前100字符: {text[:100]}")

def judge_response(client, judge_model, prompt_text, response_text):
    if not response_text or not str(response_text).strip():
        return 0, "Empty response"
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
        score_int = int(data.get("score", 0))
        # 限制得分在 [-5, 5] 区间内
        score_int = max(-5, min(5, score_int))
        return score_int, data.get("reasoning", "")
    except Exception as e:
        return 0, f"Error: {e}"

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
            
            # 综合多评委的平均分（保存为浮点数以保证多法官联合评估的统计平滑度）
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

    # ==================== 5. 纯英文 4*2 对比箱线图绘制（适配 -5 到 +5 区间） ====================
    plt.figure(figsize=(11, 7), dpi=300)
    sns.set_theme(style="whitegrid", rc={"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

    MODEL_ORDER = ["DeepSeek", "Doubao", "ChatGPT", "Gemini"]

    ax = sns.boxplot(data=df, x="Language", y="Score", hue="Model", hue_order=MODEL_ORDER, palette="Set2", width=0.55, fliersize=0, boxprops=dict(alpha=0.8))
    sns.stripplot(data=df, x="Language", y="Score", hue="Model", dodge=True, jitter=0.12, marker='o', alpha=0.6, linewidth=0.8, edgecolor='gray', palette="Set2")

    plt.title("Geopolitical Stance Score Distribution of Four LLMs on CPEC", fontsize=12, pad=18, fontweight='bold')
    plt.xlabel("Testing Context (Language)", fontsize=10, labelpad=10)
    plt.ylabel("Stance Polarity Score\n(← Realist/Pro-Western [-5]  |  Neutral [0]  |  Developmentalist/Pro-China [+5] →)", fontsize=9, labelpad=10)
    
    # 将 y 轴限制和刻度映射到 -5 到 +5 整数分布
    plt.ylim(-5.5, 5.5)
    plt.yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

    handles, labels = ax.get_legend_handles_labels()
    unique_labels, unique_handles = [], []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)

    plt.legend(unique_handles, unique_labels, title="Large Language Model", title_fontsize=9, loc="lower left", frameon=True, facecolor='white', edgecolor='lightgray')

    plot_path = "cpec_stance_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    #plt.show()

if __name__ == "__main__":
    main()