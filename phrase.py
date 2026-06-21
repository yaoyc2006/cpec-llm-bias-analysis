import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 1. 定义中文特征词典 ====================

# 1. 中方“发展主义”叙事词典 (ZH_OFFICIAL_DICT)
ZH_OFFICIAL_DICT = [
    "互联互通", "互利共赢", "全天候战略", "旗舰项目", "民生改善", 
    "改善民生", "共商共建", "共同繁荣", "高质量发展", "减贫", 
    "脱贫", "共同发展", "和平与发展", "经济繁荣", "命运共同体", 
    "就业创造", "合作共赢", "早期收获", "区域一体化", "全球发展倡议"
]

# 2. 西方“现实主义”安全化叙事词典 (ZH_REALIST_DICT)
ZH_REALIST_DICT = [
    "债务陷阱", "新殖民主义", "军事基地", "军事化", "珍珠链", 
    "印太战略", "剥夺", "环境破坏", "债务危机", "债务负担", 
    "缺乏透明度", "不透明", "安全威胁", "主权争议", "马六甲困境", 
    "战略包围", "扩张工具", "资源掠夺", "军民两用", "零和博弈"
]

CORPUS_DB = "cpec_corpus_database.xlsx"
MODEL_MAP = {"DeepSeek": "DeepSeek", "ChatGPT": "ChatGPT", "Doubao": "Doubao", "Gemini": "Gemini"}

# ==================== 2. 语料库特征统计 ====================
def main():
    if not os.path.exists(CORPUS_DB):
        print(f"未检测到语料数据库 {CORPUS_DB}。")
        return

    df_db = pd.read_excel(CORPUS_DB)
    df_db = df_db[df_db["Language"].isin(["Chinese Context", "中文回答", "cn"])]

    term_records = []
    model_true_word_counts = {}

    print("🚀 正在从统一数据库中检索中文文本特征词频...")

    for _, row in df_db.iterrows():
        model_raw = row["Model"]
        text = str(row["Full_Text"])
        word_count = int(row["Word_Count"]) # 直接调用数据库中预存的真实字数分母 [1]
        
        model_clean = MODEL_MAP.get(model_raw, model_raw)
        
        # 累加每个模型总字数
        model_true_word_counts[model_clean] = model_true_word_counts.get(model_clean, 0) + word_count

        # 统计中方叙事词频
        for term in ZH_OFFICIAL_DICT:
            count = text.count(term)
            if count > 0:
                term_records.append({"Model": model_clean, "Category": "Chinese_Official", "Term": term, "Count": count})
                
        # 统计西方叙事词频
        for term in ZH_REALIST_DICT:
            count = text.count(term)
            if count > 0:
                term_records.append({"Model": model_clean, "Category": "Western_Realist", "Term": term, "Count": count})

    if not term_records:
        print("未检索到任何特征词汇。")
        return

    df_terms = pd.DataFrame(term_records)

    # ==================== 3. 统计一：宏观话语密度统计 ====================
    macro_summary = df_terms.groupby(["Model", "Category"]).agg(Total_Hits=("Count", "sum")).reset_index()

    # 标准化密度计算
    def calc_true_density(row):
        model = row["Model"]
        hits = row["Total_Hits"]
        true_words = model_true_word_counts.get(model, 1.0)
        return (hits / true_words) * 1000

    macro_summary["Normalized_Frequency_per_1k"] = macro_summary.apply(calc_true_density, axis=1)

    # ==================== 4. 统计二：细粒度混合降序排序 ====================
    summary = df_terms.groupby(["Model", "Category", "Term"]).agg(Total_Hits=("Count", "sum")).reset_index()
    summary["Normalized_Frequency_per_1k"] = summary.apply(calc_true_density, axis=1)

    # ==================== 5. 格式化控制台输出与双 Sheet 保存 ====================
    print("\n" + "="*25 + " LLM 中西话语使用情况学术统计报表 " + "="*25)
    
    aligned_rows = []
    macro_rows = []

    for model in ["DeepSeek", "Doubao", "ChatGPT", "Gemini"]:
        print(f"\n🤖 【大语言模型: {model}】(真实生成总字数: {model_true_word_counts.get(model, 0)} 字)")
        print("=" * 86)
        
        df_model_macro = macro_summary[macro_summary["Model"] == model]
        hits_off = df_model_macro[df_model_macro["Category"] == "Chinese_Official"]["Total_Hits"].sum()
        hits_real = df_model_macro[df_model_macro["Category"] == "Western_Realist"]["Total_Hits"].sum()
        total_hits = hits_off + hits_real
        
        rate_off = df_model_macro[df_model_macro["Category"] == "Chinese_Official"]["Normalized_Frequency_per_1k"].sum()
        rate_real = df_model_macro[df_model_macro["Category"] == "Western_Realist"]["Normalized_Frequency_per_1k"].sum()
        
        prop_off = (hits_off / total_hits * 100) if total_hits > 0 else 0.0
        prop_real = (hits_real / total_hits * 100) if total_hits > 0 else 0.0
        
        print("  📊 话语体系宏观分布对比 (Macro Discourse Distribution):")
        print(f"     - 中方官方发展叙事 (Chinese Official) | 总频数: {hits_off:<3} 次 | 真实密度: {rate_off:.2f}次/千字 | 话语份额占比: {prop_off:.1f}%")
        print(f"     - 西方地缘安全叙事 (Western Realist)  | 总频数: {hits_real:<3} 次 | 真实密度: {rate_real:.2f}次/千字 | 话语份额占比: {prop_real:.1f}%")
        print("  " + "-" * 82)

        macro_rows.append({"Model": model, "Discourse_Type": "Chinese Official (中)", "Total_Hits": hits_off, "Density_per_1k": round(rate_off, 2), "Proportion": f"{prop_off:.1f}%"})
        macro_rows.append({"Model": model, "Discourse_Type": "Western Realist (西)", "Total_Hits": hits_real, "Density_per_1k": round(rate_real, 2), "Proportion": f"{prop_real:.1f}%"})

        df_model_detail = summary[summary["Model"] == model]
        df_model_sorted = df_model_detail.sort_values(by="Total_Hits", ascending=False)
        
        print(f"  📝 具体词频降序细览:")
        print(f"  {'Rank':<8} | {'特征词汇 (Term)':<18} | {'话语类别 (Category)':<24} | {'频数 (Hits)':<10} | {'每千字频数 (Rate/1k)':<12}")
        print("  " + "-" * 78)
        
        rank = 1
        for _, row_d in df_model_sorted.iterrows():
            category_cn = "中方官方发展" if row_d["Category"] == "Chinese_Official" else "西方地缘安全"
            print(f"  {rank:<8} | {row_d['Term']:<18} | {category_cn:<24} | {row_d['Total_Hits']:<10} | {row_d['Normalized_Frequency_per_1k']:.2f}")
            
            aligned_rows.append({
                "Model": model,
                "Overall_Rank": rank,
                "Term": row_d["Term"],
                "Discourse_Type": "Chinese Official (中)" if row_d["Category"] == "Chinese_Official" else "Western Realist (西)",
                "Total_Hits": row_d["Total_Hits"],
                "Normalized_Frequency_per_1k": f"{row_d['Normalized_Frequency_per_1k']:.2f}"
            })
            rank += 1
        print()

    df_final = pd.DataFrame(aligned_rows)
    df_macro_final = pd.DataFrame(macro_rows)
    
    # 写入双 Sheet 的描述性 Excel
    output_path = "discourse_chinese_mixed_sorted_summary.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        df_terms.to_excel(writer, sheet_name="Raw_Chinese_Counts", index=False)
        df_macro_final.to_excel(writer, sheet_name="Macro_Stance_Proportion", index=False)
        df_final.to_excel(writer, sheet_name="Unified_Stance_Summary", index=False)
        
    print(f"🎉 描述性统计 Excel 已更新保存至：{output_path}")

    # ==================== 6. 绘图：分组柱状图绘制 ====================
    print("正在绘制宏观话语密度对比图...")
    plt.figure(figsize=(10, 6.5), dpi=300)
    sns.set_theme(style="whitegrid", rc={"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

    plot_df = df_macro_final.copy()
    plot_df["Discourse_Type_EN"] = plot_df["Discourse_Type"].map({"Chinese Official (中)": "Chinese Official", "Western Realist (西)": "Western Realist"})
    custom_palette = {"Chinese Official": "#66c2a5", "Western Realist": "#fc8d62"}

    ax = sns.barplot(data=plot_df, x="Model", y="Density_per_1k", hue="Discourse_Type_EN", palette=custom_palette, edgecolor="gray", linewidth=0.8, alpha=0.9)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=9.5, color='black', xytext=(0, 4), textcoords='offset points', fontweight='semibold')

    plt.title("Macro Discourse Density Comparison across Four LLMs on CPEC", fontsize=12, pad=18, fontweight='bold')
    plt.xlabel("Large Language Models (Chinese Context Only)", fontsize=10, labelpad=10)
    plt.ylabel("Discourse Term Density (per 1,000 characters)", fontsize=10, labelpad=10)
    plt.ylim(0, max(plot_df["Density_per_1k"]) * 1.15)

    plt.legend(title="Discourse Narrative Type", title_fontsize=9.5, loc="upper right", frameon=True, facecolor='white', edgecolor='lightgray')
    plt.tight_layout()
    
    plot_path = "cpec_discourse_density_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    #plt.show()

if __name__ == "__main__":
    main()