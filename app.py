"""
作者：曾浩龙（独立开发）
创建时间：2024 年 10 月 25 日
第三方依赖库：Gradio (https://www.gradio.app/) 与 OpenAI Python API library (https://github.com/openai/openai-python)
其他说明：本项目声明仅供学习和研究使用。
"""
import os
import platform
print(platform.python_version())

import gradio as gr
from openai import OpenAI


def demo(project_TCGA, output_language="Chinese"):
    name_English, name_Chinese = project_name_TCGA[project_TCGA]
    tcga_link = f"https://portal.gdc.cancer.gov/projects/{project_TCGA}"
    output1, output2 = None, None

    if output_language == "Chinese":
        output1 = f"✍️ 简称：{project_TCGA}\n❤️ 中文全称：{name_Chinese}\n💛 英文全称：{name_English}\n🔗 链接：{tcga_link}"
        system_instruction = f"您是公共卫生、流行病学、癌症研究和精准医学领域的专家，对{name_Chinese}有着深刻的洞察。"
        prompt_template = f"""
您的任务是撰写关于{name_Chinese}这种复杂疾病的摘要介绍。关键在于在信息的精准性与易懂性之间取得好的平衡，并确保内容引人入胜。通过合理的结构设计、清晰的语言表达以及专业与通俗版本的融合，既能满足专业人士的需求，又能帮助普通大众理解。因此，您必须充分考虑以下具体要求：
1 - 明确目标受众。在撰写时，采用分层的信息结构，以更好地满足不同读者群体的需求。具体来说，内容可分为两个主要层次：一是为专家和研究人员提供详尽的分析与解释；二是面向普通大众，给出通俗易懂的概述。
2 - 简明扼要的引言部分。引言部分应通俗易懂，概述{name_Chinese}的基本信息，并以通俗易懂的方式向普通大众解释。关键要点如下：（1）疾病名称与定义：简单定义{name_Chinese}，让读者知道它是什么。（2）患病人群与流行病学概述：简要提到{name_Chinese}的患病率、常见人群或特定风险因素。（3）重要性与影响：阐述关注{name_Chinese}为何重要，它对患者、社会或公共卫生的影响。
3 - 医学机制与病理生理的清晰解释。对于专业人士而言，理解{name_Chinese}的深层病理机制和生物学背景至关重要，因此必须深入探索并分析其发病和进展过程中涉及的关键分子机制及信号通路。在这一部分，虽然可以使用生物医学术语，但必须确保这些术语不会过于晦涩难懂，以免降低信息的可读性和理解度。对于普通大众来说，更适合从{name_Chinese}如何影响人体的角度进行阐述。这样能帮助他们更好地认识{name_Chinese}的本质，以及{name_Chinese}对人体健康可能产生的潜在影响。
4 - 症状与诊断。此部分需准确列出{name_Chinese}的常见表型、症状及诊断方法。对于专业人士，应提及相关检查和诊断标准；而对于大众，则需强调常见症状及常用的检测方法。
5 - 治疗方法。专业版：详细探讨治疗策略，涵盖药物、手术治疗及其他干预手段，同时结合最新的治疗指南。大众版：简要介绍治疗方法，特别是药物治疗以及生活方式改变的建议，如饮食调整和运动。
6 - 预防和生活方式的建议。这部分可以结合具体的预防措施与生活方式改变建议。专业版：深入探讨预防策略、早期筛查方法，以及饮食与生活方式如何对{name_Chinese}管理产生积极影响。大众版：提供实际可行的日常生活建议，包括健康饮食和适量运动等，旨在帮助人们通过改变生活方式来预防{name_Chinese}。
7 - 结语与前景展望。在结尾部分，请简明扼要地概括{name_Chinese}所带来的影响，并展望未来。对于专业读者，可探讨未来的研究方向与治疗突破；而对于普通大众，则应分享积极信息以鼓舞人心，同时强调早期诊断与预防措施的重要性。
8 - 结构和语言。结构：应设置清晰的标题与副标题，确保表达逻辑好。语言：在涉及专业内容时，可适当使用专业术语，但需确保解释清晰；面向普通读者时，应采用类比或简化语言进行说明。
""".strip()

    else:
        output1 = f"✍️ Abbreviation: {project_TCGA}\n❤️ Full name in Chinese: {name_Chinese}\n💛 Full Name in English: {name_English}\n🔗 Link: {tcga_link}"
        system_instruction = f"You are an expert in the fields of public health, epidemiology, cancer research, and precision medicine, with deep insights into {name_English}."
        prompt_template = f"""
Your task is to thoroughly analyze the basic information about {name_English}, the phenotypes of {name_English} and effective preventions for this disease, and to explore in depth the key molecular mechanisms and signaling pathways involved in its initiation and progression.
Let's think step by step.
""".strip()

    try:
        # 要实例化一个 OpenAI 对象，你需要设置 OpenAI API Key、Base URL、最大重试次数以及超时限制时间。
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["API_BASE"],
            max_retries=3,
            timeout=60,
        )

        # 调用 client.chat.completions.create，设置关键参数。
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # gpt-4o-mini-2024-07-18, gpt-4-turbo
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt_template},
            ],
            n=1,
            seed=42,
            temperature=0.50,
            max_tokens=3072 if output_language == "Chinese" else 1024,
            logprobs=False,
            # top_logprobs=3,
            presence_penalty=0.20,
            frequency_penalty=0.20,
        )

        resp_text = chat_completion.choices[0].message.content.strip()
        # 在普通文本框不能用 "**" 渲染加粗，Markdown 才可以。因此，将输入字符串中所有的 "**" 替换为 ""。
        # if "**" in resp_text:
        #     resp_text = resp_text.replace("**", "")
        # if "# " in resp_text:
        #     resp_text = resp_text.replace("# ", "")
        # if "#" in resp_text:
        #     resp_text = resp_text.replace("#", "")

        if output_language == "Chinese":
            # "🤖 请注意：以下内容通过提示工程驱动的 GPT-4 Turbo 生成\n\n"
            output2 = "" + resp_text
        else:
            # "🤖 Note: The following content is generated by the GPT-4 Turbo driven by Prompt Engineering\n\n"
            output2 = "" + resp_text

    except Exception as e:
        print(str(e), "Response Error")
        return output1, "Response Error"

    return output1, output2


# TCGA 有 33 种癌症类型
project_name_TCGA = {
    "TCGA-ACC": ["adrenocortical carcinoma", "肾上腺皮质癌"],
    "TCGA-BLCA": ["bladder urothelial carcinoma", "膀胱尿路上皮癌"],
    "TCGA-BRCA": ["breast invasive carcinoma", "浸润性乳腺癌"],
    "TCGA-CESC": [
        "cervical squamous cell carcinoma and endocervical adenocarcinoma",
        "宫颈鳞状细胞癌与宫颈内膜腺癌",
    ],
    "TCGA-CHOL": ["cholangiocarcinoma", "胆管癌"],
    "TCGA-COAD": ["colon adenocarcinoma", "结肠腺癌"],
    "TCGA-DLBC": [
        "lymphoid neoplasm diffuse large B-cell lymphoma",
        "弥漫性大 B 细胞淋巴瘤",
    ],
    "TCGA-ESCA": ["esophageal carcinoma", "食道癌"],
    "TCGA-GBM": ["glioblastoma multiforme", "多形性胶质母细胞瘤"],
    "TCGA-HNSC": ["head and neck squamous cell carcinoma", "头颈部鳞状细胞癌"],
    "TCGA-KICH": ["kidney chromophobe", "肾嫌色细胞癌"],
    "TCGA-KIRC": ["kidney renal clear cell carcinoma", "肾透明细胞癌"],
    "TCGA-KIRP": ["kidney renal papillary cell carcinoma", "乳头状肾细胞癌"],
    "TCGA-LAML": ["acute myeloid leukemia", "急性髓系白血病"],
    "TCGA-LGG": ["brain lower grade glioma", "低级别脑胶质瘤"],
    "TCGA-LIHC": ["liver hepatocellular carcinoma", "肝细胞癌"],
    "TCGA-LUAD": ["lung adenocarcinoma", "肺腺癌"],
    "TCGA-LUSC": ["lung squamous cell carcinoma", "肺鳞状细胞癌"],
    "TCGA-MESO": ["mesothelioma", "间皮瘤"],
    "TCGA-OV": ["ovarian serous cystadenocarcinoma", "卵巢浆液性囊腺癌"],
    "TCGA-PAAD": ["pancreatic adenocarcinoma", "胰腺腺癌"],
    "TCGA-PCPG": ["pheochromocytoma and paraganglioma", "嗜铬细胞瘤和副神经节瘤"],
    "TCGA-PRAD": ["prostate adenocarcinoma", "前列腺腺癌"],
    "TCGA-READ": ["rectum adenocarcinoma", "直肠腺癌"],
    "TCGA-SARC": ["sarcoma", "肉瘤"],
    "TCGA-SKCM": ["skin cutaneous melanoma", "皮肤黑色素瘤"],
    "TCGA-STAD": ["stomach adenocarcinoma", "胃腺癌"],
    "TCGA-TGCT": ["testicular germ cell tumors", "睾丸生殖细胞肿瘤"],
    "TCGA-THCA": ["thyroid carcinoma", "甲状腺癌"],
    "TCGA-THYM": ["thymoma", "胸腺瘤"],
    "TCGA-UCEC": ["uterine corpus endometrial carcinoma", "子宫体子宫内膜癌"],
    "TCGA-UCS": ["uterine carcinosarcoma", "子宫癌肉瘤"],
    "TCGA-UVM": ["uveal melanoma", "眼内（葡萄膜）黑色素瘤"],
}
# print(len(project_name_TCGA.keys()))
# input_query = input("请输入您要查询的 TCGA 项目名称：")
# print(project_name_TCGA[input_query])
# print([k for k in project_name_TCGA.keys()])

# 支持 Markdown 和 HTML 内容格式：
# Abbreviations, Full Names and Descriptions of All Cancer Types Covered by TCGA Project.
# desc = """<h1 align="center" style="font-family: Latin Modern Math, sans-serif; font-size: 22px; color: #00FF7F;">🎉 Abbreviations, Full Names and Descriptions of All Cancer Types Covered by TCGA Project 🧬</h1>"""

desc = """<h1 align="center" style="font-family: KaiTi, sans-serif; font-size: 22px; color: #00FF7F;">🎉 TCGA 项目涉及的所有癌症类型的缩写、中英文全称和描述 🧬</h1>"""
outputs = [
    gr.Textbox(
        label="🔎 1. 您查询的 TCGA 项目的癌症类型", show_copy_button=True
    ),  # 1. The Full Name of The Cancer Type Queried.
    gr.Textbox(
        label="👩‍⚕️ 2. 迅速了解这种癌症类型的信息",
        show_copy_button=True,
    ),  # 2. Insight Into The Cancer Type Being Queried. A Quick Look At The Cancer Type Being Queried
]
my_demo = gr.Interface(
    fn=demo,
    inputs=[
        gr.Dropdown(
            choices=[k for k in project_name_TCGA.keys()],
            value="TCGA-READ",
            allow_custom_value=False,
            label="⌨️ 请输入您要查询的 TCGA 项目名称，如 TCGA-READ",
        ),  # Please enter the name of the TCGA project you want to query, such as TCGA-READ.
        gr.Dropdown(
            choices=["Chinese", "English"],
            value="Chinese",
            allow_custom_value=False,
            label="👨‍💻 输出语言目前仅支持中文和英文",
        ),
    ],
    outputs=outputs,
    submit_btn=gr.Button("提交", variant="primary"),
    clear_btn=gr.Button("清除", variant="secondary"),
    cache_examples=True,
    examples=[["TCGA-READ", "Chinese"], ["TCGA-COAD", "English"]],
    description=desc,
    theme="JohnSmith9982/small_and_pretty",
)
my_demo.launch(show_api=False, show_error=True)
