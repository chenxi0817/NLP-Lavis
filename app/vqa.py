"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# 导入Streamlit库，用于构建Web应用
import streamlit as st
# 从app模块导入加载演示图像的函数和设备配置
from app import load_demo_image, device
# 从app.utils模块导入加载模型缓存的函数
from app.utils import load_model_cache
# 从lavis.processors模块导入加载处理器的函数
from lavis.processors import load_processor
# 从PIL库导入Image模块，用于处理图像
from PIL import Image


# 定义视觉问答应用的主函数
def app():
    # 在侧边栏添加一个下拉选择框，让用户选择使用的模型，目前仅支持BLIP模型
    model_type = st.sidebar.selectbox("Model:", ["BLIP"])

    # ===== 页面布局 =====
    # 在页面中添加一个居中的标题，显示应用名称
    st.markdown(
        "<h1 style='text-align: center;'>Visual Question Answering</h1>",
        unsafe_allow_html=True
    )

    # 定义文件上传提示信息
    instructions = """Try the provided image or upload your own:"
    # 添加一个文件上传组件，让用户可以选择上传自己的图像或使用提供的示例图像
    file = st.file_uploader(instructions)

    # 将页面分为两列，用于分别显示图像和问题、答案区域
    col1, col2 = st.columns(2)

    # 在第一列添加一个标题，显示图像区域
    col1.header("Image")
    # 如果用户上传了文件，则打开并转换为RGB格式的图像
    if file:
        raw_img = Image.open(file).convert("RGB")
    # 否则，加载默认的演示图像
    else:
        raw_img = load_demo_image()

    # 获取原始图像的宽度和高度
    w, h = raw_img.size
    # 计算缩放因子，将图像宽度缩放到720像素
    scaling_factor = 720 / w
    # 根据缩放因子调整图像大小
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))

    # 在第一列显示缩放后的图像，并自适应列宽
    col1.image(resized_image, use_column_width=True)
    # 在第二列添加一个标题，显示问题输入区域
    col2.header("Question")

    # 在第二列添加一个文本输入框，让用户输入问题，并提供默认问题
    user_question = col2.text_input("Input your question!", "What are objects there?")
    # 在页面中添加一个提交按钮
    qa_button = st.button("Submit")

    # 在第二列添加一个标题，显示答案输出区域
    col2.header("Answer")

    # ===== 事件处理 =====
    # 加载BLIP图像评估处理器，并设置图像大小为480x480
    vis_processor = load_processor("blip_image_eval").build(image_size=480)
    # 加载BLIP问题处理器
    text_processor = load_processor("blip_question").build()

    # 如果用户点击了提交按钮
    if qa_button:
        # 如果用户选择的模型是BLIP系列
        if model_type.startswith("BLIP"):
            # 从缓存中加载BLIP视觉问答模型，使用VQAv2数据集进行评估
            model = load_model_cache(
                "blip_vqa", model_type="vqav2", is_eval=True, device=device
            )

            # 使用图像处理器对原始图像进行处理，并添加一个批次维度，然后移动到指定设备上
            img = vis_processor(raw_img).unsqueeze(0).to(device)
            # 使用文本处理器对用户输入的问题进行处理
            question = text_processor(user_question)

            # 构建视觉问答输入样本，包含处理后的图像和问题
            vqa_samples = {"image": img, "text_input": [question]}
            # 使用模型预测问题的答案，采用生成式推理方法
            answers = model.predict_answers(vqa_samples, inference_method="generate")

            # 在第二列显示模型预测的答案，并自适应列宽
            col2.write("\n".join(answers), use_column_width=True)
