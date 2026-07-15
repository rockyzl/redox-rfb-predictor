"""Small local web interface for Redox RFB Predictor.

Run with:
    streamlit run app.py
"""

import streamlit as st
from rdkit import Chem

from redox_rfb import predict


st.set_page_config(page_title="Redox RFB Predictor", page_icon="⚡", layout="centered")

st.title("⚡ Redox RFB Predictor")
st.caption("用分子结构预测水系有机液流电池候选物的氧化还原电位")

st.markdown(
    "输入一个 SMILES。模型输出的是 **RedDB DFT 参考尺度**上的预测电位（V vs SHE），"
    "适合在相近的醌类和含氮芳香分子中做筛选与排序；它不是实验测量值。"
)

examples = {
    "对苯醌（默认）": "O=C1C=CC(=O)C=C1",
    "吡啶": "c1ccncc1",
    "蒽醌": "O=C1c2ccccc2C(=O)c2ccccc21",
}

with st.form("prediction"):
    choice = st.selectbox("示例分子", list(examples))
    smiles = st.text_input("SMILES", value=examples[choice])
    model_label = st.radio(
        "模型",
        options=["快速：RDKit + Morgan fingerprint", "混合：RDKit + xTB"],
        help="xTB 混合模型会先做一次量子化学近似计算，通常需要 10–60 秒。",
    )
    submitted = st.form_submit_button("预测电位", type="primary")

if submitted:
    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if not smiles or mol is None:
        st.error("请输入有效的 SMILES，例如：O=C1C=CC(=O)C=C1")
    else:
        model = "hybrid" if model_label.startswith("混合") else "rdkit"
        label = "RDKit + xTB" if model == "hybrid" else "RDKit + Morgan fingerprint"
        try:
            with st.spinner("正在运行 xTB…" if model == "hybrid" else "正在计算分子特征…"):
                value = predict(smiles, model=model)
            st.metric("预测氧化还原电位", f"{value:.3f} V vs SHE")
            st.success(f"模型：{label}")
        except Exception as exc:
            st.error(f"预测未完成：{exc}")

st.divider()
st.subheader("这两个模型有什么区别？")
st.markdown(
    "- **快速模型**：从 RDKit 描述符和结构指纹中学习，约毫秒级，适合大批量初筛。\n"
    "- **混合模型**：额外计算 xTB 的 HOMO、LUMO、能隙等电子结构特征，较慢，适合对少量候选分子复筛。\n"
    "- 两者都在 RedDB 的 DFT 标签上训练；若分子远离醌类/含氮芳香训练域，结果属于外推，应谨慎解释。"
)
