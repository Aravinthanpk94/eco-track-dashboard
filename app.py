import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------

st.set_page_config(page_title="EcoTrack Dashboard", layout="wide")

st.title("🌍 EcoTrack - Smart Sustainability Dashboard")

# ---------------- ML MODEL ----------------

X_train = [
    [50,40,10,5,5],
    [30,20,5,20,15],
    [10,5,2,30,25],
    [40,30,5,10,5],
    [15,10,3,25,20],
]
y_train = [0,2,4,1,3]

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# ---------------- SESSION STATE ----------------

for key in ["dry","wet","ewaste","reuse","reduce","eco_score"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if key != "eco_score" else 100

CARBON_WASTE = 0.5
CARBON_REUSE = 0.7
CARBON_REDUCE = 1.0

# ---------------- INPUT SECTION ----------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("♻ Waste Input")
    st.session_state.dry += st.number_input("Add Dry Waste", 0)
    st.session_state.wet += st.number_input("Add Wet Waste", 0)
    st.session_state.ewaste += st.number_input("Add E-Waste", 0)

with col2:
    st.subheader("🔁 Reuse & Reduce")
    st.session_state.reuse += st.number_input("Add Reuse Items", 0)
    st.session_state.reduce += st.number_input("Add Reduce Items", 0)

# ---------------- CALCULATIONS ----------------

total_waste = st.session_state.dry + st.session_state.wet + st.session_state.ewaste

carbon = (
    st.session_state.reuse * CARBON_REUSE +
    st.session_state.reduce * CARBON_REDUCE -
    total_waste * CARBON_WASTE
)

eco_score = max(
    0,
    min(100,
        100
        - total_waste * 0.7
        + st.session_state.reuse * 0.8
        + st.session_state.reduce * 1.0)
)

# ---------------- IMPACT CLASSIFICATION ----------------

if carbon > 5:
    impact = "🌟 Strong Positive Environmental Impact"
    impact_color = "green"
    fact = "Equivalent to multiple trees absorbing CO₂ annually."
elif carbon > 1:
    impact = "👍 Positive Environmental Contribution"
    impact_color = "lightgreen"
    fact = "You are reducing landfill and production emissions."
elif carbon >= -1:
    impact = "⚖ Neutral Environmental Impact"
    impact_color = "gray"
    fact = "Balanced performance. Increase reduction for stronger impact."
elif carbon >= -5:
    impact = "⚠ Negative Environmental Impact"
    impact_color = "orange"
    fact = "Waste patterns are increasing landfill emissions."
else:
    impact = "🚨 High Carbon Footprint"
    impact_color = "red"
    fact = "Comparable to driving several kilometers in a petrol vehicle."

# ---------------- ECO BADGE ----------------

if eco_score >= 90:
    badge = "🌟 Platinum"
elif eco_score >= 75:
    badge = "🥇 Gold"
elif eco_score >= 60:
    badge = "🥈 Silver"
elif eco_score >= 40:
    badge = "🥉 Bronze"
else:
    badge = "🔴 Needs Improvement"

# ---------------- ML PREDICTION ----------------

features = np.array([[st.session_state.dry,
                      st.session_state.wet,
                      st.session_state.ewaste,
                      st.session_state.reuse,
                      st.session_state.reduce]])

prediction = model.predict(features)[0]

ai_messages = {
    0: "Reduce consumption significantly.",
    1: "Increase reuse habits.",
    2: "Improve reduction strategy.",
    3: "Good sustainable behavior.",
    4: "Excellent eco performance!"
}

# ---------------- DASHBOARD ----------------

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.metric("Sustainability Score", round(eco_score,1))
    st.metric("Carbon Impact (kg CO₂)", round(carbon,2))
    st.markdown(f"### {badge}")

with col4:
    st.markdown(f"### {impact}")
    st.write(fact)
    st.markdown(f"**AI Insight:** {ai_messages[prediction]}")

# ---------------- GRAPH ----------------

st.divider()
st.subheader("📊 Sustainability Overview")

fig, ax = plt.subplots(figsize=(6,3))
categories = ["Dry","Wet","E-Waste","Reuse","Reduce"]
values = [
    st.session_state.dry,
    st.session_state.wet,
    st.session_state.ewaste,
    st.session_state.reuse,
    st.session_state.reduce
]

ax.bar(categories, values)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)

st.pyplot(fig)