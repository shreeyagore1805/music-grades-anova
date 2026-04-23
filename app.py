import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

st.title("🎵 Does Music Affect Grades? — ANOVA Study")

# ── Load Data ──────────────────────────────────────────────
SHEET_ID = "1tORUTI7wzhTjRULPB_oKDWJigni8sagR0cihpdpsIzw"
url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

df = pd.read_csv(url, usecols=[2, 3, 4, 5, 6])
df.columns = ["study_hours", "music", "genre", "grade", "course"]
# ── Clean Data (FIX) ───────────────────────────────────────
df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
df = df.dropna()  # IMPORTANT: drop NaNs after conversion

# ── ANOVA ──────────────────────────────────────────────────
st.subheader("📊 One-Way ANOVA Result")

# Extra safety: ensure genre is string
df["genre"] = df["genre"].astype(str)

model = ols('grade ~ C(genre)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=1)

st.dataframe(anova_table)

# ✅ FIX: dynamically find p-value column
p_col = [col for col in anova_table.columns if "PR" in col or "p" in col.lower()][0]
p_value = anova_table[p_col].iloc[0]

# ── Result ────────────────────────────────────────────────
if p_value < 0.05:
    st.success(f"p-value = {p_value:.4f} → REJECT H0: Music type significantly affects grades.")
else:
    st.warning(f"p-value = {p_value:.4f} → FAIL TO REJECT H0: No significant effect found.")

# ── Tukey HSD ──────────────────────────────────────────────
st.subheader("🔍 Tukey HSD Post-Hoc Test")
tukey = pairwise_tukeyhsd(df["grade"], groups=df["genre"])
st.text(str(tukey._results_table))

# ── Chart ──────────────────────────────────────────────────
st.subheader("📈 Average Grade by Music Type")
fig, ax = plt.subplots()
df.groupby("genre")["grade"].mean().plot(kind="bar", color="steelblue", edgecolor="black", ax=ax)
ax.set_xlabel("Music Type")
ax.set_ylabel("Average Grade (%)")
ax.set_title("Average Grade by Music Genre")
st.pyplot(fig)

# Assuming you already calculated your p_value
st.write(f"### P-value: {p_value:.4f}")

# The Decision Logic
alpha = 0.05

if p_value < alpha:
    st.success("### Result: Reject the Null Hypothesis")
    st.write("There is a **significant difference** in grades between the groups. Music appears to have an impact.")
else:
    st.error("### Result: Fail to Reject the Null Hypothesis")
    st.write("There is **no significant difference** in grades. Any observed differences are likely due to random chance.")
