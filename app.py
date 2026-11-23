import streamlit as st
import eda, toxicity

st.set_page_config(page_title="YouTube Toxicity Analyzer", layout="wide")
st.title("YouTube Comment Toxicity Analyzer")

@st.cache_data
def load():
    return eda.load_frames()

df = load()

channels = st.sidebar.multiselect("Channels", sorted(df["channel"].dropna().unique()),
                                  default=sorted(df["channel"].dropna().unique()))
df = df[df["channel"].isin(channels)]

c1, c2, c3 = st.columns(3)
c1.metric("Comments", f"{len(df):,}")
c2.metric("Toxic comments", f"{int(df['is_toxic'].sum()):,}")
c3.metric("Toxicity rate", f"{df['is_toxic'].mean():.1%}")

left, right = st.columns(2)
with left:
    st.subheader("Toxicity rate by channel")
    chan = eda.by_channel(df)
    st.bar_chart(chan["toxicity_rate"])
    st.dataframe(chan)
with right:
    st.subheader("Toxicity rate by topic")
    topic = eda.by_topic(df)
    st.bar_chart(topic["toxicity_rate"])
    st.dataframe(topic)

st.subheader("Toxicity vs. video views")
res, pv = eda.toxicity_vs_views(df)
pv = pv.assign(log_views=lambda x: x["view_count"].apply(lambda v: __import__("math").log1p(v)))
st.scatter_chart(pv, x="log_views", y="toxicity_rate", color="channel")
st.json(res)

st.subheader("Most toxic comments")
st.dataframe(df.sort_values("toxicity_score", ascending=False)
               [["channel", "text_clean", "toxicity_score"]].head(20))

st.subheader("Try it yourself")
txt = st.text_area("Type a comment to score")
if st.button("Analyze") and txt.strip():
    r = toxicity.predict(txt)
    st.write(f"Toxicity score: **{r['toxicity']}**, toxic: **{r['is_toxic']}**")
    st.json(r["categories"])
