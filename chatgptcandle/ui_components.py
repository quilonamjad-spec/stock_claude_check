"""Reusable Streamlit presentation helpers."""

import streamlit as st

def colour(bias):
    return "#16a34a" if bias=="Bullish" else "#dc2626" if bias=="Bearish" else "#6b7280"

def metric(label, value, bias="Neutral"):
    st.markdown(
        f"<div style='color:#9ca3af;font-size:.82rem'>{label}</div>"
        f"<div style='font-size:1.55rem;font-weight:700;color:{colour(bias)}'>{value}</div>",
        unsafe_allow_html=True)

def box(title, lines, bias):
    c=colour(bias)
    html="".join(f"<div style='margin:4px 0'>{x}</div>" for x in lines)
    st.markdown(
        f"<div style='border-left:5px solid {c};border-radius:8px;"
        f"padding:12px 14px;background:rgba(127,127,127,.08);min-height:130px>"
        f"<div style='font-size:1.05rem;font-weight:700;color:{c};margin-bottom:7px'>{title}</div>"
        f"{html}</div>", unsafe_allow_html=True)
