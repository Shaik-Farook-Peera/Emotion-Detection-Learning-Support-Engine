import streamlit as st

from services.bilstm_model import BiLSTMClassifier
from services.bert_model import BERTClassifier


@st.cache_resource
def get_bilstm():

    return BiLSTMClassifier()


@st.cache_resource
def get_bert():

    return BERTClassifier()