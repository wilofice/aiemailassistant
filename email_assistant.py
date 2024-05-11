#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 19:46:15 2024

@author: galahassa
"""

import streamlit as st
from corellm import generate_response

st.title('AI Email Assitant')

def get_llm_response(input_text):
    ai_response = generate_response(input_text)
    st.info(ai_response)
    
with st.form('my_form'):
    text = st.text_area('Enter text:', 'Ask anything about your emails')
    submitted = st.form_submit_button('Submit')
    if submitted:
        get_llm_response(text)
    