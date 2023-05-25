# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 01:27:45 2023

@author: shivangi
"""

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import os

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

import pycaret
from pycaret.classification import setup, compare_models, pull, save_model, ClassificationExperiment
from pycaret.regression import setup, compare_models, pull, save_model, RegressionExperiment


#Giving the title to the application
st.title("Machine Learning Web App")
#adding the image
st.image("https://www.bing.com/th/id/OGC.83e7a2516286a0a1d673cf5ecdbe5079?pid=1.7&rurl=https%3a%2f%2fmiro.medium.com%2fmax%2f2878%2f0*M50IPKZz58Fyy178.gif&ehk=X%2bw%2fZv4edQ2rNLNF5ByjruaIOvF8rMGAExCjtHWWwx4%3d")


#
if os.path.exists("sourcev.csv"):
    df = pd.read_csv("sourcev.csv",index_col=None)

with st.sidebar:
    st.header("Welcome to MLapp")
    st.subheader("Following Application is made for machine learning models.")
    st.caption("Choose your parameters :- ")
    choose=st.radio("👇",["Dataset","Analysis","Training","Download"])

# upload the dataset from the directory
if choose=="Dataset":
    st.write("Please upload your dataset here.")
    dataset_value = st.file_uploader("Upload your dataset here")
   
    if dataset_value:
        df = pd.read_csv(dataset_value, index_col=None)
        df.to_csv("sourcev.csv", index = None)
        st.dataframe(df)
# Performing Exploratory data analysis
if choose=="Analysis":
    st.subheader("Perform profiling on Dataset")
    if st.sidebar.button("Do Analysis"):
        profile_report = df.profile_report()
        st_profile_report(profile_report)
# train the model with either regression or classification   
if choose=="Training":
    st.header("Train your model here:-")
    choice = st.sidebar.selectbox("Select your Technique:", ["Classification","Regression"])
 # Select here the variable which you want to classify further.
    target = st.selectbox("Select you Target Variable",df.columns)
    if choice=="Classification":
        if st.sidebar.button("Classification Train"):
            s1 = ClassificationExperiment()
            s1.setup(data=df, target=target)
            setup_df = s1.pull()
            st.info("The Setup data is as follows:")
            st.table(setup_df)
 #comparing the models          
            best_model1 = s1.compare_models()
            compare_model = s1.pull()
            st.info("The Comparison of models is as folows:")
            st.table(compare_model)
#get the best model            
            best_model1
            s1.save_model(best_model1,"Machine Learning Model")
    else:
        if st.sidebar.button("Regression Train"):
            s2 = RegressionExperiment()
            s2.setup(data=df, target=target)
            setup_df = s2.pull()
            st.info("The Setup data is as follows:")
            st.table(setup_df)
           
            best_model2 = s2.compare_models()
            compare_model = s2.pull()
            st.info("The Comparison of models is as folows:")
            st.table(compare_model)
           
            best_model2
            s2.save_model(best_model2,"Machine Learning Model")

if choose =="Download":
    with open("Machine Learning model.pkl",'rb') as f:
        st.caption("Download your model from here:")
        st.download_button("Download the file",f,"Machine Learning model.pkl")

