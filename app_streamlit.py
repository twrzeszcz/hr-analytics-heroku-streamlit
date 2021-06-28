import pandas as pd
import streamlit as st
import numpy as np
import pickle
import cv2
import gc
import seaborn as sns
import matplotlib.pyplot as plt
from utils import custom_transformer, skewness_remover
from pycaret.classification import *


st.set_option('deprecation.showPyplotGlobalUse', False)


def main_section():
    st.title('HR Analytics Project')
    background_im = cv2.imread('images/background.jpeg')
    st.image(cv2.cvtColor(background_im, cv2.COLOR_BGR2RGB), use_column_width=True)
    st.markdown('**Data Analysis** section contains some basic information about the train data and allows to perform EDA '
                'with multiple visualization options. In the **Model Performance** section evaluation on the thest data is performed'
                'and some metrics are shown. **Feature Importances** section contains calculated feature importances based on the SHAP values. '
                'Finally **Prediction Service** allows to make predictions on the user input.')
    del background_im
    gc.collect()

@st.cache
def load_preprocessor():
    prep_pipe = pickle.load(open('prep_pipe.pkl', 'rb'))
    return prep_pipe

@st.cache
def load_data():
    df = pd.read_csv('aug_train.csv')
    return df

def load_models(model_type=None):
    if model_type == 'inference':
        stacking_model = load_model('stacking_clf')
        return stacking_model
    else:
        cat_model = load_model('catboost')
        return cat_model

@st.cache
def prep_data():
    df = load_data()
    df_copy = df.copy()
    df_copy.drop_duplicates(subset=df_copy.columns[1:], inplace=True)
    prep_pipe = load_preprocessor()
    to_transform = df_copy.drop('target', axis=1)
    df_caret = prep_pipe.transform(to_transform)
    df_caret['target'] = df_copy['target'].copy()

    setup(data=df_caret, target='target',
                train_size=0.85,
                normalize=True,
                fold=5,
                data_split_stratify=True,
                fix_imbalance=True,
                remove_multicollinearity=True,
                session_id=4000,
                silent=True,
                html=False)

    return df_copy, df_caret, df, to_transform

def data_analysis():
    st.title('Data Analysis')
    if st.sidebar.checkbox('Load dataset'):
        df = load_data()
        st.success('Data successfully loaded')
    if st.sidebar.checkbox('Show data'):
        st.dataframe(df)
    if st.sidebar.checkbox('Display shape'):
        st.write('Size of the data: ', df.shape)
    if st.sidebar.checkbox('Display data types'):
        st.write('Data types', df.dtypes)
    if st.sidebar.checkbox('Display missing values'):
        st.write('Missing values', df.isna().sum())
    if st.sidebar.checkbox('Display duplicated rows'):
        st.write('Number of duplicates: ', df.duplicated(subset=df.columns[1:]).sum())
    if st.sidebar.checkbox('Display unique values'):
        st.write('Unique values for each feature: ', df.nunique())
        selected_columns_unique = st.sidebar.selectbox('Select features', df.columns)
        st.write('Unique values for {}'.format(selected_columns_unique), df[selected_columns_unique].unique())
    if st.sidebar.checkbox('Display correlations heatmap'):
        fig, ax = plt.subplots()
        sns.heatmap(df[df.describe().columns[1:]].corr(), annot=True, ax=ax)
        st.pyplot(fig)
        del fig, ax
    if st.sidebar.checkbox('Display distributions and boxplots'):
        selected_columns_for_distributions_boxplots = st.sidebar.multiselect('Select your preferred numerical features',
                                                                     df.describe().columns)
        for i in selected_columns_for_distributions_boxplots:
            fig, ax = plt.subplots()
            sns.displot(df[i], kde=True)
            st.pyplot()

            fig_1, ax_1 = plt.subplots()
            sns.boxplot(x=df[i])
            st.pyplot()
    if st.sidebar.checkbox('Display countplot'):
        selected_columns_for_countplot = st.sidebar.multiselect('Select your preferred categorical features',
                                                        list(df.select_dtypes('object').columns) + ['target'])
        df_copy = df.copy()
        for col in df_copy.isna().sum()[df_copy.isna().sum() != 0].index:
            df_copy.loc[df_copy[col].isna(), col] = 'Not specified'
        plt.rc('legend', fontsize='medium')
        fig, ax = plt.subplots()
        sns.countplot(data=df_copy, y=selected_columns_for_countplot[0], hue=selected_columns_for_countplot[1], ax=ax)
        st.pyplot()
        ax.legend(loc='best')
        del df_copy

    if st.sidebar.checkbox('Display barplot'):
        selected_columns_for_barplot = st.sidebar.multiselect('Select 2 categorical features',
                                                                     list(df.select_dtypes('object').columns) + ['target'])
        df_copy = df.copy()
        for col in df_copy.isna().sum()[df_copy.isna().sum() != 0].index:
            df_copy.loc[df_copy[col].isna(), col] = 'Not specified'
        selected_column_numerical = st.sidebar.selectbox('Select numerical feature', df.describe().columns[1:-1])
        fig, ax = plt.subplots()
        sns.barplot(data=df_copy, x=selected_columns_for_barplot[0], y=selected_column_numerical, hue=selected_columns_for_barplot[1])
        st.pyplot()
        del df_copy

    if st.sidebar.checkbox('Display scatterplot'):
        df_copy = df.copy()
        for col in df_copy.isna().sum()[df_copy.isna().sum() != 0].index:
            df_copy.loc[df_copy[col].isna(), col] = 'Not specified'
        selected_column_categorical = st.sidebar.selectbox('Select categorical feature',
                                                           list(df.select_dtypes('object').columns) + ['target'])
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_copy, x='city_development_index', y='training_hours',
                    hue=selected_column_categorical)
        st.pyplot()
        del df_copy


def model_performance():
    st.title('Model Performance')

    _1, _2, _3, _4 = prep_data()

    stacking_model = load_models(model_type='inference')
    plot_model(stacking_model.get_params()['trained_model'],
               'auc',
               display_format='streamlit')
    plot_model(stacking_model.get_params()['trained_model'],
               'confusion_matrix',
               display_format='streamlit')
    plot_model(stacking_model.get_params()['trained_model'],
               'class_report',
               display_format='streamlit')
    plot_model(stacking_model.get_params()['trained_model'],
               'error',
               display_format='streamlit')

    del _1, _2, _3, _4, stacking_model


def feature_importances():
    st.title('Feature Importances')

    _1, _2, _3, _4 = prep_data()
    cat_model = load_models()
    if st.sidebar.checkbox('Display feature importances'):
        interpret_model(cat_model.get_params()['trained_model'])
        st.pyplot()
    if st.sidebar.checkbox('Display SHAP values for a feature'):
        feature_name = st.text_input('Select feature from the general plot')
        interpret_model(cat_model.get_params()['trained_model'],
                        plot='correlation',
                        feature=feature_name)
        st.pyplot()

    del _1, _2, _3, _4


def prediction_service():
    df = load_data()
    features = []
    features.append(st.text_input('Enrollee Id', 0))
    features.append(st.selectbox('Select City', df['city'].unique()))
    features.append(float(st.text_input('City development index', 0)))
    features.append(st.selectbox('Select Gender', df['gender'].unique()))
    features.append(st.selectbox('Select Relevent experience', df['relevent_experience'].unique()))
    features.append(st.selectbox('Select Enrolled university', df['enrolled_university'].unique()))
    features.append(st.selectbox('Select Education level', df['education_level'].unique()))
    features.append(st.selectbox('Select Major discipline', df['major_discipline'].unique()))
    features.append(st.selectbox('Select Experience', df['experience'].unique()))
    features.append(st.selectbox('Select Company size', df['company_size'].unique()))
    features.append(st.selectbox('Select Company type', df['company_type'].unique()))
    features.append(st.selectbox('Select Last new job', df['last_new_job'].unique()))
    features.append(float(st.text_input('Training hours', 0)))
    if st.sidebar.button('Predict'):
        df_test = pd.DataFrame([features], columns=df.columns[:-1])
        stacking_model = load_models(model_type='inference')
        prep_pipe = load_preprocessor()
        df_test_prep = prep_pipe.transform(df_test)
        st.info('Predicted class: {}'.format(predict_model(stacking_model, data=df_test_prep)['Label'].values[0]))
        del stacking_model

activities = ['Main', 'Data Analysis', 'Model Performance', 'Feature Importances', 'Prediction Service', 'About']
option = st.sidebar.selectbox('Select Option', activities)

if option == 'Main':
    main_section()

if option == 'Data Analysis':
    data_analysis()
    gc.collect()

if option == 'Model Performance':
    model_performance()
    gc.collect()

if option == 'Feature Importances':
    feature_importances()
    gc.collect()

if option == 'Prediction Service':
    prediction_service()
    gc.collect()

if option == 'About':
    st.title('About')
    st.write('This is an interactive website for the HR Analytics Project. Data was taken from kaggle.')