import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
import plotly.express as px


import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
# template taken from:
# https://blog.streamlit.io/crafting-a-dashboard-app-in-python-using-streamlit/

# the usual loading process

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
data_path = os.path.join(current_dir, 'data')
out_path = os.path.join(current_dir, 'outputs')     # For the output
# Read the files
df = pd.read_csv(os.path.join(data_path, 'data_predicted.csv'))
df.head()

# we're using the original data, so we'll clean it a bit
# we'll also select a subset which doesn't have missing values. Not great, but it'll be a bit easier
# construction_year
df['construction_year'] = df['construction_year'].replace(0, np.nan)
# gps_height
df['gps_height'] = df['gps_height'].apply(lambda x: np.nan if x <= 0 else x)
df['longitude'] = df['longitude'].replace(0, np.nan)
df['latitude'] = df['latitude'].where(df['latitude'] < -0.5, np.nan)

# select a subset
varlist = ('quantity', 'latitude', 'longitude', 'construction_year',
    'amount_tsh', 'extraction_type_class', 'gps_height', 'source',
    'region', 'ward', 'lga', 'basin',  'water_quality', 'permit',
    'status_group', 'predicted')
df_small = df[np.intersect1d(df.columns, varlist)].dropna()
df_small = df_small.iloc[0:5000,:]
# create accuracy value
accuracy = df.status_group == df.predicted
df_small = df_small.assign(correct = lambda df_small: df.status_group == df.predicted)
df_heat = df_small.groupby('region').correct.sum()

# there's gotta be an easier way, but this works
f_f = (df_small.status_group == 'functional') & (df_small.predicted == 'funtional')
f_fnr = (df_small.status_group == 'functional') & (df_small.predicted == 'funtional needs repair')
f_nf = (df_small.status_group == 'functional') & (df_small.predicted == 'not funtional')

fnr_f = (df_small.status_group == 'functional needs repair') & (df_small.predicted == 'funtional')
fnr_fnr = (df_small.status_group == 'functional needs repair') & (df_small.predicted == 'funtional needs repair')
fnr_nf = (df_small.status_group == 'functional needs repair') & (df_small.predicted == 'not funtional')

nf_f = (df_small.status_group == 'non functional') & (df_small.predicted == 'funtional')
nf_fnr = (df_small.status_group == 'non functional') & (df_small.predicted == 'funtional needs repair')
nf_nf = (df_small.status_group == 'non functional') & (df_small.predicted == 'not funtional')

df_small['f_f'] = f_f
df_small['f_fnr'] = f_fnr
df_small['f_nf'] = f_nf
df_small['fnr_f'] = fnr_f
df_small['fnr_fnr'] = fnr_fnr
df_small['fnr_nf'] = fnr_nf
df_small['nf_f'] = nf_f
df_small['nf_fnr'] = nf_fnr
df_small['nf_nf'] =nf_nf

df_f = pd.DataFrame({"Region": df_small['region'], "Functional": f_f, "Functional Needs Repair": f_fnr, "Not Functional": f_nf})
df_fnr = pd.DataFrame({"Region": df_small['region'], "Functional": fnr_f, "Functional Needs Repair": fnr_fnr, "Not Functional": fnr_nf})
df_nf = pd.DataFrame({"Region": df_small['region'], "Functional": nf_f, "Functional Needs Repair": nf_fnr, "Not Functional": nf_nf})

# for the heatmap
df_heat = df_small.groupby('region').correct.sum()


# create accuracy value
accuracy = df.status_group == df.predicted
df_small = df_small.assign(correct = lambda df_small: df.status_group == df.predicted)

# set up page config, here is where that gets changed
st.set_page_config(
    page_title="Get Pumped Up Dashboard",
    page_icon="🤠",
    layout="wide",
    initial_sidebar_state="expanded")

alt.theme.enable("dark")


###################
###################


# build the sidebar, get the dropdown menues
with st.sidebar:
    st.title('Get Pumped Up Dashboard')

    region_list = list(df_small.region.unique())
    status_list = list(df_small.status_group.unique())

    selected_region = st.selectbox('Select a region', region_list, index=len(region_list)-1)
    df_selected_region = df_small[df_small.region == selected_region]
    df_selected_region_sorted = df_selected_region.sort_values(by="correct", ascending=False)

    selected_status = st.selectbox('Select a status', status_list, index=len(status_list)-1)
    df_selected_status = df_small[df_small.status_group == selected_status]
    df_selected_status_sorted = df_selected_status.sort_values(by="correct", ascending=False)
    if selected_status == "functional":
        df_selected_status_correct = df_f
    elif selected_status == "functional needs repair":
        df_selected_status_correct = df_fnr
    else:
        df_selected_status_correct = df_nf

    selected_feature = st.selectbox('Select a feature', varlist, index=len(status_list)-1)
    df_selected_feature = df_small[selected_feature]

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)


# makes a heatmap i think?
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
            y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Number Correct", titleFontSize=19, titlePadding=16, titleFontWeight=900, labelAngle=0)),
            x=alt.X(f'{input_x}:O', axis=alt.Axis(title="Region", titleFontSize=19, titlePadding=16, titleFontWeight=900)),
            color=alt.Color(f'max({input_color}):Q',
                             legend=None,
                             scale=alt.Scale(scheme=input_color_theme)),
            stroke=alt.value('black'),
            strokeWidth=alt.value(0.25),
        ).properties(width=900
        ).configure_axis(
        labelFontSize=14,
        titleFontSize=14
        )
    # height=300
    return heatmap


# shaded map, play around with it?
def make_choropleth(input_df, input_id, input_column, input_color_theme):
    choropleth = px.choropleth(input_df, color=input_column,
                               locationmode = 'country names',
                               locations = ["Tanzania"],
                               color_continuous_scale=input_color_theme,
                               range_color=(0, max(df_selected_region.correct)),
                               scope="Tanzania",
                               labels={'correct':'Correctly classified'}
                              )
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    return choropleth

# little donut widget
# first func calculates difference
def calculate_correct_difference(input_df, input_region):
    select_region_data = input_df[input_df['region'] == input_region].reset_index()
    select_region_data['percent_correct'] = select_region_data.correct.sum()/len(select_region_data)
    return pd.concat([select_region_data.region, select_region_data.status_group, select_region_data.predicted,
                      select_region_data.correct, select_region_data.percent_correct], axis=1).sort_values(by="percent_correct", ascending=False)
#return pd.concat([selected_year_data.states, selected_year_data.id, selected_year_data.population,
#                  selected_year_data.population_difference], axis=1).sort_values(by="population_difference", ascending=False)


# seconf func makes the pretty images
def make_donut(input_response, input_text, input_color):
  if input_color == 'blue':
      chart_color = ['#29b5e8', '#155F7A']
  if input_color == 'green':
      chart_color = ['#27AE60', '#12783D']
  if input_color == 'orange':
      chart_color = ['#F39C12', '#875A12']
  if input_color == 'red':
      chart_color = ['#E74C3C', '#781F16']

  source = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100-input_response, input_response]
  })
  source_bg = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100, 0]
  })

  plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          #domain=['A', 'B'],
                          domain=[input_text, ''],
                          # range=['#29b5e8', '#155F7A']),  # 31333F
                          range=chart_color),
                      legend=None),
  ).properties(width=130, height=130)

  text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
  plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          # domain=['A', 'B'],
                          domain=[input_text, ''],
                          range=chart_color),  # 31333F
                      legend=None),
  ).properties(width=130, height=130)
  return plot_bg + plot + text


# now, more layout things:
# number of columns
col = st.columns((1.5, 4.5, 2), gap='medium')

# column 0
with col[0]:
    st.markdown('#### Classification Performance')

    df_correct_difference_sorted = round(calculate_correct_difference(df_small, selected_region)*100, 2)

    if selected_status == 'functional':
        first_region_name = df_correct_difference_sorted.region.iloc[0]
        first_region_correct = df_correct_difference_sorted.correct.iloc[0]
        first_region_delta = round(df_small[(df_small.region == selected_region) & (df_small.status_group == 'functional')].correct.sum()/len(df_small)*100, 2)
    elif selected_status == 'non functional':
        last_region_name = df_correct_difference_sorted.region.iloc[-1]
#        last_region_correct = format_number(df_correct_difference_sorted.correct.iloc[-1])
        first_region_delta = round(df_small[(df_small.region == selected_region) & (df_small.status_group == 'non functional')].correct.sum()/len(df_small)*100, 2)
    elif selected_status == 'functional needs repair':
        first_region_name = df_correct_difference_sorted.region.iloc[0]
        first_region_correct = df_correct_difference_sorted.correct.iloc[0]
        first_region_delta = round(df_small[(df_small.region == selected_region) & (df_small.status_group == 'functional needs repair')].correct.sum()/len(df_small)*100, 2)
    else:
        last_state_name = '-'
        last_state_population = '-'
        last_state_delta = ''
    st.metric(label=selected_region, value=first_region_delta)




    st.markdown('#### True values classified:')

    if selected_status == 'functional':
        # filter the regional values to reflect outcomes
        # X_Y is true value X, classified as Y
        f_f_raw = (df_small.status_group == 'functional') & (df_small.predicted == 'funtional') & (df_small.region == selected_region)
        f_f = round(f_f_raw.sum()/len(f_f_raw), 3)*100
        f_fnr_raw = (df_small.status_group == 'functional') & (df_small.predicted == 'funtional needs repair') & (df_small.region == selected_region)
        f_fnr = round(f_f_raw.sum()/len(f_fnr_raw), 3)*100
        f_nf_raw = (df_small.status_group == 'functional') & (df_small.predicted == 'non functional') & (df_small.region == selected_region)
        f_nf = round(f_f_raw.sum()/len(f_nf_raw), 3)*100
        donut_chart_upper = make_donut(f_f, 'Correctly classified!', 'green')
        donut_chart_middle = make_donut(f_fnr, 'True value: functional needs repair', 'blue')
        donut_chart_lower = make_donut(f_nf, 'True value: not functional', 'red')
    elif selected_status == 'functional needs repair':
        fnr_f_raw = (df_small.status_group == 'functional needs repair') & (df_small.predicted == 'funtional') & (df_small.region == selected_region)
        fnr_f = round(fnr_f_raw.sum()/len(fnr_f_raw), 3)*100
        fnr_fnr_raw = (df_small.status_group == 'functional needs repair') & (df_small.predicted == 'funtional needs repair') & (df_small.region == selected_region)
        fnr_fnr = round(fnr_fnr_raw.sum()/len(fnr_fnr_raw), 3)*100
        fnr_nf_raw = (df_small.status_group == 'functional needs repair') & (df_small.predicted == 'non funtional') & (df_small.region == selected_region)
        fnr_nf = round(fnr_nf_raw.sum()/len(fnr_nf_raw), 3)*100
        donut_chart_upper = make_donut(fnr_fnr, 'Correctly classified!', 'green')
        donut_chart_middle= make_donut(fnr_f, 'True value: functional', 'blue')
        donut_chart_lower = make_donut(fnr_nf, 'True value: non functional', 'red')
    elif selected_status == 'non functional':
        nf_f_raw = (df_correct_difference_sorted.status_group == 'non functional') & (df_correct_difference_sorted.predicted == 'funtional')
        nf_f = round(nf_f_raw.sum()/len(nf_f_raw), 3)*100
        nf_fnr_raw = (df_correct_difference_sorted.status_group == 'non functional') & (df_correct_difference_sorted.predicted == 'funtional needs repair')
        nf_fnr = round(nf_fnr_raw.sum()/len(nf_fnr_raw), 3)*100
        nf_nf_raw = (df_correct_difference_sorted.status_group == 'non functional') & (df_correct_difference_sorted.predicted == 'not funtional')
        nf_nf = round(nf_nf_raw.sum()/len(nf_nf_raw), 3)*100
        donut_chart_upper = make_donut(nf_fnr, 'Correctly classified!', 'green')
        donut_chart_middle= make_donut(nf_f, 'True value: functional', 'blue')
        donut_chart_lower = make_donut(nf_nf, 'True value: functional needs repair', 'red')
    else:
        df_nf_fnr = 0
        df_nf_f = 0
        df_nf = 0
        donut_chart_upper = make_donut(nf_fnr, 'As functional!', 'green')
        donut_chart_middle= make_donut(nf_f, 'As functional needs repair:', 'blue')
        donut_chart_lower = make_donut(nf_nf, 'As non functional', 'red')

    accuracy_col = st.columns((0.2, 1, 0.2))
    with accuracy_col[1]:
        st.write('Functional')
        st.altair_chart(donut_chart_upper)
        st.write('Functional needs repair')
        st.altair_chart(donut_chart_middle)
        st.write('Non functional')
        st.altair_chart(donut_chart_lower)


# column 1

with col[1]:
    st.markdown('#### Total Accuracy')

#    choropleth = make_choropleth(df_selected_status, 'region', 'correct', selected_color_theme)
#    st.plotly_chart(choropleth, use_container_width=True)

    # heatmap = make_heatmap(df_small, 'status', 'region', df_heat, selected_color_theme)
    # st.altair_chart(heatmap, use_container_width=True)


# column 2

with col[2]:
    st.markdown('#### Top Thinges') # link to SHAP here??

    st.dataframe(df_correct_difference_sorted,
                 column_order=("regions", "correct"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "regions": st.column_config.TextColumn(
                        "Regions",
                    ),
                    "correct": st.column_config.ProgressColumn(
                        "Number Correct",
                        format="%f",
                        min_value=0,
                        max_value=max(df_selected_region_sorted.correct),
                     )}
                 )

    with st.expander('About', expanded=True):
        st.write('''
            - Model: XGBoost
            - :orange[**Performance**]: based on a subset of the test data
            - :orange[**Something**]: else goes here
            ''')



















