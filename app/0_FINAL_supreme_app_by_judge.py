import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import matplotlib.font_manager as font_manager
from PIL import Image

Title_html = """
    <style>
         body {
             background-image: url("https://i.ibb.co/KzTLw6r/SCOTUS.jpg");
             background-color: #cccccc;
        }

        .title h1{
          user-select: none;
          font-size: 60px;
          color: white;
          background: repeating-linear-gradient(-45deg, red 0%, blue 10%, red 20%, blue 30%, red 40%, blue 50%,
          red 60%, blue 70%, red 80%, blue 90%);
          background-size: 600vw 600vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 15s linear infinite forwards;
        }
        @keyframes slide {
          0%{
            background-position-x: 0%;
          }
          100%{
            background-position-x: 600vw;
          }
        }

        span.c {
            display: block;
            width: 100px;
            height: 100px;
            padding: 5px;
            border: 1px solid blue;    
            background-color: yellow; 

    </style> 
    
    <div class="title">
        <h1>Fantasy SCOTUS</h1>
    </div>
    """
st.markdown(Title_html, unsafe_allow_html=True) #Title rendering


# Call on existing data sets - one dataframe with supreme court vote data, one
# that correlates the issues to the proper number
supreme = pd.read_csv('data/supreme_1023.csv')
justice_details = pd.read_csv('data/JusticeDetails.csv')
popular_issues = pd.read_csv('data/popular_issues_1026.csv')
sample_cases = pd.read_csv('data/web_app_sample_cases.csv')

# Set list of issue descriptions
issueDescripts = popular_issues['issueDescript']
judgeSelects = justice_details['justiceName']
caseNames = sample_cases['case_name']

# Sets list of average MQ per term 
#terms_plus_MQ = supreme.groupby('term').mean()['subCourt_MQ']

'''
# Select an Issue to Vote On:
'''
 
issue = st.selectbox('Issue Description:', issueDescripts)
st.text("")
 
'''
# OR 
# Select a Landmark Historical Case:
'''

caseTitle = st.selectbox('Case Title:', caseNames)
''' '''
''' '''
if issue != '<select>' or caseTitle !='<select>':

    st.sidebar.write('# Select your Fantasy SCOTUS:')

    judge1 = st.sidebar.selectbox('Select your Chief Justice', judgeSelects,index=32)
    judge2 = st.sidebar.selectbox('Select Associate Justice 1', judgeSelects,index=29)
    judge3 = st.sidebar.selectbox('Select Associate Justice 2', judgeSelects, index=30)
    judge4 = st.sidebar.selectbox('Select Associate Justice 3', judgeSelects, index=31)
    judge5 = st.sidebar.selectbox('Select Associate Justice 4', judgeSelects, index=33)
    judge6 = st.sidebar.selectbox('Select Associate Justice 5', judgeSelects, index=34)
    judge7 = st.sidebar.selectbox('Select Associate Justice 6', judgeSelects, index=35)
    judge8 = st.sidebar.selectbox('Select Associate Justice 7', judgeSelects, index=36)
    judge9 = st.sidebar.selectbox('Select Associate Justice 8', judgeSelects, index=37)
    st.text("")
    st.text("")

    # First prompt, with 10 selections

    # Second prompt, asking for a year
    #term = st.slider('Select a year: ', 1946, 2019)

    # Define issue number and area number from category based on user choice
    issue_number = popular_issues[popular_issues['issueDescript'] == issue]['issueNumber']
    issue_area = popular_issues[popular_issues['issueDescript'] == issue]['issueArea']
    judges = [judge1, judge2, judge3, judge4, judge5, judge6, judge7, judge8, judge9]
    MQ_list = []
    for judge in judges:
        judge_mq = justice_details[justice_details['justiceName'] == judge]['MQ'].values
        MQ_list.append(judge_mq)
    Summed_MQ = sum(MQ_list)
    Summed_MQ = float(Summed_MQ)

    # Open model and apply to given parameters

    three_factor_model = pickle.load(open('models/decision_tree_3_factor_continuous_model.sav', 'rb'))
    five_factor_model = pickle.load(open('models/random_forest_5_factor_continuous_model.sav', 'rb'))
    numbers = np.array([int(Summed_MQ),int(issue_number),int(issue_area)])
    numbers = numbers.reshape(1,-1)

    result = three_factor_model.predict(numbers)

    # Classify numbers in model 

    if result == 1:
        result = 'Liberal'
    if result == 0:
        result = 'Conservative'

    Summed_MQ = np.round(Summed_MQ,3)

    if Summed_MQ > 8:
        MQ_rating = 'Extremely Conservative'
    elif 9 > Summed_MQ > 4:
        MQ_rating = 'Moderately Conservative'
    elif 5 > Summed_MQ > 0:
        MQ_rating = 'Slightly Conservative'
    elif 1 > Summed_MQ > -5 :
        MQ_rating = 'Slightly Liberal'
    elif -4 > Summed_MQ > -9:
        MQ_rating = 'Moderately Liberal'
    elif -8 > Summed_MQ:
        MQ_rating = 'Extremely Liberal'

    st.write(
        f'''

        #  FSCOTUS Bias: 
        # {Summed_MQ}, {MQ_rating}'''

        )
    ''' '''
    ''' '''
    ''' '''
    ''' '''

    # Graph representing each justice's MQ over time


    for i, judge in enumerate(judges):
        abbrev = justice_details[justice_details['justiceName'] == judge]['justiceAbbrev'].values[0]
        judge_data = supreme[supreme.justiceName_x == abbrev].groupby('term').mean()
        judge_data.reset_index(inplace=True)
        judge_data['years_in_office'] = judge_data['term'] - judge_data['term'].min()

        if i == 0:
            fig, ax = plt.subplots(figsize=[20,14])
            plt.ylim(-6,6)
            ax = sns.lineplot(data=judge_data, x='years_in_office', y='martin_quinn', linewidth=4)
            sns.set(font='Georgia')
            sns.scatterplot(data=judge_data, x='years_in_office', y='martin_quinn', s=100)
        else:
            sns.lineplot(data=judge_data, x='years_in_office', y='martin_quinn', linewidth=4)
            sns.scatterplot(data=judge_data, x='years_in_office', y='martin_quinn', s=100)

    ax.axhline(Summed_MQ, ls='--')

    ax.set_title('\n\nFantasy SCOTUS Martin-Quinn over time (MQ = {Summed_MQ})\n\n'.format(Summed_MQ=Summed_MQ), 
    fontsize=30, style='oblique')
    ax.set_xlabel('\nYears in Office', fontsize=22)
    ax.set_ylabel('Martin-\nQuinn\n Score', fontsize=22, rotation = 0, horizontalalignment='right')
    ax.set_yticklabels(['Extremely Liberal - -6', -4, -2, 0, 2, 4, 'Extremely Conservative - 6'],{'fontsize': 18})
    ax.set_xticklabels([-5, 0,5,10,15,20,25,30],{'fontsize': 18})
    font = font_manager.FontProperties(weight='bold')
    ax.legend([judge for judge in judges], fontsize=12, prop=font, facecolor='white')
    st.pyplot(fig);

    ''' '''
    ''' '''





    # Creates reference table of judge names


    df = pd.DataFrame({'Justices': judges, 'MQ': MQ_list})
    df = df.sort_values('MQ')

    def highlight_survived(s):
        return ['background-color: red']*len(s) if s.MQ>0 else ['background-color: lightblue']*len(s)

    st.dataframe(df.style.apply(highlight_survived, axis=1))

#result = loaded_model.score(X_test, Y_test)
#print(result)

    # Results
    ''' '''
    ''' '''
    ''' # Case Results: '''
sample_cases = pd.read_csv('data/web_app_sample_cases.csv')

if issue != '<select>':
        st.write(
        f'''
 

        - ## Your FSCOTUS Vote Outcome is: {result}
        - ## FSCOTUS Court MQ: {Summed_MQ}, {MQ_rating}'''

        ''' 
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Additional Resources:

        - ## [The Supreme Court Database](http://scdb.wustl.edu/) - The number one data source for this app, the SCD has an easy-to-use and thorough interface to investigate individual cases dating back to 1791.
        - ## [Myers-Quinn Score](https://mqscores.lsa.umich.edu/measures.php) - This website contains more information about the Myers-Quinn score, and scores for all justices since 1937.
        - ## [Supreme Court Official Website](https://www.supremecourt.gov/) - Official source of information on SCOTUS, including opinions, news, and further research tools.
        ''')
        
elif caseTitle !='<select>':
        actual_outcome = sample_cases[sample_cases['case_name']==caseTitle]['decisionDirection']
        actual_outcome = actual_outcome.values

        year = sample_cases[sample_cases['case_name']==caseTitle]['term'].values
        year = int(year[0])

        if actual_outcome == 1:
            actual_outcome = 'Liberal'
        if actual_outcome == 0:
            actual_outcome = 'Conservative'
        st.write(
        f'''


        - ## **Your FSCOTUS Vote Outcome is: {result}**
        - ## **In {year}, the Real Vote Outcome was: {actual_outcome}**
        - ## **FSCOTUS Court MQ: {Summed_MQ}, {MQ_rating}**'''

        ''' 
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Additional Resources:

        - ## [The Supreme Court Database](http://scdb.wustl.edu/) - The number one data source for this app, the SCD has an easy-to-use and thorough interface to investigate individual cases dating back to 1791.
        - ## [Myers-Quinn Score](https://mqscores.lsa.umich.edu/measures.php) - This website contains more information about the Myers-Quinn score, and scores for all justices since 1937.
        - ## [Supreme Court Official Website](https://www.supremecourt.gov/) - Official source of information on SCOTUS, including opinions, news, and further research tools.
        ''')


 









# input_data = pd.DataFrame({'sqrft': [sqrft], 'beds': [beds]})
# pred = lr.predict(input_data)[0]
# st.write(
# f'Predicted Sale Price of House: ${pred:.2f}'
# )




# st.write(
# '''
# You can use markdown syntax to style your text

# Headers:
# # Main Title 
# ## Sub Title 
# ### header 

# **bold text**

# *italics*

# Ordered List

# 1. Apples 
# 2. Oranges 
# 3. Bananas 

# [This is a link](https://docs.streamlit.io/en/stable/getting_started.html)


# ''')



# st.write(
# '''
# ## Seattle Home Prices
# We can import data into our streamlit app using pandas read_csv then display the resulting dataframe with st.dataframe()

# ''')

# data = pd.read_csv('SeattleHomePrices.csv')
# data = data.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'})
# st.dataframe(data)

# st.write(
# '''
# ### Graphing and Buttons
# Lets graph some of our data with matplotlib. We can also add buttons to add interactivity to our app.
# '''
# )
# show_graph = st.checkbox('Show Graph', value=True)

# fig, ax = plt.subplots()

# ax.hist(data['PRICE'])
# ax.set_title('Distribution of House Prices in $100,000s')




# if show_graph:
# 	st.pyplot(fig)

# st.write(
# '''
# ### Mapping and Filtering Our Data
# We can also use streamlits built in mapping functionality.
# We can use a slider to filter for houses within a particular price range as well.
# '''
# )

# price_input = st.slider('House Price Filter', int(data['PRICE'].min()), int(data['PRICE'].max()), 100000 )

# price_filter = data['PRICE'] < price_input
# st.map(data.loc[price_filter, ['lat', 'lon']])

# st.write(
# '''
# ## Train a linear Regression Model
# Create a model to predict house price from sqft and number of beds
# '''
# ) 
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# clean_data = data.dropna(subset=['PRICE', 'SQUARE FEET', 'BEDS'])

# X = clean_data[['SQUARE FEET', 'BEDS']]
# y = clean_data['PRICE']

# X_train, X_test, y_train, y_test = train_test_split(X, y)

# lr = LinearRegression()

# lr.fit(X_train, y_train)
# st.write(f'R2: {lr.score(X_test, y_test):.2f}')

# st.write(
