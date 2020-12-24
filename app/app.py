# MASTER IMPORT LIST

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import matplotlib.font_manager as font_manager
from PIL import Image

# INITIALIZE SESSION


# TITLE HTML RENDERING

Title_html = """

    <style>
         body {
             background-image: url("https://i.ibb.co/KzTLw6r/SCOTUS.jpg");
             background-color: #cccccc;
        }
        
        .blue {
          color: lightcoral;
        }
        .red {
          color: lightblue;
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
        
        .casedescript h3{
        font-size: 12px;
        font-style: italic;
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
st.markdown(Title_html, unsafe_allow_html=True) 
st.markdown("App developed by **Nick Wilders, Data Scientist** at Metis - [GitHub Repository](https://github.com/nickwilders/scotus_project) | [LinkedIn](https://www.linkedin.com/in/nick-wilders-7a75555b/)")
st.markdown("**DISCLAIMER:** This tool is **for educational purposes**, and biases reflect the popular opinion of the ideological group (liberal or conservative) ** at the time the justice served on the court. ** ")



# IMPORT DATA AND ASSIGN VARIABLES

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



# SIDEBAR 

# sidebar style rendering
st.markdown(
f'''
    <style>
        .sidebar .sidebar-content {{
            width: 400px;
            background-image: linear-gradient(#f7c3c1,#f7c3c1);
            color: blue;
        }}
    </style>
''',
unsafe_allow_html=True
)

# sidebar text / dropdown menus
st.sidebar.write('# Select your Fantasy SCOTUS:  \n'
                ' ###')


original_sidebar = st.sidebar.button('Restore Oct 2020 SCOTUS (default)', key='new_sidebar')
st.sidebar.write('###')
if not original_sidebar:
    judge1 = st.sidebar.selectbox(f'Chief Justice', judgeSelects,index=32)
    st.sidebar.write(f'**{judge1}**')
    judge2 = st.sidebar.selectbox('Select Associate Justice 1', judgeSelects,index=29)
    st.sidebar.write(f'**{judge2}**')
    judge3 = st.sidebar.selectbox('Select Associate Justice 2', judgeSelects, index=30)
    st.sidebar.write(f'**{judge3}**')
    judge4 = st.sidebar.selectbox('Select Associate Justice 3', judgeSelects, index=31)
    st.sidebar.write(f'**{judge4}**')
    judge5 = st.sidebar.selectbox('Select Associate Justice 4', judgeSelects, index=33)
    st.sidebar.write(f'**{judge5}**')
    judge6 = st.sidebar.selectbox('Select Associate Justice 5', judgeSelects, index=34)
    st.sidebar.write(f'**{judge6}**')
    judge7 = st.sidebar.selectbox('Select Associate Justice 6', judgeSelects, index=35)
    st.sidebar.write(f'**{judge7}**')
    judge8 = st.sidebar.selectbox('Select Associate Justice 7', judgeSelects, index=36)
    st.sidebar.write(f'**{judge8}**')
    judge9 = st.sidebar.selectbox('Select Associate Justice 8', judgeSelects, index=37)
    st.sidebar.write(f'**{judge9}**')
if original_sidebar:
    judge1 = st.sidebar.selectbox(f'Chief Justice', judgeSelects,index=32, key='judge1')
    st.sidebar.write(f'**{judge1}**')
    judge2 = st.sidebar.selectbox('Select Associate Justice 1', judgeSelects,index=29, key='judge2')
    st.sidebar.write(f'**{judge2}**')
    judge3 = st.sidebar.selectbox('Select Associate Justice 2', judgeSelects, index=30, key='judge3')
    st.sidebar.write(f'**{judge3}**')
    judge4 = st.sidebar.selectbox('Select Associate Justice 3', judgeSelects, index=31, key='judge4')
    st.sidebar.write(f'**{judge4}**')
    judge5 = st.sidebar.selectbox('Select Associate Justice 4', judgeSelects, index=33, key='judge5')
    st.sidebar.write(f'**{judge5}**')
    judge6 = st.sidebar.selectbox('Select Associate Justice 5', judgeSelects, index=34, key='judge6')
    st.sidebar.write(f'**{judge6}**')
    judge7 = st.sidebar.selectbox('Select Associate Justice 6', judgeSelects, index=35, key='judge7')
    st.sidebar.write(f'**{judge7}**')
    judge8 = st.sidebar.selectbox('Select Associate Justice 7', judgeSelects, index=36, key='judge8')
    st.sidebar.write(f'**{judge8}**')
    judge9 = st.sidebar.selectbox('Select Associate Justice 8', judgeSelects, index=37, key='judge9')
    st.sidebar.write(f'**{judge9}**')

st.sidebar.write(''' ''')
st.sidebar.write(''' ''')
st.sidebar.write(''' ''')
st.sidebar.write(f"# "
                " "
                
                " Your Fantasy SCOTUS ")
st.sidebar.write(f"{judge1}")
st.sidebar.write(f"{judge2}")
st.sidebar.write(f"{judge3}")
st.sidebar.write(f"{judge4}")
st.sidebar.write(f"{judge5}")
st.sidebar.write(f"{judge6}")
st.sidebar.write(f"{judge7}")
st.sidebar.write(f"{judge8}")
st.sidebar.write(f"{judge9}")



# SELECT BOXES FOR CASE / ISSUE
# Historical Selectbox
'''
# Select a Landmark Historical Case:
Predictions estimated within **97% accuracy.**

'''
caseTitle = st.selectbox('Case Title:', caseNames)

st.text("")
# Issue Selectbox
'''
## OR 
# Select an Issue to Vote On:
Predictions estimated within **84% accuracy.**
'''
# New Variable Assignment based on selection
issue = st.selectbox('Issue Description:', issueDescripts)
case_issue = sample_cases[sample_cases['case_name']==caseTitle]['case_issue'].values[0]
case_descript = sample_cases[sample_cases['case_name']==caseTitle]['case_descript'].values[0]
actual_answer = sample_cases[sample_cases['case_name']==caseTitle]['actual_answer'].values[0]
link = sample_cases[sample_cases['case_name']==caseTitle]['link'].values[0]



# IF / ELIF STATEMENT - ENSURE THAT BOTH BOXES DO NOT HAVE A VALUE

if issue != '<select>' and caseTitle != '<select>':
    st.markdown("# Sneaky! Please select either an issue or landmark historical case.")
elif issue != '<select>' or caseTitle !='<select>':

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

    Summed_MQ_number = np.round(Summed_MQ,2)
    Summed_MQ_percentage = np.round(abs(Summed_MQ_number/12),3)*100
    
    
    # IF ISSUE IS SELECTED:
    
    if issue != '<select>':
        st.write(
        f'''
        # *Your Results:*


        ## Your SCOTUS Vote Outcome would be **{result}**
        ## FSCOTUS Court MQ: **{Summed_MQ_percentage}% {result}**
        
        
        
        '''
        ''' # ''')
        
        
        
    # IF CASE IS SELECTED:
    
    elif caseTitle !='<select>':
        actual_outcome = sample_cases[sample_cases['case_name']==caseTitle]['decisionDirection'].values

        year = sample_cases[sample_cases['case_name']==caseTitle]['term'].values
        year = int(year[0])
        
        # assigns disposition of the ACTUAL court decision
        if actual_outcome == 1:
            actual_outcome = 'Liberal'
        elif actual_outcome == 0:
            actual_outcome = 'Conservative'
             
        # assigns whether Fantasy SCOTUS would maintain or overturn the decision
        if result == 'Conservative':
            if actual_outcome == 'Conservative':
                new_result = 'MAINTAIN'
            elif actual_outcome == 'Liberal':
                new_result = 'OVERTURN'
        elif result == 'Liberal':
            if actual_outcome == 'Conservative':
                new_result = 'OVERTURN'
            elif actual_outcome == 'Liberal':
                new_result = 'MAINTAIN'
                
        # assigns disposition of Fantasy SCOTUS decision
        if Summed_MQ > 0:
            MQ_rating = 'Conservative'
        else:
            MQ_rating = ' Liberal'
        
        # write text of decision results
        st.write(
            f'''
            # 
            # * Your Results: *
            ## **Case Name / Issue:**
            ### ** [{caseTitle}]({link}) **

            ** {case_issue} **
            ''')
        st.write(
        f'''
        ## **True Outcome:**
        ### In {year}, SCOTUS voted **{actual_answer}**, a **{actual_outcome}** outcome.
        ## **Your Outcome (Bias: ~{Summed_MQ_percentage}% {result}**):
        ### Your Fantasy SCOTUS, in {year}, would likely **{new_result}** that decision
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         
         '''
        )
        
        
        
    # BIAS VISUALIZATION FOR EITHER OPTION
    
    st.write(
        f'''
        # *Bias Visualization*
        The graph below indicates the change of Martin-Quinn score for each individual justice over their time on the Supreme Court. **This score is not issue-specific, but represents their overall disposition.**  \n
        A score of **6** represents an **extreme conservative bias**, and **-6** indicates an **extreme liberal bias**. These scores are summed to represent the **Court Martin-Quinn court score**. 
        #### Pro Tip: **click the arrows in the top right corner to expand this visual!**
        ####'''
    ''' '''
    ''' '''
        )
    
    
    
    # VISUALIZATION CODE

    for i, judge in enumerate(judges):
        abbrev = justice_details[justice_details['justiceName'] == judge]['justiceAbbrev'].values[0]
        judge_data = supreme[supreme.justiceName_x == abbrev].groupby('term').mean()
        judge_data.reset_index(inplace=True)
        judge_data['years_in_office'] = judge_data['term'] - judge_data['term'].min()

        if i == 0:
            fig, ax = plt.subplots(figsize=[22,18])
            plt.ylim(-6,6)
            ax = sns.lineplot(data=judge_data, x='years_in_office', y='martin_quinn', linewidth=4)
            sns.set(font='Georgia')
            sns.scatterplot(data=judge_data, x='years_in_office', y='martin_quinn', s=100)
        else:
            sns.lineplot(data=judge_data, x='years_in_office', y='martin_quinn', linewidth=4)
            sns.scatterplot(data=judge_data, x='years_in_office', y='martin_quinn', s=100)
    sns.set_style('whitegrid')
    ax.axhline(Summed_MQ, ls='--')
    ax.set_title('\n\nFantasy SCOTUS Martin-Quinn over time (Martin-Quinn = {Summed_MQ_number})\n\n'.format(Summed_MQ_number=Summed_MQ_number), 
    fontsize=35, style='oblique')
    plt.xlim(0,35)
    ax.set_xlabel('\nYears in Office', fontsize=22)
    ax.set_ylabel('Martin-\nQuinn\n Score', fontsize=22, rotation = 0, horizontalalignment='right')
    ax.set_yticklabels(['Extremely Liberal - -6', -4, -2, 0, 2, 4, 'Extremely Conservative - 6'],{'fontsize': 20})
    ax.set_xticklabels(['',5,10,15,20,25,30],{'fontsize': 20})
    legend = ax.legend([judge for judge in judges], bbox_to_anchor=(-.43, .94), loc=2, borderaxespad=0, fontsize=20)
    font = font_manager.FontProperties(weight='bold', size=16)

    # set labels at end of line
    for line, name in zip(ax.lines, judges):
        y = line.get_ydata()[-1]
        x = line.get_xdata()[-1]+1
        text = ax.annotate(np.round(y,2),
                           weight='bold',
                           fontsize=25,
                           xy=(x, y),
                           xytext=(0, 0),
                           color=line.get_color(),
                           xycoords=(ax.get_xaxis_transform(),
                                     ax.get_yaxis_transform()),
                           textcoords="offset points")
        text_width = (text.get_window_extent(
            fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
        #ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)

    # ax.legend([judge for judge in judges], prop=font, borderaxespad=12, loc=1, facecolor='white')
    st.pyplot(fig);

    
    
    # JUSTICE TABLE
    
    st.write('''## '''
             ''' ''')
    st.write(''' # *Justices and Biases:*''')
    
    st.write(''' **Liberal Bias (-6 to 0)** - Blue; **Conservative Bias (0 to 6)** - Red ''')
             
    st.write('''#### '''
             ''' ''')

    df = pd.DataFrame({'Justices': judges, 'Bias_MQ': MQ_list})
    df = df.sort_values('Bias_MQ')

    def highlight_survived(s):
        return ['background-color: salmon']*len(s) if s.Bias_MQ>0 else ['background-color: lightblue']*len(s)

    st.dataframe(df.style.apply(highlight_survived, axis=1))

    
    # ADDITIONAL RESOURCES
    
    st.write('''## '''
             ''' ''')

    st.write(
            ''' '''
            ''' '''
            '''
            # *Additional Resources:*

            - ### [The Supreme Court Database](http://scdb.wustl.edu/) - The number one data source for this app, the SCD has an easy-to-use and thorough interface to investigate individual cases dating back to 1791.
            - ### [Martin-Quinn Score](https://mqscores.lsa.umich.edu/measures.php) - This website contains more information about the Martin-Quinn score, and scores for all justices since 1937.
            - ### [Supreme Court Official Website](https://www.supremecourt.gov/) - Official source of information on SCOTUS, including opinions, news, and further research tools. 
            ''')
