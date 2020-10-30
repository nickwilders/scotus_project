import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
from PIL import Image

st.write('''# Fantasy SCOTUS
Project Goals

- Output who is in the court what the court "might decide" and what the actual court DID decide
''')

'''
## Fantasy SCOTUS decides on the following issue:
'''


# Call on existing data sets - one dataframe with supreme court vote data, one
# that correlates the issues to the proper number
supreme = pd.read_csv('data/supreme_1023.csv')
popular_issues = pd.read_csv('data/popular_issues_1026.csv')

# mq_by_term = supreme.groupby('term').mean()['subCourt_MQ']
# unique_terms = supreme.term.unique()

# unique_mq = supreme.subCourt_MQ.unique()

# plt.figure(figsize=(16,10))
# plt.ylim(-12,12)
# plt.xlim(1946,2020)
# ax = sns.lineplot(x=unique_terms, y=mq_by_term, linewidth=5, color='black');

# ax.set_title('Change of Court MQ over time \n', fontsize=25, style='oblique')
# ax.set_xlabel('Year', fontsize=16)
# ax.set_ylabel('Court MQ', fontsize=16, rotation = 0, horizontalalignment='right')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)

# ax.axhline(y=0, linewidth=2, color='white', alpha=.7, linestyle='--')
# ax.axvspan(1946, 1952, color='red', alpha=0.4, lw=0)
# ax.axvspan(1952, 1968, color='blue', alpha=0.6, lw=0)
# ax.axvspan(1968, 1985, color='red', alpha=0.2, lw=0)
# ax.axvspan(1985, 2004, color='red', alpha=0.5, lw=0)
# ax.axvspan(2004, 2020, color='blue', alpha=0.2, lw=0)
# ax.axvline(x=1952, linewidth=2, color='white', alpha=1, linestyle='--')
# ax.axvline(x=1968, linewidth=2, color='white', alpha=1, linestyle='--')
# ax.axvline(x=1985, linewidth=2, color='white', alpha=1, linestyle='--')
# ax.axvline(x=2004, linewidth=2, color='white', alpha=1, linestyle='--');

# st.line_chart(ax)

#plt.savefig('Court_MQ_over_time_by_Court')

# Set list of issue descriptions
issueDescripts = popular_issues['issueDescript']
# Sets list of average MQ per term 
terms_plus_MQ = supreme.groupby('term').mean()['subCourt_MQ']

# First prompt, with 10 selections
issue = st.selectbox('Select an issue to vote on:', issueDescripts)
# Second prompt, asking for a year
term = st.slider('Select a year: ', 1946, 2019)

# Defining year confines for Bias graphs
if term < 1953:
    image = Image.open('images/1. Bias_by_Issue_in_Vinson_Court.png')
    st.image(image,use_column_width=True)
elif term > 1952 and term < 1969:
    image = Image.open('images/2. Bias_by_Issue_in_Warren_Court.png')
    st.image(image,use_column_width=True)
elif term > 1968 and term < 1985:
    image = Image.open('images/3. Bias_by_Issue_in_Burger_Court.png')
    st.image(image,use_column_width=True)
elif term > 1985 and term < 2005:
    image = Image.open('images/4. Bias_by_Issue_in_Rehnquist_Court.png')
    st.image(image,use_column_width=True)
elif term > 2004:
    image = Image.open('images/5. Bias_by_Issue_in_Roberts_Court.png')
    st.image(image,use_column_width=True)

# Define issue number and area number from category based on user choice
issue_number = popular_issues[popular_issues['issueDescript'] == issue]['issueNumber']
issue_area = popular_issues[popular_issues['issueDescript'] == issue]['issueArea']

# Define MDQ based on year selection
MQ = terms_plus_MQ[terms_plus_MQ.index == term]

model = pickle.load(open('models/knn_continuous_model.sav', 'rb'))
numbers = np.array([int(MQ),int(issue_number),int(issue_area)])
numbers = numbers.reshape(1,-1)

result = model.predict(numbers)

if result == 1:
    result = 'Liberal'
if result == 0:
    result = 'Conservative'

MQ_int=int(MQ)

if MQ_int > 8:
    MQ_rating = 'Extremely Conservative'
elif 9 > MQ_int > 4:
    MQ_rating = 'Moderately Conservative'
elif 5 > MQ_int > 0:
    MQ_rating = 'Slightly Conservative'
elif 1 > MQ_int > -5 :
    MQ_rating = 'Slightly Liberal'
elif -4 > MQ_int > -9:
    MQ_rating = 'Moderately Liberal'
elif -8 > MQ_int:
    MQ_rating = 'Extremely Liberal'

MQ = np.round(MQ.values,3)

st.write(
f'''- ### Predicted Vote Outcome: {result}
- ### Court MDQ: {MQ}, {MQ_rating}'''

)



#result = loaded_model.score(X_test, Y_test)
#print(result)




 









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
