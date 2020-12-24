<h2>Fantasy SCOTUS Web App</h2>
<br>
<a href='http://www.fantasyscotus.herokuapp.com'>Heroku-Deployed Streamlit App</a> 
<br> 
<a href="https://towardsdatascience.com/a-not-quite-fantasy-scotus-92a6e43739b3">TDS Article</a>
<br><br>

Thank you for checking out the project repo for the Fantasy SCOTUS web app! This app uses a Random Forest classificaiton model to predict the disposition of a case in the United States Supreme court. This can be applied to new theoretical cases, in addition to landmask cases throughout time - what matters is the designated court deciding.
<br>
<br>
The intention of this app is to show the influence that a judge's political bias has on decision making. Although there is an even 50/50 balance of liberal vs conservative decisions over the years, those numbers are changed significantly based on issue, and based on the political disposition of the Court (as most clearly designated in the Visualizations notebook, found in the "data" folder).
<br>
<h3>US Supreme Court Database</h3>
<br>
The primary database was provided by the <a href='http://scdb.wustl.edu/'>US Supreme Court Database</a> project, maintained by Washington University Law. Their most robust dataset holds 122,000 entries, for every vote made on the Supreme Court since 1949. There is data existing before 1949, but it was determined to be outside of the scope of this project. 
<h3>Martin-Quinn Score</h3>
<br>
The Martin-Quinn score is a rating of the liberal or conservative leaning of a particular justice (explained fully and, in context, at the <a href='https://mqscores.lsa.umich.edu/'>Martin-Quinn score webpage</a>. It is a simple scale from -6 (<i>Extremely Liberal</i>) to 6 (<i>Extremely Conservative</i>), with 0 being truly moderate. This score changes over time, as demonstrated by the graph below (with the current SCOTUS):<br>
<br>
<img src='https://i.ibb.co/2Pk3jz7/Current-SCOTUS.png'>
<br>
This study makes use of a <b><i>Court MQ</i></b>, which represents the political leaning of a court in a certain year. This is calculated simply by adding the scores together, assuming that the positive conservative scores and negative liberal scores would cancel each other out to a more representative value. The change in that score can be found in the Visualiizations notebook, along with the Slide Deck presentation.
<br>
<h3>Model Development</h3>
<br>
After testing many model types, a Random Forest model was selected. The Random Forest model gave the highest accuracy on both test and validation sets, and also made the most sense with this sort of data. Although the values were treated as continuous variables to account for new entry (using the app), they essentially function as categorical variables, where Random Forest excels as a model. Other models developed but not selected include Decision Tree (a less efficient duplicate of Random Forest include KNN and SVC (comparable success), Logistic Regression and Gaussian Naive Bayes (moderate success). 
<br>
<br>
For further questions about the model, dataset, or app, please contact Nick Wilders at nawilders@gmail.com. 
