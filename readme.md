![Zillow-Group-Brand-Logos_010422-01.png](attachment:Zillow-Group-Brand-Logos_010422-01.png)


# Zillow_ Project
by cristian ibarra 

# Project Goals:

Find drivers for single families house on what effect house market?

Deliver a report that a non-data scientist can read through and understand what steps were taken, why and what was the outcome?



# Project steps:

Step 1: Understanding the Problem.

Step 2: Data Extraction.

Step 3: Data Cleaning.

Step 4: Exploratory Data Analysis.

Step 5: Feature Selection.

Step 6: Testing the Models.

Step 7: Deploying the Model.
# Project Planning:

•Create README.md with data dictionary, project and business goals, come up with initial hypotheses.

•Acquire data from the Codeup Database and create a function to automate this process. Save the function in an acquire.py file to import into the Final Report Notebook.

•Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a function to automate the process, store the function in a prepare.py module, and prepare data in Final Report Notebook by importing and using the funtion.

•Clearly define four hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.

•Establish a baseline accuracy and document well.

•Evaluate models on train and validate datasets.

•Choose the model with that performs the best and evaluate that single model on the test dataset.

•Document conclusions, takeaways, and next steps in the Final Report Notebook.

# Hypotheses:
`Hypothesis(1)-Do bedrooms increase price:`

Ho-Does more Bed rooms increase the cost of the houses?

Ha-Bed rooms has no effect on pricing of house

`Hypothesis(2)-Does squarefeet effect cost of housing:`

Ho-Would the amount of squarefeet effect the cost of housing?

Ha-squarefeet as no effect on housing


`Hypothesis(3)-What decade does the market love:`

Ho-Would decade effect total cost of the house ?

Ha-Decade has no effect on total cost of market 

`Hypothesis(4)-Do Bathrooms have a big effect on pricing:`

Ho-Does more bathrooms increase the cost of the houses?

Ha-Bathrooms has no effect on pricing of house

`Hypothesis(5)-Does county have a impact on taxtotal:`

Ho-Does county have a impact on taxtotal?

Ha-county doesnt have a impact on taxtotal

# Goal:
Find what're key drivers on property value for single family properties???

# Data Dictionary/Findings:

### # Data Used-Zillow

|Attribute|Old keys|        Data type   |       Definition   |
| -------- |-------- | -------- | -------- | 
|Bedrooms |bedroomcnt|float |  Number of bedrooms in home |
|Bathrooms |bathroomcnt|float | Number of bathrooms in home including fractional bathrooms|
|Squarefeet |calculatedfinishedsquarefeet|float | Calculated total finished living area of the home |
|TaxesTotal |taxvaluedollarcnt|float|The total property tax assessed for that assessment year 
|Year |yearbuilt|float |The Year the principal residence was built  |
|county |regionidcounty |float |The total tax assessed value of the parcel|
|Zip|regionidzip|object | Federal Information Processing Standard code |
|latitude|latitude|float|cordinates
|longitude|longitude|float|cordinates
|TotalRooms|N/A|float|bathrooms and bedrooms combined 
|location|N/A|object|area houses are located in 
|Decade|N/A|int   |Years slice into half a centary|
|los_angelos|N/A|int|1 for yes and 0 for no
|orange|N/A|int|1 for yes and 0 for no
|ventura|N/A|int|1 for yes and 0 for no

# Modeling:

---
# VALIDATE:
|MODEL | Val_rmse| Val_r2 |
| ----- | ----- | ----- |
|Lars_alpha(2)|223655.138918|0.268033|
|Depth(1) |223670.191995|0.267934|
|Depth(2) |218523.795577|0.301213|



# TRAIN:
|MODEL | Train_rmse |Train_r2|
| ----- | ----- | ----- |
|Lars_alpha(2)|220486.586943|0.288122|
|Depth(1)|220490.325968|0.288098|
|Depth(2)|215830.734471|0.317869|

# Project description
1)Why this project-
This project would help determind what're the main key aspect of prices increase in single family properties thru out the centuries.

2)Why is important-
So we could predict the price increase of single family properties

3)How does this help you- 
This would help all of us on understanding how and why our single family properties are increasing in high rate.

# Conclusion/Recommnedations/Next Steps:
`Conclusion:`

• We could conclude that bedrooms,bathrooms,decade,county,location,squarefeet have a effect on taxtotal

• In conclusion tax total could be effect by many columns and many things so what should we do ?

• We could conclude that the linear regression model perform the best with a r rate of .31

`Recommendations:`

•Addding more data about the areas surronding the houses for example School,Malls,parks,rivers,lakes,hills,views, and much more so would could obtain a more accurate model.

•Would love to keep the cost of something the same but when something is in high demand usually means price would increase. 

`Next Steps:`

• I would love to dive into more column in the zillow data set. How a room squarefeet could effect the house more then a basic rooom or how a garage could add more value then a pool??. This data has so much potential but i would just be digging myself into a rabbit hole.

• I would check 2018 and see how the market has change from the 2017 market of single families houses

# Deliverables:
To access the correct MySQL database, you will need credentials to access to the CodeUp database. This database is accessed in this project via an env.py file. Add the below to your env.py and fill in your individual access information as strings:

user = 'your_user_name'
password = 'your_password'
host = 'the_codeup_db'

1-Readme (.md)-uses as a guide 

2-Acquire (.py)-download

3-Modeling (.py)-download

4-Prepare (.py)-download

5-Final Report (.ipynb)-download 

