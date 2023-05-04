# In this file, the first baseline characteristics of surgeons and procedures performed in the SPIRIT study are analysed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import file containing Castor export
path_export = r"C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage 3 IC\Python analyses\SPIRIT_excel_export.xlsx"
df_surgeons = pd.read_excel(path_export, sheet_name="Study results")
df_data = pd.read_excel(path_export, sheet_name="Testreport")
df_data = df_data.drop(labels=[7,151], axis=0) #drop rows with only NaN's for questionnaire

#Rename columns and drop unnecessary columns
df_surgeons = df_surgeons.drop(columns={'Participant Status','Site Abbreviation'})
df_surgeons.rename(columns={'CiId': 'Surgeon_ID', 'CiSpecialisme': 'Specialism','CiGender':'Gender','CiErvaring':'Experience','CiNiveau':'Level'}, inplace=True)
df_data.rename(columns={'Agchirid': 'Surgeon_ID', 'Agpatid': 'Patient_ID', 'Agstartok':'Start_OR','Ageindeok':'End_OR','AGQuestdatr':'Date_questionnaire','AGOKtype':'Surgery_type','Agurgentie':'Urgence','VraagLMH':'Risk_LMH','VraagProcent':'Risk_percentage','VraagInschatting':'Confidence','VraagExtra':'Additional_actions','VraagLeeftijd':'Patient Age','Vraaggeslacht':'Gender_patient',
                        'VraagMedicatie':'Medication','VraagIndicatie':'Indication','VraagDuurok':'Duration OR','VraagType':'Surgery Type','VraagComorb#cardiovascular':'Comorb: cardiovas.','VraagComorb#pulmonaal':'Comorb: pulm.', 'VraagComorb#maligniteit':'Comorb: Malign.','VraagComorb#frailty':'Comorb: Frailty', 'VraagComorb#anders':'Comorb: other'}, inplace=True) 

def descriptive(df_surgeons):
    f_gender = (df_surgeons['Gender'].value_counts()[2])
    m_gender = (df_surgeons['Gender'].value_counts()[1])
    median_experience = np.round(df_surgeons['Experience'].median(), decimals=2)
    q1, q3 = np.round(df_surgeons['Experience'].quantile([0.25,0.75]))
    iqr = q3 - q1
    std_experience = np.round(df_surgeons['Experience'].std(), decimals=2)
    level1 = (df_surgeons['Level'].value_counts()[1])
    level2 = (df_surgeons['Level'].value_counts()[2])
    ortho = (df_surgeons['Specialism'].value_counts()[1])
    neuro = (df_surgeons['Specialism'].value_counts()[2])
    uro = (df_surgeons['Specialism'].value_counts()[3])
    gyn = (df_surgeons['Specialism'].value_counts()[4])
    vas1 = (df_surgeons['Specialism'].value_counts()[8])
    gastro = (df_surgeons['Specialism'].value_counts()[9])
    trauma = (df_surgeons['Specialism'].value_counts()[10])
    trans = (df_surgeons['Specialism'].value_counts()[12])
    dict_table = {'Amount of surgeons': [f'N={len(df_surgeons)}'],
                  'Gender': [f'{np.round(f_gender, decimals=0)} females, {np.round(m_gender, decimals=0)} males'],
                  'Years of experience (median + IQR)': [f'{median_experience}, ({q1} - {q3}) (median + IQR)'],
                  'Level': [f'{level1} in training, {level2} specialists'],
                  'Specialism':[f'{ortho} orthopedics, {neuro} neurology, {uro} urology, {gyn} gynaecology, {vas1} vascular, {gastro} gastrointestinal, {trauma} trauma, {trans} transplantation' ]}
    df_characteristics = pd.DataFrame.from_dict(dict_table, orient='index')
    return df_characteristics

# Create a table with the descriptive information about participating surgeons
characteristics = descriptive(df_surgeons)

#Count number of questionnaires filled by surgeons
surgeons = df_surgeons['Surgeon_ID'].tolist() #create a list of all surgeon ID's
surgeons_filled = df_data['Surgeon_ID'].unique().tolist() #create a list of all surgeons who have filled a questionnaire
answered = list(set(surgeons) & set(surgeons_filled))
surgeonID = []
for surgeon in answered:
    surgeon_id = df_data['Surgeon_ID'].value_counts()[surgeon]
    surgeonID.append(surgeon_id)

surgeonID = pd.DataFrame(surgeonID)
surgeonID['Surgeon'] = answered
surgeonID['Filled questionnaires'] = surgeonID.iloc[:,0]
surgeonID = surgeonID.drop(labels=0, axis = 1)

plt.close("all")
plt.bar(surgeonID['Surgeon'],surgeonID['Filled questionnaires'],align='center')
plt.title('Number of questionnaires answered by every surgeon', fontsize=15)
plt.xticks([i for i in surgeons_filled])
plt.yticks(np.arange(0,surgeonID['Filled questionnaires'].max()+1,1))
plt.xlabel('Surgeon ID', fontsize=15)
plt.ylabel('Completed questionnaires', fontsize=15)
plt.grid(True)
plt.show()


# replace number of risk categories with category for risk classifications
df_data['Risk_LMH'] = df_data['Risk_LMH'].replace([1], 'Very low')
df_data['Risk_LMH'] = df_data['Risk_LMH'].replace([2], 'Low')
df_data['Risk_LMH'] = df_data['Risk_LMH'].replace([3], 'Medium')
df_data['Risk_LMH'] = df_data['Risk_LMH'].replace([4], 'High')
df_data['Risk_LMH'] = df_data['Risk_LMH'].replace([5], 'Very high')

## V1 Create histograms of the risk classifications: Absolute numbers
cat_order = ['Very low', 'Low', 'Medium', 'High', 'Very high']
bar_color = '#3776ab'
#df_data.Risk_LMH.value_counts().plot.bar() # Old version of barplot without right order
#df_data.hist(column='Risk_LMH', bins=5) # Only numerical values, use if categories are 1/2/3/4/5
sns.countplot(x='Risk_LMH', data=df_data, order=cat_order, color=bar_color)
plt.title('Postoperative Infection Risk (Category)',fontsize=15)
plt.xlabel('Risk classification', fontsize=15)
plt.ylabel('Number of estimations',fontsize=15)
df_data.hist(column='Risk_percentage', bins=range(0,100,5))
plt.title('Postoperative Infection Risk (Percentage)',fontsize=15)
plt.xlabel('Percentage Infection Risk',fontsize=15)
plt.ylabel('Number of estimations',fontsize=15)
plt.xticks(np.arange(0,105,5))
plt.show()

## V2 Create histograms of the risk classifications: Percentages
# Calculate the percentage of each Risk_LMH category
cat_perc = df_data['Risk_LMH'].value_counts(normalize=True) * 100
# Set the order of the categories based on cat_order variable
cat_order = ['Very low', 'Low', 'Medium', 'High', 'Very high']
cat_perc = cat_perc[cat_order]
# Plot the count of Risk_LMH categories as a percentage of the total count
sns.barplot(x=cat_perc.index, y=cat_perc.values, color=bar_color)
plt.title('Postoperative Infection Risk (Category)', fontsize=15)
plt.xlabel('Risk classification', fontsize=15)
plt.ylabel('Percentage of all estimations (%)', fontsize=15)
plt.yticks(np.arange(0,cat_perc.max()+1,5))
plt.grid(True)
plt.show()
# Plot the histogram of Risk_percentage as a percentage of the total count
# Calculate the percentage of estimations in each bin
counts, bins = np.histogram(df_data['Risk_percentage'], bins=range(0,100,5))
percentage = counts / len(df_data) * 100
# Plot the histogram with percentage values
plt.bar(bins[:-1], percentage, width=5)
plt.title('Postoperative Infection Risk (Percentage)',fontsize=15)
plt.xlabel('Estimated Percentage Infection Risk',fontsize=15)
plt.ylabel('Percentage of all estimations (%)',fontsize=15)
plt.xticks(np.arange(0,105,5))
plt.grid(True)
plt.show()

## For every risk category, analyse the level of experience of the surgeons
# First, create a merged dataframe with both the characteristics of surgeons and surgery questionnaires
df_new = df_data.merge(df_surgeons, on='Surgeon_ID',how='inner')
df_new['Level'] = df_new['Level'].replace([1], 'One')
df_new['Level'] = df_new['Level'].replace([2], 'Two')
# Create seperate dataframes for surgeons in training and specialists with their risk classification
level_1 = df_new.query("Level=='One'")["Risk_LMH"]
level_2 = df_new.query("Level=='Two'")["Risk_LMH"]
# Count the number of times a certain risk category is assigned
totalcount1 = len(level_1.dropna())
count1 = ((level_1.value_counts())/totalcount1)*100 #calculate as percentage instead of absolute number
count1 = count1.to_frame().reset_index()
totalcount2 = len(level_2.dropna())
count2 = ((level_2.value_counts())/totalcount2)*100 #calculate as percentage instead of absolute number
count2 = count2.to_frame().reset_index()
# Merge the frequences of all risk categories for surgeons in training and specialists in one dataframe 
count = count1.merge(count2, on='index', how='outer')
count = count.rename(columns={'Risk_LMH_x':'In training', 'Risk_LMH_y':'Specialist'})
count = count.reindex([2,0,1,3,4])
count.plot.bar(x='index', title='Percentage of infection estimations for every risk category')
plt.grid(True)
plt.yticks(np.arange(0,110,10))
plt.xlabel('Risk classification', fontsize=15)
plt.ylabel('Percentage of estimations (%)',fontsize=15)
plt.show()

# V1 Evaluate overlap of risk percentages and risk categories
# First, calculate average and std of percentages for every category (very low/low/medium/high/very high)
perc_cat = df_data.groupby('Risk_LMH')['Risk_percentage'].mean().reset_index(name='Percentages mean')
perc_std = df_data.groupby('Risk_LMH')['Risk_percentage'].std().reset_index(name='Percentages Std')
# Merge the two dataframes to get average and std in a column per category to be able to create a plot
score = perc_cat.merge(perc_std, on='Risk_LMH',how='outer')
score = score.reindex([4,1,2,0,3])
plt.errorbar(score['Risk_LMH'],score['Percentages mean'],score['Percentages Std'],linestyle='None',marker='^')
for x,y in zip(score['Risk_LMH'],score['Percentages mean']):
    y3 = np.round(y,decimals=2)
    label = f"({y3})"

    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='left') # horizontal alignment can be left, right or center
plt.xlabel('Risk category',fontsize=15)
plt.ylabel('Mean (± std) Risk Percentage',fontsize=15)
plt.title('Average (± std) risk percentage for every risk category',fontsize=15)
plt.grid(True)
plt.show()

# V2 Evaluate overlap of risk percentages and risk categories with distribution plot
# Sort the dataframe by target
cat_0 = df_data.loc[df_data['Risk_LMH'] == 'Very low']
cat_1 = df_data.loc[df_data['Risk_LMH'] == 'Low']
cat_2 = df_data.loc[df_data['Risk_LMH'] == 'Medium']
cat_3 = df_data.loc[df_data['Risk_LMH'] == 'High']
cat_4 = df_data.loc[df_data['Risk_LMH'] == 'Very high']

sns.distplot(cat_0[['Risk_percentage']], hist=False, rug=True, label='Very low')
sns.distplot(cat_1[['Risk_percentage']], hist=False, rug=True, label ='Low')
sns.distplot(cat_2[['Risk_percentage']], hist=False, rug=True, label='Medium')
sns.distplot(cat_3[['Risk_percentage']], hist=False, rug=True, label='High')
sns.distplot(cat_4[['Risk_percentage']], hist=False, rug=True, label='Very high')
plt.legend()
plt.title('Distribution plot of risk percentage for every category')
plt.xlabel('Estimated Risk Percentage')
plt.show()

#Evaluate overlap of confidence levels and level of experience
# Create seperate dataframes for surgeons in training and specialists with their confidence levels
conf_1 = df_new.query("Level=='One'")["Confidence"]
conf_2 = df_new.query("Level=='Two'")["Confidence"]
# Count the number of times a certain risk category is assigned
totalconf1 = len(conf_1.dropna())
confcount1 = ((conf_1.value_counts())/totalconf1)*100
confcount1 = confcount1.to_frame().reset_index()
confcount1['index'] = confcount1['index'].replace([1.0], 'Unsure')
confcount1['index'] = confcount1['index'].replace([2.0], 'Sure')
confcount1['index'] = confcount1['index'].replace([3.0], 'Very sure')
totalconf2 = len(conf_2.dropna())
confcount2 = ((conf_2.value_counts())/totalconf2)*100 # Note: One value is NaN 
confcount2 = confcount2.to_frame().reset_index()
confcount2['index'] = confcount2['index'].replace([1.0], 'Unsure')
confcount2['index'] = confcount2['index'].replace([2.0], 'Sure')
confcount2['index'] = confcount2['index'].replace([3.0], 'Very sure')
# Merge the frequences of all risk categories for surgeons in training and specialists in one dataframe 
confcount = confcount1.merge(confcount2, on='index', how='outer')
confcount = confcount.rename(columns={'Confidence_x':'In training', 'Confidence_y':'Specialist'})
confcount = confcount.reindex([1,0,2])
confcount.plot.bar(x='index')
plt.grid(True)
plt.title('Confidence level of surgeons for estimated infection risks, grouped by experience', fontsize=15)
plt.yticks(np.arange(0,110,10))
plt.xlabel('Confidence level', fontsize=15)
plt.ylabel('Percentage of estimations (%)',fontsize=15)
plt.show()

# Evaluate confidence level and risk category
df_new2 = df_new
df_new2['Confidence'] = df_new['Confidence'].replace([1], 'One')
df_new2['Confidence'] = df_new['Confidence'].replace([2], 'Two')
df_new2['Confidence'] = df_new['Confidence'].replace([3], 'Three')
# Create seperate dataframes for the three levels of confidence
group_1 = df_new2.query("Confidence=='One'")["Risk_LMH"]
group_2 = df_new2.query("Confidence=='Two'")["Risk_LMH"]
group_3 = df_new2.query("Confidence=='Three'")["Risk_LMH"]
# Count the number of times a certain risk percentage is assigned per level of confidence 
totalcountgr1 = len(group_1.dropna()) # Drop possible NaN values before calculating percentages 
countgr1 = ((group_1.value_counts())/totalcountgr1)*100 # Calculate percentages instead of absolute numbers
countgr1 = countgr1.to_frame().reset_index()
totalcountgr2 = len(group_2.dropna())
countgr2 = ((group_2.value_counts())/totalcountgr2)*100
countgr2 = countgr2.to_frame().reset_index()
totalcountgr3 = len(group_3.dropna())
countgr3 = ((group_3.value_counts())/totalcountgr3)*100
countgr3 = countgr3.to_frame().reset_index()
# Merge the frequences of all risk categories for surgeons in training and specialists in one dataframe 
countgr = countgr1.merge((countgr2.merge(countgr3,on='index', how='outer')), on='index', how='outer')
countgr = countgr.rename(columns={'Risk_LMH':'Unsure','Risk_LMH_x':'Sure', 'Risk_LMH_y':'Very sure'})
countgr = countgr.reindex([3,1,0,2,4])
clrs = ['#F97306','#3776ab','#15B01A']
countgr.plot.bar(x='index', color=clrs)
plt.grid(True)
plt.title('Confidence level for every estimation of a risk category', fontsize=15)
plt.xlabel('Risk classification (Category)', fontsize=15)
plt.ylabel('Percentage (%) of estimations for that confidence level',fontsize=15)
plt.yticks(np.arange(0,110,10))
plt.show()



# How many times were additional actions performed based on the risk estimate? 
additional_action = df_data['Additional_actions'].value_counts()[1]

# How many surgeries were elective and how many were acute? 
urgence_acute = df_data['Urgence'].value_counts()[1]
urgence_elective = df_data['Urgence'].value_counts()[2]

# V1 Create a piechart for risk factors on which the risk estimates are based
# Frist, create a dataframe (df_data2) with only the predictive factors in columns
df_data2 = df_data.iloc[:,16:27].replace([2],[1]) # Do not discriminate the level of importance, only count if factor was relevant or not
df_sum2 = df_data.sum() # Sum with level of importance (1 or 2) included
df_sum = df_data2.sum() # Sum without level of importance included (only values of 1 if relevant or NaN if not relevant)
df_sum3 = pd.DataFrame(df_data2.sum(), columns=['Frequency'])
df_sum3['Factor'] = df_sum3.index
df_sum3.groupby(['Factor']).sum().plot(kind='pie', y='Frequency',autopct='%1.0f%%')
plt.title('Prospective Risk Factors')
plt.legend(bbox_to_anchor=(1,0.5),loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
#plt.show()

# V2 Create a horizontal barplot of the relevant risk factors on which the estimates are based 
# First, group categories low-very low and high-very high
df_new3 = df_new2
df_new3['Risk_LMH'] = df_new3['Risk_LMH'].replace('Very low', 'Low') 
df_new3['Risk_LMH'] = df_new3['Risk_LMH'].replace('Very high', 'High') 
df_new3.iloc[:, 16:27] = df_new3.iloc[:, 16:27].replace(2, 1)
# Define the categories of interest
categories = ['Low', 'Medium', 'High']
# Initialize an empty nested dictionary to store the results
results = []
# Iterate over the columns in the range 16:27, these columns include the relevant risk factors 
for col in df_new3.iloc[:, 16:27]:
    # Iterate over the categories of interest
    for category in categories:
        # Count the number of occurrences of 1 in the current column with the specified category
        counts = ((df_new3[col] == 1) & (df_new3['Risk_LMH'] == category)).sum()
        # Add the counts for the current column and category to the results list
        results.append({'column': col, 'category': category, 'count': counts})

# Create a DataFrame from the results list
df_results = pd.DataFrame(results)
# Group the results by column and category, and sum the counts
df_plot = df_results.groupby(['column', 'category']).sum().reset_index()
df_plot['total_count'] = df_plot.groupby('category')['count'].transform('sum')
# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))
# Define the x-positions of the bars for each category
x_positions = np.arange(len(df_plot['column'].unique()))
colors = ['tab:green','tab:orange','tab:red']

# Iterate over the categories of interest
for i, category in enumerate(categories):
    # Filter the results for the current category
    df_category = df_plot[df_plot['category'] == category]
    # Set the x-position of the bars for the current category
    x_pos = x_positions + (i - len(categories)/2 + 0.5) * 0.2
    # Create the horizontal bar plot for the current category
    ax.barh(y=x_pos, width=df_category['count'], height=0.2,
            color=colors[i].format(i), alpha=0.8, label=category)
    # Add labels to the bars
    for j, val in enumerate(df_category['count']):
        ax.text(val, x_pos[j], str(val))

# Set the y-tick labels to the unique values in 'column'
ax.set_yticks(x_positions)
ax.set_yticklabels(df_plot['column'].unique())
# Add labels and legends
ax.set_xlabel('Count',fontsize=15)
ax.set_ylabel('Risk factor',fontsize=15)
ax.set_title('Relevant factors for estimation of postoperative infection risk',fontsize=15)
ax.legend(fontsize=15)
plt.show()

## V3 Calculate the same plot, but with percentages and not absolute numbers
# First calculate the percentage that a factor was relevant for the total estimations within a risk category
df_plot['percentage'] = (df_plot['count']/df_plot['total_count']*100).round(decimals=2)
# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))
# Define the x-positions of the bars for each category
x_positions = np.arange(len(df_plot['column'].unique()))
colors = ['tab:green','tab:orange','tab:red']

# Iterate over the categories of interest
for i, category in enumerate(categories):
    # Filter the results for the current category
    df_category = df_plot[df_plot['category'] == category]
    # Set the x-position of the bars for the current category
    x_pos = x_positions + (i - len(categories)/2 + 0.5) * 0.2
    # Create the horizontal bar plot for the current category
    ax.barh(y=x_pos, width=df_category['percentage'], height=0.2,
            color=colors[i], alpha=0.8, label=category)
    # Add labels to the bars
    for j, val in enumerate(df_category['percentage']):
        ax.text(val, x_pos[j], str(val))

# Set the y-tick labels to the unique values in 'column'
ax.set_yticks(x_positions)
ax.set_yticklabels(df_plot['column'].unique())
# Add labels and legends
ax.set_xlabel('Percentage of risk category', fontsize=15)
ax.set_ylabel('Risk factor',fontsize=15)
ax.set_title('Relevant factors for estimation of postoperative infection risk (% per category)',fontsize=15)
ax.legend(fontsize=15)
plt.show()

