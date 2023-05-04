# In this file, the first results regarding postoperative infections registered in HiX will be analysed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

#Import file containing HiX export
path_export = r"C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage 3 IC\Python analyses\SPIRIT_linked_to_outcomesV2.xlsx"
df_outcomes = pd.read_excel(path_export)
#Import file containing Castor data
path_export = r"C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage 3 IC\Python analyses\SPIRIT_excel_export.xlsx"
df_data = pd.read_excel(path_export, sheet_name="Testreport")
df_data = df_data.drop(labels=[7,151], axis=0) #drop rows with only NaN's for questionnaire
df_data.rename(columns={'Agchirid': 'Surgeon_ID', 'Agpatid': 'SPIRIT_PATIENT', 'Agstartok':'Start_OR','Ageindeok':'End_OR','AGQuestdatr':'Date_questionnaire','AGOKtype':'Surgery_type','Agurgentie':'Urgence','VraagLMH':'Risk_LMH','VraagProcent':'Risk_percentage','VraagInschatting':'Confidence','VraagExtra':'Additional_actions','VraagLeeftijd':'Patient Age','Vraaggeslacht':'Gender_patient',
                        'VraagMedicatie':'Medication','VraagIndicatie':'Indication','VraagDuurok':'Duration OR','VraagType':'Surgery Type','VraagComorb#cardiovascular':'Comorb: cardiovas.','VraagComorb#pulmonaal':'Comorb: pulm.', 'VraagComorb#maligniteit':'Comorb: Malign.','VraagComorb#frailty':'Comorb: Frailty', 'VraagComorb#anders':'Comorb: other'}, inplace=True) 

# Create a dataframe with the expecations of surgeons from Castor and the outcomes of HiX
df_out_pred = df_outcomes.merge(df_data, on='SPIRIT_PATIENT', how='inner')

# Calculate the number of infections from patients present in the HiX file
infections = (df_outcomes['infection30d'].value_counts()[1])
inf_perc = np.round((infections/len(df_outcomes))*100, decimals=2) #calculates the percentage postoperative infections
# Calculate the number of infections from patients present in the combined dataframe (present in Castor and HiX)
infections2 = (df_out_pred['infection30d'].value_counts()[1])
inf_perc2 = np.round((infections2/len(df_out_pred))*100, decimals=2) #calculates the percentage postoperative infections

# Create a ROC curve based on the risk percentages 
fpr, tpr, thresholds = roc_curve(df_out_pred['infection30d'],df_out_pred['Risk_percentage'])
roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr,label='ROC curve Percentage (AUC = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('ROC Curve Risk Percentages') #Uncomment if separate curves are plotted
#plt.show()


# Create a ROC curve based on the risk categories
fpr2, tpr2, thresholds2 = roc_curve(df_out_pred['infection30d'],df_out_pred['Risk_LMH'])
roc_auc2 = auc(fpr2, tpr2)
plt.plot(fpr2,tpr2,label='ROC curve Category (AUC = %0.2f)' % roc_auc2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('ROC Curve Risk Category') #Uncomment if separate curves are plotted
plt.title('ROC Curve Postoperative Infection Predictions')
plt.legend()
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,1.1,0.1))
plt.show()


# Create ROC-curves based on risk categories, but rewrite categories as percentages
df_out_pred['Risk_LMH2']= df_out_pred['Risk_LMH'].replace(1,10)
df_out_pred['Risk_LMH2']= df_out_pred['Risk_LMH2'].replace(2,30)
df_out_pred['Risk_LMH2']= df_out_pred['Risk_LMH2'].replace(3,50)
df_out_pred['Risk_LMH2']= df_out_pred['Risk_LMH2'].replace(4,70)
df_out_pred['Risk_LMH2']= df_out_pred['Risk_LMH2'].replace(5,90)

fpr3, tpr3, thresholds3 = roc_curve(df_out_pred['infection30d'],df_out_pred['Risk_LMH2'])
roc_auc3 = auc(fpr3, tpr3)
plt.plot(fpr3,tpr3,label='ROC curve Category (AUC = %0.2f)' % roc_auc3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('ROC Curve Risk Category') #Uncomment if separate curves are plotted
plt.title('ROC Curve Postoperative Infection Predictions (with altered numbers)')
plt.legend()
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,1.1,0.1))
plt.show()

# Create calibration curve for the percentual estimation
true_prob, pred_prob = calibration_curve(df_out_pred['infection30d'], df_out_pred['Risk_percentage']/100, n_bins=10)
# Create calibration curve for the categorical estimation, based on the categories as percentage as calculated above 
true_prob2, pred_prob2 = calibration_curve(df_out_pred['infection30d'], df_out_pred['Risk_LMH2']/100, n_bins=10)

# Plot the calibration curve
plt.plot(pred_prob, true_prob, marker='o', linestyle='-', label='Surgeons Risk Estimation (%)')
plt.plot(pred_prob2, true_prob2, marker='*', linestyle='-', label='Surgeons Risk Estimation (Category)')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Predicted probability')
plt.ylabel('True probability')
plt.title('Calibration Plot For Surgeons Predictions of Postoperative Infection')
plt.legend()
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,1.1,0.1))
plt.show()


