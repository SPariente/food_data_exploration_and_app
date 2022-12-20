import streamlit as st

#Import data exploitation libraries
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

#Import image exploitation libraries
import cv2
from pyzbar.pyzbar import decode
import time

#Import functions
from st_functions import *


st.title('Food recommandation app')

#Import data
@st.cache
def import_data():
    #Import food database
    data = pd.read_feather('data/FoodData_final.feather')

    #Import recommanded nutritional values database
    means_kcal = pd.read_excel('data/daily_needs_anses.xlsx', sheet_name = 'means', index_col = 'indicator')

    #Reorder and rename columns
    numeric_fields = [
        'energy-kcal_100g',
        'saturated-fat_100g',
        'non-saturated-fat_100g',
        'sugars_100g',
        'other-carbohydrates_100g',
        'proteins_100g',
        'sodium_100g'
    ]

    means_kcal = means_kcal[['mean']]
    means_kcal = means_kcal.loc[[x.replace('_100g',"") for x in numeric_fields],:]
    
    return data, means_kcal, numeric_fields

data_load_state = st.text('Loading dataset...')

FoodData, means_kcal, numeric_fields = import_data()

data_load_state = st.text("Loading complete")

#Create separate tabs
ov_tab, rec_tab = st.tabs(['Overview', 'Recommended food'])

with ov_tab:
#Overview tab
    #Initialize new list
    if st.button("New product list"):
         st.session_state['barcodes_list'] = []
    else: #If no button clicked, check for existing list, and if not create empty one
        if 'barcodes_list' not in st.session_state:
            st.session_state['barcodes_list'] = []


    #Add barcode to list
    if st.button("Add new product"):
         st.session_state['barcodes_list'] = add_barcode(st.session_state['barcodes_list'], FoodData, 'code')


    #Select if all foods or only non-perishables are to be included
    only_perishables = st.checkbox('Use only perishable products for nutritional values',
                                   value=True)
    if not only_perishables:
        excluded_groups = []
    else: #Definition of nutritional groups to exclude from computation of nutritional values
        excluded_groups = ['sweets', 'fats', 'protein complements', 'salts']

    if len(st.session_state['barcodes_list'])>0: #Check if barcodes have been entered
        #Use the function
        actuals, val_actuals, val_target, r_actuals, gap_target, r_best, n_products=nutr_values_products(FoodData, means_kcal['mean'], st.session_state['barcodes_list'], numeric_fields, excluded_groups)

        #Define labels for visualization
        label_vals = [
            'Energy',
            'Saturated fats',
            'Unsat. fats',
            'Sugars',
            'Fibers / carbs',
            'Proteins',
            'Sodium'
        ]

        #Define units to show for visualization
        units = ['kcal']+['g']*(len(label_vals)-1)

        #Show figures side by side

        fig_title1, fig_title2 = st.columns(2)
        with fig_title1:
            st.header("Nutritional values of the basket")

        with fig_title2:
            st.header("Basket composition")
            
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            fig = show_basket_values(r_actuals, val_actuals, val_target, n_products, label_vals, units)
            st.plotly_chart(fig, use_container_width=True)

        with fig_col2:
            fig = show_basket_composition(actuals)
            st.plotly_chart(fig, use_container_width=True)
    
with rec_tab:
    if len(st.session_state['barcodes_list'])>0: #Check if barcodes have been entered
        #Advanced settings expander
        with st.expander("Advanced settings"):
            n_top = st.slider("How many products to recommend:",
                              min_value=1,
                              max_value=10,
                              step=1, 
                              value=3)
            dist_type = st.selectbox("Which distance to use to find the best product",
                                     ['cosine', 'euclidian'])
        
        #Select food group from which to recommend products
        groups_list = np.append(
            ['any group'],
            FoodData['pnns_groups_1'].unique()
        )

        selected_group = st.selectbox(
            "Select food group from which to recommend products",
            groups_list
        )

        #Get mapping of excluded groups
        map_excl_group = ~FoodData['pnns_groups_2'].isin(excluded_groups)

        if selected_group != 'any group':
            map_group = FoodData['pnns_groups_1'] == selected_group

            subgroups_list = np.append(
                ["any subgroup"],
                FoodData.loc[map_group, 'pnns_groups_2'].unique()
            )

            selected_subgroup = st.selectbox(
                "Select subgroup from which to recommend products",
                subgroups_list
            )

            if selected_subgroup != 'any subgroup':
                map_subgroup = FoodData['pnns_groups_2'] == selected_subgroup
                FoodData_subset = FoodData.loc[map_group&map_subgroup,:]

            else:
                FoodData_subset = FoodData.loc[map_group&map_excl_group,:]

        else:
            FoodData_subset = FoodData.loc[map_excl_group,:]

        FoodData_subset = extract_highest_grade_products(FoodData_subset)

        #Normalize numerical values into values/100 g/mL
        FoodData_subset_norm = FoodData_subset[numeric_fields].mul(FoodData_subset['quantity']/100, axis=0)
        FoodData_subset_norm.columns = FoodData_subset_norm.columns.str.replace('_100g',"")
        FoodData_subset_norm.head()

        FoodData_subset_norm = subset_normalization(FoodData_subset, val_target, numeric_fields)
        best_products_found = n_top_products(FoodData, FoodData_subset_norm, r_best, n_top=n_top, dist=dist_type)

        for index in best_products_found.index:
            col1, col2 = st.columns(2)
            with col1:
                prod_sel = st.button(FoodData.loc[index, 'product_name'])
                if FoodData.loc[index, "image_small_url"] != None:
                    st.image(FoodData.loc[index, "image_small_url"])
                else:
                    st.write("No image")
            with col2:
                if prod_sel:
                    v_nutr = pd.DataFrame(FoodData.loc[index,numeric_fields])
                    v_nutr.columns = ['Nutr. val. /100g (or mL)']
                    v_nutr.index = ["Energy (kcal)",
                                    "Saturated fats (g)",
                                    "Unsaturated fats (g)",
                                    "Sugars (g)",
                                    "Fibers / carbs (g)",
                                    "Preoteins (g)",
                                    "Sodium (g)"
                                   ]
                    st.dataframe(v_nutr)
                    st.write(f"Nutriscore: {FoodData.loc[index,'nutriscore_grade'].upper()}")
                    if dist_type == 'cosine':
                        st.write(f"Recommended amount: {max(best_products_found.loc[index, 'quantity'], 0):.0f} g or mL.")