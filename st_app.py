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

#Import additional libraries
import requests
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


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

#Initialize new list
if st.button("New product list"):
     st.session_state['barcodes_list'] = []
else: #If no button clicked, check for existing list, and if not create empty one
    if 'barcodes_list' not in st.session_state:
        st.session_state['barcodes_list'] = []

def add_barcode(barcode_list):
    """
    Function to open a webcam, read a barcode, check if the product is
    found in the database, and if so add it to a list of barcodes for
    further use.
    """
    
    cam_viz = st.image([]) #Open visualization area
    vc = cv2.VideoCapture(0) #Set capture mode
    barcode = False #Clear read barcode
    barcodes_list_new =  barcode_list.copy() #Copy the input list

    if vc.isOpened(): #Check for feed
        retval, frame = vc.read()
    else:
        retval = False

    while retval: #Exit on camera close
        retval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27 or barcode: #exit on ESC key or barcode detection
            break

        time.sleep(0.5) #Use 2 frames per second
        barcode = decode(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_viz.image(frame)

    vc.release()
    cam_viz.empty()
    
    if barcode[0].data in FoodData['code'].astype(bytes).values:
        barcodes_list_new.append(barcode[0].data)
        return barcodes_list_new
    
    else:
        st.text(f"Aucun produit dont le code barre est {int(barcode[0].data)} n'a pu être identifié.")
        return barcodes_list_new

#Add barcode to list
if st.button("Add new product"):
     st.session_state['barcodes_list'] = add_barcode(st.session_state['barcodes_list'])

def find_codes(codes, data, code_col):
    
    '''
    Fonction qui extrait l'ensemble des données des produits d'après une liste de code barres.
    
    Paramètres:
    -----------
    - codes : list / itérable de codes barres
    - data : dataframe contenant les données à extraire, dont les codes barres
    - code_col : nom de colonne (str) de la colonne de data contenant les codes barres
    
    Résultats:
    ----------
    Dataframe contenant l'ensemble des produits identifiés par les codes barres.    
    '''
    
    actuals = pd.DataFrame()
    
    for code in codes:
        actuals = pd.concat([actuals, data.loc[data[code_col].astype(bytes) == code,:]])
    
    return actuals



def nutr_values_products(data_products, balanced_values, barcodes_list, excluded_groups_list=None):
    
    """
    Function to:
    - Find nutritional values associated with products in barcodes list
    - Exclude some select food products from computation of nutritional values
    - Compute (sum) the nutritional values found for non-excluded products
    - Compute target value for each nutritional item based on energy in products
    - Compute the gap to target (ratio and absolute)
    - Computate the ideal product values to reach target (by difference to actuals)
    """
    
    actuals = find_codes(barcodes_list, data_products, 'code')
    
    if excluded_groups_list is not None:
        map_excl_group = ~actuals['pnns_groups_2'].isin(excluded_groups_list)
        val_actuals = (actuals.loc[map_excl_group,'quantity']/100).dot(actuals.loc[map_excl_group,numeric_fields])
        n_products = len(actuals.loc[map_excl_group,'quantity']/100)
    
    else:
        val_actuals = (actuals.loc[:,'quantity']/100).dot(actuals.loc[:,numeric_fields])
        n_products = len(actuals)
    
    val_actuals.index = val_actuals.index.str.replace('_100g',"")
    
    val_target = balanced_values * val_actuals["energy-kcal"]
    
    r_actuals = val_actuals/val_target
    
    gap_target = val_actuals - val_target
    
    r_best = 1 - r_actuals
    
    return actuals, val_actuals, val_target, r_actuals, gap_target, r_best, n_products

def show_basket_values(actuals_norm, actuals, target, n_products, labels, units):
    
    """
    Function to visualize the nutritional values of the current basket.
    """
    
    fig = px.bar(
        data_frame=pd.DataFrame({
            'actuals':actuals_norm,
            'gap_target':target - actuals,
            'target': target
        }),
        orientation='h',
        color=actuals_norm.index,
        width=500,
        height=500,
        text=[f"<b>{val:,.0f} {unit}</b>" for val, unit in zip(actuals, units)],
        custom_data=['gap_target', 'target'],
        title=f"Valeur nutritionnelle totale du panier<br>({n_products} produits retenus)"
    )

    fig.add_vline(
        x=1, 
        line={
            'dash':'dash',
            'color':'black',
            'width': 3
        }
    )

    fig.update_yaxes(
        ticktext=labels,
        tickvals=actuals_norm.index,
        title=None,
        linewidth=5,
        color='black'
    )

    fig.update_xaxes(
        ticktext=['<b>Panier<br>équilibré</b>'],
        color='black',
        tickvals=[1],
        title=None,
        linewidth=2,
        linecolor=None,
        showgrid=False
    )

    fig.update_traces(
        hovertemplate='%{label}<br>Cible : %{customdata[1]:.0f} g<br>Ecart à la cible : %{customdata[0]:.0f} g<extra></extra>',
        textposition='inside',
        insidetextanchor="start"
    )
    fig.update_traces(
        hovertemplate='%{label}<extra></extra>', selector={'name':'energy-kcal'}
    )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family="Source Sans Pro",
        font_size=15,
        hoverlabel_font_family="Source Sans Pro",
        title_font_color='black',
        title_x=0.5
    )

    return fig

def show_basket_composition(actuals):
    
    """
    Visualization by product type.
    """
    
    # Dataframe contenant le nombre de produits par catégorie
    sb_values = pd.DataFrame(actuals.loc[:,['pnns_groups_1','pnns_groups_2']].value_counts(), columns=['values'])
    sb_values.reset_index(inplace=True)
    
    # Visualisation en sunburst chart
    fig = px.sunburst(
        data_frame=sb_values,
        values='values',
        path=['pnns_groups_1','pnns_groups_2'],
        width=500,
        height=500,
        color='pnns_groups_1',
        title=f"Composition du panier ({len(actuals)} produits)"
    )

    fig.update_traces(
        sort=True,
        textfont_color='black',
        textinfo='label+value',
        insidetextorientation='radial',
        hovertemplate='%{label} : %{value} produit(s)<extra></extra>'
    )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family="Source Sans Pro",
        font_size=15,
        hoverlabel_font_family="Source Sans Pro",
        title_font_color='black',
        title_x=0.5
    )
    
    return fig

def extract_highest_grade_products(data):
    
    """
    Function to keep only the highest nutriscore products
    """
    
    grades = data['nutriscore_grade'].dropna().unique()
    grades.sort()
    map_grade = data['nutriscore_grade'] == grades[0]
    data_subset = data.loc[map_grade,:]
    
    return data_subset

def subset_normalization(data, norm_vector):
    
    """
    Projection of dataset into space where unit vector = norm_vector
    """
    
    data = data[numeric_fields].mul(data['quantity']/100, axis=0)
    data.columns = data.columns.str.replace('_100g',"")
    
    data_norm = data/norm_vector
    
    return data_norm


def n_top_eucl(full_data, data_subset, ideal_vector, n_top=3):
    
    """
    Returns the n top products to bridge the identified gap,
    based on euclidian distance
    """
    
    data = data_subset.copy()
    distance = np.linalg.norm(data - ideal_vector, axis=1)
    data['distance'] = distance
    
    n_best_products = data.nsmallest(n=n_top, columns='distance')
    n_best_products['quantity'] = full_data.loc[n_best_products.index, 'quantity']
    
    return n_best_products

def n_top_cosine(full_data, data_subset, ideal_vector, n_top=3):    
    
    """
    Returns the n top products to bridge the identified gap,
    based on cosine similarity
    """
    
    data = data_subset.copy()
    ideal_vector = np.array(ideal_vector).reshape(1, -1)
    distance = cosine_similarity(data, ideal_vector)
    data['distance'] = distance
    
    n_best_products = data.nlargest(n=n_top, columns='distance')
    
    # Calcul de la quantité idéale de produit en 2 temps
    ideal_quantity_mul = n_best_products['distance']*np.linalg.norm(ideal_vector)
    n_best_products['quantity'] = full_data.loc[n_best_products.index,'quantity']*ideal_quantity_mul
    
    return n_best_products

def n_top_products(full_data, data_subset, ideal_vector, n_top=3, dist="cosine"):
    
    """
    Function to allow selection of distance to use (default = cosine)
    """
    
    if dist == "cosine":
        return n_top_cosine(full_data, data_subset, ideal_vector, n_top=3)
    
    if dist == "eucl":
        return n_top_eucl(full_data, data_subset, ideal_vector, n_top=3)
    
    else:
        raise ValueError("La distance doit être 'cosine' ou 'eucl'.")




#Select if all foods or only non-perishables are to be included
only_perishables = st.checkbox('Use only perishable products',
                               value=True)
if not only_perishables:
    excluded_groups = None
else: #Definition of nutritional groups to exclude from computation of nutritional values
    excluded_groups = ['sweets', 'fats', 'protein complements', 'salts']

if len(st.session_state['barcodes_list'])>0: 
    #Use the function
    actuals, val_actuals, val_target, r_actuals, gap_target, r_best, n_products=nutr_values_products(FoodData, means_kcal['mean'], st.session_state['barcodes_list'], excluded_groups)

    #Define labels for visualization
    label_vals = [
        'Calories',
        'Graisses saturées',
        'Graisses non-saturées',
        'Sucres',
        'Autres glucides',
        'Protéines',
        'Sodium'
    ]

    #Define units to show for visualization
    units = ['kcal']+['g']*(len(label_vals)-1)

    #Show figure
    fig = show_basket_values(r_actuals, val_actuals, val_target, n_products, label_vals, units)
    st.plotly_chart(fig)
    
    

    fig = show_basket_composition(actuals)
    st.plotly_chart(fig)
    
    
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
    
    FoodData_subset_norm = subset_normalization(FoodData_subset, val_target)
    best_products_test = n_top_products(FoodData, FoodData_subset_norm, r_best, n_top=3)
    
    for index in best_products_test.index:
        st.image(FoodData.loc[index, "image_small_url"],
                 caption=
                )