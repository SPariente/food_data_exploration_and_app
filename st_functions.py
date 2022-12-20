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


def add_barcode(barcode_list, data, code_col):
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
    
    if barcode[0].data in data[code_col].astype(bytes).values:
        barcodes_list_new.append(barcode[0].data)
        return barcodes_list_new
    
    else:
        st.text(f"Aucun produit dont le code barre est {int(barcode[0].data)} n'a pu être identifié.")
        return barcodes_list_new

        
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


def nutr_values_products(data_products, balanced_values, barcodes_list, numeric_fields, excluded_groups_list=None):
    
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
        title=f"Total nutritional values of the basket<br>({n_products} products considered)"
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
        ticktext=['<b>Balanced<br>basket</b>'],
        color='black',
        tickvals=[1],
        title=None,
        linewidth=2,
        linecolor=None,
        showgrid=False
    )

    fig.update_traces(
        hovertemplate='%{label}<br>Cible : %{customdata[1]:.0f} g<br>Gap to target: %{customdata[0]:.0f} g<extra></extra>',
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
        title=f"Basket composition ({len(actuals)} products)"
    )

    fig.update_traces(
        sort=True,
        textfont_color='black',
        textinfo='label+value',
        insidetextorientation='radial',
        hovertemplate='%{label} : %{value} products<extra></extra>'
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


def subset_normalization(data, norm_vector, numeric_fields):
    
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
    # ideal_vector = np.array(ideal_vector).reshape(1, -1)
    # distance = cosine_similarity(data, ideal_vector)
    distance = [np.dot(ideal_vector, sample)/(np.linalg.norm(ideal_vector)*np.linalg.norm(sample))\
                if (np.linalg.norm(ideal_vector)*np.linalg.norm(sample)!=0) else 0\
                for sample in data.values]
    data['distance'] = distance
    
    n_best_products = data.nlargest(n = n_top, columns = 'distance')
    
    # Calcul de la quantité idéale de produit en 2 temps
    ideal_quantity_mul = n_best_products['distance']*np.linalg.norm(ideal_vector)
    n_best_products['quantity'] = full_data.loc[n_best_products.index,'quantity']*ideal_quantity_mul
    
    return n_best_products


def n_top_products(full_data, data_subset, ideal_vector, n_top=3, dist="cosine"):
    
    """
    Function to allow selection of distance to use (default = cosine)
    """
    
    if dist == "cosine":
        return n_top_cosine(full_data, data_subset, ideal_vector, n_top=n_top)
    
    if dist == "euclidian":
        return n_top_eucl(full_data, data_subset, ideal_vector, n_top=n_top)
    
    else:
        raise ValueError("La distance doit être 'cosine' ou 'euclidian'.")