import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.colors as mcolors
import squarify as sqrf
import streamlit as st
import seaborn as sns

from babel.dates import format_datetime
from datetime import datetime
from Levenshtein import distance as levdist
from PIL import Image
from textwrap import wrap
from wordcloud import WordCloud, get_single_color_func


path_to_df = "processed_data.pkl"
df = pd.read_pickle(path_to_df)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

viz_prono = "Pronostics"
results = "Résultats"
sexes = ["Fille", "Garçon"]

dict_cheveux = {
    "ordre_cheveux": [
        "Maxi chevelure !",
        "Chevelure classique",
        "Juste assez pour ne pas être chauve...",
        "Pas de cheveux !",
    ],
    "ordre_couleur": ["Noirs", "Bruns", "Roux", "Blonds"],
    "couleurs_cheveux": ["black", "saddlebrown", "darkorange", "gold"],
    "longueur_cheveux": [4, 2, 0.5, 0],
    "longueur_cheveux_pts": [1, 0.7, 0.2, 0],
    "dist_couleur": {
        "Roux": 0,
        "Blonds": 0.3,
        "Bruns": 0.7,
        "Noirs": 1,
    },
    "ordre_style":[
        "Tendance", # : dans le top 50
        "Classique", #  : y en a toujours eu un peu et il y en aura toujours
        "Ancien", # qui revient (un peu) à la mode
        "Rare", #  : dans le top 100 mais plutôt en partant de la fin
        "Inventé" #  : première fois en France si ce n'est dans le monde
    ],
    "dist_style":{
        "Tendance" : 1,
        "Classique" : 0.8,
        "Ancien" : 0.5,
        "Rare" : 0.2,
        "Inventé" : 0
    },
}

dict_baby = {
    "birthday": pd.to_datetime("12/10/2024 22:28:00", dayfirst=True),
    "prenom": "Arwen",
    "sexe": "Fille",
    "taille": 45,
    "poids": 2715,
    "longueur_cheveux": "Maxi chevelure !",
    "couleur_cheveux": "Noirs",
    "couleur_yeux": "Bleu foncé comme un bébé",
    "style_prenom": "Rare : dans le top 100 mais plutôt en partant de la fin",
    "color": "indigo",
}

dict_R = {
    'birthday' : pd.to_datetime('23/01/2022 03:36:00', dayfirst=True),
    'prenom' : 'Raphaël',
    'sexe' : 'Garçon',
    'taille': 47,
    'poids' : 2565,
    'longueur_cheveux' : 'Maxi chevelure !',
    'couleur_cheveux' : 'Noirs',
    'color' : 'green',
}

dict_F = {
    'prenom' : 'Florian',
    'sexe' : 'Garçon',
    'taille': 50,
    'poids' : 3290,
    'longueur_cheveux' : 'Maxi chevelure !',
    'couleur_cheveux' : 'Noirs',
    'color' : 'darkred'
}

dict_H = {
    'prenom' : 'Hélène',
    'sexe' : 'Fille',
    'taille': 48,
    'poids' : 3170,
    'longueur_cheveux' : 'Maxi chevelure !',
    'couleur_cheveux' : 'Noirs',
    'color' : 'blue'
}

## ALTAIR
selector = alt.selection_point(empty=True, fields=["sexe"])
base = alt.Chart(df).add_params(selector).interactive()

zoom_possible = """ ✨ Zoom et filtre garçon/fille possible + infobulles si vous êtes sur ordinateur. ✨  
"""


def portrait_robot(data):
    sexe_majo = data["sexe"].mode()[0]

    if sexe_majo == "Fille":
        prenom_majo = data.loc[data["sexe"] == sexe_majo]["prenom_fem"].mode()[0]
    else:
        prenom_majo = data.loc[data["sexe"] == sexe_majo]["prenom_masc"].mode()[0]

    dict_profile = {
        "birthday": data.date.median(),
        "prenom": prenom_majo,
        "sexe": sexe_majo,
        "taille": data.taille.median(),
        "poids": data.poids.median(),
        "longueur_cheveux": data["longueur_cheveux"].mode()[0],
        "couleur_cheveux": data["couleur_cheveux"].mode()[0],
        "couleur_yeux": data["couleur_yeux"].mode()[0],
        "style_prenom": data["style_prenom"].mode()[0],
    }

    return dict_profile


def display_birth_announcement(dict_baby, sexes, colors_gender, dict_cheveux):
    sex_color = colors_gender[sexes.index(dict_baby["sexe"])]
    winning_color_hair = dict_cheveux["couleurs_cheveux"][
        dict_cheveux["ordre_couleur"].index(dict_baby["couleur_cheveux"])
    ]

    jour = format_datetime(dict_baby["birthday"], "EEEE d MMMM yyyy 'à' H'h'mm ", locale="fr")
    fille = "e" if dict_baby["sexe"] == "Fille" else ""
    # pronom = "elle" if dict_baby["sexe"] == "Fille" else "il"

    size = 0.4
    outer_colors = [winning_color_hair, "white"]
    inner_color = ["pink"]

    fig, ax = plt.subplots()

    ax.pie(
        [0.5, 0.5],
        radius=1,
        colors=outer_colors,
        wedgeprops=dict(width=size, edgecolor="w"),
    )

    ax.pie([1], radius=1 - size, colors=inner_color, wedgeprops=dict(edgecolor="w"))

    rect = plt.Rectangle(
        # (left - lower corner), width, height
        (0.2, -0.1),
        size + 0.22,
        1.05,
        fill=False,
        color=sex_color,
        lw=4,
        zorder=-100,
        transform=fig.transFigure,
        figure=fig,
        linestyle="--",
        capstyle='round',
        sketch_params=1.1,
    )
    fig.patches.extend([rect])

    plt.text(0, 1.15, f"C'est un{fille} {dict_baby['sexe']} !".upper(), ha="center", fontsize=10, fontfamily="serif")
    plt.text(0, -1, f"~ {dict_baby['prenom']} ~".upper(), ha="center", fontsize=25, color=sex_color, fontfamily="serif")
    plt.text(0, -1.3, f"Né{fille} le {jour}".upper(), ha="center", fontweight="ultralight", fontfamily="serif")
    plt.text(
        0, -1.5, f"{int(dict_baby['taille'])} cm - {int(dict_baby['poids']):1,} kg", ha="center", fontfamily="serif"
    )

    if dict_baby["longueur_cheveux"] == "Pas de cheveux !":
        hair_text = f"{dict_baby['prenom']} est chauve,"
    else:
        hair_text = f"{dict_baby['prenom']} a les cheveux {dict_baby['couleur_cheveux'].lower()} ({dict_baby['longueur_cheveux'].lower()}),"

    eye_text = f"ses yeux sont {' '.join(dict_baby['couleur_yeux'].lower().split(' ')[:2])}"
    name_text = f"et son prénom est {dict_baby['style_prenom'].lower().split(' ')[0]}"

    plt.text(0, -1.85, f"{hair_text}\n{eye_text} {name_text}.", ha="center", fontsize=10, fontfamily="serif", color="grey")

    return fig


def formatting_pct(pct, allvals):
    absolute = int(round(pct / 100.0 * np.sum(allvals)))
    return "{:d}\n({:.1f}%)".format(absolute, pct)


def sex_predictions(data, colors_gender_dict):
    labels = data.value_counts().index.tolist()
    colors_gender = [colors_gender_dict[x] for x in labels]
    fig, ax = plt.subplots(figsize=(4, 4))
    _, _, autotexts = ax.pie(
        data.value_counts(),
        labels=labels,
        labeldistance=1.15,
        wedgeprops={"linewidth": 3, "edgecolor": "white"},
        autopct=lambda pct: formatting_pct(pct, [x for x in data.value_counts()]),
        textprops={
            "size": "larger",
        },
        colors=colors_gender,
    )
    autotexts[0].set_color("white")
    autotexts[1].set_color("white")
    ax.axis("equal")
    return fig


def display_birthdate_pred(base, condition_color_gender):
    expected_d_day = (
        alt.Chart(
            pd.DataFrame(
                {
                    "Date de naissance": pd.date_range(
                        start="2024-11-04T23:00:00",
                        end="2024-11-05T22:59:59",
                        periods=100,
                    )
                }
            )
        )
        .mark_tick(thickness=2, size=80, color="lightgrey", opacity=1)
        .encode(x="Date de naissance:T")
        .interactive()
    )

    d_day = (
        alt.Chart(pd.DataFrame([dict_baby]))
        .mark_tick(thickness=2, size=80, opacity=1, color="indigo")
        .encode(
            alt.X("birthday:T"),
            tooltip=[
                alt.Tooltip("prenom", title=" "),
                alt.Tooltip(
                    "birthday",
                    title="  ",
                    format=format_datetime(
                        dict_baby["birthday"],
                        "EEEE d MMMM yyyy 'à' H'h'mm ",
                        locale="fr",
                    ),
                ),
            ],
        )
        .interactive()
    )

    pred_ticks = (
        base.mark_tick(thickness=2, size=80, opacity=0.5)
        .encode(
            alt.X("date:T", title="Date de naissance"),
            color=condition_color_gender,
            tooltip=[
                alt.Tooltip("date", title="Date prédite "),  # , format='%d-%m à %H:%M'
                alt.Tooltip("heure", title="à "),
                alt.Tooltip("prenom", title="par "),
                alt.Tooltip("nom", title=" "),
            ],
        )
        .properties(height=135)
        .interactive()
    )

    return expected_d_day, d_day, pred_ticks


def size_charts(base, condition_color_gender, selector):  # colors_scale_family
    taille_scale = alt.Scale(domain=(40, 60))
    poids_scale = alt.Scale(domain=(2000, 5000))
    tick_axis = alt.Axis(labels=False, ticks=False)
    area_args = {"opacity": 0.3, "interpolate": "step"}

    points = base.mark_circle().encode(
        alt.Y("taille", scale=taille_scale, title="Taille (cm)"),
        alt.X("poids", scale=poids_scale, title="Poids (g)"),
        color=condition_color_gender,
        tooltip=[
            alt.Tooltip("poids", title="Poids "),
            alt.Tooltip("taille", title="Taille "),
            alt.Tooltip("prenom", title="par "),
            alt.Tooltip("nom", title=" "),
        ],
    )

    points_F = alt.Chart(pd.DataFrame([dict_F])
    ).mark_point(size=50, color=dict_F['color']).encode(
        alt.Y('taille', scale=taille_scale, title='Taille (cm)'),
        alt.X('poids', scale=poids_scale, title='Poids (g)'),
        tooltip=[
            alt.Tooltip('prenom', title=" "),
            alt.Tooltip('poids', title="Poids "), 
            alt.Tooltip('taille', title='Taille '), 
        ]       
    ).interactive().transform_filter(selector)

    points_H = alt.Chart(pd.DataFrame([dict_H])
    ).mark_point(size=50, color=dict_H['color']).encode(
        alt.Y('taille', scale=taille_scale, title='Taille (cm)'),
        alt.X('poids', scale=poids_scale, title='Poids (g)'),
        tooltip=[
            alt.Tooltip('prenom', title=" "),
            alt.Tooltip('poids', title="Poids "), 
            alt.Tooltip('taille', title='Taille '), 
        ]      
    ).interactive().transform_filter(selector)

    points_R = alt.Chart(pd.DataFrame([dict_R])
    ).mark_point(size=50, color=dict_R['color']).encode(
        alt.Y('taille', scale=taille_scale, title='Taille (cm)'),
        alt.X('poids', scale=poids_scale, title='Poids (g)'),
        tooltip=[
            alt.Tooltip('prenom', title=" "),
            alt.Tooltip('poids', title="Poids "), 
            alt.Tooltip('taille', title='Taille '), 
        ]      
    ).interactive().transform_filter(selector)

    points_baby = (
        alt.Chart(pd.DataFrame([dict_baby]))
        .mark_point(size=50, color=dict_baby["color"])
        .encode(
            alt.Y("taille", scale=taille_scale, title="Taille (cm)"),
            alt.X("poids", scale=poids_scale, title="Poids (g)"),
            tooltip=[
                alt.Tooltip("prenom", title=" "),
                alt.Tooltip("poids", title="Poids "),
                alt.Tooltip("taille", title="Taille "),
            ],
        )
        .interactive()
        .transform_filter(selector)
    )

    points = alt.layer(points, points_H, points_F, points_R, points_baby).interactive()
    
    right_chart = (
        base.mark_area(**area_args)
        .encode(
            alt.Y(
                "taille:Q",
                bin=alt.Bin(maxbins=20, extent=taille_scale.domain),
                stack=None,
                title="",
                axis=tick_axis,
            ),
            alt.X("count()", stack=None, title=""),
            color=condition_color_gender,
            tooltip=[
                alt.Tooltip("taille", title="Taille"),
                alt.Tooltip("count()", title="Nb "),
            ],
        )
        .properties(width=100)
        .transform_filter(selector)
    )

    top_chart = base.mark_tick().encode(
        alt.Y("sexe", axis=tick_axis, title=""),
        alt.X("poids", axis=tick_axis, scale=poids_scale, title=""),
        tooltip=alt.Tooltip("poids"),
        color=condition_color_gender,
    )

    return top_chart, points, right_chart


def size_charts_2(base, condition_color_gender, selector): #colors_scale_family
    taille_scale = alt.Scale(domain=(40, 60))
    poids_scale = alt.Scale(domain=(2000, 5000))
    tick_axis = alt.Axis(labels=False, ticks=False)
    area_args = {'opacity': .3, 'interpolate': 'step'}

    points = base.mark_circle().encode(
        alt.Y('taille', scale=taille_scale, title='Taille (cm)'),
        alt.X('poids', scale=poids_scale, title='Poids (g)'),
        color=condition_color_gender,
        tooltip=[
            alt.Tooltip('poids', title="Poids "), 
            alt.Tooltip('taille', title='Taille '), 
            alt.Tooltip('prenom', title="par "),
            alt.Tooltip('nom', title=" ")
        ]
    )

    points_F = alt.Chart(pd.DataFrame([dict_F])
    ).mark_point(size=50, color=dict_F['color']).encode(
        alt.Y('taille', scale=taille_scale, title='Taille (cm)'),
        alt.X('poids', scale=poids_scale, title='Poids (g)'),
        tooltip=[
            alt.Tooltip('prenom', title=" "),
            alt.Tooltip('poids', title="Poids "), 
            alt.Tooltip('taille', title='Taille '), 
        ]       
    ).interactive().transform_filter(selector)

    points_H = alt.Chart(pd.DataFrame([dict_H])
    ).mark_point(size=50, color=dict_H['color']).encode(
        alt.Y('taille', scale=taille_scale, title='Taille (cm)'),
        alt.X('poids', scale=poids_scale, title='Poids (g)'),
        tooltip=[
            alt.Tooltip('prenom', title=" "),
            alt.Tooltip('poids', title="Poids "), 
            alt.Tooltip('taille', title='Taille '), 
        ]      
    ).interactive().transform_filter(selector)

    points_R = alt.Chart(pd.DataFrame([dict_baby])
    ).mark_point(size=50, color=dict_baby['color']).encode(
        alt.Y('taille', scale=taille_scale, title='Taille (cm)'),
        alt.X('poids', scale=poids_scale, title='Poids (g)'),
        tooltip=[
            alt.Tooltip('prenom', title=" "),
            alt.Tooltip('poids', title="Poids "), 
            alt.Tooltip('taille', title='Taille '), 
        ]       
    ).interactive().transform_filter(selector)

    # points_all = alt.Chart(pd.DataFrame([dict_F, dict_H, dict_baby])
    # ).mark_point(size=50).encode(
    #     alt.Y('taille', title='Taille (cm)'),
    #     alt.X('poids', title='Poids (g)'),
    #     color=alt.Color('prenom', scale=colors_scale_family),
    #     tooltip=[
    #         alt.Tooltip('prenom', title=" "),
    #         alt.Tooltip('poids', title="Poids "), 
    #         alt.Tooltip('taille', title='Taille '), 
    #     ]       
    # ).interactive()

    points = alt.layer(points, points_H, points_F, points_R).interactive()
    #points = alt.layer(points, points_all).interactive()

    right_chart = base.mark_area(**area_args).encode(
        alt.Y('taille:Q',
            bin=alt.Bin(maxbins=20, extent=taille_scale.domain),
            stack=None,
            title='', axis=tick_axis,
            ),
        alt.X('count()', stack=None, title=''),
        color=condition_color_gender,
        tooltip=[alt.Tooltip('taille', title='Taille'), alt.Tooltip('count()', title='Nb ')]
    ).properties(width=100).transform_filter(selector)

    top_chart = base.mark_tick().encode(
        alt.Y('sexe', axis=tick_axis, title=''),
        alt.X('poids', axis=tick_axis, scale=poids_scale, title=''),
        tooltip=alt.Tooltip('poids'),
        color=condition_color_gender
    )

    return top_chart, points, right_chart

def calcul_angles(angles, proportion, acc):
    current_angle = proportion * np.pi
    acc.append(current_angle)
    if len(angles) == 0:
        angles.append(current_angle / 2)
    else:
        angles.append(angles[-1] + acc[-2] / 2 + current_angle / 2)

    return angles, acc


def make_hair_bars(df, dict_cheveux):
    width = []
    radii = []
    angles = []
    acc = []
    theta = []
    colors = []
    nb_labels = []

    for longueur in dict_cheveux["ordre_cheveux"]:
        nb = len(df.loc[(df["longueur_cheveux"] == longueur)])
        angles, acc = calcul_angles(angles, nb / len(df), acc)

        for couleur in dict_cheveux["ordre_couleur"]:
            nb = len(df.loc[(df["longueur_cheveux"] == longueur) & (df["couleur_cheveux"] == couleur)])
            if nb > 0:
                nb_labels.append("{} - {}".format(couleur, nb))
                radii.append(dict_cheveux["longueur_cheveux"][dict_cheveux["ordre_cheveux"].index(longueur)])
                colors.append(dict_cheveux["couleurs_cheveux"][dict_cheveux["ordre_couleur"].index(couleur)])
                theta, width = calcul_angles(theta, nb / len(df), width)

    return width, radii, angles, theta, colors, nb_labels


def cool_hair_plot(df, dict_cheveux):
    width, radii, angles, theta, colors, nb_labels = make_hair_bars(df, dict_cheveux)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0, 0, 0.8, 0.8], polar=True)
    bottom = 5
    bars = ax.bar(theta, radii, width=(np.array(width) - 0.01), bottom=bottom, color=colors)

    # mise en forme du graphique
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(angles)

    # nom label trop long
    hair_labels = dict_cheveux["ordre_cheveux"].copy()
    hair_labels[0] = "Maxi Chevelure"

    ax.set_xticklabels(["\n".join(wrap(r, 11, break_long_words=False)) for r in hair_labels])
    ax.set_yticks([0, 10])
    ax.set_yticklabels([])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.spines["start"].set_color("none")
    ax.spines["end"].set_color("none")
    ax.spines["polar"].set_color("none")

    # affichage des labels + rotation pour lisibilité et opacité
    rotations = np.rad2deg(theta)
    for x, bar, rotation, label in zip(theta, bars, rotations, nb_labels):
        new_rotation = 180 + rotation if rotation > 90 else rotation
        ax.text(
            x,
            bottom + bar.get_height() + 1,
            label,
            size=7,
            ha="center",
            va="center",
            rotation=new_rotation,
            rotation_mode="anchor",
        )
        bar.set_alpha(0.7)

    return fig


def eye_plot(data, color_yeux_dict):
    labels = data.value_counts().index.tolist()
    colors_eyes = [color_yeux_dict[x] for x in labels]
    fig, ax = plt.subplots(figsize=(4, 4))
    _, _, autotexts = ax.pie(
        data.value_counts(),
        labels=labels,
        labeldistance=1.15,
        autopct=lambda pct: formatting_pct(pct, [x for x in data.value_counts()]),
        # textprops={
        #     "size": "larger",
        # },
        colors=colors_eyes,
    )
    autotexts[0].set_color("white")
    autotexts[1].set_color("white")
    ax.axis("equal")

    ax.pie([1], radius=0.6, colors="w", wedgeprops={"alpha": 0.3})
    ax.pie([1], radius=0.4, colors="k")
    # plt.subplots_adjust(bottom=0, right=0.8, top=0.9)

  
    # marche trop bien pour faire un effet d'iris
    # fig1, ax1 = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # # Generate data for iris-like effect
    # theta = np.linspace(0, 6*np.pi, 400)
    # radii = 1.5 + 0.2 * np.sin(15 * theta) + 2 * np.random.rand(400)

    # # Plot with gradient and texture to mimic iris
    # plt.plot(theta, radii, color='skyblue', linewidth=2)
    # plt.fill(theta, radii, color='skyblue', alpha=0.5)

    # # Add darker rings to simulate iris texture
    # for i in range(8):
    #     inner_radii = radii - 0.05 * (i+1)
    #     plt.plot(theta, inner_radii, color='darkblue', alpha=0.3, linewidth=1)

    # # Remove default polar plot styling
    # ax1.set_yticklabels([])
    # ax1.set_xticklabels([])
    # ax1.xaxis.set_visible(False)
    # ax1.yaxis.set_visible(False)
    # ax1.patch.set_alpha(0.5)
    # ax1.set_facecolor(None)

    # # fig.patches.extend([fig1])

    # plt.subplots_adjust(bottom=0, left=0.5, right=1, top=0.9)

    return fig


def create_sequential_colormap(start_color, end_color, num_colors=256):
    """
    Create a custom sequential colormap between two colors.
    
    Parameters:
    - start_color: Starting color (hex or RGB)
    - end_color: Ending color (hex or RGB)
    - num_colors: Number of color gradations in the colormap
    
    Returns:
    - matplotlib.colors.LinearSegmentedColormap
    """
    # Convert colors to RGB if they're in hex
    if isinstance(start_color, str):
        start_color = mcolors.to_rgb(start_color)
    if isinstance(end_color, str):
        end_color = mcolors.to_rgb(end_color)
    
    # Create color gradient
    colors = [
        (start_color[0] + (end_color[0] - start_color[0]) * i/(num_colors-1),
         start_color[1] + (end_color[1] - start_color[1]) * i/(num_colors-1),
         start_color[2] + (end_color[2] - start_color[2]) * i/(num_colors-1))
        for i in range(num_colors)
    ]

    return mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)


def visualize_name_style_distribution(prognosis_series, colors_gender):
    """
    Create multiple visualizations of prognosis distribution using Matplotlib
    
    :param prognosis_series: Pandas Series with prognosis categories and their counts
    :return: Matplotlib figure objects
    """
    # Convert Series to dictionary if it's not already
    if not isinstance(prognosis_series, dict):
        prognosis_data = prognosis_series.value_counts().to_dict()
    else:
        prognosis_data = prognosis_series

    # Prepare the data
    categories = list(prognosis_data.keys())
    counts = list(prognosis_data.values())
    total = sum(counts)
    
    # Color palette
    custom_cmap = create_sequential_colormap(colors_gender[0], colors_gender[1])
    colors = [custom_cmap(x) for x in np.linspace(0, 1, len(counts))]
    # colors = plt.cm.custom_cmap(np.linspace(0, 1, len(categories)))

    # 2. Treemap Visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    sqrf.plot(sizes=counts, 
                label=[f'{cat.split(" ")[0]} : {count} ({count/total*100:.1f}%)' 
                        for cat, count in zip(categories, counts)],
                color=colors,
                text_kwargs={'fontsize':12},
                alpha=0.8,
                )
    ax.axis('off')

    return fig


def distance_name_plot(dict_cheveux):
    text = []
    points = []

    for style_name in dict_cheveux["dist_style"]:
        dist = dict_cheveux["dist_style"][style_name]
        text.append(f"{style_name} : {np.round(dist, 2)}")
        points.append(dist)

    # Choose some nice levels
    levels = np.tile([3, 2, 1], int(np.ceil(len(points) / 3)))[: len(dict_cheveux["dist_style"])]

    # Create figure and plot a stem plot
    fig, ax = plt.subplots(figsize=(8, 2), constrained_layout=True)
    ax.set(title="")

    markerline, _, _ = ax.stem(
        points,
        levels,
        linefmt="C3-",
        basefmt="k-",
        # use_line_collection=True
    )

    plt.setp(markerline, mec="k", mfc="w", zorder=3)

    # Shift the markers to the baseline by replacing the y-data by zeros.
    markerline.set_ydata(np.zeros(len(points)))

    # annotate lines
    vert = np.array(["top", "bottom"])[(levels > 0).astype(int)]
    for d, l, r, va in zip(points, levels, text, vert):
        ax.annotate(
            r,
            xy=(d, l),
            xytext=(-3, np.sign(l) * 3),
            textcoords="offset points",
            va=va,
            ha="left",
        )

    # remove y axis and spines
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    for spine in ["left", "top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.margins(y=0.1)

    return fig


def distance_hair_plot(dict_cheveux):
    text = []
    points = []

    for longueur_cheveux in dict_cheveux["ordre_cheveux"]:
        dist_index = dict_cheveux["ordre_cheveux"].index(longueur_cheveux)
        dist = dict_cheveux["longueur_cheveux_pts"][dist_index]
        pts = np.round(dist, 2)
        text.append(f"{longueur_cheveux} : {pts}")
        points.append(pts)

    for couleur in dict_cheveux["dist_couleur"]:
        dist = dict_cheveux["dist_couleur"][couleur]
        text.append(f"{couleur} : {np.round(dist, 2)}")
        points.append(dist)

    # Choose some nice levels
    levels_chev = np.tile([-3, -1, -2], int(np.ceil(len(points) / 3)))[: len(dict_cheveux["ordre_cheveux"])]
    levels_coul = np.tile([3, 2, 1], int(np.ceil(len(points) / 3)))[: len(dict_cheveux["dist_couleur"])]

    levels = np.append(levels_chev, levels_coul)

    # Create figure and plot a stem plot
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.set(title="")

    markerline, _, _ = ax.stem(
        points,
        levels,
        linefmt="C3-",
        basefmt="k-",
        # use_line_collection=True
    )

    plt.setp(markerline, mec="k", mfc="w", zorder=3)

    # Shift the markers to the baseline by replacing the y-data by zeros.
    markerline.set_ydata(np.zeros(len(points)))

    # annotate lines
    vert = np.array(["top", "bottom"])[(levels > 0).astype(int)]
    for d, l, r, va in zip(points, levels, text, vert):
        ax.annotate(
            r,
            xy=(d, l),
            xytext=(-3, np.sign(l) * 3),
            textcoords="offset points",
            va=va,
            ha="left",
        )

    # remove y axis and spines
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    for spine in ["left", "top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.margins(y=0.1)

    return fig


def count_all_different_names(series: pd.Series):
    return pd.Series(series.dropna()).value_counts()


def get_dict_from_freq(freq_masc, freq_fem):
    dict_masc = dict(freq_masc)
    dict_fem = dict(freq_fem)

    dict_both = {k: dict_masc.get(k, 0) + dict_fem.get(k, 0) for k in set(dict_masc) | set(dict_fem)}

    return dict_masc, dict_fem, dict_both


def most_voted_names(freq_masc, freq_fem):
    _, _, dict_both = get_dict_from_freq(freq_masc, freq_fem)

    max_freq = max(dict_both.values())
    prenoms_max = [name for name, freq in dict_both.items() if freq == max_freq]
    nb_freq = len(prenoms_max)

    return max_freq, prenoms_max, nb_freq


def both_gender_cloud(freq_masc, freq_fem, colors_gender, mask_path):
    mask = np.array(Image.open(mask_path))
    mask[mask == 0] = 255

    _, dict_fem, dict_both = get_dict_from_freq(freq_masc, freq_fem)

    color_to_words = {
        dict_baby["color"]: [dict_baby["prenom"]],
        colors_gender[0]: list(dict_fem.keys()),
    }
    grouped_color_func = GroupedColorFunc(color_to_words, default_color=colors_gender[1])

    wordcloud = WordCloud(
        random_state=30,
        background_color="white",
        max_font_size=45,
        mask=mask,
        relative_scaling=0.7,
        color_func=grouped_color_func,
    ).generate_from_frequencies(dict_both)

    fig = plt.figure(figsize=(16, 9))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    return fig


def one_gender_cloud(freq, color, mask_path):
    mask = np.array(Image.open(mask_path))
    mask[mask == 0] = 255

    color_to_words = {dict_baby["color"]: [dict_baby["prenom"]]}
    grouped_color_func = GroupedColorFunc(color_to_words, default_color=color)

    wordcloud = WordCloud(
        random_state=6,
        background_color="white",
        max_font_size=45,
        mask=mask,
        color_func=grouped_color_func,
    ).generate_from_frequencies(dict(freq))

    fig = plt.figure(figsize=(16, 9))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    return fig


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
    to certain words based on the color to words mapping
    Parameters
    ----------
    color_to_words : dict(str -> list(str))
      A dictionary that maps a color to the list of words.
    default_color : str
      Color that will be assigned to a word that's not a member
      of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color for (color, words) in color_to_words.items() for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
    specified colors to certain words based on the color to words mapping.
    Uses wordcloud.get_single_color_func
    Parameters
    ----------
    color_to_words : dict(str -> list(str))
      A dictionary that maps a color to the list of words.
    default_color : str
      Color that will be assigned to a word that's not a member
      of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words)) for (color, words) in color_to_words.items()
        ]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(color_func for (color_func, words) in self.color_func_to_words if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)


def set_hair_style_points(dict_baby, dict_cheveux):
    index_longueur_cheveux_baby = dict_cheveux["ordre_cheveux"].index(dict_baby["longueur_cheveux"])
    dist_longueur_baby = dict_cheveux["longueur_cheveux_pts"][index_longueur_cheveux_baby]
    coeff_cheveux = {}
    for longueur_cheveux in dict_cheveux["ordre_cheveux"]:
        dist_index = dict_cheveux["ordre_cheveux"].index(longueur_cheveux)
        dist = abs(dist_longueur_baby - dict_cheveux["longueur_cheveux_pts"][dist_index])
        coeff_cheveux[longueur_cheveux] = np.round(1 - dist, 2)

    dist_couleur = dict_cheveux["dist_couleur"]
    coeff_couleur = {}
    for couleur in dict_cheveux["ordre_couleur"]:
        coeff_couleur[couleur] = 1 - (abs(dist_couleur[dict_baby["couleur_cheveux"]] - dist_couleur[couleur]))

    dist_style = dict_cheveux["dist_style"]
    coeff_style = {}
    for style in dict_cheveux["ordre_style"]:
        coeff_style[style] = 1 - (abs(dist_style[dict_baby["style_prenom"]] - dist_style[style]))

    return coeff_cheveux, coeff_couleur, coeff_style


def calculate_scores(df_pred, dict_baby):
    df = df_pred.drop(columns=["heure", "birthday_day", "birthday_time"], errors="ignore")
    df = df.replace(np.nan, "", regex=True)
    df.prenom_masc = df.prenom_masc.astype(str)
    df.prenom_fem = df.prenom_fem.astype(str)

    coeff_cheveux, coeff_couleur, coeff_style = set_hair_style_points(dict_baby, dict_cheveux)
    is_boy = dict_baby["sexe"] == "Garçon"

    name_dist = lambda x: (
        1
        if levdist(dict_baby["prenom"].lower(), x.lower().strip()) < 1
        else np.maximum(
            1 - 0.2 * (levdist(dict_baby["prenom"].lower(), x.lower().strip()) - 2) - 0.05 * np.abs(len(x) - 7) ** 0.5,
            0,
        )
    )

    # name_dist = lambda x: (levdist(dict_baby["prenom"].lower(), x.lower().strip()))

    delta = np.abs(df.date - dict_baby["birthday"]).astype("int64") * 1e-15
    df["score_date"] = np.exp(-0.8 * delta)
    
    df["score_sexe"] = (df.sexe == dict_baby["sexe"]) * (1 - df["sexe"].value_counts()[dict_baby["sexe"]] / len(df))

    df["score_poids"] = np.exp(-(((dict_baby["poids"] - df.poids) / 500) ** 2))
    df["score_taille"] = np.maximum(0, (1 - 0.15 * np.abs(dict_baby["taille"] - df.taille)))

    df["score_cheveux"] = df.longueur_cheveux.apply(lambda x: coeff_cheveux[x])
    df["score_couleur"] = df.couleur_cheveux.apply(lambda x: coeff_couleur[x])
    df["score_cheveux"] = 0.5 * (df.score_cheveux + df.score_couleur)
    df.drop("score_couleur", axis=1, inplace=True)

    coeff_yeux = 1 - df["couleur_yeux"].value_counts()[dict_baby["couleur_yeux"]] / len(df)
    df["score_yeux"] = (df.couleur_yeux == dict_baby["couleur_yeux"]) * coeff_yeux

    df["score_prenom"] = np.maximum(
        df.prenom_masc.apply(name_dist) * (1 if is_boy else 0.8),
        df.prenom_fem.apply(name_dist) * (0.8 if is_boy else 1),
    )
    df["score_style"] = df.style_prenom.apply(lambda x: coeff_style[x.split(" ")[0]])
    df["score_prenom"] = 0.8 * df.score_prenom + 0.2 * df.score_style
    df.drop("score_style", axis=1, inplace=True)

    scores = [col for col in df.columns if "score" in col]
    df["score"] = df[scores].sum(1)

    place = np.argsort(df.score.to_numpy())[::-1].tolist()
    df["place"] = [place.index(i) + 1 for i in range(len(df))]

    df = df.sort_values(by="score", ascending=False)
    df = df.set_index("place")

    return df.round(3)


def beautify_df(df):
    mapper = {
        "prenom": "Prénom",
        "nom": "Nom",
        "sexe": "Sexe",
        "date": "Date de naissance",
        "poids": "Poids",
        "taille": "Taille",
        "longueur_cheveux": "Longueur des cheveux",
        "couleur_cheveux": "Couleur des cheveux",
        "prenom_masc": "Prénom masculin",
        "prenom_fem": "Prénom féminin",
        "score_date": "Score date",
        "score_sexe": "Score sexe",
        "score_prenom": "Score prénom",
        "score_poids": "Score poids",
        "score_taille": "Score taille",
        "score_cheveux": "Score cheveux",
        "score_yeux": "Score yeux",
        "score": "Score total",
        "place": "Classement",
        "couleur_yeux" : "Couleur des yeux",
        "style_prenom" : "Style du prénom",
    }
    df = df.rename(columns=mapper)
    cols = df.columns.tolist()
    cols.pop(cols.index("prenom_masc_clean"))
    cols.pop(cols.index("prenom_fem_clean"))

    # cols.pop(cols.index("Prénom"))
    # cols.pop(cols.index("Nom"))

    return df[cols]


def df_styler(df):
    styler = df.style.format(
        {"Date de naissance": lambda t: format_datetime(t, "d MMMM yyyy 'à' H'h'mm ", locale="fr")}
    )

    return styler


def function_select(df):
    options = {}
    for _, row in df.iterrows():
        options[f""] = pd.Series(row)
    return options


def scores_participant(serie_participant, len_df):
    st.markdown(
        f"<center><b>~ &nbsp; ~ &nbsp; Prédictions de {serie_participant['Prénom']} {serie_participant['Nom']} &nbsp; ~ &nbsp; ~</b></center>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<center><b>Score total</b> : {serie_participant['Score total']} pts</center>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<center><b>Classement</b> : {serie_participant.place} / {len_df}</center>",
        unsafe_allow_html=True,
    )

    st.write("&nbsp;")

    col0, col1 = st.columns((2, 1))
    col0.markdown(f"**Sexe** : {serie_participant['Sexe']}")
    col1.markdown(f"{serie_participant['Score sexe']} pt")

    col0, col1 = st.columns((2, 1))
    date_predite = format_datetime(
        datetime.strptime(serie_participant["Date de naissance"], "%Y-%m-%d %H:%M:%S"),
        "d MMMM yyyy 'à' H'h'mm ",
        locale="fr",
    )
    col0.markdown(f"**Date de naissance** : {date_predite}")
    col1.markdown(f"{serie_participant['Score date']} pt")

    col0, col1 = st.columns((2, 1))
    col0.markdown(f"**Poids** : {serie_participant['Poids']} g")
    col1.markdown(f"{serie_participant['Score poids']} pt")

    col0, col1 = st.columns((2, 1))
    col0.markdown(f"**Taille** : {serie_participant['Taille']} cm")
    col1.markdown(f"{serie_participant['Score taille']} pt")

    col0, col1 = st.columns((2, 1))
    col0.markdown(
        f"**Cheveux** : {serie_participant['Couleur des cheveux']} &nbsp; - &nbsp; {serie_participant['Longueur des cheveux']}"
    )
    col1.markdown(f"{serie_participant['Score cheveux']} pt")

    col0, col1 = st.columns((2, 1))
    col0.markdown(
        f"**Yeux** : {serie_participant['Couleur des yeux']}"
    )
    col1.markdown(f"{serie_participant['Score yeux']} pt")


    col0, col1 = st.columns((2, 1))
    double_prenom = (
        True
        if len(serie_participant["Prénom masculin"]) > 1 and len(serie_participant["Prénom féminin"]) > 1
        else False
    )
    col0.markdown(
        f"""
        **Prénom{'s' if double_prenom else ''}** : {serie_participant['Prénom masculin']} 
        {'&nbsp; & &nbsp;' if double_prenom else "&nbsp;"} {serie_participant['Prénom féminin']} &nbsp; - &nbsp; {serie_participant['Style du prénom'].split(' ')[0]}
        """
    )
    col1.markdown(f"""{serie_participant['Score prénom']} pt""")


def plot_score_distributions(df, columns=None, figsize=(15, 15), bins=10, output_file=None):
    """
    Generate histograms for score distributions across specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing score columns
    columns : list, optional
        List of column names to plot. If None, will use all numeric columns.
    figsize : tuple, optional
        Size of the entire figure (width, height)
    bins : int, optional
        Number of bins for the histograms
    output_file : str, optional
        Path to save the figure. If None, will display the plot.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure with score distribution histograms
    """
    # If no columns specified, find numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Calculate number of rows and columns for subplots
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if len(columns) > 1 else [axes]
    
    # Plot histograms
    for i, col in enumerate(columns):
        sns.histplot(df[col], bins=bins, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('Frequency')
    
    # Remove extra subplots if any
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    # if output_file:
    #     plt.savefig(output_file)
    # else:
    #     plt.show()
    
    return fig


def create_olympic_podium(df, value_column):
    """
    Create an Olympic-style podium visualization with first place in the center
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    value_column : str
        Name of the column to use for ranking
    
    Returns:
    --------
    plotly.Figure
        Olympic podium visualization
    """
    # Sort and select top 3 entries
    top_3 = df.nlargest(3, value_column).reset_index(drop=True)
    
    # Reorder to have first place in the middle
    # Second place on the left, third place on the right
    reordered = [
        top_3.iloc[0],  # First place (center)
        top_3.iloc[1],  # Second place (left)
        top_3.iloc[2]   # Third place (right)
    ]
    
    # Define podium heights (Olympic-style)
    heights = [150, 100, 75]
    colors = ['gold', 'silver', '#CD7F32']
    
    # Create figure
    fig = go.Figure()
    
    # Add podium blocks with specific x-positions
    x_positions = [0, -2, -1]  # Center, Left, Right
    
    for i in range(3):
        fig.add_trace(go.Bar(
            x=[reordered[i]['Prénom']],
            y=[heights[i]],
            name=reordered[i]['Prénom'],
            marker_color=colors[i],
            text=[f"<b>{reordered[i]['Prénom']}</b><br>{reordered[i][value_column]:.2f}"],
            textposition='outside',  # Place text outside the bar
            textfont=dict(size=13),  # Adjust text size
            hoverinfo=' z',
            width=0.5,
            cliponaxis = False,
            offset=x_positions[i] +0.5  # Adjust positioning
        ))
    
    # Customize layout to look like a podium
    fig.update_layout(
        title='Podium',
        # xaxis_title='Entries',
        # yaxis_title='Performance',
        height=200,
        # width=600,
        margin=dict(t=40, b=50, l=20, r=20),
        showlegend=False,
        # bargap=1,
        plot_bgcolor='white',
        xaxis=dict(
            tickangle=0,
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=[
                f"🥈 {reordered[1]['Prénom']}",  # Second place
                f"🥇 {reordered[0]['Prénom']}",  # First place
                f"🥉 {reordered[2]['Prénom']}"   # Third place
            ],
        ),
        yaxis=dict(
            showline=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False),
    )
    
    # Positioning to create podium effect
    fig.update_traces(
        width=1,
        marker_line_width=0.8,
        marker_line_color='black'
    )
    
    return fig


footer = """<style>
a:link , a:visited{
color: red;
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: gray;
text-align: center;
}
</style>
<div class="footer">
<p>Made with 💖 by Hélène T.</p>
</div>
"""
