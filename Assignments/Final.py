"""
Name: Luke Wininger
CS230: Section 4
Data: Fortune 500
URL:
Description:
This program allows the user to explore the Fortune 500 dataset by viewing corporate headquarter locations based on
certain criteria and comparing company information such as location, revenue, profit, and number of employees. This
program also allows the user to search companies by rank and have certain information be returned.
"""
import streamlit as st
import csv
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
import numpy as np
from pydeck import data_utils as du


# read in data
def read_data():
    return pd.read_csv("Fortune500.csv")


# Mapping function
def map_data(df, ranks, state, city):
    # filters data within rank range
    df = df.loc[df['RANK'] >= ranks[0]]
    df = df.loc[df['RANK'] <= ranks[1]]

    # conditions for whether to filter by state and city
    if state != "US":
        df = df.loc[df['STATE'] == state]
    if city != "ALL":
        df = df.loc[df['CITY'] == city]

    # makes dataframe for mapping and dataframe for table display
    map_df = df.filter(['NAME', 'LATITUDE', 'LONGITUDE'])
    show_points = df[['RANK', 'NAME', 'CITY', 'STATE', 'EMPLOYEES', 'REVENUES', 'PROFIT']]
    st.write(show_points)
    map_df = map_df.rename(columns={'LATITUDE': 'latitude', 'LONGITUDE': 'longitude'})

    # automatically sets initial viewstate to include relevant locations from pyplot documentation
    test_view = du.viewport_helpers.compute_view(map_df[['longitude', 'latitude']], view_proportion=1)

    # maps scatter layer with scaled point sizes
    layer = pdk.Layer('ScatterplotLayer',
                      data=map_df,
                      get_position='[longitude, latitude]',
                      radius_scale=20,
                      radius_min_pixels=3,
                      radius_max_pixels=5,
                      get_color=[255, 0, 0],
                      pickable=True)

    # uses html to set hover text for map points
    tool_tip = {'html': 'Listing:<br/><b>{NAME}</b>', 'style': {'backgroundCOLOR': 'blue', 'color': 'white'}}

    # creates the deck and create map
    map = pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',
                   initial_view_state=test_view,
                   layers=[layer],
                   tooltip=tool_tip)
    st.pydeck_chart(map)


# Barplot of top companies
def top_bar(df, pick):
    # creates separate plot figure
    plt.figure()

    # dataframe with necessary columns
    df = df[['NAME', 'RANK', 'REVENUES']]

    # filters dataframe by selected rank constraint
    df = df.loc[df['RANK'] <= pick].sort_values(by='RANK')

    # makes list of x and y values to graph
    lst = df.values.tolist()
    x = [(comp[0]) for comp in lst]
    y = [(comp[2]) for comp in lst]

    # creates graph plot and labels
    plt.title(f"Revenues of Top {pick} Companies")
    plt.bar(x, y, color="green")
    plt.ylabel("Revenue ($ Millions)")
    plt.xlabel("Companies")
    plt.xticks(fontsize=7, rotation=90)

    return plt


# gets list of cities from inputted state selection
def get_cities(df, state):
    # filters dataframe by state
    df = df.loc[df['STATE'] == state]

    # makes array of all cities remaining in dataframe
    list_cities = np.array(df['CITY'].values.tolist())

    # removes repeated cities.  adds option at position one to select all cities. returns array.
    list_cities = np.unique(list_cities)
    list_cities = np.insert(list_cities, 0, "ALL")
    return list_cities


# makes unique array of states
def get_states(df):
    list_states = np.array(df['STATE'].values.tolist())
    list_states = np.unique(list_states)

    # inserts default 'US' option for not selecting a state
    list_states = np.insert(list_states, 0, "US")
    return list_states


# search a company by their rank
def company_data(df, rank_choice):
    # finds inputted value in dataframe. sets rank as index
    df = df.loc[df['RANK'] == rank_choice].set_index('RANK')

    # retrieves first row from dataframe. returns as a series
    info = df.iloc[0]
    st.subheader(f"Data for Company Ranked {info.name} ({info['NAME']})")

    # removes unwanted indexes from data
    info = info.drop(index=['FID', 'X', 'Y', 'OBJECTID', 'SOURCE', 'PRC', 'COUNTYFIPS'])

    st.write(info)
    # for loop to print out data on each line
    for line in info.index:
        st.write(f"{line}: {info[line]}")


# compare data of several companies at once
def company_compare(df, selections):
    # defines list for later use
    list_dicts = []

    # goes thru chosen selections
    for line in selections:
        # makes copy of dataframe
        diction = df.copy()

        # finds data of company and gets first row
        diction = df.loc[diction['RANK'] == line]
        info = diction.iloc[0]
        # converts series to dict and adds dict to list_dicts
        info = info.to_dict()
        list_dicts.append(info)

    # makes dataframe from list of dicts. drops unwanted columns
    frame = pd.DataFrame(list_dicts)
    frame = frame.drop(columns=['FID', 'X', 'Y', 'OBJECTID', 'SOURCE', 'PRC', 'COUNTYFIPS'])

    # rotates table axis and displays at streamlit table

    frame = frame.transpose()
    st.table(frame)


# shows relationship and trendline between revenue and company size
def regression(df):
    # makes new plot figure
    plt.figure()

    # takes necessary columns and makes array of values
    df = df[['EMPLOYEES', 'REVENUES']]
    lst = df.values.tolist()
    arr = np.array(lst)

    # defines data for scatter layer and line
    x = [(comp[0]) for comp in lst]
    y = [(comp[1]) for comp in lst]
    test1 = arr[:, 0]
    test2 = arr[:, 1]

    # calculates coefficients for best-fit line. defines equation of line. found in numpy documentation
    poly = np.polyfit(test1, test2, 1)
    best_line = poly[1] + poly[0] * test1

    # makes labels, ticks, and plots line
    plt.title("Company Revenue vs Size (With Regression Line)")
    plt.plot(test1, best_line, color="red")
    plt.ylabel("Revenue ($ Millions)")
    plt.xlabel("Employees")
    plt.xticks(fontsize=7, rotation=25)
    plt.ticklabel_format(style="plain")

    # plots scatter and returns both to plot in main function
    plt.scatter(x, y, color="green")
    return plt


# makes pivot table of revenues grouped by state
def pivots(df):
    # rounds column values to whole numbers
    df = df.round({'REVENUES': 0, 'PROFIT': 0})

    # makes pivot table with aggregate sum function, grouped by state as index
    piv_data = df.pivot_table(values=['REVENUES', 'PROFIT'], index=['STATE'], aggfunc=sum)
    st.write(piv_data)


def main():
    full_data = read_data()
    st.title("Fortune 500 Companies")
    st.sidebar.write("Please select options to filter the data.")

    # select which portion of application to view
    view = st.sidebar.radio("Select your view",
                                ("General",
                                 "Map", "Company Search",
                                 "Compare Companies"))
    list_states = get_states(full_data)
    if view == "General":
        # calls functions for pivot table, regression analysis, and top companies by revenue
        st.subheader("Financials by State")
        pivots(full_data)
        st.subheader("Company Revenues vs Size")
        st.pyplot(regression(full_data))

        # slider to pick how many companies to view
        st.subheader(f"Top companies")
        pick = st.slider("How many companies would you like to see?", 2, 20, 5)
        st.pyplot(top_bar(full_data, pick))

    elif view == "Map":
        # select box to choose a state or leave on US.  only lets choose a city if a state is selected
        state = st.selectbox("Select Filter State", list_states)
        if state != "US":
            # calls get_cities function to get list of cities in chosen state. else default as 'ALL'
            city = st.selectbox("Select City", get_cities(full_data, state))
        else:
            city = "ALL"
        # slider to choose upper and lower rank values. calls map function
        values = st.slider("Select a rank range", 1, 500, (1, 500))
        map_data(full_data, values, state, city)
    elif view == "Company Search":
        # number input to choose rank of company to search. calls company data function to display.
        rank_choice = st.number_input(label="Select the Rank of the Company", value=1, step=1, min_value=1, max_value=500)
        company_data(full_data, rank_choice)
    elif view == "Compare Companies":
        # user selects multiple ranks from 1-500.  if a value is selected, calls company compare function.
        selects = st.multiselect("Select ranks to compare", range(1, 500))
        if selects:
            company_compare(full_data, selects)


main()


