import streamlit as st
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import plotly.express as px

#configure page
st.set_page_config(page_title="NYPD Strategic Command Center", layout="wide")

@st.cache_data
#load data function with caching to optimize performance
def load_data():
    df_preds = pd.read_csv('crime_predictions_streamlit.csv')
    df_preds['date'] = pd.to_datetime(df_preds['date'])
    df_actuals = pd.read_csv('nyc_crime_data_cleaned.csv', low_memory=False)
    df_actuals['CMPLNT_FR_DT'] = pd.to_datetime(df_actuals['CMPLNT_FR_DT'])
    df_actuals['Hour'] = pd.to_datetime(df_actuals['CMPLNT_FR_TM'], format='%H:%M:%S').dt.hour
    df_actuals['DayOfWeek'] = df_actuals['CMPLNT_FR_DT'].dt.day_name()
    centroids = pd.read_csv('precinct_centroids.csv')
    return df_preds, df_actuals, centroids

#run optimization function using Gurobi to solve the strategic deployment problem
def run_optimization(input_df, target_date, budget, floor, cap, efficiency, weight):
    day_data = input_df[input_df['date'] == pd.to_datetime(target_date)].copy()
    if day_data.empty: return None, None
    m = gp.Model("Strategic_Deployment"); m.setParam('OutputFlag', 0)
    precincts = day_data['ADDR_PCT_CD'].unique()
    shifts = day_data['shift_window'].unique()
    x = m.addVars(precincts, shifts, vtype=GRB.INTEGER, lb=floor, ub=cap, name="units")
    risk = m.addVars(precincts, shifts, lb=0, name="risk")
    m.addConstr(gp.quicksum(x[p, s] for p in precincts for s in shifts) <= budget)
    for _, row in day_data.iterrows():
        p, s, pred = row['ADDR_PCT_CD'], row['shift_window'], row['predicted_incidents']
        m.addConstr(risk[p, s] >= pred - (x[p, s] * efficiency))
    obj = gp.quicksum(risk[p, s] * (weight if day_data[(day_data['ADDR_PCT_CD']==p) & (day_data['shift_window']==s)]['high_risk'].iloc[0] == 1 else 1.0) for p in precincts for s in shifts)
    m.setObjective(obj, GRB.MINIMIZE); m.optimize()
    if m.status == GRB.OPTIMAL:
        res = [{'ADDR_PCT_CD': p, 'Shift': day_data[day_data['shift_window']==s]['shift_label'].iloc[0], 'Officers': int(x[p, s].X)} for p in precincts for s in shifts]
        return pd.DataFrame(res), m.objVal
    return None, None

#initialization and data loading
df_preds, df_actuals, centroids = load_data()

#function to clear all widget states and reset to defaults
def reset_widgets():
    st.session_state["budget_slider"] = 350
    st.session_state["floor_input"] = 0
    st.session_state["weight_slider"] = 2.0
    st.session_state["hist_date"] = [df_actuals['CMPLNT_FR_DT'].min().date(), df_actuals['CMPLNT_FR_DT'].max().date()]

#create sidebar with filters and optimization parameters
st.sidebar.header("Global Filters")

#historical data range filter
hist_date_range = st.sidebar.date_input(
    "Historical Date Range", 
    value=st.session_state.get("hist_date", [df_actuals['CMPLNT_FR_DT'].min(), df_actuals['CMPLNT_FR_DT'].max()]),
    key="hist_date"
)

st.sidebar.divider()
st.sidebar.header("Optimization Parameters")
st.sidebar.caption("Adjust the parameters for the strategic deployment optimization model. Use the reset button to return to default settings.")

#users can adjust the budget, safety floor, and hotspot priority weight for the optimization model
user_budget = st.sidebar.slider(
    "Officer-Shift Budget", 100, 1000, 
    value=st.session_state.get("budget_slider", 350),
    key="budget_slider",
    help="Total 4-hour shifts available per 24h period."
)

user_floor = st.sidebar.number_input(
    "Safety Floor (Min Units)", 0, 2, 
    value=st.session_state.get("floor_input", 0),
    key="floor_input",
    help="Minimum officers assigned to every precinct-shift."
)

user_weight = st.sidebar.slider(
    "Hotspot Priority Weight", 1.0, 10.0, 
    value=st.session_state.get("weight_slider", 2.0),
    key="weight_slider",
    help="Multiplier for 'High Risk' windows."
)

#Reset button
if st.sidebar.button("Reset to Defaults", type="secondary", on_click=reset_widgets):
    st.rerun()

#Main dashboard layout with tabs for historical analysis and strategic deployment
st.title("NYPD Brooklyn Command Center")
tab1, tab2 = st.tabs(["Historical Analysis (Actuals)", "Strategic Deployment (Expected)"])

with tab1:
    #ensure valid date range is selected for historical analysis and filter the actuals data accordingly
    if isinstance(hist_date_range, list) or len(hist_date_range) == 2:
        h_mask = (df_actuals['CMPLNT_FR_DT'] >= pd.to_datetime(hist_date_range[0])) & \
                 (df_actuals['CMPLNT_FR_DT'] <= pd.to_datetime(hist_date_range[1]))
        f_hist = df_actuals[h_mask]
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Geospatial Crime Density")
            fig_map = px.density_mapbox(f_hist, lat='Latitude', lon='Longitude', radius=7, zoom=10, mapbox_style="carto-positron")
            st.plotly_chart(fig_map, use_container_width=True)
        with c2:
            st.metric("Total Incidents", f"{len(f_hist):,}")
            st.subheader("Crime Category Distribution")
            top_crimes = f_hist['OFNS_DESC'].value_counts().nlargest(10).reset_index()
            fig_bar = px.bar(top_crimes, x='count', y='OFNS_DESC', orientation='h', color='count', color_continuous_scale="Reds")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()
        c3, c4 = st.columns([1, 1])
        with c3:
            st.subheader("Temporal Hotspots (Hour vs. Day)")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heat_data = f_hist.groupby(['DayOfWeek', 'Hour']).size().reset_index(name='Incidents')
            heat_pivot = heat_data.pivot(index='DayOfWeek', columns='Hour', values='Incidents').reindex(day_order)
            fig_heat = px.imshow(heat_pivot, labels=dict(x="Hour of Day", y="Day of Week", color="Incidents"), x=list(range(24)), color_continuous_scale="YlOrRd")
            st.plotly_chart(fig_heat, use_container_width=True)
        with c4:
            st.subheader("Crime Location Profiling (Premises)")
            prem_data = f_hist['PREM_TYP_DESC'].value_counts().nlargest(15).reset_index()
            fig_tree = px.treemap(prem_data, path=['PREM_TYP_DESC'], values='count', color='count', color_continuous_scale="Greens")
            st.plotly_chart(fig_tree, use_container_width=True)

with tab2:
    #allow users to select a target date for optimization and run the Gurobi model to calculate the optimal patrol plan based on the predicted crime data
    st.header("Future Deployment Engine")
    target_date = st.selectbox("Select Target Date", sorted(df_preds['date'].dt.date.unique(), reverse=True))
    
    if st.button("CALCULATE OPTIMAL PATROL PLAN", type="primary", use_container_width=True):
        with st.spinner("Solving Gurobi Model..."):
            plan, obj_val = run_optimization(df_preds, target_date, user_budget, user_floor, 60, 0.05, user_weight)
        
        if plan is not None:
            map_data = plan.groupby('ADDR_PCT_CD')['Officers'].sum().reset_index().merge(centroids, on='ADDR_PCT_CD')
            day_data = df_preds[df_preds['date'] == pd.to_datetime(target_date)].copy()
            total_raw = day_data['predicted_incidents'].sum()
            day_data['weight'] = day_data['high_risk'].apply(lambda x: user_weight if x == 1 else 1.0)
            total_weighted = (day_data['predicted_incidents'] * day_data['weight']).sum()
            coverage = (1 - (obj_val / total_weighted)) * 100 if total_weighted > 0 else 0
            
            st.success(f"Optimal Plan for {target_date} Generated.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Incident Volume", f"{total_raw:.2f}", help="Total statistical probability of crime.")
            m2.metric("Optimal Risk Score", f"{obj_val:.2f}", help="Remaining weighted risk.")
            m3.metric("Predicted Crime Coverage", f"{coverage:.1f}%", help="Weighted risk mitigated.")
            
            cl1, cl2 = st.columns([2, 1])
            with cl1:
                st.subheader("Strategic Deployment Map")
                fig_opt = px.scatter_mapbox(map_data, lat='Latitude', lon='Longitude', size='Officers', color='Officers', color_continuous_scale="Viridis", zoom=10, mapbox_style="carto-darkmatter")
                st.plotly_chart(fig_opt, use_container_width=True)
            with cl2:
                st.subheader("Deployment Schedule Detail")
                st.dataframe(plan.sort_values(by='Officers', ascending=False), use_container_width=True)