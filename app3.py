import streamlit as st
import streamlit as st
import pandas as pd
import plotly.express as px
from queryController import LLMQueries
from modules import getAgentData,getagentLLMToolmapping,prepGraphData,process_trace_data,calcCost
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide", page_title="TredenceAgentOpsAnalytics",page_icon="logo.png")
st.markdown("""
<style>
/* General Layout */
.block-container {
    padding: 0rem 0.5rem 0.5rem 0.5rem;
}

/* Reusable Box Style */
.dashboard-box {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    background-color: #ffffff;
    margin-bottom: 16px;
}

/* Header Box Style */
.title-box {
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 10px 15px;
    background-color: #f5f6fa;
    margin-bottom: 0.5rem;
}
.title-box h5 {
    font-size: 18px;
    font-weight: 600;
    margin: 0;
}

/* Smaller inputs */
.stSelectbox > div, .stDateInput > div > input {
    font-size: 14px;
    height: 34px !important;
    padding: 2px 8px;
}
.stCheckbox {
    margin-top: 30px !important;
}

h6 {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 6px;
}
/* Hide Streamlit's built-in header */
header, [data-testid="stHeader"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Reduce vertical spacing between Streamlit rows */
[data-testid="stVerticalBlock"] {
    gap: 0.2rem !important;
}
[data-testid="stHorizontalBlock"] {
    gap: 0.25rem !important;  /* or even 0.1rem */
    margin-bottom: 0.2rem !important;
}
/* Optional: reduce plot title spacing */
.plot-title {
    margin-bottom: 0.3rem;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Box container for charts and tables */
.plot-box {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 16px;
    background-color: #ffffff;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
    transition: box-shadow 0.3s ease;
}
.plot-box:hover {
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
}

/* Plot or table titles inside the box */
.plot-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)
llm = LLMQueries()

#st.markdown("<div class='boxed-section'><h5 style='margin: 0;'>LLM Observability Dashboard</h5></div>", unsafe_allow_html=True)

# st.markdown("""
# <div style='background-color:lightblue; padding:1px 1px; border-radius:1px; margin-bottom:1px'>
#     <h5 style='margin:0; font-size:19px;'>Tredence Agent Ops Platform</h5>
# </div>
# """, unsafe_allow_html=True)
# st.markdown("""
#     <div style='background-color:lightblue; padding:10px; border-radius:5px; margin-bottom:10px'>
#         <h5 style='margin:0; font-size:19px; color:black;'>Tredence Agent Ops Platform</h5>
#     </div>
# """, unsafe_allow_html=True)

st.markdown("""
<div style='background-color:lightblue; padding:12px 16px; border-radius:0px; margin:0; position: relative; top: 0;'>
    <h5 style='margin:0; font-size:20px; font-weight:700; color:black;'>Tredence Agent Ops Platform</h5>
</div>
""", unsafe_allow_html=True)

with st.expander("Filters",expanded=True):
# --- Filters Box ---
    col1, col2, col3, col4 = st.columns([0.4, 0.2, 0.2, 0.2],vertical_alignment="center")
    with col1:
        selected = st.selectbox("Projects", llm.get_dropdown_options(), label_visibility="visible")
    with col2:
        min_date, max_date = llm.get_date_range_for_project(selected)
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, label_visibility="visible")
    with col3:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, label_visibility="visible")
    with col4:
        st.markdown("&nbsp;", unsafe_allow_html=True)

# --- Metric Section ---
val1 = llm.get_tile1_value(selected, start_date, end_date)
val2 = round(llm.get_tile2_value(selected, start_date, end_date), 2)
val3 = llm.get_tile3_value(selected, start_date, end_date)
val4 = llm.get_tile4_value(selected, start_date, end_date)

# metric_cols = st.columns(4,border=True)
# metric_cols[0].metric("Total LLM Calls", val1)
# metric_cols[1].metric("Avg Latency", val2)
# metric_cols[2].metric("Prompt Tokens", val3)
# metric_cols[3].metric("Completion Tokens", val4)
model_df= llm.getAttributes(selected,start_date, end_date)
agentData= getAgentData(model_df)
llm_df,tool_df=getagentLLMToolmapping(agentData)
unique_models = sorted(llm_df['child_name'].unique())
agentLLMdf=llm_df.loc[
            (llm_df["parent_span_kind"]=="AGENT") & (llm_df["child_span_kind"]=="LLM"),
            ["parent_name","child_name","token","parent_id","error","token_input",
             "token_output","status_code","status_message","timestamp"]
        ].rename(columns={
        "parent_name": "agent_name",
        "child_name": "llm_model",
        "token": "token_count",
        "parent_id": "agent_span_id",
        "error": "error_count"
        })
grouped_df = agentLLMdf.groupby(['agent_span_id', 'agent_name','llm_model',"status_code","status_message","timestamp"], as_index=False)[['token_count','error_count',"token_input","token_output"]].sum()
grouped_df= calcCost(grouped_df)
agent_new_df=grouped_df.loc[grouped_df["status_code"]=="OK",
    ["agent_name","llm_model","token_count","agent_span_id","total_cost(in dollars)","error_count","timestamp"]]
agent_new_error_df=grouped_df.loc[
    grouped_df["status_code"].isin(["UNSET", "ERROR"]),
    ["agent_name","llm_model","token_count","agent_span_id","status_message","total_cost(in dollars)","timestamp"]]
toolLLMdf=llm_df.loc[
            (llm_df["parent_span_kind"]=="TOOL") & (llm_df["child_span_kind"]=="LLM"),
            ["parent_name","child_name","token","parent_id","error","token_input",
            "token_output","status_code","status_message","timestamp"]
        ].rename(columns={
        "parent_name": "tool_name",
        "child_name": "llm_model",
        "token": "token_count",
        "parent_id": "tool_span_id",
        "error": "error_count"
        })
grouped_df = toolLLMdf.groupby(['tool_span_id','tool_name', 'llm_model',"status_code","status_message","timestamp"], as_index=False)[['token_count','error_count',"token_input","token_output"]].sum()
grouped_df= calcCost(grouped_df)
tool_new_df=grouped_df.loc[
    grouped_df["status_code"]=="OK",
    ["tool_name","llm_model","token_count","tool_span_id","error_count","total_cost(in dollars)","timestamp"]]
tool_new_error_df=grouped_df.loc[
    grouped_df["status_code"].isin(["UNSET", "ERROR"]),
    ["tool_name","llm_model","token_count","tool_span_id","error_count","total_cost(in dollars)","timestamp"]]       
total_agent_new_df=agent_new_df.groupby(['agent_name', 'llm_model'], as_index=False)[["token_count","total_cost(in dollars)"]].sum()
total_tool_new_df=tool_new_df.groupby(['tool_name', 'llm_model'], as_index=False)[["token_count","total_cost(in dollars)"]].sum()
col1, col2,col4 = st.columns([1.3,1.3,4.7],border=True) 
with col1:
    st.metric("Total LLM Calls", val1)
    st.divider()
    st.metric("Avg Latency", val2)
    st.divider()
    st.metric("Total Agent Cost (in $)",round(agent_new_df["total_cost(in dollars)"].sum(),4))

with col2:
    st.metric("Prompt Tokens", val3)
    st.divider()
    st.metric("Completion \n Tokens", val4) 
    st.divider() 
    st.metric("Total Tool cost (in $)",round(tool_new_df["total_cost(in dollars)"].sum(),4))
with col4:
    tab1,tab2,tab3,tab4,tab5, tab6=st.tabs(["Cumulative agent token usage","Cumulative tool token usage","Agent token usage", "Tool token usage","Error analysis - agent", "Error analysis - tool"])
    with tab1:
        st.dataframe(total_agent_new_df,use_container_width=True,hide_index=True,height=270)
    with tab2:  
        st.dataframe(total_tool_new_df,use_container_width=True,hide_index=True,height=270)
    with tab3:
         st.dataframe(agent_new_df,use_container_width=True,hide_index=True,height=270)
    with tab4:
        st.dataframe(tool_new_df,use_container_width=True,hide_index=True,height=270)
    with tab5:
        st.dataframe(agent_new_error_df,use_container_width=True,hide_index=True,height=270)
    with tab6: 
        st.dataframe(tool_new_error_df,use_container_width=True,hide_index=True,height=270)       

# --- Charts ---
chart_col1, chart_col2 = st.columns(2,border=True)
with chart_col1:
    totaltraces=llm.get_metric2_data(selected,start_date,end_date)
    st.markdown("Traces Analytics")
    tab1,tab2=st.tabs(["Total traces","Traces over time"])
    status_summary = totaltraces.groupby('status')['trace_count'].sum().reset_index()
    with tab1:
        # Create the pie chart
        fig = px.pie(
            status_summary,
            names='status',
            values='trace_count',
            title='Distribution of Total Traces'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label+value')
        st.plotly_chart(fig,use_container_width=True)
    with tab2:
        fig = px.line(
            totaltraces,
            x='date',
            y='trace_count',
            color='status',
            markers=True,
            title='Trace Counts Over Time by Status'
        )
        fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Traces',
        legend_title='Status Code'
        )  
        # Render in Streamlit
        st.plotly_chart(fig, use_container_width=True)  

with chart_col2:
   # st.markdown("Agent Analytics")
    #This flow is done for crew AI
    tab2,tab3,tab4,tab1=st.tabs(["Model calls per agent","Tool call per agent","Model calls per Tool","Agent call distribution"]) 
  #  llm_df["error_status"] = llm_df["error"].apply(lambda x: "Error" if x > 0 else "No Error")
    with tab1:
        agent_model_counts_all=llm_df.loc[
            (llm_df["parent_span_kind"]=="AGENT") & (llm_df["child_span_kind"]=="LLM"),
            ["parent_name","child_name","parent_id"]
        ]
        agent_model_span_counts_all = agent_model_counts_all.groupby(['parent_name','child_name','parent_id'], as_index=False).size()
        agent_model_agg=agent_model_span_counts_all.groupby(['parent_name','child_name'], as_index=False).size()
        if agent_model_agg is not None:
            fig = px.bar(
                agent_model_agg,
                x='parent_name',
                y='size',
                color='child_name',
                barmode='group',
                labels={
                    'parent_name': 'Agent',
                    'size': 'Call Count',
                    'child_name': 'Model Name',
                },
                    title='Agent Call Distribution'
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("&nbsp;", unsafe_allow_html=True)    

    with tab2:
        llm_filtered = llm_df[llm_df['parent_span_kind']=='AGENT']
        llm_grouped=prepGraphData(llm_filtered)
        tab21,tab22,tab23=st.tabs(['All','Error','No Error'])
        with tab21:
            llm_grouped_all=llm_grouped.groupby(['parent_name', 'child_name'], as_index=False)['count'].sum()
            if llm_grouped_all is not None:
                fig = px.bar(
                    llm_grouped_all,
                    x='child_name',
                    y='count',
                    color='parent_name',
                    barmode='group',  # or 'stack',
                    labels={'child_name': 'Model Name', 'count': 'Number of Calls','parent_name':'Agent_name'},
                    title='Number of LLM calls'
                )
                st.plotly_chart(fig, use_container_width=True,key="llm_all")
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True) 
        with tab22:
            llm_grouped_error=llm_grouped[llm_grouped['error']>0]
            if llm_grouped_error is not None:
                fig = px.bar(
                    llm_grouped_error,
                    x='child_name',
                    y='count',
                    color='parent_name',
                    barmode='group',  # or 'stack',
                    labels={'child_name': 'Model Name', 'count': 'Number of Calls','parent_name':'Agent_name'},
                    title='Number of LLM calls'
                )
                st.plotly_chart(fig, use_container_width=True,key="llm_error")
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)
        with tab23:
            llm_grouped_noerror=llm_grouped[llm_grouped['error']==0]
            if llm_grouped_noerror is not None:
                fig = px.bar(
                    llm_grouped_noerror,
                    x='child_name',
                    y='count',
                    color='parent_name',
                    barmode='group',  # or 'stack',
                    labels={'child_name': 'Model Name', 'count': 'Number of Calls','parent_name':'Agent_name'},
                    title='Number of LLM calls'
                )
                st.plotly_chart(fig, use_container_width=True,key="llm_norror")
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)                                                   
    with tab3:
        tool_grouped=prepGraphData(tool_df)
        tab31,tab32,tab33=st.tabs(['All','Error','No Error'])
        with tab31:
            tool_grouped_all=tool_grouped.groupby(['parent_name', 'child_name'], as_index=False)['count'].sum()
            if tool_grouped_all is not None:
                fig = px.bar(
                    tool_grouped_all,
                    x='child_name',
                    y='count',
                    color='parent_name',
                    barmode='group',  # or 'stack',
                    labels={'child_name': 'Tool Name', 'count': 'Number of Calls','parent_name':'Agent_name'},
                    title='Number of Tool calls'
                )
                st.plotly_chart(fig, use_container_width=True, key="tool_all")
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)
        with tab32:
            tool_grouped_error=tool_grouped[tool_grouped['error']>0]
            if tool_grouped_error is not None:
                fig = px.bar(
                    tool_grouped_error,
                    x='child_name',
                    y='count',
                    color='parent_name',
                    barmode='group',  # or 'stack',
                    labels={'child_name': 'Tool Name', 'count': 'Number of Calls','parent_name':'Agent_name'},
                    title='Number of Tool calls'
                )
                st.plotly_chart(fig, use_container_width=True, key="tool_error")
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)
        with tab33:
            tool_grouped_noerror=tool_grouped[tool_grouped['error']==0]
            if tool_grouped_noerror is not None:
                fig = px.bar(
                    tool_grouped_noerror,
                    x='child_name',
                    y='count',
                    color='parent_name',
                    barmode='group',  # or 'stack',
                    labels={'child_name': 'Tool Name', 'count': 'Number of Calls','parent_name':'Agent_name'},
                    title='Number of Tool calls'
                )
                st.plotly_chart(fig, use_container_width=True, key="tool_noerror")
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)  
    with tab4:
        tool_llm_filtered = llm_df[llm_df['parent_span_kind']=='TOOL']
        tool_llm_grouped=prepGraphData(tool_llm_filtered)
        tab41,tab42,tab43=st.tabs(['All','Error','No Error'])
        with tab41:
            tool_llm_grouped_all=tool_llm_grouped.groupby(['parent_name', 'child_name'], as_index=False)['count'].sum()
            if tool_llm_grouped_all is not None:
                fig = px.bar(
                    tool_llm_grouped_all,
                    x='child_name',
                    y='count',
                    color='parent_name',
                    barmode='group',  # or 'stack',
                    labels={'child_name': 'Model Name', 'count': 'Number of Calls','parent_name':'Agent_name'},
                    title='Number of LLM calls'
                )
                st.plotly_chart(fig, use_container_width=True,key="tool_llm_all")
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True) 
        with tab42:
            tool_llm_grouped_error=tool_llm_grouped[tool_llm_grouped['error']>0]
            if tool_llm_grouped_error is not None:
                fig = px.bar(
                    tool_llm_grouped_error,
                    x='child_name',
                    y='count',
                    color='parent_name',
                    barmode='group',  # or 'stack',
                    labels={'child_name': 'Model Name', 'count': 'Number of Calls','parent_name':'Agent_name'},
                    title='Number of LLM calls'
                )
                st.plotly_chart(fig, use_container_width=True,key="tool_llm_error")
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)
        with tab43:
            tool_llm_grouped_noerror=tool_llm_grouped[tool_llm_grouped['error']==0]
            if tool_llm_grouped_noerror is not None:
                fig = px.bar(
                    tool_llm_grouped_noerror,
                    x='child_name',
                    y='count',
                    color='parent_name',
                    barmode='group',  # or 'stack',
                    labels={'child_name': 'Model Name', 'count': 'Number of Calls','parent_name':'Agent_name'},
                    title='Number of LLM calls'
                )
                st.plotly_chart(fig, use_container_width=True,key="tool_llm_norror")
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)  

processed_df=process_trace_data(model_df) 
st.dataframe(processed_df,
                 use_container_width=True,hide_index=True,height=300)


# # --- Interactive Bar Chart for LLM calls ---
# st.markdown("LLM Calls Breakdown by Trace ID")

# trace_ids = sorted(processed_df["trace_id"].unique())
# selected_trace_id = st.selectbox("Select Trace ID", trace_ids)

# row = processed_df[processed_df["trace_id"] == selected_trace_id].iloc[0]
# llm_calls = row["LLM calls"]

# models, counts = [], []
# for part in str(llm_calls).split(";"):
#     part = part.strip()
#     if "-" in part:
#         k, v = part.rsplit("-", 1)
#         try:
#             models.append(k.strip())
#             counts.append(int(v))
#         except ValueError:
#             pass

# if models:
#     bar_df = pd.DataFrame({"Model": models, "Count": counts})
#     bar_df["Count"] = bar_df["Count"].astype(int)

#     # === BAR CHART (force go.Bar) ===
#     fig = go.Figure()
#     colors = px.colors.qualitative.Set2

#     for i, row_bar in bar_df.iterrows():
#         fig.add_trace(
#             go.Bar(
#                 x=[row_bar["Model"]],
#                 y=[row_bar["Count"]],
#                 name=row_bar["Model"],
#                 marker_color=colors[i % len(colors)],
#                 text=[row_bar["Count"]],
#                 textposition="outside"
#             )
#         )

#     fig.update_layout(
#         barmode="group",
#         transition_duration=0,
#         xaxis_title="Model",
#         yaxis_title="Number of Calls",
#         yaxis=dict(autorange=True)
#     )

#     clicked = plotly_events(
#         fig,
#         click_event=True,
#         hover_event=False,
#         select_event=False,
#         override_height=500,
#         override_width="100%",
#         key="bar_click"
#     )

#     if clicked:
#         model_name = clicked[0]["x"]

#         latency_map = {k.strip(): v.strip() for k,v in (p.split("-",1) for p in row["latency"].split(";"))}
#         tokens_map = {k.strip(): v.strip() for k,v in (p.rsplit("-",1) for p in row["agent_token_usage"].split(";"))}

#         st.markdown("---")
#         st.subheader(f"Details for **{model_name}**")
#         c1, c2 = st.columns(2)
#         c1.metric("Latency", latency_map.get(model_name, "N/A"))
#         c2.metric("Token Usage", tokens_map.get(model_name, "N/A"))

# else:
#     st.warning("No valid LLM calls found for this trace.")


# ---------- Streamlit App ----------
st.set_page_config(page_title="LLM Calls per Trace ID", layout="wide")

# --- Layout with 2 columns ---
col1, col2 = st.columns(2, border=True)

with col1:
    st.markdown("LLM Calls per Trace ID")

    trace_ids = sorted(processed_df["trace_id"].unique())
    selected_trace_id = st.selectbox("Select Trace ID", trace_ids)

    row = processed_df[processed_df["trace_id"] == selected_trace_id].iloc[0]
    llm_calls = row["LLM calls"]

    # Parse counts
    models, counts = [], []
    for part in str(llm_calls).split(";"):
        part = part.strip()
        if "-" in part:
            k, v = part.rsplit("-", 1)
            try:
                models.append(k.strip())
                counts.append(int(v))
            except ValueError:
                pass

    if models:
        # Build lookup maps for latency/tokens from this row
        latency_map = {}
        for p in row["latency"].split(";"):
            p = p.strip()
            if "-" in p:
                k, v = p.split("-", 1)
                latency_map[k.strip()] = v.strip()

        tokens_map = {}
        for p in row["agent_token_usage"].split(";"):
            p = p.strip()
            if "-" in p:
                k, v = p.rsplit("-", 1)
                try:
                    tokens_map[k.strip()] = f"{int(v):,}"
                except ValueError:
                    tokens_map[k.strip()] = v.strip()

        colors = px.colors.qualitative.Plotly
        fig = go.Figure()

        for i, (m, c) in enumerate(zip(models, counts)):
            customdata = [[latency_map.get(m, "—"), tokens_map.get(m, "—")]]
            fig.add_trace(go.Bar(
                x=[m],
                y=[int(c)],
                name=m,
                marker_color=colors[i % len(colors)],
                text=[int(c)],
                textposition="outside",
                customdata=customdata,
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Calls: %{y}<br>" +
                    "Latency: %{customdata[0]}<br>" +
                    "Tokens: %{customdata[1]}<extra></extra>"
                )
            ))

        fig.update_layout(
            title=f"LLM Calls for Trace {selected_trace_id}",
            barmode="group",
            xaxis_title="Model",
            yaxis_title="Number of Calls"
        )

        clicked = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=500,
            override_width="100%",
            key="bar_click"
        )

        if clicked:
            model_name = clicked[0].get("x")
            st.markdown("---")
            st.subheader(f"Details for **{model_name}**")

            lat_val = latency_map.get(model_name, "N/A")
            tok_val = tokens_map.get(model_name, "N/A")

            c1, c2 = st.columns(2)
            c1.metric("Latency", lat_val)
            c2.metric("Token Usage", tok_val)
    else:
        st.warning("No valid LLM calls found for this trace.")


with col2:
    st.markdown("Error Distribution")

    if {"status_code", "status_message"}.issubset(processed_df.columns):
        error_df = processed_df.loc[processed_df["status_code"] == "ERROR", ["status_message"]].copy()

        if not error_df.empty:
            def extract_error(msg: str) -> str:
                """Extract word after first '.' and before first ':'."""
                if not isinstance(msg, str):
                    return None
                if "." in msg and ":" in msg:
                    try:
                        part = msg.split(".", 1)[1]   # after first '.'
                        return part.split(":", 1)[0].strip()
                    except Exception:
                        return None
                return None

            # Apply transformation
            error_df["error_message"] = error_df["status_message"].apply(extract_error)

            # Drop rows where extraction failed
            error_df = error_df.dropna(subset=["error_message"])

            if not error_df.empty:
                error_summary = (
                    error_df.groupby("error_message")
                            .size()
                            .reset_index(name="count")
                            .sort_values("count", ascending=False)
                )

                fig2 = px.bar(
                    error_summary,
                    x="error_message",
                    y="count",
                    text="count",
                    title="Error Frequency",
                    color="error_message",
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                )
                fig2.update_traces(textposition="outside", cliponaxis=False)
                fig2.update_layout(
                    xaxis_title="Error Type",
                    yaxis_title="Frequency",
                    showlegend=False,
                    xaxis_tickangle=-30,
                    xaxis={'categoryorder': 'total descending'}
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No parsable error messages found.")
        else:
            st.info("No errors found in this dataset.")
    else:
        st.info("No status_code / status_message columns found.")