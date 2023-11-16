# Copyright (c) 2023 Forschungszentrum Juelich GmbH.
# This file is part of LLview. 
#
# This is an open source software distributed under the GPLv3 license. More information see the LICENSE file at the top level.
#
# Contributions must follow the Contributor License Agreement. More information see the CONTRIBUTING.md file at the top level.
#
# Contributors:
#    Filipe Guimarães (Forschungszentrum Juelich GmbH) 

import plotly.graph_objs as go
import numpy as np
import pandas as pd

def CreateOverviewFig(config,data,time_range,x1,y1,x2,y2):
  # General layout options
  layout = dict(legend=dict(
                      yanchor="top",
                      y=0.95,
                      xanchor="right",
                      x=0.78,
                      orientation="v",
                      groupclick='toggleitem',
                      bgcolor='rgba(255,255,255,0.8)',
                      bordercolor='lightgray',
                      borderwidth=1,
                      ),
                    margin=dict(l=50, r=50, t=50, b=50),
                    height=300,
                    font_family="'Liberation Sans','Arial',sans-serif",
                    title_font_family="'Liberation Sans','Arial',sans-serif",
                    title_font_color="black",
                    title = f"<b>Job-Usage Overview</b>",
                    title_x=0.5,
                    plot_bgcolor='whitesmoke',
                    modebar=dict(orientation='h',
                    ),
                    # legend_title_font_color="green"
                  )
  # Creating figure  
  fig = go.Figure(layout=layout).set_subplots(rows=1, cols=3, 
                                              horizontal_spacing=0.1, 
                                              # vertical_spacing=0.1, 
                                              shared_xaxes=False, 
                                              shared_yaxes=False,
                                              column_widths=[0.03,0.94,0.03],
                                              specs=[[{"secondary_y": False}, {"secondary_y": True,"r":-0.06},{"secondary_y": False,"l":0.06,"r":-0.06}]],
                                              # row_heights=[0.3, 0.7]
                                              )
  if x1:
    int_usage = int(data['cpu']['usage'])
    img = np.arange(int_usage).reshape((int_usage, 1))
    fig.add_trace(go.Heatmap(z=img, 
                            showscale=False, 
                            connectgaps=False, 
                            zmin=0,
                            zmax=100.0,
                            hoverinfo='skip',
                            colorscale=["#d62728","gold","#2ca02c"],
                            # colorbar=dict(outlinewidth=0.5,
                            #               outlinecolor='black',
                            #               len=0.5,
                            #               thickness=0.02,
                            #               thicknessmode='fraction',
                            #               ticks='inside',
                            #               yanchor='bottom',
                            #               x=0.2) , 
                            ), 1, 1)
    fig['layout']['xaxis'].update(dict(zeroline=False, showgrid=False, range=[-0.5,0.5], tickmode='array',tickvals=[], fixedrange=True, showticklabels=False))
    fig['layout']['yaxis'].update(dict(zeroline=False, showgrid=False, range=[0.0,100.0], tickmode='array',tickvals=[], fixedrange=True ,showticklabels=False))

    color = tuple(255*x for x in config['appearance']['colors_cmap'][7])

    hovertext = []
    for time,value in zip(x1,y1):
      delta_comp = (time - time_range[0]).components
      delta = f"{(str(delta_comp.hours)+'h:') if delta_comp.hours > 0 else ''}{delta_comp.minutes:02d}m:{delta_comp.seconds:02d}s"
      hovertext.append(f"Time: {time} ({delta})<br />{data['cpu']['usage_or_load_text']}: {value:.2f}")

    fig.add_trace(go.Scatter( x=x1,
                              y=y1, 
                              name = data['cpu']['usage_or_load_text'],
                              legendgroup = 'cpu',
                              line = {"shape": 'hvh', "color": f"rgb{color}"},
                              mode="lines+markers",
                              hoverinfo='text',
                              text=hovertext,
                              # mode = 'markers',
                              marker=dict(
                                    size=5,
                                    color=f"rgb{color}",
                                ),
                                ), 1, 2)
    fig['layout']['yaxis2'].update(dict( title=data['cpu']['overview_label'],
                                          color = f"rgb{color}",
                                          tickcolor = f"rgb{color}",
                                          range=data['cpu']['overview_range'],
                                          ))
    fig['layout']['xaxis2'].update(dict( tickformat=('%d/%m/%y\n%H:%M:%S'),
                                          range=time_range,
                                          ))
    fig.add_annotation( text = "<b>Average<br>CPU Usage</b>",
                        x = 0.0,
                        y = 100,
                        xanchor='center',
                        yanchor='bottom',
                        yshift=10,
                        xref = "x",
                        yref = "y",
                        align="center",
                        valign="middle",
                        showarrow = False,
                        font = {
                            "family": "'Liberation Sans','Arial',sans-serif",
                            "size": 14,
                        })
    fig.add_annotation( text = f"<b>{data['cpu']['usage']:.1f}%</b>",
                        x = 0.0,
                        y = 75,
                        xref = "x",
                        yref = "y",
                        xanchor='center',
                        yanchor='middle',
                        align="center",
                        valign="middle",
                        showarrow = False,
                        textangle=-90.0,
                        font = {
                            "family": "'Liberation Sans','Arial',sans-serif",
                            "size": 12,
                        })
  if x2:
    int_usage = int(data['gpu']['gpu_usage_avg'])
    img = np.arange(int_usage).reshape((int_usage, 1))
    fig.add_trace(go.Heatmap(z=img, 
                            showscale=False, 
                            connectgaps=False, 
                            zmin=0,
                            zmax=100.0,
                            hoverinfo='skip',
                            colorscale=["#d62728","gold","#2ca02c"],
                            # colorbar=dict(outlinewidth=0.5,
                            #               outlinecolor='black',
                            #               len=0.5,
                            #               thickness=0.02,
                            #               thicknessmode='fraction',
                            #               ticks='inside',
                            #               yanchor='bottom',
                            #               x=0.8) , 
                            ), 1, 3)
    fig['layout']['xaxis3'].update(dict(zeroline=False, showgrid=False, range=[-0.5,0.5], tickmode='array',tickvals=[], fixedrange=True, showticklabels=False))
    fig['layout']['yaxis4'].update(dict(zeroline=False, showgrid=False, range=[0.0,100.0], tickmode='array',tickvals=[], fixedrange=True ,showticklabels=False))

    color = tuple(255*x for x in config['appearance']['colors_cmap'][0])

    hovertext = []
    for time,value in zip(x2,y2):
      delta_comp = (time - time_range[0]).components
      delta = f"{(str(delta_comp.hours)+'h:') if delta_comp.hours > 0 else ''}{delta_comp.minutes:02d}m:{delta_comp.seconds:02d}s"
      hovertext.append(f"Time: {time} ({delta})<br />Avg. GPU Usage: {value:.2f}")

    fig.add_trace(go.Scatter( x=x2,
                              y=y2, 
                              name = 'Average GPU Usage',
                              legendgroup = 'gpu',
                              line = {"shape": 'hvh', "color":f"rgb{color}"},
                              mode="lines+markers",
                              hoverinfo='text',
                              text=hovertext,
                              # mode = 'markers',
                              marker=dict(
                                    size=5,
                                    color=f"rgb{color}",
                                ),
                                ),
                                1, 2, secondary_y=True)
    fig['layout']['yaxis3'].update(dict( title="GPU Usage (%)",
                                          color = f"rgb{color}",
                                          tickcolor = f"rgb{color}",
                                          range=[0,110],
                                          ))
    fig.add_annotation( text = "<b>Average<br>GPU Usage</b>",
                        x = 0.0,
                        y = 100,
                        xanchor='center',
                        yanchor='bottom',
                        yshift=10,
                        xref = "x3",
                        yref = "y4",
                        align="center",
                        valign="middle",
                        showarrow = False,
                        font = {
                            "family": "'Liberation Sans','Arial',sans-serif",
                            "size": 14,
                        })
    fig.add_annotation( text = f"<b>{data['gpu']['gpu_usage_avg']:.1f}%</b>",
                        x = 0.0,
                        y = 75,
                        xref = "x3",
                        yref = "y4",
                        xanchor='center',
                        yanchor='middle',
                        align="center",
                        valign="middle",
                        showarrow = False,
                        textangle=-90.0,
                        font = {
                            "family": "'Liberation Sans','Arial',sans-serif",
                            "size": 12,
                        })
  for i in range(1,5): 
    frame = dict( mirror=True,
                  ticks='inside',
                  linecolor='black',
                  showgrid=False,
                  # showline=True,
                  )
    fig['layout'][f'yaxis{i}'].update(frame)
    if i == 4: continue
    fig['layout'][f'xaxis{i}'].update(frame)

  return fig


def CreateTimeline(config,timeline_df,time_range):
  # General layout options
  layout = dict(legend=dict(
                      yanchor="top",
                      y=0.95,
                      xanchor="right",
                      x=0.78,
                      orientation="v",
                      groupclick='toggleitem',
                      bgcolor='rgba(255,255,255,0.8)',
                      bordercolor='lightgray',
                      borderwidth=1,
                      ),
                    margin=dict(l=50, r=50, t=50, b=50),
                    height=min(1000*config['timeline']['barsize']*config['timeline']['nsteps']+120,config['appearance']['max_timeline_height_html']),
                    font_family="'Liberation Sans','Arial',sans-serif",
                    title_font_family="'Liberation Sans','Arial',sans-serif",
                    title_font_color="black",
                    title = f"<b>Timeline</b>",
                    title_x=0.5,
                    plot_bgcolor='whitesmoke',
                    modebar=dict(orientation='h',
                    ),
                    # legend_title_font_color="green"
                  )
  # Creating figure  
  fig = go.Figure(layout=layout).set_subplots(rows=1, cols=1)
  hovertext = []
  for index, row in timeline_df.iterrows():
    delta_comp = row['duration'].components
    delta = f"{(str(delta_comp.hours)+'h:') if delta_comp.hours > 0 else ''}{delta_comp.minutes:02d}m:{delta_comp.seconds:02d}s"
    if 'nnodes' in row:
      hovertext.append(f"<b>{'Step '+str(row['step'])+': '+str(row['st'])+'</b><br />Name: '+str(row['name']) if str(row['step'])!='job' else 'Job '+str(row['name'])+': '+str(row['st'])+'</b>'}<br />#Nodes: {row['nnodes']} ({row['nodelist'][:27] + '...' if len(row['nodelist'])>30 else row['nodelist']})<br />{'#Tasks: '+str(row['ntasks'])+', ' if row['ntasks'] > 0 else ''}#CPUS: {row['ncpus']}<br />Start time: {row['start_time']}<br />End time: {row['end_time']}<br />Duration: {delta}<br />Return Code: {row['rc']}<br />Signal: {row['sig']}")
    else:
          hovertext.append(f"<b>Step {row['step']}: {row['st']}</b><br />Executable: {row['name']}<br />Start time: {row['start_time']}<br />End time: {row['end_time']}<br />Duration: {delta}<br />Return Code: {row['rc']}<br />Signal: {row['sig']}")
  fig.add_trace(go.Bar( x=timeline_df['duration']/1000000,
                        base=timeline_df['start_time'],
                        y=timeline_df['step'],
                        name = 'Timeline',
                        orientation='h',
                        hoverinfo='text',
                        hovertext=hovertext,
                        marker=dict(
                              color=timeline_df['colorhtml'],
                              line=dict(color=timeline_df['edgecolorhtml']),
                          ),
                          ))
  fig['layout']['yaxis'].update(dict( title=f'Step',
                                      range=[config['timeline']['nsteps']-0.5,-0.5],
                                      ))
  fig['layout']['xaxis'].update(dict( tickformat=('%d/%m/%y\n%H:%M:%S'),
                                      type="date",
                                      range=time_range,
                                      ))

  # fig.add_annotation( text = "<b>Average<br>CPU Usage</b>",
  #                     x = 0.0,
  #                     y = 100,
  #                     xanchor='center',
  #                     yanchor='bottom',
  #                     yshift=10,
  #                     xref = "x",
  #                     yref = "y",
  #                     align="center",
  #                     valign="middle",
  #                     showarrow = False,
  #                     font = {
  #                         "family": "'Liberation Sans','Arial',sans-serif",
  #                         "size": 14,
  #                     })
  frame = dict( mirror=True,
                ticks='inside',
                linecolor='black',
                showgrid=False,
                # showline=True,
                )
  fig['layout'][f'yaxis'].update(frame)
  fig['layout'][f'xaxis'].update(frame)

  return fig



def CreatePlotlyFig(config,
                    graph,
                    report,
                    time_range,
                    nodes,
                    cticks,
                    x,
                    y,
                    z,
                    df_avg_time,
                    errmin_time,
                    errmax_time,
                    df_avg_node,
                    errmin_node,
                    errmax_node,
                    min_value,
                    avg_value,
                    max_value):
  """
  This function generates a 2x2 figure using plotly
  (to be exported to html later)
  """
  # General layout options
  layout = dict(legend=dict(
                      yanchor="middle",
                      y=0.68,
                      xanchor="center",
                      x=0.79,
                      orientation="h",
                      groupclick='toggleitem',
                      bgcolor='rgba(0,0,0,0)',
                      ),
                    # xaxis=dict(showgrid=False),
                    # yaxis=dict(showgrid=False),
                    margin=dict(l=20, r=20, t=50, b=50),
                    # width=1000,
                    height=500,
                    font_family="'Liberation Sans','Arial',sans-serif",
                    title_font_family="'Liberation Sans','Arial',sans-serif",
                    title_font_color="black",
                    title = "<b>{}: {}</b>".format(report['type'].replace('\\',''),report['graphs'][graph]),
                    title_x=0.5,
                    plot_bgcolor='whitesmoke',
                    modebar=dict(orientation='h',
                    ),
                    # legend_title_font_color="green"
                  )
  # Creating figure  
  fig = go.Figure(layout=layout).set_subplots(rows=2, cols=2, 
                                              horizontal_spacing=0.03, 
                                              vertical_spacing=0.1, 
                                              shared_xaxes=False, 
                                              shared_yaxes=False,
                                              column_widths=[0.8, 0.2],
                                              row_heights=[0.3, 0.7]
                                              )

  for i in range(1,5): 
    frame = dict( mirror=True,
                  ticks='inside',
                  linecolor='black',
                  showgrid=False,
                  # showline=True,
                  )
    fig['layout'][f'xaxis{i}'].update(frame)
    fig['layout'][f'yaxis{i}'].update(frame)
  if report['headers'][graph] == 'gpu_clkr':
    # Filtering NaN values to avoid weird min/max plots
    x_time = x[errmax_time.notna()]
    y_nodes = nodes[errmax_node.notna()]
    errmax_time = errmax_time[errmax_time.notna()]
    errmin_time = errmin_time[errmin_time.notna()]
    errmax_node = errmax_node[errmax_node.notna()]
    errmin_node = errmin_node[errmin_node.notna()]
  else:
    x_time = x
    y_nodes = nodes

  annotations = []
  # Add dropdown menus
  button_y = 0.683
  # Colorscale definitions due to possible bug on the potly library
  # See: https://github.com/plotly/plotly.py/issues/3860
  # Hawaii and lajolla from https://www.fabiocrameri.ch/colourmaps/
  colorscales = {}
  colorscales['hawaii'] = [[0.0,"#8C0273"],[0.1111111111111111,"#922A59"],[0.2222222222222222,"#964742"],[0.3333333333333333,"#996330"],[0.4444444444444444,"#9D831E"],[0.5555555555555556,"#97A92A"],[0.6666666666666666,"#80C55F"],[0.7777777777777778,"#66D89C"],[0.8888888888888888,"#6CEBDB"],[1.0,"#B3F2FD"]]
  colorscales['lajolla'] = [[0.0,"#1A1A01"],[0.1111111111111111,"#422818"],[0.2222222222222222,"#73382F"],[0.3333333333333333,"#A54742"],[0.4444444444444444,"#D2624D"],[0.5555555555555556,"#E48751"],[0.6666666666666666,"#ECA855"],[0.7777777777777778,"#F4CC68"],[0.8888888888888888,"#FFFFCC"],[1.0,"#fcffa4"]]
  colorscales['Reds'] = [[0.0,"rgb(103,0,13)"],[0.125,"rgb(165,15,21)"],[0.25,"rgb(203,24,29)"],[0.375,"rgb(239,59,44)"],[0.5,"rgb(251,106,74)"],[0.625,"rgb(252,146,114)"],[0.75,"rgb(252,187,161)"],[0.875,"rgb(254,224,210)"],[1.0,"rgb(255,245,240)"]]
  colorscales['RdBu'] = [[0.0,"rgb(103,0,31)"],[0.1,"rgb(178,24,43)"],[0.2,"rgb(214,96,77)"],[0.3,"rgb(244,165,130)"],[0.4,"rgb(253,219,199)"],[0.5,"rgb(247,247,247)"],[0.6,"rgb(209,229,240)"],[0.7,"rgb(146,197,222)"],[0.8,"rgb(67,147,195)"],[0.9,"rgb(33,102,172)"],[1.0,"rgb(5,48,97)"]]
  dropdown_buttons = [
          dict(
              buttons=list([
                  dict(
                      args=[{ 'marker.colorscale':[colorscales['hawaii'],"",colorscales['hawaii']],
                              'colorscale':["",colorscales['hawaii'],""]},[1,2,4]],
                      label="Hawaii",
                      method="restyle"
                  ),
                  dict(
                      args=[{ 'marker.colorscale':["Cividis","","Cividis"],
                              'colorscale':["","Cividis",""]},[1,2,4]],
                      label="Cividis",
                      method="restyle"
                  ),
                  dict(
                      args=[{ 'marker.colorscale':[colorscales['lajolla'],"",colorscales['lajolla']],
                              'colorscale':["",colorscales['lajolla'],""]},[1,2,4]],
                      label="Lajolla",
                      method="restyle"
                  ),
                  dict(
                      args=[{ 'marker.colorscale':[colorscales['RdBu'],"",colorscales['RdBu']],
                              'colorscale':["",colorscales['RdBu'],""]},[1,2,4]],
                      label="RdBu",
                      method="restyle"
                  ),
                  dict(
                      args=[{ 'marker.colorscale':["Blues","","Blues"],
                              'colorscale':["","Blues",""]},[1,2,4]],
                      label="Blues",
                      method="restyle"
                  ),
                  dict(
                      args=[{ 'marker.colorscale':["Greens","","Greens"],
                              'colorscale':["","Greens",""]},[1,2,4]],
                      label="Greens",
                      method="restyle"
                  ),
                  dict(
                      args=[{ 'marker.colorscale':[colorscales['Reds'],"",colorscales['Reds']],
                              'colorscale':["",colorscales['Reds'],""]},[1,2,4]],
                      label="Reds",
                      method="restyle"
                  ),
              ]),
              direction="down",
              pad={"r": 0, "t": 0},
              showactive=True,
              x=0.05,
              xanchor="left",
              y=button_y,
              yanchor="middle"
          ),
          dict(
              buttons=list([
                  dict(
                      args=[{ 'marker.reversescale':[False,"",False],
                              'reversescale':["",False,""]},[1,2,4]],
                      label="False",
                      method="restyle"
                  ),
                  dict(
                      args=[{ 'marker.reversescale':[True,"",True],
                              'reversescale':["",True,""]},[1,2,4]],
                      label="True",
                      method="restyle"
                  )
              ]),
              direction="down",
              pad={"r": 0, "t": 0},
              showactive=True,
              x=0.25,
              xanchor="left",
              y=button_y,
              yanchor="middle"
          ),
      ]
  # if ([min_value,max_value] != report['lim'][graph]):
  if (not 0.95*report['lim'][graph][0] < min_value < 1.05*report['lim'][graph][0]) and (not 0.95*report['lim'][graph][1] < max_value < 1.05*report['lim'][graph][1]):
    dropdown_buttons.append(dict(
              buttons=list([
                  dict(
                      args=[{ 
                              'marker.cauto': [False,"",False],
                              'zauto': ["",False,""]
                            }, {
                              'yaxis.range': report['lim'][graph],
                              'xaxis4.range': report['lim'][graph],
                            },
                            [1,2,4]],
                      label="System",
                      method="update"
                  ),
                  dict(
                      args=[{ 
                              'marker.cauto': [True,"",True],
                              'zauto': ["",True,""]
                            }, {
                              'yaxis.range': [min_value,max_value],
                              'xaxis4.range': [min_value,max_value],
                            },
                            [1,2,4]],
                      label="Job",
                      method="update"
                  ),
              ]),
              direction="down",
              pad={"r": 0, "t": 0},
              showactive=True,
              x=0.45,
              xanchor="left",
              y=button_y,
              yanchor="middle"
          ))
    annotations.append(dict( text="Limits", 
                        x=0.45, 
                        xref="paper", 
                        y=button_y,
                        yref="paper", 
                        xanchor="right", 
                        yanchor="middle",
                        align="right",
                        valign="middle",
                        showarrow=False,
                        )) 

  fig.update_layout(updatemenus=dropdown_buttons)

  #(1,1) - NODE-AVERAGE PER TIME
  fig.add_trace(go.Scatter( x=pd.concat([x_time,x_time.reindex(index=x_time.index[::-1])]),
                            y=pd.concat([ errmax_time,errmin_time.reindex(index=errmin_time.index[::-1]) ]), 
                            name = 'min/max',
                            legendgroup = 'time',
                            line = {"shape": 'hvh', "color": config['appearance']['minmax_color']},
                            fillcolor = config['appearance']['minmax_color'],
                            mode="none",
                            fill='toself',
                              ), 1, 1)
  hovertext = []
  delta = []
  if report['x'][graph] == 'ts':
    for time,value in zip(x,df_avg_time):
      delta_comp = (time - x[0]).components
      delta.append(f"{(str(delta_comp.hours)+'h:') if delta_comp.hours > 0 else ''}{delta_comp.minutes:02d}m:{delta_comp.seconds:02d}s")
      if np.isnan(value):
        hovertext.append("")
      else:
        if report['log'][graph]:
          hovertext.append(f"Time: {time} ({delta[-1]})<br />Avg. per time: {int(value)}")
        else:
          hovertext.append(f"Time: {time} ({delta[-1]})<br />Avg. per time: {value:.2f}")
  else:
    for xval,value in zip(x,df_avg_time):
      if np.isnan(value):
        hovertext.append("")
      else:
        if report['log'][graph]:
          hovertext.append(f"{report['xlabel'][graph]}: {xval}<br />Avg. per {report['xlabel'][graph]}: {int(value)}")
        else:
          hovertext.append(f"{report['xlabel'][graph]}: {xval}<br />Avg. per {report['xlabel'][graph]}: {value:.2f}")

  fig.add_trace(go.Scatter( x=x,
                            y=df_avg_time, 
                            name = f"Avg. per {report['xlabel'][graph] if report['xlabel'][graph] else 'time'}      ",
                            legendgroup = 'time',
                            line = {"shape": 'hvh', "color":"black"},
                            mode="lines+markers",
                            # mode = 'markers',
                            hoverinfo='text',
                            text=hovertext,
                            marker=dict(
                                  size=5,
                                  cmin=(np.log2(report['lim'][graph][0]) if report['log'][graph] else report['lim'][graph][0]),
                                  cmax=(np.log2(report['lim'][graph][1]) if report['log'][graph] else report['lim'][graph][1]),
                                  color=(np.log2(df_avg_time) if report['log'][graph] else df_avg_time),
                                  colorscale=colorscales[report['cmap'][graph].replace('cmc.','').replace('_r','')] if report['cmap'][graph].replace('cmc.','').replace('_r','') in colorscales else report['cmap'][graph],
                              ),
                              ), 1, 1)
  fig['layout'][f'xaxis1'].update(dict(showticklabels=False,range=time_range if report['x'][graph] == 'ts' else [x.min(),x.max()]))
  fig['layout'][f'yaxis1'].update(dict(title=f"{report['unit'][graph]}", range=report['lim'][graph]))
  if report['log'][graph]:
    fig['layout'][f'yaxis1'].update(dict(type="log", dtick=np.log10(report['log'][graph]), range=[np.log10(i) for i in report['lim'][graph]], tickvals=cticks))

  #(1,2) - TEXT INFO
  fig['layout'][f'xaxis2'].update(dict(range=[0.0,1.0], ticks="", fixedrange=True, showticklabels=False))
  fig['layout'][f'yaxis2'].update(dict(range=[0.0,1.0], ticks="", fixedrange=True, showticklabels=False))
  if report['note'][graph] == "":
    annotations.extend([
                  dict(
                  text = f"<b>avg: {avg_value:.1f} {report['unit'][graph]}</b>",
                  x = 0.5,
                  y = 0.5,
                  xref = "x2",
                  yref = "y2",
                  align="center",
                  valign="middle",
                  showarrow = False,
                  font = {
                      "family": "'Liberation Sans','Arial',sans-serif",
                      "size": 16,
                  }),
                  dict(
                  text = f"min: {min_value:.1f} {report['unit'][graph]}",
                  x = 0.5,
                  y = 0.15,
                  xref = "x2",
                  yref = "y2",
                  align="center",
                  valign="middle",
                  showarrow = False,
                  font = {
                      "family": "'Liberation Sans','Arial',sans-serif",
                      "size": 14,
                  }),
                  dict(
                  text = f"max: {max_value:.1f} {report['unit'][graph]}",
                  x = 0.5,
                  y = 0.85,
                  xref = "x2",
                  yref = "y2",
                  align="center",
                  valign="middle",
                  showarrow = False,
                  font = {
                      "family": "'Liberation Sans','Arial',sans-serif",
                      "size": 14,
                  })
                  ])
  else:
    annotations.extend([
                  dict(
                  text = report['note'][graph].replace("\n","<br>"),
                  x = 0.5,
                  y = 0.5,
                  xref = "x2",
                  yref = "y2",
                  align="center",
                  valign="middle",
                  showarrow = False,
                  font = {
                      "family": "'Liberation Sans','Arial',sans-serif",
                      "size": 12,
                  })
                  ])
  annotations.extend([
                  # Dropdown labels
                  dict( text="Colorscale", 
                        x=0.05, 
                        xref="paper", 
                        y=button_y,
                        yref="paper", 
                        xanchor="right", 
                        yanchor="middle",
                        align="right",
                        valign="middle",
                        showarrow=False,
                        ),
                  dict( text="Reverse", 
                        x=0.25, 
                        xref="paper", 
                        y=button_y,
                        yref="paper", 
                        xanchor="right", 
                        yanchor="middle",
                        align="right",
                        valign="middle",
                        showarrow=False,
                        )
                      ])
  fig['layout']['annotations'] = annotations
  #(2,1) - COLORPLOT
  fig['layout'][f'xaxis3'].update(dict( matches='x', tickformat=('%d/%m/%y\n%H:%M:%S')) if report['x'][graph] == 'ts' else dict( matches='x' ))
  fig['layout'][f'yaxis3'].update(dict( matches='y4', 
                                        title=report['ylabel'][graph],
                                        nticks = 16,
                                        tickmode = 'auto',
                                        # tickfont={"family":"'Liberation Sans','Arial',sans-serif"},
                                        # tickvals = y,
                                        # ticktext = nodes,
                                        ))
  # Adding x label, when present
  if report['xlabel'][graph]:
    fig['layout'][f'xaxis3'].update(dict( title=report['xlabel'][graph] ))

  colorbar = dict(title=f"{report['unit'][graph]}",
                                          outlinewidth=0.5,
                                          outlinecolor='black',
                                          len=0.7,
                                          thickness=0.02,
                                          thicknessmode='fraction',
                                          ticks='inside',
                                          yanchor='bottom',
                                          y=-0.025)
  if report['log'][graph]:
    colorbar = dict(**colorbar,tickvals=np.log2(cticks),ticktext=cticks)

  hovertext = []
  if report['x'][graph] == 'ts':
    for yi, yy in enumerate(nodes):
      hovertext.append([])
      for xi, xx in enumerate(x):
        if np.isnan(z[yi][xi]):
          hovertext[-1].append("")
        else:
          if report['log'][graph]:
            hovertext[-1].append(f"Time: {xx} ({delta[xi]})<br />{report['ylabel'][graph]}: {yy}<br />Value: {int(z[yi][xi])}")
          else:
            hovertext[-1].append(f"Time: {xx} ({delta[xi]})<br />{report['ylabel'][graph]}: {yy}<br />Value: {z[yi][xi]:.2f}")
  else:
    for yi, yy in enumerate(nodes):
      hovertext.append([])
      for xi, xx in enumerate(x):
        if np.isnan(z[yi][xi]):
          hovertext[-1].append("")
        else:
          if report['log'][graph]:
            hovertext[-1].append(f"{report['xlabel'][graph]}: {xx}<br />{report['ylabel'][graph]}: {yy}<br />Value: {int(z[yi][xi])}")
          else:
            hovertext[-1].append(f"{report['xlabel'][graph]}: {xx}<br />{report['ylabel'][graph]}: {yy}<br />Value: {z[yi][xi]:.2f}")

  fig.add_trace(go.Heatmap( x=x,
                            y=nodes,
                            z=(np.log2(z) if report['log'][graph] else z),
                            showscale=True, 
                            connectgaps=False, 
                            zmin=(np.log2(report['lim'][graph][0]) if report['log'][graph] else report['lim'][graph][0]),
                            zmax=(np.log2(report['lim'][graph][1]) if report['log'][graph] else report['lim'][graph][1]),
                            name=f"{report['graphs'][graph]}",
                            hoverinfo='text',
                            text=hovertext,
                            # zsmooth='best',
                            colorbar=colorbar, 
                            colorscale=colorscales[report['cmap'][graph].replace('cmc.','').replace('_r','')] if report['cmap'][graph].replace('cmc.','').replace('_r','') in colorscales else report['cmap'][graph]), 2, 1)
                            # colorscale=["#d62728","gold","#2ca02c"]), 2, 1)

  #(2,2) - TIME-AVERAGE PER NODE
  fig.add_trace(go.Scatter( x=pd.concat([errmax_node,errmin_node.reindex(index=errmin_node.index[::-1])]), 
                            y=np.hstack([y_nodes,y_nodes[::-1]]),
                            name = 'min/max',
                            legendgroup = 'node',
                            line = {"shape": 'vhv', "color": config['appearance']['minmax_color']},
                            fillcolor = config['appearance']['minmax_color'],
                            mode="none",
                            fill='toself',
                              ), 2, 2)
  hovertext = []
  for node,value in zip(nodes,df_avg_node):
    if np.isnan(value):
      hovertext.append("")
    else:
      if report['log'][graph]:
        hovertext.append(f"{report['ylabel'][graph]}: {node}<br />Avg. per {report['ylabel'][graph]}: {int(value)}")
      else:
        hovertext.append(f"{report['ylabel'][graph]}: {node}<br />Avg. per {report['ylabel'][graph]}: {value:.2f}")

  fig.add_trace(go.Scatter( x=df_avg_node,
                            y=nodes,
                            name = f"Avg. per {report['ylabel'][graph]}",
                            legendgroup = 'node',
                            line = {"shape": 'vhv', "color":"black"},
                            mode="lines+markers",
                            hoverinfo='text',
                            text=hovertext,
                            marker=dict(
                                size=5,
                                cmin=(np.log2(report['lim'][graph][0]) if report['log'][graph] else report['lim'][graph][0]),
                                cmax=(np.log2(report['lim'][graph][1]) if report['log'][graph] else report['lim'][graph][1]),
                                color=(np.log2(df_avg_node) if report['log'][graph] else df_avg_node),
                                colorscale=colorscales[report['cmap'][graph].replace('cmc.','').replace('_r','')] if report['cmap'][graph].replace('cmc.','').replace('_r','') in colorscales else report['cmap'][graph],
                              )), 2, 2)
  fig['layout'][f'xaxis4'].update(dict(title=f"{report['unit'][graph]}", range=report['lim'][graph]))
  fig['layout'][f'yaxis4'].update(dict(showticklabels=False,range=[y.min()-0.5,y.max()+0.5]))
  if report['log'][graph]:
    fig['layout'][f'xaxis4'].update(dict(type="log", dtick=np.log10(report['log'][graph]), range=[np.log10(i) for i in report['lim'][graph]], tickvals=cticks))


  return fig


def CreateUnifiedPlotlyFig(data,config,report,time_range):
  """
  This function generates a unified figure (with Dropdown selections) containing user-defined curves
  using plotly
  (to be exported to html later)
  """
  # General layout options
  layout = dict(legend=dict(
                      yanchor="middle",
                      y=0.68,
                      xanchor="center",
                      x=0.79,
                      orientation="h",
                      groupclick='toggleitem',
                      bgcolor='rgba(0,0,0,0)',
                      ),
                    # xaxis=dict(showgrid=False),
                    # yaxis=dict(showgrid=False),
                    margin=dict(l=20, r=100, t=50, b=50),
                    # width=1000,
                    height=400,
                    font_family="'Liberation Sans','Arial',sans-serif",
                    title_font_family="'Liberation Sans','Arial',sans-serif",
                    title_font_color="black",
                    title = "<b>{}: {}</b>".format(report['type'].replace('\\',''),", ".join(report['graphs'])),
                    title_x=0.5,
                    plot_bgcolor='whitesmoke',
                    modebar=dict(orientation='h',
                    ),
                    # showlegend=True
                    # legend_title_font_color="green"
                  )
  # Creating figure  
  fig = go.Figure(layout=layout)

  frame = dict( mirror=True,
                ticks='inside',
                linecolor='black',
                showgrid=False,
                # showline=True,
                )
  fig['layout'][f'xaxis'].update(frame)
  fig['layout'][f'yaxis'].update(frame)

  # Add dropdown menus
  # Adding different options to Dropdown
  buttons_y = []
  if 'ts' in report['x']:
    report['data']['ts'] += config['appearance']['timezonegap']
    report['data']['datetime'] = pd.to_datetime(report['data']['ts'],unit='s')

    buttons_x = [dict(
                      args=[
                            { 
                              'x': [report['data']['datetime']],
                              },
                            {
                              'xaxis.title': '',
                              'xaxis.range': time_range,
                            },
                              ],
                      label='datetime',
                      method="update"
                  )]

  for i,graph in enumerate(report['graphs']):
    graph_header = report['headers'][i]

    # Removing extra line
    report['data'].drop(report['data'][report['data'][report['y'][i]] == 'zzz'].index, inplace=True)
    # Transforming 'ts' to datetime
    x_header = report['x'][i]
    if x_header == 'ts':
      x = report['data']['datetime']
    else:
      x = report['x'][x_header]


    # Calculating range of each graph
    min_value = report['data'][graph_header].min()
    max_value = report['data'][graph_header].max()
    report['lim'][i] = [min(min_value,report['lim'][i][0]) if report['lim'][i][0] else min_value,max(max_value,report['lim'][i][1]) if report['lim'][i][1] else max_value]

    buttons_x.append(dict(
                      args=[
                            { 
                              'x': [report['data'][graph_header]],
                              },
                            {
                              'xaxis.title': graph,
                              'xaxis.range': report['lim'][i],
                            },
                              ],
                      label=graph,
                      method="update"
                  ))
    buttons_y.append(dict(
                      args=[
                            { 
                              'y': [report['data'][graph_header]],
                              'name': graph,
                              },
                            {
                              'yaxis.title': graph,
                              'yaxis.range': report['lim'][i],
                            },
                              ],
                      label=graph,
                      method="update"
                  ))
  dropdown_buttons = [dict(
            buttons=buttons_x,
            direction="up",
            showactive=True,
            x=1.0,
            xanchor="right",
            y=-0.1,
            yanchor="top"
        ),dict(
            buttons=buttons_y,
            direction="down",
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.01,
            yanchor="bottom"
        )]
  fig.update_layout(updatemenus=dropdown_buttons)

  fig.add_trace(go.Scatter( x=x,
                            y=report['data'][report['headers'][0]], 
                            name = report['graphs'][0],
                            line = {"shape": 'hvh', "color":"black"},
                            mode="lines+markers",
                            # mode = 'markers',
                            # hoverinfo='text',
                            # text=hovertext,
                            marker=dict(
                                  size=5,
                                  # cmin=(np.log2(report['lim'][graph][0]) if report['log'][graph] else report['lim'][graph][0]),
                                  # cmax=(np.log2(report['lim'][graph][1]) if report['log'][graph] else report['lim'][graph][1]),
                                  # color=(np.log2(y) if report['log'][graph] else y),
                                  # colorscale=colorscales[report['cmap'][graph].replace('cmc.','').replace('_r','')] if report['cmap'][graph].replace('cmc.','').replace('_r','') in colorscales else report['cmap'][graph],
                              ),
                              ))
  fig['layout'][f'xaxis'].update(dict(range=time_range))
  fig['layout'][f'yaxis'].update(dict(title=f"{report['graphs'][0]}", range=report['lim'][0]))

  return {'Custom': {'x': 'custom', 'graph': fig} }
