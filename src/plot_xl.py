from pandas import ExcelFile
import plotly.graph_objs as go

desc = 'ayo'
yaxis_short = 'poa'
directory = '../output/%s' % desc
ex_file = '/%s.xlsx' % yaxis_short
yaxis_title = 'Cumulative Pseudo Regret' if yaxis_short == 'cpr' else 'Probability of Optimal Action'
figure = []
results = ExcelFile(directory + ex_file).parse(sheet_name=None, index_col=0)
for i, ind_var in enumerate(sorted(results)):
  df = results[ind_var]
  x = list(range(len(df.columns)))
  line_name = ind_var
  line_hue = str(int(360 * (i / len(results))))
  y = df.mean(axis=0, numeric_only=True)
  sem = df.sem(axis=0, numeric_only=True)
  y_upper = y + sem
  y_lower = y - sem
  line_color = "hsla(" + line_hue + ",100%,40%,1)"
  error_band_color = "hsla(" + line_hue + ",100%,40%,0.125)"
  figure.extend([
      go.Scatter(
          name=line_name,
          x=x,
          y=y,
          line=dict(color=line_color, width=3),
          mode='lines',
      ),
      go.Scatter(
          name=line_name+"-upper",
          x=x,
          y=y_upper,
          mode='lines',
          marker=dict(color=error_band_color),
          line=dict(width=0),
          showlegend=False,
      ),
      go.Scatter(
          name=line_name+"-lower",
          x=x,
          y=y_lower,
          marker=dict(color=error_band_color),
          line=dict(width=0),
          mode='lines',
          fillcolor=error_band_color,
          fill='tonexty',
          showlegend=False,
      )
  ])
plotly_fig = go.Figure(figure)
plotly_fig.update_layout(
    font=dict(size=18),
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    yaxis_title=yaxis_title,
    xaxis_title="Trial",
    # title=plot_title,
)
plotly_fig.show()
plotly_fig.write_html(directory + '/%s.html' % yaxis_short)