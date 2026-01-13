import plotly.express as px
def plot_mel(mel, title):
    fig = px.imshow(
        mel,
        origin="lower",
        aspect="auto",
        title=title,
        color_continuous_scale="magma"
    )
    fig.update_layout(height=300)
    return fig

