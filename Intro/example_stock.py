import altair as alt
import ipywidgets as widgets
from vega_datasets import data

def stock():
    source = data.stocks()

    stock_picker = widgets.SelectMultiple(
        options=source.symbol.unique(),
        value=list(source.symbol.unique()),
        description='Symbols')

    # The value of symbols will come from the stock_picker.
    @widgets.interact(symbols=stock_picker)
    def render(symbols):
    selected = source[source.symbol.isin(list(symbols))]

    return alt.Chart(selected).mark_line().encode(
        x='date',
        y='price',
        color='symbol',
        strokeDash='symbol',
    )