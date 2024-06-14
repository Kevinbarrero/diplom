import pandas as pd
import plotly.graph_objects as go

import pandas_ta as ta

def generate_patterns(df):
    patterns = df.ta.cdl_pattern(name=['doji', 'inside'])
    patterns = patterns.replace(0, -5000)
    return patterns

def generate_cycles(df):
    cycles = pd.DataFrame(index=df.index)
    cycles['ebsw'] =ta.ebsw(df['close'])
    cycles['atr'] = ta.atr(df['high'], df['low'], df['close'], length=7)
    z = ta.cdl_z(df['open'], df['high'], df['low'], df['close'])
    cycles['open_z'] = z['open_Z_30_1']
    cycles['high_z'] = z['high_Z_30_1']
    cycles['low_z'] =  z['low_Z_30_1']
    cycles['close_z'] = z['close_Z_30_1']
   
    
    return cycles

def generate_stats(df):
    t = 7
    stats = pd.DataFrame(index=df.index)
    
    stats['dev'] = ta.stdev(df['close'], length=t)
    stats['ema20'] = ta.ema(df['close'], length=20)
    stats['ema50'] = ta.ema(df['close'], length=50)
    # stats['ema100'] = ta.ema(df['close'], length=100)
    # stats['ema200'] = ta.ema(df['close'], length=200)
    stats['rsi6'] = ta.rsi(df['close'], length=6)
    stats['rsi12'] = ta.rsi(df['close'], length=12)
    stats['adx'] = ta.adx(df['high'], df['low'], df['close'], length=24)['ADX_24']
    # stats['sar'] = ta.sar(df['high'], df['low'])
    bbands = ta.bbands(df['close'], length=t, std=2.0)
    # print(bbands.columns)
    stats['b_upper'] = bbands['BBU_7_2.0']
    stats['b_middle'] = bbands['BBM_7_2.0']
    stats['b_lower'] = bbands['BBL_7_2.0']
    stats['percentage_difference'] = (df['high'] - df['low']) / df['low'] * 100
    stats['percentage_change'] = df['close'].pct_change()
    stats['min_price'] = df['close'].rolling(3).min()
    stats['max_price'] = df['close'].rolling(3).max()
    stats['next_day_close'] = df['close'].shift(-1)
    
    return stats

def generate_ind(df):
    t = 7
    stats = pd.DataFrame(index=df.index)

    stats['var'] = ta.variance(df['close'], length=t)
    a = ta.aroon(df['high'], df['low'], length=t)
    # stats['aroonD'], stats['aroonU'], stats['aroonSC'] = ta.aroon(df['high'], df['low'], length=t)
    stats['aroond'] = a['AROOND_7']
    stats['aroonu'] = a['AROONU_7']
    stats['aroonc'] = a['AROONOSC_7']
    # print(aroon.columns)
    # stats['aroon'] = aroon['Aroon Up'] - aroon['Aroon Down']
    stats['bop'] = ta.bop(df['open'], df['high'], df['low'], df['close'])
    stats['cci'] = ta.cci(df['high'], df['low'], df['close'], length=t)
    stats['mom'] = ta.mom(df['close'], length=t)
    stats['roc'] = ta.roc(df['close'], length=t) 
    # stats['rocp'] = ta.rocp(df['close'], length=t)
    stats['willr'] = ta.willr(df['high'], df['low'], df['close'], length=t)
    # stats = stats.astype(float)

    return stats


def r2_metric(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

def add_percentage(value, percent):
    return value + (value * (percent / 100))

def plot_gauge(value, title='', min_value=0, max_value=100):
    # Convert min_value and max_value to numeric values
    min_value = float(min_value)
    max_value = float(max_value)

    # Create the gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_value, max_value]},
            'bar': {'color': "gray"},
            'steps': [
                {'range': [min_value, (max_value-min_value)*0.25 + min_value], 'color': "#ff6347"},
                {'range': [(max_value-min_value)*0.25 + min_value, (max_value-min_value)*0.5 + min_value], 'color': "#ffa500"},
                {'range': [(max_value-min_value)*0.5 + min_value, (max_value-min_value)*0.75 + min_value], 'color': "#ffff00"},
                {'range': [(max_value-min_value)*0.75 + min_value, max_value], 'color': "#7fff00"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=20),
        height=200
    )

    return fig
