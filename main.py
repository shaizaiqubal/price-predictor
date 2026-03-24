import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def gen_data(n_samples=1000):
    np.random.seed(50)
    size = np.random.normal(1400, 50, n_samples)
    price=size*50+ np.random.normal(0, 50, n_samples)
    return pd.DataFrame({"size": size, "price": price})

def train_model():
    df= gen_data()
    X=df[['size']]
    y=df['price']
    #X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
    model=LinearRegression()
    model.fit(X,y)
    return model

def main():
    st.title('Housing price predictor')
    st.write('Input House size')
    model=train_model()
    size=st.number_input('House size (.sqft)',min_value=500,max_value=1600, value=1500)

    df=gen_data()

    if st.button('Predict Price'):
        pred=model.predict(np.array([[size]]))
        st.success(f"Your estimated price is: ${pred[0]:,.2f}" )
        fig=px.scatter(df, x="size", y="price", title = "Size vs House price")
        fig.add_scatter(x=[size], y= [pred[0]],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Prediction')
        st.plotly_chart(fig)

if __name__=='__main__':
    main()