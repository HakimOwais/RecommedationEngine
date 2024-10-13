import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
import altair as alt  

from utils import (load_data, preprocess_data, compute_svd, top_cosine_similarity,
                   visualize_user_book_matrix, visualize_user_book_matrix_altair, similar_books)
                   

# Streamlit UI
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Home', 'Recommendations', 'Matrix Heatmap'])

if page == 'Home':
    #
    st.markdown(
        """
        <div class='center-content'>
            <img src='https://th.bing.com/th/id/OIP.EnOy3-5IcV9Zi9yJRYdATgAAAA?rs=1&pid=ImgDetMain' alt='Logo' width='400px' style="border-radius: 10px;">
        </div>
        """, 
        unsafe_allow_html=True
    )


    # Floating, multi-colored message
    st.markdown("""
        <div class="floating-text">
            <h3>Welcome to the Book Recommendation System! Use the sidebar to navigate.</h3>
        </div>
        """, unsafe_allow_html=True)

elif page == 'Recommendations':
    st.title('📖 Book Recommendations')
    # Load data
    book_df, ratings_df, user_df = load_data()

    # Preprocess data
    book_user_rating, user_book_matrix = preprocess_data(book_df, ratings_df, user_df)

    # Compute SVD
    all_user_predicted_ratings, Vt = compute_svd(user_book_matrix)

    # User input for book selection
    book_id = st.number_input('Enter the unique ID of a book to get recommendations:', min_value=0, max_value=len(book_user_rating['unique_id_book'].unique()) - 1)
    top_n = st.number_input('Number of recommendations to show:', min_value=1, max_value=10, value=3)

    if st.button('📚 Get Recommendations', key='recommend_button'):
        top_indexes = top_cosine_similarity(Vt.T[:, :50], book_id, top_n)
        recommendations = similar_books(book_user_rating, book_id, top_indexes)
        
        # Display recommendations in a table
        st.write(pd.DataFrame(recommendations))

elif page == 'Matrix Heatmap':
    st.title('🗺️ User-Book Matrix Heatmap')
    # Load data
    book_df, ratings_df, user_df = load_data()

    # Preprocess data
    book_user_rating, user_book_matrix = preprocess_data(book_df, ratings_df, user_df)

    if st.button('Show Heatmap', key='heatmap_button'):
        visualize_user_book_matrix_altair(user_book_matrix)

# Custom CSS for styling with yellow and black combination
st.markdown(
    """
    <style>
    .main {
        background-color: #f9fbfd;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #eff2f7;
    }
    .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .floating-text {
        animation: float 3s ease-in-out infinite;
        background-image: linear-gradient(45deg, yellow, black);
        -webkit-background-clip: text;
        color: transparent;
        font-size: 24px;
        text-align: center;
        margin-top: 20px;
    }
    @keyframes float {
        0% {
            transform: translatey(0px);
        }
        50% {
            transform: translatey(-10px);
        }
        100% {
            transform: translatey(0px);
        }
    }
    h1 {
        color: black;
    }
    button[data-testid="recommend_button"] {
        background-color: yellow;
        color: black;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        cursor: pointer;
    }
    button[data-testid="heatmap_button"] {
        background-color: yellow;
        color: black;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        cursor: pointer;
    }
    button:hover {
        background-color: black;
        color: yellow;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Feedback section
st.sidebar.title('Feedback')
feedback = st.sidebar.text_area('Your feedback:')
if st.sidebar.button('Submit'):
    st.sidebar.write('Thank you for your feedback!')
