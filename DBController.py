import psycopg2
import streamlit as st


def get_connection():

    return psycopg2.connect(

        host="",
        port="5432",
        dbname="",
        user="",
        password=""

    )
 