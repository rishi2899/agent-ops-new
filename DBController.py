import psycopg2
import streamlit as st


def get_connection():

    return psycopg2.connect(

        host="agentopsacc.postgres.database.azure.com",
        port="5432",
        dbname="agentopsacc",
        user="phoenixadmin",
        password="agentopsacc"

    )
 