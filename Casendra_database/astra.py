from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from dotenv import load_dotenv
import os

load_dotenv() # take environment variables from .env.



def getDBSession():
    """Create and get a Cassandra session"""

    ASTRA_CLIENT_ID = os.getenv("ASTRA_CLIENT_ID")
    ASTRA_CLIENT_SECRET = os.getenv("ASTRA_CLIENT_SECRET")

    cloud_config= {
            'secure_connect_bundle': './Casendra_database/secure-connect-backorder.zip'
    }
    auth_provider = PlainTextAuthProvider(ASTRA_CLIENT_ID, ASTRA_CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()
        
    return session

def createJSONonAstra(sessions, casendra_database):
    """Create a document on Astra using the Document API"""


    insert_query = sessions.prepare("\
                INSERT INTO information.backorder (date_added, national_inv, lead_time, sales_1_month, pieces_past_due, \
                    perf_6_month_avg, local_bo_qty, deck_risk, oe_constraint, ppap_risk, stop_auto_buy, rev_stop, prediction)\
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\
                IF NOT EXISTS\
                ")

    try:
        sessions.execute(insert_query, casendra_database)
        return "Successful"
    except Exception as e: 
        return e





