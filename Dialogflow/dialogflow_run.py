import dialogflow_v2 as dialogflow
import os

credentials_file = 'newagent-fpxjnq-08c3138e5a20.json'  

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(credentials_file)

import dialogflow_NLU



