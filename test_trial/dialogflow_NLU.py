import dialogflow_v2 as dialogflow
import os

credentials_file = 'newagent-fpxjnq-08c3138e5a20.json'  

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(credentials_file)


def create_intent(project_id, display_name, training_phrases_parts):
    """Create an intent of the given intent type."""

    intents_client = dialogflow.IntentsClient()
    parent = intents_client.project_agent_path(project_id)

    for i in range(len(display_name)):
        training_phrases = []
        for training_phrases_part in training_phrases_parts[i]:
            part = dialogflow.types.Intent.TrainingPhrase.Part(
                text=training_phrases_part)
            # Here we create a new training phrase for each provided part.
            training_phrase = dialogflow.types.Intent.TrainingPhrase(parts=[part])
            training_phrases.append(training_phrase)

        #text = dialogflow.types.Intent.Message.Text(text=message_texts)
        #message = dialogflow.types.Intent.Message(text=text)
        try:
            intent = dialogflow.types.Intent(
                display_name=display_name[i],
                training_phrases=training_phrases)

            response = intents_client.create_intent(parent, intent)
            #print('Intent created: {}'.format(response))

        except: 
            print('meet problem in intent: ', display_name[i])
            pass

        continue



def train_agent(project_id):
    client = dialogflow.AgentsClient()

    parent = client.project_path(project_id)

    response = client.train_agent(parent)



def get_intent_id(project_id, name):
    #name = display_name[2]

    intents_client = dialogflow.IntentsClient()
    parent = intents_client.project_agent_path(project_id)

    intents = intents_client.list_intents(parent)

    intent_names = [
        intent.name for intent in intents
        if intent.display_name == name]

    intent_ids = [
        intent_name.split('/')[-1] for intent_name
        in intent_names]

    return intent_ids



def detect_intent(project_id, text_to_be_analyzed):
    DIALOGFLOW_PROJECT_ID = project_id
    DIALOGFLOW_LANGUAGE_CODE = 'en'
    SESSION_ID = '1'

    #text_to_be_analyzed = "transfer money"
    text_to_be_analyzed = text_to_be_analyzed #"set alarm"


    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)

    text_input = dialogflow.types.TextInput(text=text_to_be_analyzed, language_code=DIALOGFLOW_LANGUAGE_CODE)
    query_input = dialogflow.types.QueryInput(text=text_input)

    try:
        response = session_client.detect_intent(session=session, query_input=query_input)
    except InvalidArgument:
        raise

    print("Query text:", response.query_result.query_text)
    print("Detected intent:", response.query_result.intent.display_name)
    print("Detected intent confidence:", response.query_result.intent_detection_confidence)
    #print("Fulfillment text:", response.query_result.fulfillment_text)
    
    return response.query_result.intent.display_name



def delete_all_intents(project_id):
    intents_client = dialogflow.IntentsClient()
    parent = intents_client.project_agent_path(project_id)

    intents = intents_client.list_intents(parent)
    intent_set_bf = [i.display_name for i in intents]
    print('Intent before delete: ', len(intent_set_bf))

    if len(intent_set_bf) != 0:
        intents = intents_client.list_intents(parent)
        response = intents_client.batch_delete_intents(parent, intents)
    
    print('-'*20)
    intents = intents_client.list_intents(parent)
    intent_set_aft = [i.display_name for i in intents]
    print('Intent after delete: ', len(intent_set_aft))
