import json
from pathlib import Path
import streamlit as st
from openai import OpenAI
from dotenv import dotenv_values
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup

MAX_MESSAGES = None
def chatbot_reply_with_context(user_prompt, context, memory):
    # dodaj system message
    messages = [
        {
            "role": "system",
            "content": st.session_state["chatbot_personality"],
        },
    ]

    # dodaj kontekst
    if context:
        messages.append({"role": "system", "content": context})



    # dodaj wszystkie wiadomości z pamięci
    for message in memory:
        messages.append({"role": message["role"], "content": message["content"]})

    # dodaj wiadomość użytkownika
    messages.append({"role": "user", "content": user_prompt})

    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    usage = {}
    if response.usage: 
        usage = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "role": "assistant",
        "content": response.choices[0].message.content,
        "usage": usage,
    }

def read_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Sprawdza, czy zapytanie zakończyło się sukcesem
        soup = BeautifulSoup(response.text, 'html.parser')

        # Znajdź wszystkie akapity i dołącz ich tekst
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content, url  # Zwróć treść i URL
    except requests.exceptions.RequestException as e:
        return f"Error fetching the website: {e}", None  # Zwróć błąd i None dla URL

# def read_website(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Sprawdza, czy zapytanie zakończyło się sukcesem
#         soup = BeautifulSoup(response.text, 'html.parser')

#         # Znajdź wszystkie akapity i dołącz ich tekst
#         paragraphs = soup.find_all('p')
#         content = ' '.join([p.get_text() for p in paragraphs])
#         return content
#     except requests.exceptions.RequestException as e:
#         return f"Error fetching the website: {e}"

#AUTHOR_INPUTS=...

model_pricings = {
    "gpt-4o": {
        "input_tokens": 5.00 / 1_000_000,  # per token
        "output_tokens": 15.00 / 1_000_000,  # per token
    },
    "gpt-4o-mini": {
        "input_tokens": 0.150 / 1_000_000,  # per token
        "output_tokens": 0.600 / 1_000_000,  # per token
    }
}
MODEL = "gpt-4o-mini"
USD_TO_PLN = 3.97
PRICING = model_pricings[MODEL]

env = dotenv_values(".env")
 
openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])
 
#read in program dictionary
def load_language_settings(language):
    with open('dictionary.json') as json_file:
        data = json.load(json_file)
        #language = data.get("language", "")
        dictionary = data['dictionaries'][language]
        languages = data.get("dictionaries", {}).keys()
        return dictionary, languages 

def load_language_default():
    with open('defaults.json') as json_file:
        data = json.load(json_file)
        language = data.get("language", "")
        return language
  
#read in program conversation defaults
def load_conversation_defaults(language):
    with open('defaults.json') as json_file:
        data = json.load(json_file)
        defaults = data['conversation'][language]
        
        return defaults
    
#read in program story_draft defaults
def load_story_draft_defaults(language):
    with open('defaults.json') as json_file:
        data = json.load(json_file)
        defaults = data['story_draft'][language]
        
        return defaults
  
#  
# CHATBOT
#
def chatbot_reply(user_prompt, memory):
    # dodaj system message
    messages = [
        {
            "role": "system",
            "content": st.session_state["chatbot_personality"],
        },
    ]
    # dodaj wszystkie wiadomości z pamięci
    for message in memory:
        messages.append({"role": message["role"], "content": message["content"]})

    # dodaj wiadomość użytkownika
    messages.append({"role": "user", "content": user_prompt})

    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    usage = {}
    if response.usage: 
        usage = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "role": "assistant",
        "content": response.choices[0].message.content,
        "usage": usage,
    }
 
 

DB_PATH = Path("db")
DB_CONVERSATIONS_PATH = DB_PATH / "conversations"
# db/
# ├── current.json
# ├── conversations/
# │   ├── 1.json
# │   ├── 2.json
# │   └── ...
DB_SESSIONS_PATH = DB_PATH / "sessions"
# db/
# ├── current.json
# ├── sessions/
# │   ├── 1.json
# │   ├── 2.json
# │   └── ...  
def get_program_language():
    #if not st.session_state["program_language"]:
     #   language = 


    return language



def load_conversation_to_state(conversation):
    st.session_state["id"] = conversation["id"]
    st.session_state["name"] = conversation["name"]
    st.session_state["messages"] = conversation["messages"]
    st.session_state["program_language"] = conversation["language"]
    st.session_state["chatbot_personality"] = conversation["chatbot_personality"]


#load current story draft to state 
def load_story_draft_to_state(story_draft):
    st.session_state["story_id"] = story_draft["id"]
    st.session_state["story_draft_name"] = story_draft["name"]
    st.session_state["story_draft_messages"] = story_draft["messages"]
    st.session_state["story_draft_chatbot_personality"] = story_draft["chatbot_personality"]


def load_current_conversation():
    if not DB_PATH.exists():
        DB_PATH.mkdir()
        DB_CONVERSATIONS_PATH.mkdir()
        conversation_id = 1
        conversation = {
            "id": conversation_id,
            "name": f"{SESSION} 1",
            "language": program_language,
            "chatbot_personality": DEFAULT_CONVERSATION_PERSONALITY,
            "messages": [],

            
        }

        # tworzymy nową konwersację
        with open(DB_CONVERSATIONS_PATH / f"conversation_{conversation_id}.json", "w") as f:
            f.write(json.dumps(conversation, indent=4))
            

        with open(DB_PATH / "current.json", "w") as f:
            f.write(json.dumps({
                "current_conversation_id": conversation_id,
            }, indent=4))

    else:
        # sprawdzamy, która konwersacja jest aktualna
        with open(DB_PATH / "current.json", "r") as f:
            data = json.loads(f.read())
            conversation_id = data["current_conversation_id"]

        # wczytujemy konwersację
        with open(DB_CONVERSATIONS_PATH / f"conversation_{conversation_id}.json", "r") as f:
            conversation = json.loads(f.read())

    load_conversation_to_state(conversation)
    return conversation_id

#Load current story draft 
def load_current_story_draft(story_id):
    DB_STORY_DRAFT_PATH = DB_CONVERSATIONS_PATH / f"story_draft_{story_id}.json"
    if not DB_STORY_DRAFT_PATH.exists():
        
        story_draft = {
                    "id": story_id,
                    "name": f"Story {story_id}",
                    "chatbot_personality": DEFAULT_STORY_DRAFT_PERSONALITY,
                    "messages": [],

            
        } 
        #time.sleep(1)
        # tworzymy nową konwersację
        with open(DB_STORY_DRAFT_PATH, "w") as f:
            f.write(json.dumps(story_draft, indent=4))
            

        with open(DB_PATH / "story_draft_current.json", "w") as f:
            f.write(json.dumps({
                "current_story_id": story_id,
            }, indent=4))

    else:
        # sprawdzamy, która konwersacja jest aktualna
        #with open(DB_PATH / "story_draft_current.json", "r") as f:
        #    data = json.loads(f.read())
         #   story_id = data["current_story_id"]

        # wczytujemy konwersację
        with open(DB_CONVERSATIONS_PATH / f"story_draft_{story_id}.json", "r") as f:
            story_draft = json.loads(f.read())
        #... 
    load_story_draft_to_state(story_draft)
    #st.rerun()


def save_current_conversation_messages():
    conversation_id = st.session_state["id"]
    new_messages = st.session_state["messages"]

    with open(DB_CONVERSATIONS_PATH / f"conversation_{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"conversation_{conversation_id}.json", "w") as f:
        f.write(json.dumps({
            **conversation,
            "messages": new_messages,
        }, indent=4))


def save_current_story_draft_messages():
    story_id = st.session_state["id"]
    story_draft_new_messages = st.session_state["story_draft_messages"]

    with open(DB_CONVERSATIONS_PATH / f"story_draft_{story_id}.json", "r") as f:
        story_draft = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"story_draft_{story_id}.json", "w") as f:
        f.write(json.dumps({
            **story_draft,
            "messages": story_draft_new_messages,
        }, indent=4))


def save_current_conversation_name():
    conversation_id = st.session_state["id"]
    new_conversation_name = st.session_state["new_conversation_name"]

    with open(DB_CONVERSATIONS_PATH / f"conversation_{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"conversation_{conversation_id}.json", "w") as f:
        f.write(json.dumps({
            **conversation,
            "name": new_conversation_name,
        }, indent=4))

def save_current_conversation_language():
    conversation_id = st.session_state["id"]
    new_conversation_language = st.session_state["program_language"]

    with open(DB_CONVERSATIONS_PATH / f"conversation_{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"conversation_{conversation_id}.json", "w") as f:
        f.write(json.dumps({
            **conversation,
            "language": new_conversation_language,
        }, indent=4))

def save_current_story_draft_name():
    story_id = st.session_state["id"]
    new_story_draft_name = st.session_state["new_story_draft_name"]

    with open(DB_CONVERSATIONS_PATH / f"story_draft_{story_id}.json", "r") as f:
        story_draft = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"story_draft_{story_id}.json", "w") as f:
        f.write(json.dumps({
            **story_draft,
            "name": new_story_draft_name,
        }, indent=4))

def save_current_conversation_personality():
    conversation_id = st.session_state["id"]
    new_chatbot_personality = st.session_state["new_chatbot_personality"]

    with open(DB_CONVERSATIONS_PATH / f"conversation_{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"conversation_{conversation_id}.json", "w") as f:
        f.write(json.dumps({
            **conversation,
            "chatbot_personality": new_chatbot_personality,
        }, indent=4))

def save_current_story_draft_personality():
    story_id = st.session_state["id"]
    story_draft_new_chatbot_personality = st.session_state["story_draft_new_chatbot_personality"]

    with open(DB_CONVERSATIONS_PATH / f"story_draft_{story_id}.json", "r") as f:
        story_draft = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"story_draft_{story_id}.json", "w") as f:
        f.write(json.dumps({
            **story_draft,
            "chatbot_personality": story_draft_new_chatbot_personality,
        }, indent=4))

def create_new_conversation():
    # poszukajmy ID dla naszej kolejnej konwersacji
    conversation_ids = []
    for p in DB_CONVERSATIONS_PATH.glob("conversation_*.json"):
        #conversation_ids.append(int(p.stem))


        conversation_num = p.stem.split('_')[1]
        conversation_ids.append(int(conversation_num))

    # conversation_ids zawiera wszystkie ID konwersacji
    # następna konwersacja będzie miała ID o 1 większe niż największe ID z listy
    conversation_id = max(conversation_ids) + 1
    personality = DEFAULT_CONVERSATION_PERSONALITY
    #if "chatbot_personality" in st.session_state and st.session_state["chatbot_personality"]:
    #    personality = st.session_state["chatbot_personality"]
    

    
    
    
    conversation = {
        "id": conversation_id,
        "name": f"{SESSION} {conversation_id}",
        "language": program_language,
        "chatbot_personality": personality,
        "messages": [],
    }

    # tworzymy nową konwersację
    with open(DB_CONVERSATIONS_PATH / f"conversation_{conversation_id}.json", "w") as f:
        f.write(json.dumps(conversation, indent=4))

    # która od razu staje się aktualną
    with open(DB_PATH / "current.json", "w") as f:
        f.write(json.dumps({
            "current_conversation_id": conversation_id,
        }, indent=4))

    load_conversation_to_state(conversation)
    st.rerun()

def create_new_story_draft():
    
    story_id=conversation_id
    story_draft_personality = DEFAULT_STORY_DRAFT_PERSONALITY
    if "story_draft_chatbot_personality" in st.session_state and st.session_state["story_draft_chatbot_personality"]:
        story_draft_personality = st.session_state["story_draft_chatbot_personality"]
    
    story_draft = {
        "id": story_id,
        "name": f"Story {story_id}",
        "chatbot_personality": story_draft_personality,
        "messages": [],
    }

    # tworzymy nową konwersację
    with open(DB_CONVERSATIONS_PATH / f"story_draft_{story_id}.json", "w") as f:
        f.write(json.dumps(story_draft, indent=4))

    # która od razu staje się aktualną
    with open(DB_PATH / "story_draft_current.json", "w") as f:
        f.write(json.dumps({
            "story_draft_current_conversation_id": story_id,
        }, indent=4))

    load_conversation_to_state(story_draft)
    st.rerun()

def switch_conversation(conversation_id):
    with open(DB_CONVERSATIONS_PATH / f"conversation_{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_PATH / "current.json", "w") as f:
        f.write(json.dumps({
            "current_conversation_id": conversation_id,
        }, indent=4))

    load_conversation_to_state(conversation)
    st.rerun()


def switch_story_draft(story_id):
    with open(DB_CONVERSATIONS_PATH / f"story_draft_{story_id}.json", "r") as f:
        story_draft = json.loads(f.read())

    with open(DB_PATH / "story_draft_current.json", "w") as f:
        f.write(json.dumps({
            "current_story_id": story_id,
        }, indent=4))

    load_conversation_to_state(story_draft)
    st.rerun()

def list_conversations():
    conversations = []
    for p in DB_CONVERSATIONS_PATH.glob("conversation_*.json"):
        with open(p, "r") as f:
            conversation = json.loads(f.read())
            conversations.append({
                "id": conversation["id"],
                "name": conversation["name"],
            })

    return conversations

def list_story_draft():
    story_drafts = []
    for p in DB_CONVERSATIONS_PATH.glob("story_draft_*.json"):
        with open(p, "r") as f:
            story_draft = json.loads(f.read())
            story_drafts.append({
                "id": story_draft["id"],
                "name": story_draft["name"],
            })

    return story_draft


#
# MAIN PROGRAM 
#
DEFAULT_PROGRAM_LANGUAGE=load_language_default()
if "program_language" not in st.session_state:
    program_language = DEFAULT_PROGRAM_LANGUAGE
    st.session_state["program_language"]=program_language
else:
    program_language = st.session_state["program_language"]



dictionary, program_languages =load_language_settings(program_language)

DEFAULT_CONVERSATION_PERSONALITY = load_conversation_defaults(program_language)["DEFAULT_CONVERSATION_PERSONALITY"].strip()
DEFAULT_STORY_DRAFT_PERSONALITY= load_story_draft_defaults(program_language)["DEFAULT_STORY_DRAFT_PERSONALITY"].strip()
AUTHOR_INPUTS = dictionary['AUTHOR_INPUTS']
ASSISTENT_CHAT=dictionary["ASSISTENT_CHAT"]
TITLE_AND_PLOTS = dictionary['TITLE_AND_PLOTS']
SCENES = dictionary['SCENES']
PROGRAM_NAME=dictionary["PROGRAM_NAME"]
SELECT_LANGUAGE = dictionary["SELECT_LANGUAGE"]
CURRENT_SESSION = dictionary["CURRENT_SESSION"]
SESSION_NAME = dictionary["SESSION_NAME"]
SESSION_COST_USD = dictionary["SESSION_COST_USD"]
SESSION_COST_PLN = dictionary["SESSION_COST_PLN"]
CHATBOT_PERSONALITY = dictionary["CHATBOT_PERSONALITY"]
SESSION_LIST = dictionary["SESSION_LIST"]
NEW_SESSION = dictionary["NEW_SESSION"]
LOAD = dictionary["LOAD"]
SESSION = dictionary["SESSION"]
LOAD_FILE_PROMPT = dictionary["LOAD_FILE_PROMPT"]
LOAD_FILE_HELP = dictionary["LOAD_FILE_HELP"]
FLAG = dictionary["FLAG"]

story_id=load_current_conversation()
load_current_story_draft(story_id)

st.title(f":books: {PROGRAM_NAME}")


assistent_chat, author_inputs, title_and_plots, scenes = st.tabs([ASSISTENT_CHAT, AUTHOR_INPUTS, TITLE_AND_PLOTS, SCENES])
 
with assistent_chat:

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # st.header("Czytaj zawartość strony internetowej")
    # url_input = st.text_input("Wprowadź adres URL:", "")

    # if st.button("Pobierz zawartość"):
    #     if url_input:
    #         content = read_website(url_input)
    #         st.text_area("Zawartość strony:", value=content, height=300)
    #     else:
    #         st.warning("Proszę wprowadzić adres URL.")

    # prompt = st.chat_input("O co chcesz spytać?")
    # if prompt:
    #     with st.chat_message("user"):
    #         st.markdown(prompt)

    #     st.session_state["messages"].append({"role": "user", "content": prompt})

    #     with st.chat_message("assistant"):
    #         response = chatbot_reply(prompt, memory=st.session_state["messages"][-10:])
    #         st.markdown(response["content"])

    #     st.session_state["messages"].append({"role": "assistant", "content": response["content"], "usage": response["usage"]})
    #     save_current_conversation_messages()

    # st.header("Czytaj zawartość strony internetowej")
    # url_input = st.text_input("Wprowadź adres URL:", "")

    # content = ""
    # if st.button("Pobierz zawartość"):
    #     if url_input:
    #         content = read_website(url_input)
    #         st.text_area("Zawartość strony:", value=content, height=300)
    #     else:
    #         st.warning("Proszę wprowadzić adres URL.")

    # prompt = st.chat_input("O co chcesz spytać?")
    # if prompt:
    #     with st.chat_message("user"):
    #         st.markdown(prompt)

    #     st.session_state["messages"].append({"role": "user", "content": prompt})

    #     with st.chat_message("assistant"):
    #         response = chatbot_reply_with_context(prompt, content, memory=st.session_state["messages"][-10:])
    #         st.markdown(response["content"])

    #     st.session_state["messages"].append({"role": "assistant", "content": response["content"], "usage": response["usage"]})
    #     save_current_conversation_messages()





    # Kontynuuj z chatbotem
    prompt = st.chat_input("O co chcesz spytać?")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state["messages"].append({"role": "user", "content": prompt})
        content, url = st.session_state["content_url"] 
        # Użyj URL jako kontekstu
        context = f"URL: {url}\nTreść strony: {content}"
        with st.chat_message("assistant"):
            response = chatbot_reply_with_context(prompt, context, memory=st.session_state["messages"][MAX_MESSAGES:])
        
        
            st.markdown(response["content"])

        #     st.markdown(
        # f"""
        # <div style="height: 500px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; background-color: #f1f1f1;">
        #     <pre style="white-space: pre-wrap;">{response["content"]}</pre>
        # </div>
        # """,
        # unsafe_allow_html=True)
            

        # st.markdown(
        #     f"""
        #     <div style="height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; background-color: #2e2e2e; color: white;">
        #         <pre style="white-space: pre-wrap; font-family: monospace; color: white;">{response["content"]}</pre>
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )

    # # Użycie st.markdown do stworzenia przewijalnego okna
    #     st.markdown(
    #         f"""
    #         <div style="height: 500px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; background-color: #333; color: white;">
    #             <pre style="white-space: pre-wrap; color: white;">{response["content"]}</pre>
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )
        st.session_state["messages"].append({"role": "assistant", "content": response["content"], "usage": response["usage"]})
        save_current_conversation_messages()
        st.rerun()

with st.sidebar:
    st.subheader(CURRENT_SESSION)
   #select language  
     
    program_language = st.selectbox(f"{FLAG} {SELECT_LANGUAGE}", program_languages, index=list(program_languages).index(program_language))
   
    # Nowa sekcja do wyboru modelu AI
    available_models = list(model_pricings.keys())  # Lista dostępnych modeli
    selected_model = st.selectbox("Wybierz model AI", available_models, index=available_models.index("gpt-4o-mini"))

    # Ustawienie modelu na podstawie wyboru użytkownika
    MODEL = selected_model
    PRICING = model_pricings[MODEL]  # Aktualizuj PRICING na podstawie wybranego modelu

   

    
    total_cost = 0
    for message in st.session_state.get("messages") or []:
        if "usage" in message:
            total_cost += message["usage"]["prompt_tokens"] * PRICING["input_tokens"]
            total_cost += message["usage"]["completion_tokens"] * PRICING["output_tokens"]

    c0, c1 = st.columns(2) 
    with c0: 
        st.metric(SESSION_COST_USD, f"${total_cost:.4f}") 
  
    #with c1:
    #    st.metric(SESSION_COST_PLN, f"{total_cost * USD_TO_PLN:.4f}")


    # st.session_state["chatbot_personality"] = st.text_area(
    #     CHATBOT_PERSONALITY,
    #     max_chars=5000,
    #     height=200,
    #     value=st.session_state["chatbot_personality"],
    #     key="new_chatbot_personality",
    #     on_change=save_current_conversation_personality,
    # )

# # Przycisk resetujący osobowość chatbota
#     if st.button("Resetuj osobowość chatbota"):
#         default_personality = load_conversation_defaults(program_language)["DEFAULT_CONVERSATION_PERSONALITY"].strip()
#         st.session_state["chatbot_personality"] = default_personality
#         save_current_conversation_personality()
#         st.success("Osobowość chatbota została zresetowana.")

# Przycisk resetujący osobowość chatbota
    if st.button("Resetuj osobowość chatbota"):
        default_personality = load_conversation_defaults(program_language)["DEFAULT_CONVERSATION_PERSONALITY"].strip()
        st.session_state["chatbot_personality"] = default_personality
        # Bezpośrednio zaktualizuj wartość w text_area
        st.session_state["new_chatbot_personality"] = default_personality
        save_current_conversation_personality()
        st.success("Osobowość chatbota została zresetowana.")
        st.rerun()

    st.session_state["chatbot_personality"] = st.text_area(
        CHATBOT_PERSONALITY,
        max_chars=5000,
        height=200,
        value=st.session_state.get("new_chatbot_personality", st.session_state["chatbot_personality"]),
        key="new_chatbot_personality",
        on_change=save_current_conversation_personality,
    )

    st.session_state["name"] = st.text_input(
        SESSION_NAME,
        value=st.session_state["name"], 
        key="new_conversation_name",
        on_change=save_current_conversation_name,
    )

    st.subheader(SESSION_LIST)
    if st.button(NEW_SESSION):
        create_new_conversation()

    # pokazujemy tylko top 5 konwersacji
    conversations = list_conversations()
    sorted_conversations = sorted(conversations, key=lambda x: x["id"], reverse=True)
    for conversation in sorted_conversations[:]:
        c0, c1 = st.columns([10, 3])
        with c0:
            st.write(conversation["name"])

        with c1:
            if st.button(LOAD, key=conversation["id"], disabled=conversation["id"] == st.session_state["id"]):
                switch_conversation(conversation["id"])



if st.session_state["program_language"] != program_language:
    st.session_state["program_language"] = program_language
    save_current_conversation_language()
    st.rerun()

with author_inputs:
    st.header(AUTHOR_INPUTS)
    uploaded_file=st.file_uploader(LOAD_FILE_PROMPT, type='txt',help=LOAD_FILE_HELP) 
    
    if uploaded_file is not None:
    # Odczytaj zawartość pliku
        content = uploaded_file.read().decode("utf-8")
    # Wyświetl zawartość pliku
        st.text_area("Zawartość pliku:", value=content, height=300)
 
  
    uploaded_files = st.file_uploader(LOAD_FILE_PROMPT, type='txt', help=LOAD_FILE_HELP, accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Wyświetl nazwę pliku
            st.write(f"**Załadowany plik:** {uploaded_file.name}")

            # Zainicjalizuj przycisk do wyświetlenia zawartości
            if st.button(f"Pokaż zawartość {uploaded_file.name}"):
                # Odczytaj zawartość pliku
                content = uploaded_file.read().decode("utf-8")
                # Wyświetl zawartość pliku w nowym oknie (tekście)
                st.text_area(f"Zawartość pliku: {uploaded_file.name}", value=content, height=300)

    url_input = st.text_input("Wprowadź adres URL:", "")

    content = ""
    url = ""

    if "content_url" not in st.session_state:
        st.session_state["content_url"] = content, url

    if st.button("Pobierz zawartość"):
        if url_input:
            content, url = read_website(url_input)
            st.session_state["content_url"] = content, url
            if url:
                st.text_area("Zawartość strony:", value=content, height=300)
            else:
                st.warning("Nie udało się pobrać treści ze strony.")
        else:
            st.warning("Proszę wprowadzić adres URL.")

