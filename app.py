import json
from pathlib import Path
import streamlit as st
from openai import OpenAI
from dotenv import dotenv_values
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup

#top_p = 0.9
##top_k = 50
#frequency_penalty = 0.5
presence_penalty = 0.5

should_rerun = False

MAX_MESSAGES = None
DEFAULT_TEMPERATURE = 0.7  # Możesz dostosować tę wartość


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
DEFAULT_AI_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 4096*16
DEFAULT_TOKENS = 4096
DEFAULT_MAX_TOP_P = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_FREQUENCY_PENALTY = 2.0
DEFAULT_MIN_FREQUENCY_PENALTY = -2.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_MAX_PRESENCE_PENALTY = 2.0
DEFAULT_MIN_PRESENCE_PENALTY = -2.0
DEFAULT_PRESENCE_PENALTY = 0.0

USD_TO_PLN = 3.97
PRICING = model_pricings[DEFAULT_AI_MODEL]
assistant_temperature=DEFAULT_TEMPERATURE

env = dotenv_values(".env")
 
openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])
 

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
    #messages
    #time.sleep()
    # dodaj wszystkie wiadomości z pamięci
    for message in memory:
        messages.append({"role": message["role"], "content": message["content"]})

    # dodaj wiadomość użytkownika
    messages.append({"role": "user", "content": user_prompt})

    # print("Params:\n")
    # ai_model
    # assistant_temperature
    # max_tokens
    # top_p
    # frequency_penalty
    # presence_penalty

    response = openai_client.chat.completions.create(
        model=ai_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=assistant_temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
        #top_k=top_k,
        n=3,
        #stop = ["\n", "User:", "Assistant:"],  # Przykładowe sekwencje, które mogą kończyć generowanie

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
# Funkcja do wyświetlania wiadomości z przewijaniem
def display_messages():
    all_messages = ""
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            sender = "<div style='color: lightblue;'><strong>👤 User:</strong></div>"
        else:
            sender = "<div style='color: lightgreen;'><strong>🤖 Bot:</strong></div>"

        content = f"<div>{msg['content']}</div>"
        all_messages += f"{sender}{content}<br>"

    # Wyświetlanie wiadomości w divie z przewijaniem
    st.markdown(
        f"""
        <div id="chat" style="height: 1000px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; background-color: #2e2e2e; color: white;">
            <pre style="white-space: pre-wrap; font-family: monospace; color: white;">{all_messages}</pre>
        </div>
        <script>
            var chat = document.getElementById("chat");
            chat.scrollTop = chat.scrollHeight;  // Przewiń na dół
        </script>
        """,
        unsafe_allow_html=True
    )
    


# Funkcja do wyświetlania wiadomości z przewijaniem
def display_messages1():
    all_messages = ""
    # Odwróć kolejność wiadomości, aby najnowsze były na górze
    for msg in reversed(st.session_state["messages"]):
        if msg["role"] == "user":
            sender = "<div style='color: lightblue;'><strong>👤 User:</strong></div>"
        else:
            sender = "<div style='color: lightgreen;'><strong>🤖 Bot:</strong></div>"

        content = f"<div>{msg['content']}</div>"
        all_messages += f"{sender}{content}<br>"

    # Wyświetlanie wiadomości w divie z przewijaniem
    st.markdown(
        f"""
        <div id="chat" style="height: 500px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; background-color: #2e2e2e; color: white;">
            <pre style="white-space: pre-wrap; font-family: monospace; color: white;">{all_messages}</pre>
        </div>
        <script>
            var chat = document.getElementById("chat");
            chat.scrollTop = 0;  // Przewiń na górę
        </script>
        """,
        unsafe_allow_html=True
    )
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


DB_PATH = Path("db")
DB_CONVERSATIONS_PATH = DB_PATH / "conversations"
# db/
# ├── current.json
# ├── conversations/
# │   ├── conversation_1.json
# │   ├── conversation_12.json
# │   └── ...
DB_SESSIONS_PATH = DB_PATH / "sessions"
# db/
# ├── current.json
# ├── sessions/
# │   ├── story_draft_1.json
# │   ├── story_draft_2.json
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

    #conversation["language"]
    #st.session_state["program_language"]
    #program_language = conversation["language"]
    #st.rerun()
    should_rerun = True

#load current story draft to state 
def load_story_draft_to_state(story_draft):
    st.session_state["story_id"] = story_draft["id"]
    st.session_state["story_draft_name"] = story_draft["name"]
    st.session_state["story_draft_messages"] = story_draft["messages"]
    st.session_state["story_draft_chatbot_personality"] = story_draft["chatbot_personality"]
    #st.rerun()
    should_rerun = True

def load_current_conversation():
    if not DB_PATH.exists():
        DB_PATH.mkdir()
        DB_CONVERSATIONS_PATH.mkdir()
        conversation_id = 1
        conversation = {
            "id": conversation_id,
            "name": f"{TEXT_SESSION} 1",
            "language": program_language,
            "chatbot_personality": TEXT_CONVERSATION_PERSONALITY,
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
    #st.rerun()
    should_rerun = True
    return conversation_id

#Load current story draft 
def load_current_story_draft(story_id):
    DB_STORY_DRAFT_PATH = DB_CONVERSATIONS_PATH / f"story_draft_{story_id}.json"
    if not DB_STORY_DRAFT_PATH.exists():
        
        story_draft = {
                    "id": story_id,
                    "name": f"Story {story_id}",
                    "chatbot_personality": TEXT_STORY_DRAFT_PERSONALITY,
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
    should_rerun = True


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
        #st.rerun()
        should_rerun = True


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
    personality = TEXT_CONVERSATION_PERSONALITY
    #if "chatbot_personality" in st.session_state and st.session_state["chatbot_personality"]:
    #    personality = st.session_state["chatbot_personality"]
    #st.rerun()
    should_rerun = True

    
    
    
    conversation = {
        "id": conversation_id,
        "name": f"{TEXT_SESSION} {conversation_id}",
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
    #st.rerun()
    should_rerun = True

def create_new_story_draft():
    
    story_id=conversation_id
    story_draft_personality = TEXT_STORY_DRAFT_PERSONALITY
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
    #st.rerun()
    should_rerun = True

def switch_conversation(conversation_id):
    with open(DB_CONVERSATIONS_PATH / f"conversation_{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_PATH / "current.json", "w") as f:
        f.write(json.dumps({
            "current_conversation_id": conversation_id,
        }, indent=4))
    
    load_conversation_to_state(conversation)
    
    #st.rerun()
    should_rerun = True


def switch_story_draft(story_id):
    with open(DB_CONVERSATIONS_PATH / f"story_draft_{story_id}.json", "r") as f:
        story_draft = json.loads(f.read())

    with open(DB_PATH / "story_draft_current.json", "w") as f:
        f.write(json.dumps({
            "current_story_id": story_id,
        }, indent=4))

    load_conversation_to_state(story_draft)
    #st.rerun()
    should_rerun = True

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

if "ai_model" not in st.session_state:
    ai_model = DEFAULT_AI_MODEL
    st.session_state["ai_model"]=ai_model
else:
    ai_model = st.session_state["ai_model"]

if "assistant_temperature" not in st.session_state:
    ai_model = DEFAULT_AI_MODEL
    st.session_state["assistant_temperature"]=assistant_temperature
else:
    assistant_temperature = st.session_state["assistant_temperature"]

if "max_tokens" not in st.session_state:
    max_tokens = DEFAULT_TOKENS
    st.session_state["max_tokens"]=max_tokens
else:
    max_tokens = st.session_state["max_tokens"]

if "top_p" not in st.session_state:
    top_p = DEFAULT_TOP_P
    st.session_state["top_p"]=top_p
else:
    top_p = st.session_state["top_p"]


if "frequency_penalty" not in st.session_state:
    frequency_penalty = DEFAULT_FREQUENCY_PENALTY
    st.session_state["frequency_penalty"]=frequency_penalty
else:
    frequency_penalty = st.session_state["frequency_penalty"]

if "presence_penalty" not in st.session_state:
    presence_penalty = DEFAULT_FREQUENCY_PENALTY
    st.session_state["presence_penalty"]=presence_penalty
else:
    presence_penalty = st.session_state["presence_penalty"]

   

dictionary, program_languages =load_language_settings(program_language)

TEXT_CONVERSATION_PERSONALITY = load_conversation_defaults(program_language)["TEXT_CONVERSATION_PERSONALITY"].strip() 
TEXT_STORY_DRAFT_PERSONALITY= load_story_draft_defaults(program_language)["TEXT_STORY_DRAFT_PERSONALITY"].strip()
TEXT_AUTHOR_INPUTS = dictionary['TEXT_AUTHOR_INPUTS']
TEXT_ASSISTENT_CHAT=dictionary["TEXT_ASSISTENT_CHAT"]
TEXT_TITLE_AND_PLOTS = dictionary['TEXT_TITLE_AND_PLOTS']
TEXT_SCENES = dictionary['TEXT_SCENES']
TEXT_PROGRAM_NAME=dictionary["TEXT_PROGRAM_NAME"]
TEXT_SELECT_LANGUAGE = dictionary["TEXT_SELECT_LANGUAGE"]
TEXT_CURRENT_SESSION = dictionary["TEXT_CURRENT_SESSION"]
TEXT_SESSION_NAME = dictionary["TEXT_SESSION_NAME"]
TEXT_SESSION_COST_USD = dictionary["TEXT_SESSION_COST_USD"]
TEXT_SESSION_COST_PLN = dictionary["TEXT_SESSION_COST_PLN"]
TEXT_CHATBOT_PERSONALITY = dictionary["TEXT_CHATBOT_PERSONALITY"]
TEXT_SELECT_AI_MODEL = dictionary["TEXT_SELECT_AI_MODEL"]
TEXT_SESSIONS_LIST = dictionary["TEXT_SESSIONS_LIST"]
TEXT_NEW_SESSION = dictionary["TEXT_NEW_SESSION"]
TEXT_LOAD = dictionary["TEXT_LOAD"]
TEXT_SESSION = dictionary["TEXT_SESSION"]
TEXT_LOAD_FILE_PROMPT = dictionary["TEXT_LOAD_FILE_PROMPT"]
TEXT_LOAD_FILE_HELP = dictionary["TEXT_LOAD_FILE_HELP"]
TEXT_FLAG = dictionary["TEXT_FLAG"]
TEXT_SET_TEMPERATURE = dictionary["TEXT_SET_TEMPERATURE"]
TEXT_MAX_TOKENS = dictionary["TEXT_MAX_TOKENS"] 
TEXT_TOP_P = dictionary["TEXT_TOP_P"]
TEXT_FREQUENCY_PENALTY = dictionary["TEXT_FREQUENCY_PENALTY"]
TEXT_PPRESENCE_PENALTY = dictionary["TEXT_PPRESENCE_PENALTY"]

story_id=load_current_conversation()
load_current_story_draft(story_id)

st.title(f":books: {TEXT_PROGRAM_NAME}")


assistent_chat, author_inputs, title_and_plots, scenes = st.tabs([TEXT_ASSISTENT_CHAT, TEXT_AUTHOR_INPUTS, TEXT_TITLE_AND_PLOTS, TEXT_SCENES])
 
with st.sidebar:
    st.subheader(TEXT_CURRENT_SESSION)
   #select language  
     
    program_language = st.selectbox(f"{TEXT_FLAG} {TEXT_SELECT_LANGUAGE}", program_languages, index=list(program_languages).index(program_language))
    if st.session_state["program_language"] != program_language:
        st.session_state["program_language"] = program_language
        save_current_conversation_language()
        #st.rerun()
        should_rerun = True

    # Nowa sekcja do wyboru modelu AI
    available_models = list(model_pricings.keys())  # Lista dostępnych modeli
    ai_model = st.selectbox(TEXT_SELECT_AI_MODEL, available_models, index=available_models.index("gpt-4o-mini"))

     

    # Ustawienie modelu na podstawie wyboru użytkownika
    #DEFAULT_AI_MODEL = ai_model
    PRICING = model_pricings[ai_model]  # Aktualizuj PRICING na podstawie wybranego modelu
 
 
    #DEFAULT_TEMPERATURE
    
    total_cost = 0
    total_prompt_cost = 0
    total_completion_cost = 0
    total_prompt_tokens = 0 
    total_completion_tokens = 0 
    for message in st.session_state.get("messages") or []:
        if "usage" in message:
            total_cost += message["usage"]["prompt_tokens"] * PRICING["input_tokens"]
            total_cost += message["usage"]["completion_tokens"] * PRICING["output_tokens"]
            total_prompt_cost += message["usage"]["prompt_tokens"] * PRICING["input_tokens"]
            total_completion_cost += message["usage"]["completion_tokens"] * PRICING["output_tokens"]


            total_prompt_tokens += message["usage"]["prompt_tokens"] 
            total_completion_tokens +=  message["usage"]["completion_tokens"] 

    c0, c1 = st.columns(2) 
    with c0: 
        st.metric("total_prompt_cost", f"${total_prompt_cost:.4f}")
        st.metric("total_completion_cost", f"${total_completion_cost:.4f}")
        st.metric(TEXT_SESSION_COST_USD, f"${total_cost:.4f}") 
  
    with c1:
    #    st.metric(TEXT_SESSION_COST_PLN, f"{total_cost * USD_TO_PLN:.4f}")
        st.metric("total_prompt_tokens", f"{total_prompt_tokens }")
        st.metric("total_completion_tokens", f"{total_completion_tokens }")
        st.metric("total_tokens", f"{total_prompt_tokens + total_completion_tokens }")


    # Suwak do ustawiania temperatury
    assistant_temperature = st.slider(TEXT_SET_TEMPERATURE, min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.1)

    # Suwak do ustawiania maksymalnej liczby tokenów
    max_tokens = st.slider(TEXT_MAX_TOKENS, min_value=10, max_value=DEFAULT_MAX_TOKENS, value=DEFAULT_TOKENS)

    # Suwak do ustawiania top-p (nucleus sampling)
    top_p = st.slider(TEXT_TOP_P, min_value=0.0, max_value=DEFAULT_MAX_TOP_P, value=DEFAULT_TOP_P, step=0.05)

    # Suwak do ustawiania top-k
    #top_k = st.slider("Top-k", min_value=1, max_value=100, value=50)

    # Suwak do częstotliwości kary
    frequency_penalty = st.slider(TEXT_FREQUENCY_PENALTY, min_value=DEFAULT_MIN_FREQUENCY_PENALTY, max_value=DEFAULT_MAX_FREQUENCY_PENALTY, value=DEFAULT_FREQUENCY_PENALTY, step=0.1)

    # Suwak do kary za obecność
    presence_penalty = st.slider(TEXT_PPRESENCE_PENALTY, min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

    


# Przycisk resetujący osobowość chatbota
    if st.button("Resetuj osobowość chatbota"):
        default_personality = load_conversation_defaults(program_language)["TEXT_CONVERSATION_PERSONALITY"].strip()
        st.session_state["chatbot_personality"] = default_personality
        # Bezpośrednio zaktualizuj wartość w text_area
        st.session_state["new_chatbot_personality"] = default_personality
        #st.rerun()
        save_current_conversation_personality()
        st.success("Osobowość chatbota została zresetowana.")
        time.sleep(1)
        #st.rerun()
        should_rerun = True

    st.session_state["chatbot_personality"] = st.text_area(
        TEXT_CHATBOT_PERSONALITY,
        max_chars=5000,
        height=200,
        value=st.session_state["chatbot_personality"],
        key="new_chatbot_personality",
        on_change=save_current_conversation_personality,
    )

    st.session_state["name"] = st.text_input(
        TEXT_SESSION_NAME,
        value=st.session_state["name"], 
        key="new_conversation_name",
        on_change=save_current_conversation_name,
    )

    st.subheader(TEXT_SESSIONS_LIST)
    if st.button(TEXT_NEW_SESSION):
        create_new_conversation()
        #st.rerun()
        should_rerun = True

    # pokazujemy tylko top 5 konwersacji
    conversations = list_conversations()
    sorted_conversations = sorted(conversations, key=lambda x: x["id"], reverse=True)
    for conversation in sorted_conversations[:]:
        c0, c1 = st.columns([10, 3])
        with c0:
            st.write(conversation["name"])

        with c1:
            if st.button(TEXT_LOAD, key=conversation["id"], disabled=conversation["id"] == st.session_state["id"]):
                switch_conversation(conversation["id"])
                #st.rerun()
                should_rerun = True


with assistent_chat:

    # for message in st.session_state["messages"]:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    # all_messages = ""
    # for i, msg in enumerate(st.session_state["messages"]):
    #     sender = "User" if i % 2 == 0 else "Bot"  # Przykładowe przypisanie nadawcy
    #     all_messages += f"**{sender}:** {msg}\n\n"

    # all_messages = ""
    # for msg in st.session_state["messages"]:
    #     sender = "User" if msg["role"] == "user" else "Bot"
    #     all_messages += f"**{sender}:** {msg['content']}\n\n"
    # all_messages = ""
    # for msg in st.session_state["messages"]:
    #     if msg["role"] == "user":
    #         sender = "\n👤 User\n"
    #     else:
    #         sender = "\n🤖 Bot\n"
    #     all_messages += f"**{sender}:** {msg['content']}\n\n"
    # #all_messages = "\n\n".join([msg['content'] for msg in st.session_state["messages"]])

    # all_messages = ""
    # for msg in st.session_state["messages"]:
    #     if msg["role"] == "user":
    #         sender = "\n\n👤 User\n\n"
    #     else:
    #         sender = "\n\n🤖 Bot\n\n"
    #     all_messages += f"{sender}{msg['content']}" 

    # all_messages = ""
    # for msg in st.session_state["messages"]:
    #     if msg["role"] == "user":
    #         sender = "<div style='color: blue;'><strong>👤 User:</strong></div>"
    #     else:
    #         sender = "<div style='color: green;'><strong>🤖 Bot:</strong></div>"
    #     all_messages += f"{sender}<div>{msg['content']}</div><br><br>"

    # all_messages = ""
    # for msg in st.session_state["messages"]:
    #     if msg["role"] == "user":
    #         sender = "<div style='color: blue;'><strong>👤 User:</strong></div>"
    #         content = f"<div style='color: lightblue;'>{msg['content']}</div>"
    #     else:
    #         sender = "<div style='color: green;'><strong>🤖 Bot:</strong></div>"
    #         content = f"<div style='color: lightgreen;'>{msg['content']}</div>"

    #     all_messages += f"{sender}{content}<br><br>"
    all_messages = ""
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            sender = "<div style='color: lightblue;'><strong>👤 User:</strong></div>"
            content = f"<div style='color: lightblue;'>{msg['content']}</div>"
        else:
            sender = "<div style='color: lightgreen;'><strong>🤖 Bot:</strong></div>"
            content = f"<div style='color: lightgreen;'>{msg['content']}</div>"

        all_messages += f"{sender}{content}<br>"



    # st.markdown(
    #     f"""
    #     <div style="height: 700px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; background-color: #2e2e2e; color: white;">
    #         <pre style="white-space: pre-wrap; font-family: monospace; color: white;">{all_messages}</pre>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )
    # st.markdown(
    #     f"""
    #     <div id="chat" style="height: 700px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; background-color: #2e2e2e; color: white;">
    #         <pre style="white-space: pre-wrap; font-family: monospace; color: white;">{all_messages}</pre>
    #     </div>
    #     <script>
    #         var chat = document.getElementById("chat");
    #         chat.scrollTop = chat.scrollHeight;  // Przewiń na dół
    #     </script>
    #     """,
#         unsafe_allow_html=True
# ) 

    # Wywołanie funkcji do wyświetlania wiadomości
    #st.rerun()
    #display_messages() 
    # if "last_query" in st.session_state:
    #     last_query = st.session_state["last_query"]
    #     # Ogranicz długość do 2 linii
    #     last_query_display = '\n'.join(last_query.splitlines()[:2])  # Weź tylko 2 linie
    #     st.info(f"**Ostatnie zapytanie:** {last_query_display}")
 # Wyciągnij ostatnie zapytanie z st.session_state["messages"]
    if st.session_state["messages"]:
        # Przeszukaj wiadomości, aby znaleźć ostatnie zapytanie od użytkownika
        last_user_query = next((msg["content"] for msg in reversed(st.session_state["messages"]) if msg["role"] == "user"), None)

        if last_user_query:
            # Ogranicz długość do 2 linii
            last_query_display = '\n'.join(last_user_query.splitlines()[:2])  # Weź tylko 2 linie
            st.info(f"**Ostatnie zapytanie:** {last_query_display}")       
    display_messages1() 

    prompt = st.chat_input("O co chcesz spytać?")
    if prompt:
        #with st.chat_message("user"):
        #    st.markdown(prompt)
        # Zapisz zapytanie w stanie sesji
        st.session_state["last_query"] = prompt

        st.session_state["messages"].append({"role": "user", "content": prompt})
        content, url = st.session_state["content_url"]
        context = f"URL: {url}\nTreść strony: {content}"

        response = chatbot_reply_with_context(prompt, context, memory=st.session_state["messages"][MAX_MESSAGES:])

        #with st.chat_message("assistant"):
        #    st.markdown(response["content"])

        st.session_state["messages"].append({"role": "assistant", "content": response["content"], "usage": response["usage"]})
        save_current_conversation_messages()
        
        # Ustaw tylko tu, jeżeli chcesz, aby było potrzebne rerun
        should_rerun = True 
        #st.rerun()
  

with author_inputs:
    st.header(TEXT_AUTHOR_INPUTS)
    uploaded_file=st.file_uploader(TEXT_LOAD_FILE_PROMPT, type='txt',help=TEXT_LOAD_FILE_HELP) 
    
    if uploaded_file is not None:
    # Odczytaj zawartość pliku
        content = uploaded_file.read().decode("utf-8")
    # Wyświetl zawartość pliku
        st.text_area("Zawartość pliku:", value=content, height=300)
 
  
    uploaded_files = st.file_uploader(TEXT_LOAD_FILE_PROMPT, type='txt', help=TEXT_LOAD_FILE_HELP, accept_multiple_files=True)

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


st.session_state["ai_model"] = ai_model
st.session_state["assistant_temperature"] = assistant_temperature
st.session_state["max_tokens"] = max_tokens
st.session_state["top_p"] = top_p
st.session_state["frequency_penalty"] = frequency_penalty
st.session_state["presence_penalty"] = presence_penalty

time.sleep(1)
if should_rerun: st.rerun()

