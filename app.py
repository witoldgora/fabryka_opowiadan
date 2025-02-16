import json
from pathlib import Path
import streamlit as st
from openai import OpenAI
from dotenv import dotenv_values
import time

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

#
# CONVERSATION HISTORY AND DATABASE
#
DEFAULT_PERSONALITY = """
Jesteś pomocnikiem, który odpowiada na wszystkie pytania użytkownika.
Odpowiadaj na pytania w sposób zwięzły i zrozumiały.
""".strip()

DRAFTER_PERSONALITY="""
Jesteś pomocnikiem piarza, który pomaga wymyślić dobry tytuł i  zamienić draft opowieści na plan rozdziałów, zidentyfikowacć kluczowe postacie i wątki.
""".strip()

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

def load_conversation_to_state(conversation):
    st.session_state["id"] = conversation["id"]
    st.session_state["name"] = conversation["name"]
    st.session_state["messages"] = conversation["messages"]
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
            "name": "Konwersacja 1",
            "chatbot_personality": DEFAULT_PERSONALITY,
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
                    "chatbot_personality": DRAFTER_PERSONALITY,
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
    for p in DB_CONVERSATIONS_PATH.glob("*.json"):
        conversation_ids.append(int(p.stem))

    # conversation_ids zawiera wszystkie ID konwersacji
    # następna konwersacja będzie miała ID o 1 większe niż największe ID z listy
    conversation_id = max(conversation_ids) + 1
    personality = DEFAULT_PERSONALITY
    if "chatbot_personality" in st.session_state and st.session_state["chatbot_personality"]:
        personality = st.session_state["chatbot_personality"]

    conversation = {
        "id": conversation_id,
        "name": f"Konwersacja {conversation_id}",
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
    story_draft_personality = DRAFTER_PERSONALITY
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
story_id=load_current_conversation()
load_current_story_draft(story_id)
PROGRAM_NAME="Fabryka Opowiadań"
st.title(f":books: {PROGRAM_NAME}")

#Tas start here
AUTHOR_INPUTS="Szkic opowieści"
TITLE_AND_PLOTS="Tytuł i zarys fabuły"
SCENES="Sceny"
author_inputs, title_and_plots, scenes = st.tabs([AUTHOR_INPUTS, TITLE_AND_PLOTS, SCENES])

with author_inputs:
    st.header(AUTHOR_INPUTS)

    LOAD_FILE_PROMPT="Załaduj plik tekstowy z dysku lokalnego"
    LOAD_FILE_HELP="Nowy plik nadpisze zawartość szkicu"
    uploaded_file=st.file_uploader(LOAD_FILE_PROMPT, type='txt',help=LOAD_FILE_HELP) 
   


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("O co chcesz spytać?")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = chatbot_reply(prompt, memory=st.session_state["messages"][-10:])
        st.markdown(response["content"])

    st.session_state["messages"].append({"role": "assistant", "content": response["content"], "usage": response["usage"]})
    save_current_conversation_messages()

with st.sidebar:
    st.subheader("Aktualna konwersacja")
    total_cost = 0
    for message in st.session_state.get("messages") or []:
        if "usage" in message:
            total_cost += message["usage"]["prompt_tokens"] * PRICING["input_tokens"]
            total_cost += message["usage"]["completion_tokens"] * PRICING["output_tokens"]

    c0, c1 = st.columns(2)
    with c0:
        st.metric("Koszt rozmowy (USD)", f"${total_cost:.4f}")

    with c1:
        st.metric("Koszt rozmowy (PLN)", f"{total_cost * USD_TO_PLN:.4f}")

    st.session_state["name"] = st.text_input(
        "Nazwa konwersacji",
        value=st.session_state["name"],
        key="new_conversation_name",
        on_change=save_current_conversation_name,
    )
    st.session_state["chatbot_personality"] = st.text_area(
        "Osobowość chatbota",
        max_chars=5000,
        height=200,
        value=st.session_state["chatbot_personality"],
        key="new_chatbot_personality",
        on_change=save_current_conversation_personality,
    )

    st.subheader("Konwersacje")
    if st.button("Nowa konwersacja"):
        create_new_conversation()

    # pokazujemy tylko top 5 konwersacji
    conversations = list_conversations()
    sorted_conversations = sorted(conversations, key=lambda x: x["id"], reverse=True)
    for conversation in sorted_conversations[:]:
        c0, c1 = st.columns([10, 3])
        with c0:
            st.write(conversation["name"])

        with c1:
            if st.button("załaduj", key=conversation["id"], disabled=conversation["id"] == st.session_state["id"]):
                switch_conversation(conversation["id"])
