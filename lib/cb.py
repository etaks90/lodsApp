import openai, os, json
from lib.utils import *
logger = logging.getLogger('logger')

def ask_question_cb(q, history, engine):

    print(os.environ["openai__api_type"])
    print(os.environ["openai__api_base"])
    print(os.environ["openai__api_version"])
    print(os.environ["openai__api_key"])
    print(os.environ["openai__engine"])

    openai.api_type = os.getenv("openai__api_type")
    openai.api_base = os.getenv("openai__api_base")
    openai.api_version = os.getenv("openai__api_version")
    openai.api_key = os.getenv("openai__api_key")

    for k in ["api_type", "api_base", "api_version", "api_key"]:
        print(f"{k} : {getattr(openai, k)}")

    # Add user message to history  
    history.append({"role":"user","content":f"{q}"})  
          
    if os.getenv("openai__mock_chatbot") == "1":
        print("MOCKED RESULT")
        history.append({"role":"assistant","content":"MOCKED RESULT"}) 
    else:
        print(f"USE ENGINE {engine}")
        completion = openai.ChatCompletion.create(  
            engine=engine,  
            messages=history,  # Use history instead of message_text  
            temperature=0.7,  
            max_tokens=800,  
            top_p=0.95,  
            frequency_penalty=0,  
            presence_penalty=0,  
            stop=None  
        )  
        # Add assistant message to history  
        history.append({"role":"assistant","content":completion.choices[0].message['content']})  
    
    # problem with size, only take most recent.
    return [history[0], history[-1]]