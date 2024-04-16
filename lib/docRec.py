from doctr.models import ocr_predictor
import json, openai, logging, os
from pydantic import BaseModel
from typing import Any, Dict
import pandas as pd
from lib.utils import *

# load only once
model = ocr_predictor(det_arch = 'db_resnet50', reco_arch = "crnn_vgg16_bn", pretrained = True)

logger = logging.getLogger('logger')

import json, openai, logging, os
from pydantic import BaseModel
from typing import Any, Dict
import pandas as pd
from lib.utils import *

# load only once


logger = logging.getLogger('logger')

def remove_geometry_entries(data):
    if isinstance(data, dict):
        return {k: remove_geometry_entries(v) for k, v in data.items() if k not in ("geometry", "confidence", "artefacts") and not (isinstance(v, dict) and "geometry" in v)}
    elif isinstance(data, list):
        return [remove_geometry_entries(item) for item in data]
    else:
        return data
    
def text_representation(data):
    data = json.loads(data)["pages"][0]["blocks"]
    combined_values = []
    for item in data:
        combined_line = []
        for line in item['lines']:
            for word in line['words']:
                combined_line.append(word['value'])
        combined_values.append(" ".join(combined_line))
    return "\n".join(combined_values)

def j_from_img(img_path):
    # loads so long, only load if needed
    try:
        img = DocumentFile.from_images(img_path)
    except:
        from doctr.io import DocumentFile
        img = DocumentFile.from_images(img_path)
    result = model(img)
    output = result.export()
    return json.dumps(output)

def img2txt(img_path):
    j = j_from_img(img_path)
    j = remove_geometry_entries(j)
    return text_representation(j)

class Response(BaseModel):
    role: str
    content: str

def ask_question(q: str, engine, mock) -> Dict[str, Any]:
    logger.debug(f"ASK QUESTION {q}")
    if mock == "1":
        print(f"MOCK RESULTS")
        logger.debug(f"MOCK RESULTS")
        return {'role': 'assistant', 'content': '{\n  "Dishes": [\n    "Yam Iten",\n    "S.Soup Lala Item (XL)",\n    "Fish Item",\n    "Omellete Item",\n    "White Rice",\n    "Vege Item (M)",\n    "Beverage"\n  ],\n  "Total amount": 179.50,\n  "Bill Number": "B1llH-V001-541021",\n  "Server": "113 CASHIER"\n}'}
    
    message_text = [{"role": "system", "content": "You are an AI assistant to answer questions."},
                    {"role": "user", "content": q}]
    
    openai.api_type = os.getenv("openai__api_type")
    openai.api_base = os.getenv("openai__api_base")
    openai.api_version = os.getenv("openai__api_version")
    openai.api_key = os.getenv("openai__api_key")

    for k in ["api_type", "api_base", "api_version", "api_key"]:
        print(f"{k} : {getattr(openai, k)}")

    completion = openai.ChatCompletion.create(
        #engine="gpt-35-turbo",
        #engine="gpt4-testing",gpt-35-turbo-16k
        engine=engine,
        messages=message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    response_content = completion.choices[0].message['content']
    return Response(role="assistant", content=response_content).dict()

def imtxt2df(img_txt, list_fields, engine, mock):
    q = f"Extract {','.join(list_fields)} from info and provide result as json: {img_txt}."
    llm_r = ask_question(q, engine, mock)
    print(llm_r)
    df = pd.DataFrame([json.loads(llm_r["content"])])
    for col in df.columns:
            # Check if the column contains lists
            if isinstance(df[col].iloc[0], list):
                # Explode the column if it contains lists
                df = df.explode(col)
    
    return df

def fpimg2df(fp, list_fields, engine, mock, skip_after_first = None):
    j__df = {}
    for img_path in [os.path.join(fp, f) for f in os.listdir(fp) if os.path.isfile(os.path.join(fp, f))]:
        logger.debug(f"READ IMAGE {img_path}")
        img_txt = img2txt(img_path)
        j__df[img_path] = imtxt2df(img_txt, list_fields, engine, mock)
        j__df[img_path]["img_path"] = img_path
        if skip_after_first:
            break
    
    return pd.concat(j__df.values(), ignore_index=True)