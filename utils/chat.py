import openai

import os
import json


def json_completion(prompt, json_load=True):
    client = _getClient()
    if isinstance(prompt, str):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    elif isinstance(prompt, list):
        if len(prompt) == 1:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt[0]}
            ]
        elif len(prompt) == 2:
            messages = [
                {"role": "system", "content": prompt[0]},
                {"role": "user", "content": prompt[1]}
            ]
        else:
            raise ValueError(f"If you provide a prompt list, len(prompt) should be 1 or 2. Get len(prompt)={len(prompt)}")
    else:
        raise ValueError(f"Prompt type should either be str or list. Get type(prompt)={type(prompt)}")
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    response_str = completion.choices[0].message.content
    response_str = response_str.lstrip("```json").rstrip("```").strip()
    
    if not json_load:
        return response_str
    else:
        try:
            response_json = json.loads(response_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"错误位置: 第{e.lineno}行, 第{e.colno}列")
            print(f"错误详情: {e.msg}")
            response_json = {
                "response": response_str
            }
        return response_json
    
    
    
def _getClient():
    client = openai.OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['OPENAI_BASE_URL']
    )  
    
    return client