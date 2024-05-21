# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import sys
import json
import requests


if __name__ == "__main__":
    url = sys.argv[1]
    url = 'http://' + url + '/api'
    headers = {'Content-Type': 'application/json'}

    # while True:
        # sentence = input("Enter prompt: ")
        # tokens_to_generate = int(eval(input("Enter number of tokens to generate: ")))
    with open('output_texts.txt') as f:
        inputs  = f.read()
    eos_token = "</s>"
    # segmented_inputs = inputs.split(eos_token)
    # print(sentence)

    tokens_to_generate = 1
    # for seg in segmented_inputs:
    index = 1000
    while index < len(inputs):
        seg = inputs[:min(index, len(inputs))]

        # sentence = "</s>".join(segmented_inputs)[:-1000]
        sentence = seg[3:].strip()

        data = {"prompts": [sentence], "tokens_to_generate": tokens_to_generate}
        response = requests.put(url, data=json.dumps(data), headers=headers)
        # print(response.json())

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()['message']}")
        else:
            response = response.json()
            print(f"{min(index, len(inputs))}: Loss:  {response['loss']}")
            size = len(response['text'])
            # print(respon"e['text'][:min(size,200)])
        index+=1000


    
