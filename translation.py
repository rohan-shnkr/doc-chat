import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"
headers = {"Authorization": f"Bearer {os.getenv('HF_ACCESS_TOKEN')}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def translate_text(text, src_lang, tgt_lang):
	if src_lang == tgt_lang:
		return [{'translation_text': text}]
	
	output = query({
			"inputs": text,
			"parameters": {
				"src_lang": src_lang,
				"tgt_lang": tgt_lang
			}
		})

	return output

if __name__ == "__main__":
	print(f"Output: {translate_text('I do not like to eat pizza', 'eng_Latn', 'hin_Deva')}")
